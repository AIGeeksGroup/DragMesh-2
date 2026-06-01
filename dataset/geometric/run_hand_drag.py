"""
Run a hand-drag interaction on GAPartNet articulated objects.

Pipeline
--------
1. Load scene: SMPLX hand (virtual joints) + table + articulated object
2. GAPartNet annotations → locate the *handle* of a draggable part
3. Ergonomic grasp planner → wrist orientation so palm faces handle,
   fingers wrap perpendicular to bar
4. Force-closure grasp → finger DOF targets computed from handle thickness
5. Approach → pre-shape → grasp → drag → release
6. Save per-frame hand & object state

Usage
-----
    python dataset/geometric/run_hand_drag.py                     # with viewer
    python dataset/geometric/run_hand_drag.py --headless          # batch mode
"""

import argparse
import json
import os
import sys

# Isaac Gym MUST be imported before torch
from isaacgym import gymapi  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation as Rot

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dataset.geometric.hand_object_gym import (
    HandObjectGym,
    N_HAND_DOFS, IDX_FINGER,
    find_handle_for_part,
    get_mobility_joint_info,
    get_urdf_base_rotation,
    compute_wrist_orientation,
    compute_wrist_position,
    compute_force_closure_targets,
    _pre_shape_targets,
    _open_hand_targets,
)
import yaml


def read_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def lerp(a, b, t):
    """Linear interpolation between arrays *a* and *b*."""
    return a * (1.0 - t) + b * t


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env/hand_config.yaml")
    parser.add_argument("--headless", action="store_true", default=False)
    args, _ = parser.parse_known_args()

    cfgs = read_yaml_config(args.config)
    cfgs["HEADLESS"] = args.headless or cfgs.get("HEADLESS", False)
    drag_cfg = cfgs["drag"]
    save_cfg = cfgs["save"]
    acfg = cfgs["asset"]
    arti_scale = acfg["arti_obj_scale"]
    gapart_ids = [str(gid) for gid in acfg["arti_gapartnet_ids"]]

    for gapart_id in gapart_ids:
        print(f"\n{'='*60}")
        print(f"  Processing GAPartNet object: {gapart_id}")
        print(f"{'='*60}")

        # ── Init ──
        gym = HandObjectGym(cfgs)
        gym.get_gapartnet_anno()
        gym.run_steps(50, refresh_obs=True)

        cates  = gym.gapart_cates[0]
        bboxes = gym.gapart_init_bboxes[0]

        # ── Select target part (slider_drawer preferred) ──
        target_idx = None
        for pi, cat in enumerate(cates):
            if "slider" in cat or "drawer" in cat:
                target_idx = pi
                break
        if target_idx is None:
            for pi, cat in enumerate(cates):
                if "door" in cat:
                    target_idx = pi
                    break
        if target_idx is None:
            target_idx = len(cates) - 1
        print(f"  Target part: {target_idx} ({cates[target_idx]})")

        # ── Locate handle and compute grasp geometry ──
        obj_pos = np.array(gym.arti_init_obj_pos_list[0])
        obj_rot_quat = np.array(gym.arti_init_obj_rot_list[0])
        obj_rot_mat = Rot.from_quat(obj_rot_quat).as_matrix()

        # The URDF has a fixed joint (typically rpy 90° 0 -90°) between
        # "base" and the body link.  The mobility_v2.json axis/origin are
        # in the GAPartNet canonical frame (before this rotation), but
        # Isaac Gym applies it when loading the URDF.  We must apply the
        # same rotation to mobility data (axis_direction, axis_origin).
        # NOTE: bbox annotations already produce correct world positions
        # with just obj_rot_mat, so R_urdf must NOT be applied to bboxes.
        R_urdf = get_urdf_base_rotation(
            gapart_id, acfg["asset_root"], acfg["arti_obj_root"]
        )
        R_mobility = obj_rot_mat @ R_urdf   # for mobility axis/origin only
        print(f"    URDF base rotation loaded (det={np.linalg.det(R_urdf):.4f})")

        handle_centre, handle_long, handle_normal, handle_short, handle_thickness = \
            find_handle_for_part(
                target_idx, cates, bboxes, arti_scale, obj_pos, obj_rot_mat
            )

        print(f"    Handle centre:   {handle_centre}")
        print(f"    Handle normal:   {handle_normal}")
        print(f"    Handle long:     {handle_long}")
        print(f"    Handle thickness: {handle_thickness:.4f} m")

        # ── Query joint type from mobility data ──
        link_names = gym.gapart_link_names[0]
        target_link = link_names[target_idx]
        joint_info = get_mobility_joint_info(
            gapart_id, target_link,
            acfg["asset_root"], acfg["arti_obj_root"],
        )

        if joint_info is not None:
            motion_type = joint_info["joint_type"]
        else:
            # Fallback: infer from part category
            motion_type = "hinge" if "door" in cates[target_idx] else "slider"
        is_revolute = (motion_type == "hinge")
        print(f"    Motion type:     {motion_type} ({'revolute' if is_revolute else 'prismatic'})")

        # ── Compute world-frame joint axis ──
        # Use R_mobility (= obj_rot_mat @ R_urdf) because mobility_v2.json
        # axis/origin are in the GAPartNet canonical frame, not the URDF frame.
        if joint_info is not None:
            axis_local = joint_info["axis_direction"]
            axis_world = R_mobility @ axis_local
            axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)

        if is_revolute and joint_info is not None:
            pivot_local = joint_info["axis_origin"]
            # Transform to world frame (canonical → URDF → world)
            pivot_world = arti_scale * (R_mobility @ pivot_local) + obj_pos
            drag_angle_deg = drag_cfg.get("drag_angle", 60)
            drag_angle_rad = np.radians(drag_angle_deg)
            print(f"    Pivot (world):   {pivot_world.round(4)}")
            print(f"    Axis  (world):   {axis_world.round(4)}")
            print(f"    Drag angle:      {drag_angle_deg}°")

        if not is_revolute and joint_info is not None:                                  
            # Use the actual URDF slide axis instead of the bbox normal.         
            # Ensure the drag direction is consistent with "outward" (opening).         
            slide_axis = axis_world.copy()                                       
            if np.dot(slide_axis, handle_normal) < 0:                            
                slide_axis = -slide_axis                                         
            print(f"    Slide axis (world): {slide_axis.round(4)}")              
            print(f"    (bbox normal was:   {handle_normal.round(4)})")          
            handle_normal = slide_axis 

        # ── Ergonomic wrist orientation ──
        wrist_euler = compute_wrist_orientation(handle_long, handle_normal)

        # ── Offset grasp target so fingers wrap around the handle ──
        # Push the palm centre well past the handle bar so that the
        # handle sits *inside* the grip (between the palm and the curled
        # fingertips) rather than resting on top of the knuckles.
        # The +Z_hand shift must clear all MCP collision geometry
        # (Pinky is worst-case at ~24.5 mm from palm centre).
        R_hand = Rot.from_euler("XYZ", wrist_euler).as_matrix()
        palm_z_world = R_hand[:, 2]          # +Z_hand in world frame
        palm_y_world = R_hand[:, 1]          # +Y_hand in world frame (finger direction)
        handle_centre = (handle_centre
                         + palm_z_world * (handle_thickness / 2.0 + 0.030)
                         + handle_normal * 0.025
                         + palm_y_world * 0.008)

        # ── Wrist position so palm surface contacts the handle ──
        grasp_wrist_xyz = compute_wrist_position(handle_centre, wrist_euler)

        # ── Force-closure finger targets ──
        grasp_fingers = compute_force_closure_targets(handle_thickness)
        open_fingers  = _open_hand_targets()

        print(f"    Wrist euler (XYZ): {np.degrees(wrist_euler).round(1)}°")
        print(f"    Grasp wrist pos:   {grasp_wrist_xyz.round(4)}")

        # ── Output dir ──
        save_dir = os.path.join(save_cfg["root"], gapart_id)
        os.makedirs(save_dir, exist_ok=True)
        trajectory = []

        def record(phase, step):
            if save_cfg["save_hand_state"]:
                fr = {"phase": phase, "step": step}
                fr.update(gym.get_hand_state())
                if save_cfg["save_object_state"]:
                    fr.update(gym.get_arti_state())
                trajectory.append(fr)

        # ═══════════════════════════════════════════
        #  Phase 1 — Place hand at handle with force-closure grip
        # ═══════════════════════════════════════════
        n_settle = drag_cfg["grasp_settle_steps"]
        print(f"\n  Phase 1: Place hand at handle + force-closure ({n_settle} steps)")

        gym.set_full_hand_state(grasp_wrist_xyz, wrist_euler, grasp_fingers)
        for step in range(n_settle):
            gym.run_steps(2, refresh_obs=True)
            record("grasp", step)

        # ═══════════════════════════════════════════
        #  Phase 2 — Drag (linear or rotational)
        # ═══════════════════════════════════════════
        n_drag = drag_cfg["drag_steps"]

        if is_revolute and joint_info is not None:
            # ── Revolute drag: arc trajectory around hinge axis ──
            print(f"  Phase 2: Revolute drag {drag_angle_deg}° ({n_drag} steps)")

            for step in range(n_drag):
                frac = (step + 1) / n_drag
                theta = frac * drag_angle_rad

                # Incremental rotation around the hinge axis
                R_delta = Rot.from_rotvec(theta * axis_world).as_matrix()

                # Rotate handle position around the pivot
                new_handle = pivot_world + R_delta @ (handle_centre - pivot_world)

                # Rotate the wrist orientation by the same rotation
                R_initial = Rot.from_euler("XYZ", wrist_euler).as_matrix()
                R_new = R_delta @ R_initial
                new_wrist_euler = Rot.from_matrix(R_new).as_euler("XYZ")

                xyz = compute_wrist_position(new_handle, new_wrist_euler)
                gym.set_full_hand_state(xyz, new_wrist_euler, grasp_fingers)
                gym.run_steps(2, refresh_obs=True)
                record("drag", step)

                if save_cfg.get("save_video") and step % 5 == 0:
                    gym.save_camera_image(
                        os.path.join(save_dir, "video", f"drag_{step:04d}.png")
                    )
        else:
            # ── Prismatic drag: linear translation along normal ──
            drag_dist = drag_cfg["drag_distance"]
            print(f"  Phase 2: Linear drag {drag_dist:.3f} m ({n_drag} steps)")

            for step in range(n_drag):
                frac = (step + 1) / n_drag
                new_handle = handle_centre + frac * drag_dist * handle_normal
                xyz = compute_wrist_position(new_handle, wrist_euler)
                gym.set_full_hand_state(xyz, wrist_euler, grasp_fingers)
                gym.run_steps(2, refresh_obs=True)
                record("drag", step)

                if save_cfg.get("save_video") and step % 5 == 0:
                    gym.save_camera_image(
                        os.path.join(save_dir, "video", f"drag_{step:04d}.png")
                    )

        # ═══════════════════════════════════════════
        #  Phase 3 — Release & retract
        # ═══════════════════════════════════════════
        n_post = drag_cfg["post_drag_steps"]
        print(f"  Phase 3: Release ({n_post} steps)")

        if is_revolute and joint_info is not None:
            # Final state after full rotation
            R_final = Rot.from_rotvec(drag_angle_rad * axis_world).as_matrix()
            final_handle = pivot_world + R_final @ (handle_centre - pivot_world)
            R_final_wrist = Rot.from_matrix(
                R_final @ Rot.from_euler("XYZ", wrist_euler).as_matrix()
            ).as_euler("XYZ")
            # Retract outward along the rotated handle normal
            final_normal = R_final @ handle_normal
            retract_pos = compute_wrist_position(
                final_handle + 0.15 * final_normal, R_final_wrist
            )
        else:
            final_handle = handle_centre + drag_dist * handle_normal
            R_final_wrist = wrist_euler
            final_normal = handle_normal
            retract_pos = compute_wrist_position(
                final_handle + 0.15 * handle_normal, wrist_euler
            )

        for step in range(n_post):
            frac = (step + 1) / n_post
            fingers = lerp(grasp_fingers, open_fingers, min(frac * 2, 1.0))
            xyz = lerp(
                compute_wrist_position(final_handle, R_final_wrist),
                retract_pos, frac
            )
            gym.set_full_hand_state(xyz, R_final_wrist, fingers)
            gym.run_steps(2, refresh_obs=True)
            record("release", step)

        # ── Save trajectory ──
        traj_path = os.path.join(save_dir, "trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        print(f"\n  Saved {len(trajectory)} frames → {traj_path}")

        if not cfgs["HEADLESS"]:
            print("  Running 500 extra steps for visual inspection …")
            gym.run_steps(500, refresh_obs=False)

        gym.clean_up()
        del gym

    print("\nDone.")


if __name__ == "__main__":
    main()
