"""
HandObjectGym – Isaac Gym environment that loads an SMPLX right hand
(with virtual prismatic joints for free positioning) together with
GAPartNet articulated objects.

DOF layout (51 total):
  0-2   : virtual_x / virtual_y / virtual_z  (prismatic positioning)
  3-5   : R_Wrist_x / R_Wrist_y / R_Wrist_z  (wrist rotation, intrinsic XYZ)
  6-50  : finger joints (5 fingers × 3 phalanges × 3 axes)

SMPLX right-hand anatomy (rest pose, all DOFs = 0):
  -Y : finger extension direction (fingertips point -Y)
  -Z : palm normal            (palm faces -Z)
  +X : thumb / radial side
  Palm centre ≈ (−0.003, −0.054, −0.004) from R_Wrist origin
"""

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
import math
import numpy as np
import torch
import os
import json
from scipy.spatial.transform import Rotation as Rot

# ──────────────────── DOF constants ────────────────────
N_VIRTUAL   = 3   # prismatic x, y, z
N_WRIST     = 3   # wrist rotation x, y, z
N_FINGER    = 45  # 5 fingers × 3 phalanges × 3 axes
N_HAND_DOFS = N_VIRTUAL + N_WRIST + N_FINGER  # 51

IDX_VIRTUAL = slice(0, 3)
IDX_WRIST   = slice(3, 6)
IDX_FINGER  = slice(6, 51)

# Palm centre in the R_Wrist link frame (from URDF inertial origin)
PALM_OFFSET_LOCAL = np.array([-0.003, -0.054, -0.004])

# Per-finger flexion DOF indices (_x joints that curl the finger)
# Layout per finger: [phalanx1_x, phalanx1_y, phalanx1_z,
#                     phalanx2_x, phalanx2_y, phalanx2_z,
#                     phalanx3_x, phalanx3_y, phalanx3_z]
# The _x component is the main flexion axis.
FINGER_FLEXION = {
    "index":  {"mcp": 6,  "pip": 9,  "dip": 12},
    "middle": {"mcp": 15, "pip": 18, "dip": 21},
    "pinky":  {"mcp": 24, "pip": 27, "dip": 30},
    "ring":   {"mcp": 33, "pip": 36, "dip": 39},
}
# Finger spread DOF indices (the _z joint of phalanx 1)
FINGER_SPREAD = {"index": 8, "middle": 17, "pinky": 26, "ring": 35}

# Thumb DOF indices with meaningful range
THUMB = {
    "abd":  42,  # Thumb1_x  abduction   [-0.785, 0.785]
    "flex": 43,  # Thumb1_y  main flexion [-0.785, 2.269]
    "rot":  44,  # Thumb1_z  rotation     [-1.222, 0.524]
    "pip":  47,  # Thumb2_z  PIP flexion  [-0.960, 0.349]
    "dip":  50,  # Thumb3_z  DIP flexion  [-1.396, 0.262]
}


# ──────────────────── Grasp presets ────────────────────

def _open_hand_targets() -> np.ndarray:
    """Fully open hand — all finger DOFs at 0."""
    return np.zeros(N_HAND_DOFS, dtype=np.float32)


def _pre_shape_targets() -> np.ndarray:
    """
    Natural pre-grasp hand shape: fingers slightly flexed, thumb
    slightly opposed.  Looks ergonomic during approach.
    """
    t = np.zeros(N_HAND_DOFS, dtype=np.float32)
    for f in FINGER_FLEXION.values():
        t[f["mcp"]] = 0.30
        t[f["pip"]] = 0.20
        t[f["dip"]] = 0.10
    t[THUMB["abd"]]  =  0.15
    t[THUMB["flex"]] =  0.40
    t[THUMB["rot"]]  = -0.20
    t[THUMB["pip"]]  = -0.15
    t[THUMB["dip"]]  = -0.20
    return t


def compute_force_closure_targets(handle_thickness: float) -> np.ndarray:
    """
    Compute finger DOF targets that achieve force closure on a bar of
    the given *handle_thickness* (metres).

    Force-closure principle for a cylindrical/bar power grasp:
      • Four fingers wrap from one side (palm side), contacting the bar
        on their palmar surfaces.
      • Thumb opposes from the other side.
      • MCP, PIP, DIP flexion angles are chosen so the finger arcs
        envelope the bar cross-section.

    Flexion angles scale with bar size: thicker bar → more curl.
    Segment lengths (from URDF): proximal ≈ 3.2 cm, middle ≈ 2.3 cm,
    distal ≈ 1.6 cm.  For a bar of half-thickness r, the MCP angle
    for the proximal segment to subtend the bar is roughly
        θ ≈ arcsin(r / L_segment)
    but we add a safety margin for stable force closure.
    """
    t = np.zeros(N_HAND_DOFS, dtype=np.float32)

    # Clamp to reasonable range
    ht = np.clip(handle_thickness, 0.005, 0.06)
    r = ht / 2.0

    # Segment lengths (metres)
    L1, L2, L3 = 0.032, 0.023, 0.016

    # Flexion angles — base + geometry-dependent term + squeeze margin.
    # The base angles are set high enough that the proximal phalanges
    # reach past the handle (which sits well below the MCP joints in a
    # deep wrap-around grip) and the squeeze margin generates inward
    # contact force that resists slip.
    squeeze = 0.20
    mcp_angle = np.clip(0.70 + np.arcsin(min(r / L1, 0.95)) + squeeze, 0.70, 1.55)
    pip_angle = np.clip(0.75 + np.arcsin(min(r / L2, 0.95)) + squeeze, 0.75, 1.60)
    dip_angle = np.clip(0.40 + np.arcsin(min(r / L3, 0.95)) + squeeze, 0.40, 1.20)

    # Four-finger curl
    for f in FINGER_FLEXION.values():
        t[f["mcp"]] = mcp_angle
        t[f["pip"]] = pip_angle
        t[f["dip"]] = dip_angle

    # Thumb opposition — must reach across the bar to the opposite side
    # Thumb flexion (y) drives the tip across; abduction (x) opens the
    # thumb plane; rotation (z) pronates the thumb into contact.
    # Conservative curl to avoid thumb tip clipping through the bar.
    t[THUMB["abd"]]  =  0.35               # moderate abduction
    t[THUMB["flex"]] =  1.2 + r * 8.0      # more flex for thicker bar
    t[THUMB["rot"]]  = -0.45               # pronate for palmar contact
    t[THUMB["pip"]]  = -0.45 - r * 3.0     # PIP curl
    t[THUMB["dip"]]  = -0.60 - r * 3.0     # DIP curl

    # Clamp thumb to joint limits
    t[THUMB["abd"]]  = np.clip(t[THUMB["abd"]],  -0.785, 0.785)
    t[THUMB["flex"]] = np.clip(t[THUMB["flex"]], -0.785, 2.269)
    t[THUMB["rot"]]  = np.clip(t[THUMB["rot"]],  -1.222, 0.524)
    t[THUMB["pip"]]  = np.clip(t[THUMB["pip"]],  -0.960, 0.349)
    t[THUMB["dip"]]  = np.clip(t[THUMB["dip"]],  -1.396, 0.262)

    return t


# ──────────────────── Grasp geometry helpers ────────────────────

def compute_wrist_orientation(handle_long: np.ndarray,
                              handle_normal: np.ndarray) -> np.ndarray:
    """
    Compute wrist Euler angles (intrinsic XYZ) that orient the hand
    ergonomically for grasping a bar handle.

    Desired mapping (hand frame → world frame):
      X_hand  →  handle_long     (bar runs across palm)
      Y_hand  →  handle_normal   (fingers extend −Y = toward object surface)
      Z_hand  →  X_hand × Y_hand (palm faces −Z)

    The wrist-to-knuckle direction (−Y in SMPLX) is kept perpendicular
    to the handle's long axis and parallel to the handle's depth
    direction.  Fingers point toward the object so they can curl
    around the bar; the palm faces upward for an underhand power grasp.

    Returns:
        np.ndarray of shape (3,) — [rx, ry, rz] in radians
    """
    x_axis = handle_long / (np.linalg.norm(handle_long) + 1e-12)
    y_axis = handle_normal / (np.linalg.norm(handle_normal) + 1e-12)

    # Ensure orthogonality
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        # handle_long ∥ handle_normal — pick an arbitrary perpendicular
        arb = np.array([0, 0, 1]) if abs(x_axis[2]) < 0.9 else np.array([1, 0, 0])
        z_axis = np.cross(x_axis, arb)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
    else:
        z_axis /= z_norm
        # Re-orthogonalise y
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

    # Rotation matrix: columns are the hand-frame axes in world coords
    R_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

    # Intrinsic XYZ Euler angles (matches URDF chain: Rx → Ry → Rz)
    rpy = Rot.from_matrix(R_mat).as_euler("XYZ")
    return rpy


def compute_wrist_position(handle_centre: np.ndarray,
                            wrist_euler: np.ndarray) -> np.ndarray:
    """
    Compute the virtual-joint XYZ so that the **palm centre** (not the
    wrist joint) ends up at *handle_centre*.

    virtual_xyz positions R_Wrist_Base, which co-locates with R_Wrist
    (all wrist joint origins are 0,0,0).  The palm centre is offset
    from R_Wrist by PALM_OFFSET_LOCAL in the hand's local frame.

    wrist_world = handle_centre − R_hand @ palm_offset
    """
    R_hand = Rot.from_euler("XYZ", wrist_euler).as_matrix()
    palm_world_offset = R_hand @ PALM_OFFSET_LOCAL
    return handle_centre - palm_world_offset


# # ══════════════════════════════════════════════════════════════
# #  URDF base-frame rotation
# # ══════════════════════════════════════════════════════════════

def get_urdf_base_rotation(gapart_id, asset_root, gapartnet_root):
    """
    Compute the rotation from the GAPartNet canonical frame to the
    URDF base (world) frame by accumulating fixed-joint rotations.

    GAPartNet URDFs place a fixed joint with a non-identity rotation
    (typically rpy="90° 0 -90°") between the ``base`` link and the
    body link.  The mobility_v2.json axes are expressed in the
    canonical frame (body link's frame).  Isaac Gym applies this
    rotation when it loads the URDF, so we must apply the same
    rotation to mobility axis/origin data.

    Returns:
        np.ndarray (3, 3) — rotation matrix  R  such that
        ``p_world = R @ p_canonical``  (before scale and actor pose).
    """
    import xml.etree.ElementTree as ET

    urdf_path = os.path.join(
        asset_root, gapartnet_root, str(gapart_id),
        "mobility_annotation_gapartnet.urdf",
    )
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build parent → [(child, type, rpy)] map
    children_of = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        jtype = joint.attrib.get("type", "fixed")
        origin = joint.find("origin")
        rpy = [0.0, 0.0, 0.0]
        if origin is not None and origin.attrib.get("rpy"):
            rpy = [float(v) for v in origin.attrib["rpy"].split()]
        children_of.setdefault(parent, []).append((child, jtype, rpy))

    # Walk from "base" through fixed joints, accumulating rotation.
    # Stop at the first non-fixed joint (the body link).
    R = np.eye(3)
    current = "base"
    while current in children_of:
        advanced = False
        for child, jtype, rpy in children_of[current]:
            if jtype == "fixed":
                # URDF rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
                R_joint = Rot.from_euler(
                    "ZYX", [rpy[2], rpy[1], rpy[0]]
                ).as_matrix()
                R = R @ R_joint
                current = child
                advanced = True
                break
        if not advanced:
            break

    return R


# ══════════════════════════════════════════════════════════════
#  Joint-type query from GAPartNet mobility data
# ══════════════════════════════════════════════════════════════

def get_mobility_joint_info(gapart_id, link_name, asset_root, gapartnet_root):
    """
    Look up the joint type and axis for a given link from mobility_v2.json.

    For links with a fixed joint (e.g. a handle rigidly attached to a door),
    traverses up the kinematic tree until a non-fixed joint is found.

    Returns:
        dict with keys:
            joint_type  : "hinge" | "slider" | "heavy"
            axis_origin : np.ndarray (3,) — pivot point (object local frame)
            axis_direction : np.ndarray (3,) — rotation/slide axis (object local frame)
            limit       : (float, float) — (lower, upper) in the dataset's units
        or None if no mobility data is found.
    """
    mob_path = os.path.join(
        asset_root, gapartnet_root, str(gapart_id), "mobility_v2.json"
    )
    if not os.path.exists(mob_path):
        return None

    mob = json.loads(open(mob_path).read())
    mob_by_id = {entry["id"]: entry for entry in mob}

    # Extract numeric id from link_name (e.g. "link_0" → 0)
    link_id = int(link_name.split("_")[-1])

    # Walk up the tree: if this link's joint is fixed, follow parent
    visited = set()
    current_id = link_id
    while current_id in mob_by_id and current_id not in visited:
        visited.add(current_id)
        entry = mob_by_id[current_id]
        jtype = entry["joint"]
        if jtype in ("hinge", "slider"):
            jdata = entry["jointData"]
            return {
                "joint_type": jtype,
                "axis_origin": np.array(jdata["axis"]["origin"]),
                "axis_direction": np.array(jdata["axis"]["direction"]),
                "limit": (jdata["limit"]["a"], jdata["limit"]["b"]),
            }
        # Fixed or heavy — try parent
        parent_id = entry.get("parent", -1)
        if parent_id < 0:
            break
        current_id = parent_id

    return None


# ══════════════════════════════════════════════════════════════
#  Handle-finding from GAPartNet annotations
# ══════════════════════════════════════════════════════════════

def find_handle_for_part(target_part_idx: int,
                         cates: list,
                         bboxes: np.ndarray,
                         scale: float,
                         obj_pos: np.ndarray,
                         obj_rot_mat: np.ndarray):
    """
    Given a target part (e.g. slider_drawer), find the associated
    ``line_fixed_handle`` by spatial proximity.

    Returns:
        handle_centre  (3,)  — world-frame centre of the handle bar
        handle_long    (3,)  — unit vector along the bar's long axis
        handle_normal  (3,)  — unit vector pointing outward from the object
        handle_short   (3,)  — unit vector along the bar's short axis
        handle_thickness float — bar thickness in metres (world scale)

    If no ``line_fixed_handle`` is found, falls back to the target
    part's own front face (the original behaviour).
    """
    # World-frame front-face centre of the target part
    target_bbox_w = bboxes[target_part_idx] * scale @ obj_rot_mat.T + obj_pos
    target_front_centre = target_bbox_w[:4].mean(axis=0)

    # Collect all handle annotations
    handle_indices = [
        i for i, c in enumerate(cates) if "handle" in c.lower()
    ]

    if handle_indices:
        # Pick the handle whose centre is closest to the target front face
        best_idx = None
        best_dist = float("inf")
        for hi in handle_indices:
            hbbox_w = bboxes[hi] * scale @ obj_rot_mat.T + obj_pos
            hcentre = hbbox_w.mean(axis=0)
            d = np.linalg.norm(hcentre - target_front_centre)
            if d < best_dist:
                best_dist = d
                best_idx = hi
        bbox_w = bboxes[best_idx] * scale @ obj_rot_mat.T + obj_pos
        source = f"handle (part {best_idx}, {cates[best_idx]})"
    else:
        bbox_w = target_bbox_w
        source = f"target front face (no handle annotation found)"

    print(f"    Grasp target: {source}")

    # Decompose the 8-corner OBB
    #   Vertices 0-3: front face, 4-7: back face  (GAPartNet convention)
    #   Edge 0→1 = long axis, Edge 0→3 = short axis, Edge 4→0 = normal
    front_centre = bbox_w[:4].mean(axis=0)
    back_centre  = bbox_w[4:].mean(axis=0)
    handle_centre = (front_centre + back_centre) / 2.0

    out = bbox_w[0] - bbox_w[4]
    long_ = bbox_w[0] - bbox_w[1]
    short_ = bbox_w[0] - bbox_w[3]

    thickness = np.linalg.norm(out)
    handle_normal = out / (np.linalg.norm(out) + 1e-12)
    handle_long   = long_ / (np.linalg.norm(long_) + 1e-12)
    handle_short  = short_ / (np.linalg.norm(short_) + 1e-12)

    return handle_centre, handle_long, handle_normal, handle_short, thickness


# ══════════════════════════════════════════════════════════════
#  Main class
# ══════════════════════════════════════════════════════════════

class HandObjectGym:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.num_envs = cfgs["num_envs"]
        self.num_per_row = max(1, int(math.sqrt(self.num_envs)))
        self.spacing = cfgs["env_spacing"]
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        self.headless = cfgs["HEADLESS"]

        # Viewer look-at positions are always needed (even without camera
        # sensors) so the viewer knows where to point.
        self.cam_poss = cfgs["cam"]["cam_poss"]
        self.cam_targets = cfgs["cam"]["cam_targets"]

        self.use_cam = cfgs["cam"]["use_cam"]
        if self.use_cam:
            self.cam_w = cfgs["cam"]["cam_w"]
            self.cam_h = cfgs["cam"]["cam_h"]
            self.cam_far_plane = cfgs["cam"]["cam_far_plane"]
            self.cam_near_plane = cfgs["cam"]["cam_near_plane"]
            self.horizontal_fov = cfgs["cam"]["cam_horizontal_fov"]
            self.num_cam = len(self.cam_poss)

        # ── Isaac Gym ──
        self.gym = gymapi.acquire_gym()
        self.args = gymutil.parse_arguments(
            description="HandDrag",
            custom_parameters=[
                {"name": "--mode",     "type": str, "default": ""},
                {"name": "--device",   "type": str, "default": "cuda"},
                {"name": "--headless", "action": "store_true", "default": False},
            ],
        )
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else "cpu"

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 4
        sim_params.physx.num_position_iterations = 12
        sim_params.physx.num_velocity_iterations = 4
        sim_params.physx.contact_offset = 0.005
        sim_params.physx.rest_offset = 0.001

        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            sim_params,
        )
        assert self.sim is not None

        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            assert self.viewer is not None

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        self.prepare_hand_asset()
        self.prepare_table_asset()
        if cfgs["USE_ARTI"]:
            self.prepare_arti_obj_assets()
        self.load_envs()
        self.init_observation()
        self.run_steps(5, refresh_obs=True)

    # ─────────────── Asset preparation ───────────────

    def prepare_hand_asset(self):
        hcfg = self.cfgs["hand"]
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.disable_gravity = True
        opts.collapse_fixed_joints = False
        opts.thickness = 0.001
        opts.vhacd_enabled = True
        opts.vhacd_params = gymapi.VhacdParams()
        opts.vhacd_params.resolution = 100000
        opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        opts.flip_visual_attachments = False

        self.hand_asset = self.gym.load_asset(
            self.sim, hcfg["hand_asset_root"], hcfg["urdf"], opts
        )
        self.hand_num_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        assert self.hand_num_dofs == N_HAND_DOFS, (
            f"Expected {N_HAND_DOFS} DOFs, got {self.hand_num_dofs}"
        )

        props = self.gym.get_asset_dof_properties(self.hand_asset)
        # Virtual prismatic — very stiff
        for i in range(N_VIRTUAL):
            props["driveMode"][i] = gymapi.DOF_MODE_POS
            props["stiffness"][i] = 1e5
            props["damping"][i]   = 1e3
        # Wrist rotation — stiff
        for i in range(N_VIRTUAL, N_VIRTUAL + N_WRIST):
            props["driveMode"][i] = gymapi.DOF_MODE_POS
            props["stiffness"][i] = 5e3
            props["damping"][i]   = 200.0
        # Finger joints — high stiffness for stable grasping
        for i in range(N_VIRTUAL + N_WRIST, N_HAND_DOFS):
            props["driveMode"][i] = gymapi.DOF_MODE_POS
            props["stiffness"][i] = 8e3
            props["damping"][i]   = 200.0
            props["effort"][i]    = 5.0
        self.hand_dof_props = props

        self.hand_default_dof_pos = np.zeros(self.hand_num_dofs, dtype=np.float32)
        self.hand_default_dof_state = np.zeros(
            self.hand_num_dofs, gymapi.DofState.dtype
        )
        self.hand_default_dof_state["pos"] = self.hand_default_dof_pos

        link_dict = self.gym.get_asset_rigid_body_dict(self.hand_asset)
        self.hand_num_links = len(link_dict)
        self.wrist_link_name = "R_Wrist"
        assert self.wrist_link_name in link_dict
        self.wrist_link_index = link_dict[self.wrist_link_name]
        print(f"Hand DOFs: {self.hand_num_dofs}, links: {self.hand_num_links}")

    def prepare_table_asset(self):
        tp = self.cfgs["asset"]["table_pose_p"]
        ts = self.cfgs["asset"]["table_scale"]
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(*tp)
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        self.table_asset = self.gym.create_box(
            self.sim, ts[0], ts[1], ts[2], opts
        )

    def prepare_arti_obj_assets(self):
        acfg = self.cfgs["asset"]
        self.asset_root = acfg["asset_root"]
        self.gapartnet_ids = acfg["arti_gapartnet_ids"]
        self.gapartnet_root = acfg["arti_obj_root"]

        paths = [
            f"{self.gapartnet_root}/{gid}/mobility_annotation_gapartnet.urdf"
            for gid in self.gapartnet_ids
        ]
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.collapse_fixed_joints = True
        opts.armature = 0.005
        opts.vhacd_enabled = True
        opts.vhacd_params = gymapi.VhacdParams()
        opts.vhacd_params.resolution = 100000
        opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        opts.disable_gravity = False
        opts.flip_visual_attachments = False

        self.arti_obj_assets = [
            self.gym.load_asset(self.sim, self.asset_root, p, opts) for p in paths
        ]
        self.arti_obj_asset = self.arti_obj_assets[0]
        self.arti_obj_num_dofs = self.gym.get_asset_dof_count(self.arti_obj_asset)
        self.arti_obj_num_links = len(
            self.gym.get_asset_rigid_body_dict(self.arti_obj_asset)
        )
        print(f"Arti obj DOFs: {self.arti_obj_num_dofs}, "
              f"links: {self.arti_obj_num_links}")

        self.arti_obj_dof_props = self.gym.get_asset_dof_properties(
            self.arti_obj_asset
        )
        self.arti_obj_dof_props["damping"][:] = 10.0
        self.arti_obj_dof_props["driveMode"][:] = gymapi.DOF_MODE_NONE

        self.arti_obj_default_dof_state = np.zeros(
            self.arti_obj_num_dofs, gymapi.DofState.dtype
        )
        self.arti_obj_default_dof_state["pos"] = self.arti_obj_dof_props["lower"]

    # ─────────────── Environment loading ───────────────

    def load_envs(self):
        self.envs = []
        self.hand_actors = []
        self.arti_actors = []
        self.wrist_idxs = []
        self.arti_init_obj_pos_list = []
        self.arti_init_obj_rot_list = []

        acfg = self.cfgs["asset"]
        arti_pose_p = acfg["arti_obj_pose_ps"][0]
        arti_rot = acfg["arti_rotation"]
        arti_scale = acfg["arti_obj_scale"]
        hand_scale = self.cfgs["hand"]["hand_scale"]

        for i in range(self.num_envs):
            env = self.gym.create_env(
                self.sim, self.env_lower, self.env_upper, self.num_per_row
            )
            self.envs.append(env)

            # Hand
            hand_pose = gymapi.Transform()
            hand_pose.p = gymapi.Vec3(0, 0, 0)
            hh = self.gym.create_actor(
                env, self.hand_asset, hand_pose, "hand", i, 2, 0
            )
            self.gym.set_actor_dof_properties(env, hh, self.hand_dof_props)
            self.gym.set_actor_dof_states(
                env, hh, self.hand_default_dof_state, gymapi.STATE_ALL
            )
            self.gym.set_actor_dof_position_targets(
                env, hh, self.hand_default_dof_pos
            )
            self.gym.set_actor_scale(env, hh, hand_scale)
            self.hand_actors.append(hh)

            sp = self.gym.get_actor_rigid_shape_properties(env, hh)
            for s in sp:
                s.friction = 5.0
                s.contact_offset = 0.005
                s.thickness = 0.002
            self.gym.set_actor_rigid_shape_properties(env, hh, sp)

            self.wrist_idxs.append(
                self.gym.find_actor_rigid_body_index(
                    env, hh, self.wrist_link_name, gymapi.DOMAIN_SIM
                )
            )

            # Table
            self.gym.create_actor(
                env, self.table_asset, self.table_pose, "table", i, 0, 0
            )

            # Articulated object
            if self.cfgs["USE_ARTI"]:
                ap = gymapi.Transform()
                ap.p = gymapi.Vec3(*arti_pose_p)
                ap.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0, 0, 1), arti_rot / 180.0 * math.pi
                )
                self.arti_init_obj_pos_list.append(
                    [ap.p.x, ap.p.y, ap.p.z]
                )
                self.arti_init_obj_rot_list.append(
                    [ap.r.x, ap.r.y, ap.r.z, ap.r.w]
                )
                ah = self.gym.create_actor(
                    env, self.arti_obj_asset, ap, "arti_actor", i, 1, 0
                )
                self.gym.set_actor_dof_properties(
                    env, ah, self.arti_obj_dof_props
                )
                self.gym.set_actor_dof_states(
                    env, ah,
                    self.arti_obj_default_dof_state, gymapi.STATE_ALL,
                )
                self.gym.set_actor_scale(env, ah, arti_scale)
                osp = self.gym.get_actor_rigid_shape_properties(env, ah)
                for s in osp:
                    s.friction = 5.0
                    s.contact_offset = 0.02
                    s.thickness = 0.2
                self.gym.set_actor_rigid_shape_properties(env, ah, osp)
                self.arti_actors.append(ah)

            # Cameras
            if self.use_cam:
                if i == 0:
                    self.cams = []
                    self.rgb_tensors = []
                    self.depth_tensors = []
                cp = gymapi.CameraProperties()
                cp.width = self.cam_w
                cp.height = self.cam_h
                cp.far_plane = self.cam_far_plane
                cp.near_plane = self.cam_near_plane
                cp.horizontal_fov = self.horizontal_fov
                cp.enable_tensors = True
                ec, er, ed = [], [], []
                for ci in range(self.num_cam):
                    ch = self.gym.create_camera_sensor(env, cp)
                    self.gym.set_camera_location(
                        ch, env,
                        gymapi.Vec3(*self.cam_poss[ci]),
                        gymapi.Vec3(*self.cam_targets[ci]),
                    )
                    ec.append(ch)
                    er.append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim, env, ch, gymapi.IMAGE_COLOR)))
                    ed.append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim, env, ch, gymapi.IMAGE_DEPTH)))
                self.cams.append(ec)
                self.rgb_tensors.append(er)
                self.depth_tensors.append(ed)

        if not self.headless:
            self.gym.viewer_camera_look_at(
                self.viewer, self.envs[0],
                gymapi.Vec3(*self.cam_poss[0]),
                gymapi.Vec3(*self.cam_targets[0]),
            )
        self.gym.prepare_sim(self.sim)

    # ─────────────── Observation ───────────────

    def init_observation(self):
        self.rb_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        self.dof_states = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )
        self.total_dofs_per_env = int(self.dof_states.shape[0] / self.num_envs)
        self.dof_pos = self.dof_states[:, 0].view(
            self.num_envs, self.total_dofs_per_env, 1
        )
        self.dof_vel = self.dof_states[:, 1].view(
            self.num_envs, self.total_dofs_per_env, 1
        )

    def refresh_observation(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.wrist_pos = self.rb_states[self.wrist_idxs, :3]
        self.wrist_rot = self.rb_states[self.wrist_idxs, 3:7]

    # ─────────────── Hand control ───────────────

    def get_current_hand_targets(self) -> np.ndarray:
        return self.dof_pos[0, :N_HAND_DOFS, 0].cpu().numpy().copy()

    def set_hand_dof_targets(self, targets: np.ndarray):
        pa = torch.zeros(
            self.num_envs, self.total_dofs_per_env, device=self.device
        )
        pa[:] = self.dof_pos.squeeze(-1)
        pa[:, :N_HAND_DOFS] = torch.tensor(
            targets, dtype=torch.float32, device=self.device
        )
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(pa)
        )

    def set_hand_pose(self, xyz, wrist_rpy=None):
        """Position the hand via virtual joints; optionally set wrist rotation."""
        t = self.get_current_hand_targets()
        t[0], t[1], t[2] = xyz[0], xyz[1], xyz[2]
        if wrist_rpy is not None:
            t[3], t[4], t[5] = wrist_rpy[0], wrist_rpy[1], wrist_rpy[2]
        self.set_hand_dof_targets(t)

    def set_finger_targets(self, finger_targets: np.ndarray):
        """
        Set finger DOF targets (45-dim) while preserving current
        position and wrist.
        """
        t = self.get_current_hand_targets()
        t[IDX_FINGER] = finger_targets[IDX_FINGER]
        self.set_hand_dof_targets(t)

    def set_full_hand_state(self, xyz, wrist_rpy, finger_targets):
        """Set position, wrist orientation, and finger DOFs in one call."""
        t = np.zeros(N_HAND_DOFS, dtype=np.float32)
        t[0], t[1], t[2] = xyz[0], xyz[1], xyz[2]
        t[3], t[4], t[5] = wrist_rpy[0], wrist_rpy[1], wrist_rpy[2]
        t[IDX_FINGER] = finger_targets[IDX_FINGER]
        self.set_hand_dof_targets(t)

    def open_hand(self):
        t = self.get_current_hand_targets()
        t[IDX_FINGER] = _open_hand_targets()[IDX_FINGER]
        self.set_hand_dof_targets(t)

    def pre_shape_hand(self):
        t = self.get_current_hand_targets()
        t[IDX_FINGER] = _pre_shape_targets()[IDX_FINGER]
        self.set_hand_dof_targets(t)

    def close_hand(self, handle_thickness: float = 0.02):
        t = self.get_current_hand_targets()
        fc = compute_force_closure_targets(handle_thickness)
        t[IDX_FINGER] = fc[IDX_FINGER]
        self.set_hand_dof_targets(t)

    # ─────────────── GAPartNet annotations ───────────────

    def get_gapartnet_anno(self):
        self.gapart_raw_valid_annos = []
        self.gapart_init_bboxes = []
        self.gapart_cates = []
        self.gapart_link_names = []
        for gid in self.gapartnet_ids:
            anno_path = os.path.join(
                self.asset_root, self.gapartnet_root, str(gid),
                "link_annotation_gapartnet.json",
            )
            anno = json.loads(open(anno_path).read())
            valid = [a for a in anno if a["is_gapart"]]
            self.gapart_raw_valid_annos.append(valid)
            self.gapart_cates.append([a["category"] for a in valid])
            self.gapart_init_bboxes.append(
                np.array([np.asarray(a["bbox"]) for a in valid])
            )
            self.gapart_link_names.append([a["link_name"] for a in valid])

    # ─────────────── Simulation ───────────────

    def run_steps(self, n: int = 1, refresh_obs: bool = True):
        for _ in range(n):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            if not self.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        if refresh_obs:
            self.refresh_observation()

    def save_camera_image(self, path, env_idx=0, cam_idx=0):
        self.gym.render_all_camera_sensors(self.sim)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.gym.write_camera_image_to_file(
            self.sim, self.envs[env_idx], self.cams[env_idx][cam_idx],
            gymapi.IMAGE_COLOR, path,
        )

    # ─────────────── Data collection ───────────────

    def get_hand_state(self) -> dict:
        self.refresh_observation()
        hd = self.dof_pos[0, :N_HAND_DOFS, 0].cpu().numpy()
        return {
            "virtual_xyz":  hd[:3].tolist(),
            "wrist_rpy":    hd[3:6].tolist(),
            "finger_dofs":  hd[6:].tolist(),
            "wrist_pos":    self.wrist_pos[0].cpu().numpy().tolist(),
            "wrist_rot":    self.wrist_rot[0].cpu().numpy().tolist(),
        }

    def get_arti_state(self) -> dict:
        return {
            "joint_positions":
                self.dof_pos[0, N_HAND_DOFS:, 0].cpu().numpy().tolist()
        }

    def clean_up(self):
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
