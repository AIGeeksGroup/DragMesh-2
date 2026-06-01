"""
Trajectory-tracking baseline for the HandDrag dataset.

This is not a learned policy and not an object-state replay. It uses the
dataset hand pose at frame t+1 as the next PD target and lets the articulated
object move only through physics/contact.

Usage:
    conda activate isaacgym
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    cd /path/to/DragMesh-2

    python scripts/track_trajectory_baseline.py --object_id 45661 --no-headless

By default this script uses direct hand PD targets, matching how
dataset/geometric/run_hand_drag.py generated the dataset. Use --control_mode delta_action to
exercise the PPO action interface instead.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

# Isaac Gym MUST be imported before torch.
from isaacgym import gymapi, gymtorch  # noqa: F401
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.geometric.hand_object_gym import HandObjectGym, N_HAND_DOFS  # noqa: E402
from ppo.hand_drag_task import HandDragTask  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track dataset hand trajectory through the HandDrag env"
    )
    parser.add_argument("--config", default="configs/env/hand_config.yaml")
    parser.add_argument("--object_id", default="45661")
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--target_joint_idx", type=int, default=None,
                        help="Object DOF to score; default auto-detects max trajectory motion")
    parser.add_argument("--handle_link_name", default=None,
                        help="Rigid body link used for palm-handle distance; default auto-detects the target part")
    parser.add_argument("--sim_steps_per_frame", type=int, default=2)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--hold_final_steps", type=int, default=0)
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Trajectory frame used for initial state")
    parser.add_argument("--phase", default=None,
                        help="Track the first contiguous block of a trajectory phase, e.g. drag")
    parser.add_argument("--control_mode", choices=("pd_target", "delta_action"),
                        default="pd_target")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_episode_length", type=int, default=100000)
    parser.add_argument("--ema_alpha", type=float, default=1.0,
                        help="Action EMA alpha in eval mode; 1 disables EMA")
    parser.add_argument("--break_on_done", action="store_true")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show Isaac Gym viewer")
    parser.add_argument("--log_csv", default=None,
                        help="Optional CSV path for per-step metrics")
    parser.add_argument("--summary_json", default=None,
                        help="Optional JSON path for aggregate metrics")
    return parser.parse_known_args()


def load_config(args):
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["HEADLESS"] = not args.no_headless
    cfg["num_envs"] = 1
    cfg["cam"]["use_cam"] = False
    cfg["asset"]["arti_gapartnet_ids"] = [int(args.object_id)]
    return cfg


def load_trajectory(args):
    if args.trajectory:
        path = Path(args.trajectory)
        print(f"  [trajectory] using (--trajectory): {path}")
    else:
        manifest_path = Path(os.environ.get(
            "HAND_DRAG_MANIFEST",
            str(PROJECT_ROOT / "data" / "manifest.csv"),
        ))
        path = None
        if manifest_path.exists():
            import csv
            with manifest_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if (
                        row.get("object_id") == str(args.object_id)
                        and row.get("enabled", "1").strip() == "1"
                    ):
                        path = Path(row["trajectory"])
                        print(f"  [trajectory] using (manifest:{manifest_path}): {path}")
                        break
        if path is None and os.environ.get("HAND_DRAG_ALLOW_DEFAULT_TRAJECTORY") == "1":
            path = PROJECT_ROOT / "output" / "hand_drag" / args.object_id / "trajectory.json"
            print(f"  [trajectory] using (legacy output/hand_drag fallback): {path}")
        if path is None:
            raise FileNotFoundError(
                f"No canonical trajectory for object_id={args.object_id}. "
                "Pass --trajectory or set HAND_DRAG_MANIFEST."
            )
    with open(path) as f:
        trajectory = json.load(f)
    if len(trajectory) < 2:
        raise ValueError(f"Need at least two trajectory frames: {path}")
    return path, trajectory


def resolve_handle_link_name(env: HandObjectGym, requested: str):
    """Match dataset/geometric/run_hand_drag.py target-part selection for collapsed handles."""
    if requested:
        return requested

    cates = env.gapart_cates[0]
    link_names = env.gapart_link_names[0]
    if not cates:
        raise ValueError("No valid GAPartNet annotations found for handle auto-detection")

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

    link_name = link_names[target_idx]
    print(f"  [auto] handle_link_name = {link_name} ({cates[target_idx]})")
    return link_name


def resolve_frame_window(args, trajectory):
    if args.phase is not None:
        start_frame = None
        for idx, frame in enumerate(trajectory[:-1]):
            if frame.get("phase") == args.phase:
                start_frame = idx
                break
        if start_frame is None:
            phases = sorted({frame.get("phase", "") for frame in trajectory})
            raise ValueError(
                f"Phase {args.phase!r} not found in trajectory; available phases: {phases}"
            )

        end_frame = start_frame + 1
        while (
            end_frame < len(trajectory)
            and trajectory[end_frame].get("phase") == args.phase
        ):
            end_frame += 1
    else:
        start_frame = max(0, min(args.start_frame, len(trajectory) - 2))
        end_frame = len(trajectory)

    if args.max_frames is not None:
        end_frame = min(end_frame, start_frame + max(2, args.max_frames))

    if end_frame - start_frame < 2:
        raise ValueError(
            f"Need at least two frames in selected window, got {start_frame}->{end_frame - 1}"
        )
    return start_frame, end_frame


def set_task_state_from_traj(task: HandDragTask, frame_idx: int):
    """Set hand and object to one dataset frame for initialisation only."""
    env_ids = torch.arange(task.num_envs, device=task.device)
    hand_pos = task.traj_hand[frame_idx].unsqueeze(0).repeat(task.num_envs, 1)
    arti_pos = task.traj_arti[frame_idx].unsqueeze(0).repeat(task.num_envs, 1)

    task.dof_pos[env_ids, :N_HAND_DOFS, 0] = hand_pos
    task.dof_vel[env_ids, :N_HAND_DOFS, 0] = 0.0
    task.dof_pos[env_ids, N_HAND_DOFS:, 0] = arti_pos
    task.dof_vel[env_ids, N_HAND_DOFS:, 0] = 0.0
    task.pos_targets[env_ids, :N_HAND_DOFS] = hand_pos

    hand_ids = task.hand_actor_idxs[env_ids]
    arti_ids = task.arti_actor_idxs[env_ids]
    all_ids = torch.cat([hand_ids, arti_ids])

    task.gym.set_dof_state_tensor_indexed(
        task.sim,
        gymtorch.unwrap_tensor(task.dof_states),
        gymtorch.unwrap_tensor(all_ids),
        len(all_ids),
    )
    task.gym.set_dof_position_target_tensor_indexed(
        task.sim,
        gymtorch.unwrap_tensor(task.pos_targets),
        gymtorch.unwrap_tensor(hand_ids),
        len(hand_ids),
    )

    task.progress_buf.zero_()
    task.actions.zero_()
    task.filtered_actions.zero_()
    task.done_buf.zero_()
    task.compute_observations()
    task.detach_armed_buf[:] = task.palm_to_handle_dist <= task.detach_dist
    task.prev_target_joint[:] = task.dof_pos[
        :, N_HAND_DOFS + task.target_joint_idx, 0
    ]


def simulate_once(task: HandDragTask, action: torch.Tensor):
    task.pre_physics_step(action)
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._render_viewer()
    task.compute_observations()
    reward, done = task.compute_reward()
    return reward, done


def simulate_pd_target_once(task: HandDragTask, target_hand: torch.Tensor):
    """Set the hand PD target directly, then step physics once."""
    task.pos_targets[:, :N_HAND_DOFS] = target_hand
    task.gym.set_dof_position_target_tensor(
        task.sim, gymtorch.unwrap_tensor(task.pos_targets),
    )

    cur_hand = task.dof_pos[:, :N_HAND_DOFS, 0]
    pseudo_action = ((target_hand - cur_hand) / task.max_delta).clamp(-1.0, 1.0)
    task.actions = pseudo_action
    task.filtered_actions.copy_(pseudo_action)

    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._render_viewer()
    task.compute_observations()
    reward, done = task.compute_reward()
    return reward, done


def make_metric_row(task, step, frame_idx, substep, phase, expert_joint, reward, done):
    cur_joint = task.dof_pos[0, N_HAND_DOFS + task.target_joint_idx, 0]
    start_joint = task.traj_arti[0, task.target_joint_idx]
    expert_goal = task.traj_arti[:, task.target_joint_idx].max()
    denom = torch.clamp(expert_goal - start_joint, min=1e-6)
    progress = (cur_joint - start_joint) / denom
    action_abs_max = task.actions[0].abs().max()
    action_l2 = torch.linalg.vector_norm(task.actions[0])

    return {
        "step": step,
        "frame_idx": frame_idx,
        "substep": substep,
        "phase": phase,
        "target_joint": float(cur_joint.item()),
        "expert_joint": float(expert_joint),
        "normalized_progress": float(progress.item()),
        "palm_to_handle_dist": float(task.palm_to_handle_dist[0].item()),
        "reward": float(reward[0].item()),
        "done": int(bool(done[0].item())),
        "success": int(bool(task._success_flag[0].item())),
        "detach_armed": int(bool(task.detach_armed_buf[0].item())),
        "action_abs_max": float(action_abs_max.item()),
        "action_l2": float(action_l2.item()),
    }


def write_summary_json(path, summary):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    args, gym_args = parse_args()
    sys.argv = [sys.argv[0]] + gym_args

    cfg = load_config(args)
    traj_path, trajectory = load_trajectory(args)

    env = HandObjectGym(cfg)
    env.get_gapartnet_anno()
    env.run_steps(50, refresh_obs=True)
    handle_link_name = resolve_handle_link_name(env, args.handle_link_name)

    task = HandDragTask(
        env,
        trajectory_path=str(traj_path),
        target_joint_idx=args.target_joint_idx,
        handle_link_name=handle_link_name,
        include_handle_rot=True,
        is_eval_mode=True,
        epoch_log_path=None,
    )
    task.eval_action_ema_alpha = float(args.ema_alpha)
    task.max_episode_length = int(args.max_episode_length)

    try:
        start_frame, end_frame = resolve_frame_window(args, trajectory)
        set_task_state_from_traj(task, frame_idx=start_frame)
        for _ in range(max(0, args.settle_steps)):
            task.gym.simulate(task.sim)
            task.gym.fetch_results(task.sim, True)
            task._render_viewer()
        task.compute_observations()
        task.prev_target_joint[:] = task.dof_pos[
            :, N_HAND_DOFS + task.target_joint_idx, 0
        ]

        rows = []
        first_done = None
        step = 0

        for frame_idx in range(start_frame + 1, end_frame):
            target_hand = task.traj_hand[frame_idx].unsqueeze(0)
            expert_joint = task.traj_arti[frame_idx, task.target_joint_idx].item()
            phase = trajectory[frame_idx].get("phase", "")

            for substep in range(max(1, args.sim_steps_per_frame)):
                if args.control_mode == "pd_target":
                    reward, done = simulate_pd_target_once(task, target_hand)
                else:
                    cur_hand = task.dof_pos[:, :N_HAND_DOFS, 0]
                    action = (target_hand - cur_hand) / task.max_delta
                    action = action.clamp(-1.0, 1.0)
                    reward, done = simulate_once(task, action)

                row = make_metric_row(
                    task, step, frame_idx, substep, phase, expert_joint, reward, done
                )
                rows.append(row)
                if row["done"] and first_done is None:
                    first_done = dict(row)
                step += 1

                if args.break_on_done and bool(done[0].item()):
                    break
            if args.break_on_done and first_done is not None:
                break

        final_target = task.traj_hand[end_frame - 1].unsqueeze(0)
        for hold_idx in range(max(0, args.hold_final_steps)):
            if args.control_mode == "pd_target":
                reward, done = simulate_pd_target_once(task, final_target)
            else:
                cur_hand = task.dof_pos[:, :N_HAND_DOFS, 0]
                action = ((final_target - cur_hand) / task.max_delta).clamp(-1.0, 1.0)
                reward, done = simulate_once(task, action)
            row = make_metric_row(
                task,
                step,
                end_frame - 1,
                hold_idx,
                "hold_final",
                task.traj_arti[end_frame - 1, task.target_joint_idx].item(),
                reward,
                done,
            )
            rows.append(row)
            if row["done"] and first_done is None:
                first_done = dict(row)
            step += 1
            if args.break_on_done and bool(done[0].item()):
                break

        if args.log_csv:
            log_path = Path(args.log_csv)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        final = rows[-1]
        success_rate = sum(r["success"] for r in rows) / max(1, len(rows))
        mean_dist = sum(r["palm_to_handle_dist"] for r in rows) / max(1, len(rows))
        selected_start_joint = task.traj_arti[start_frame, task.target_joint_idx].item()
        selected_end_joint = task.traj_arti[end_frame - 1, task.target_joint_idx].item()
        selected_expert_delta = selected_end_joint - selected_start_joint
        summary = {
            "object_id": str(args.object_id),
            "trajectory": str(traj_path),
            "control_mode": args.control_mode,
            "target_joint_idx": int(task.target_joint_idx),
            "handle_link_name": handle_link_name,
            "phase": args.phase,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame - 1),
            "num_trajectory_frames": int(len(trajectory)),
            "sim_steps": int(len(rows)),
            "final_target_joint": final["target_joint"],
            "expert_target_joint": final["expert_joint"],
            "selected_expert_delta": float(selected_expert_delta),
            "normalized_progress": final["normalized_progress"],
            "mean_palm_handle_dist": float(mean_dist),
            "success_step_fraction": float(success_rate),
            "first_done": first_done,
        }
        if args.summary_json:
            write_summary_json(args.summary_json, summary)

        print("\nTrajectory tracking baseline")
        print(f"  trajectory: {traj_path}")
        print(f"  control mode: {args.control_mode}")
        print(f"  target joint idx: {task.target_joint_idx}")
        print(f"  handle link: {handle_link_name}")
        print(f"  frames used: {start_frame}->{end_frame - 1}/{len(trajectory) - 1}")
        print(f"  sim steps: {len(rows)}")
        print(f"  final target joint: {final['target_joint']:.6f}")
        print(f"  expert target joint: {final['expert_joint']:.6f}")
        print(f"  selected expert delta: {selected_expert_delta:.6f}")
        print(f"  normalized progress: {final['normalized_progress']:.3f}")
        print(f"  mean palm-handle dist: {mean_dist:.4f}")
        print(f"  success-step fraction: {success_rate:.3f}")
        if abs(selected_expert_delta) < 1e-6:
            print("  warning: selected frame window has near-zero expert joint motion")
        if first_done is not None:
            print(
                "  first done: "
                f"step={first_done['step']} frame={first_done['frame_idx']} "
                f"success={first_done['success']} reward={first_done['reward']:.3f}"
            )
        if args.log_csv:
            print(f"  wrote metrics: {args.log_csv}")
        if args.summary_json:
            print(f"  wrote summary: {args.summary_json}")
    finally:
        env.clean_up()


if __name__ == "__main__":
    main()
