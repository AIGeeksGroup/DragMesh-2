"""
rl_games IVecEnv wrapper for HandDragTask.

Bridges HandDragTask (Isaac Gym GPU tensors) to the rl_games PPO runner.
"""

import torch
import numpy as np
from gym import spaces
from rl_games.common import vecenv

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _infer_target_joint_idx_from_trajectory(trajectory_path):
    """Infer the articulated DOF that the trajectory actually moves."""
    if not trajectory_path:
        return None
    import json
    import numpy as _np

    with open(trajectory_path) as f:
        trajectory = json.load(f)
    if not trajectory:
        raise ValueError(f"Empty trajectory file: {trajectory_path}")
    joints = _np.asarray(
        [frame.get("joint_positions", []) for frame in trajectory],
        dtype=_np.float64,
    )
    if joints.ndim != 2 or joints.shape[1] == 0:
        raise ValueError(f"No joint_positions in trajectory: {trajectory_path}")
    motion_range = joints.max(axis=0) - joints.min(axis=0)
    return int(motion_range.argmax())


def _joint_child_link_from_urdf(env_config, object_id, target_joint_idx):
    """Map trajectory DOF index to the moving rigid link in mobility.urdf."""
    if env_config is None or target_joint_idx is None:
        return None
    from pathlib import Path
    import xml.etree.ElementTree as ET

    asset_cfg = env_config.get("asset", {}) if isinstance(env_config, dict) else {}
    asset_root = asset_cfg.get("asset_root")
    arti_root = asset_cfg.get("arti_obj_root", "gapartnet_example")
    if object_id is None:
        ids = asset_cfg.get("arti_gapartnet_ids") or []
        object_id = ids[0] if ids else None
    if not asset_root or object_id is None:
        return None

    obj_dir = Path(asset_root) / str(arti_root) / str(object_id)
    for name in ("mobility.urdf", "mobility_annotation_gapartnet.urdf", "mobility_texture_gapartnet.urdf"):
        urdf = obj_dir / name
        if not urdf.exists():
            continue
        try:
            root = ET.parse(urdf).getroot()
        except Exception as exc:
            print(f"  [auto][WARN] failed to parse URDF {urdf}: {exc}")
            continue
        movable = []
        for joint in root.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            child = joint.find("child")
            if child is None:
                continue
            movable.append((joint.get("name"), child.get("link")))
        if 0 <= int(target_joint_idx) < len(movable):
            joint_name, child_link = movable[int(target_joint_idx)]
            return child_link, joint_name, str(urdf)
    return None


def resolve_handle_link_name(
    env,
    requested=None,
    target_joint_idx=None,
    trajectory_path=None,
    env_config=None,
    object_id=None,
):
    """Resolve the moving part link that corresponds to the trajectory DOF.

    Historical code picked the first drawer/door annotation, which is wrong for
    multi-DOF objects. The safe default is now:
      trajectory -> target_joint_idx -> mobility.urdf child link.
    """
    if requested:
        return requested

    if target_joint_idx is None and trajectory_path:
        try:
            target_joint_idx = _infer_target_joint_idx_from_trajectory(trajectory_path)
        except Exception as exc:
            print(f"  [auto][WARN] failed to infer target_joint_idx from trajectory: {exc}")
            target_joint_idx = None

    cates = env.gapart_cates[0]
    link_names = env.gapart_link_names[0]
    if not cates:
        raise ValueError("No valid GAPartNet annotations found for handle auto-detection")

    mapped = _joint_child_link_from_urdf(env_config, object_id, target_joint_idx)
    if mapped is not None:
        link_name, joint_name, urdf = mapped
        if link_name in link_names:
            cat = cates[link_names.index(link_name)]
            print(
                f"  [auto] handle_link_name = {link_name} ({cat}, "
                f"matched target_joint_idx={target_joint_idx}, joint={joint_name})"
            )
            return link_name
        print(
            f"  [auto][WARN] URDF child link {link_name} for "
            f"target_joint_idx={target_joint_idx} not found in GAPartNet annotations; "
            "falling back to semantic auto-detect"
        )

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
    print(f"  [auto] handle_link_name = {link_name} ({cates[target_idx]}, semantic fallback)")
    return link_name


class HandDragVecEnv(vecenv.IVecEnv):
    """
    IVecEnv adapter — translates between rl_games dict-based interface
    and HandDragTask's tensor-based interface.
    """

    def __init__(self, config_name, num_actors, **kwargs):
        from isaacgym import gymapi  # noqa: F401  (must import before torch)
        from dataset.geometric.hand_object_gym import HandObjectGym
        from ppo.hand_drag_task import HandDragTask

        task_cfg = kwargs.get("task_config", {})

        # -- Build base Isaac Gym environment --
        cfgs = task_cfg["env_config"]
        headless = task_cfg.get("headless", True)
        cfgs["HEADLESS"] = headless
        cfgs["num_envs"] = num_actors

        # Disable camera sensors for RL training — they consume huge GPU
        # memory (64 envs × 2 cams × 1080×720 × 2 tensors ≈ 760 MB)
        # and aren't needed for the observation pipeline.
        # The viewer still works fine without camera sensors.
        cfgs["cam"]["use_cam"] = False

        # Sync the loaded object with the trajectory's object ID so the
        # RSI states match the actual URDF in the scene.
        object_id = task_cfg.get("object_id")
        if object_id is not None:
            cfgs["asset"]["arti_gapartnet_ids"] = [int(object_id)]

        env = HandObjectGym(cfgs)
        env.get_gapartnet_anno()
        env.run_steps(50, refresh_obs=True)
        target_joint_idx = task_cfg.get("target_joint_idx")
        if target_joint_idx is None:
            target_joint_idx = _infer_target_joint_idx_from_trajectory(
                task_cfg["trajectory_path"]
            )
            print(f"  [auto] target_joint_idx = {target_joint_idx} (from trajectory, before task init)")
        handle_link_name = resolve_handle_link_name(
            env,
            task_cfg.get("handle_link_name"),
            target_joint_idx=target_joint_idx,
            trajectory_path=task_cfg["trajectory_path"],
            env_config=cfgs,
            object_id=object_id,
        )

        # -- Wrap with RL task --
        self.task = HandDragTask(
            env,
            trajectory_path=task_cfg["trajectory_path"],
            target_joint_idx=target_joint_idx,
            handle_link_name=handle_link_name,
            include_handle_rot=task_cfg.get("include_handle_rot", True),
            is_eval_mode=task_cfg.get("is_eval_mode", False),
            epoch_log_path=task_cfg.get("epoch_log_path"),
            include_prev_action_in_history=task_cfg.get(
                "include_prev_action_in_history", True,
            ),
            physical_regularization=task_cfg.get("physical_regularization"),
            dynamics_randomization=task_cfg.get("dynamics_randomization"),
            physical_auxiliary=task_cfg.get("physical_auxiliary"),
            aram=task_cfg.get("aram"),
            reconfig_reward=task_cfg.get("reconfig_reward"),
        )

        self.num_envs = num_actors
        self.num_agents = 1  # single-agent

        # TensorBoard writer for per-epoch reward components. Writes into
        # the same summaries/ dir that rl_games uses so every scalar
        # (ours + rl_games') shows up on the same TB run.
        self._tb_writer = None
        self._tb_log_dir = None
        epoch_log_path = task_cfg.get("epoch_log_path")
        if epoch_log_path:
            self._tb_log_dir = os.path.join(
                os.path.dirname(epoch_log_path), "summaries"
            )

        # Gym spaces for rl_games
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.task.obs_dim,), dtype=np.float32,
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.task.act_dim,), dtype=np.float32,
        )
        self.observation_space = self._observation_space
        self.action_space = self._action_space
        self.value_size = 1

    # -- IVecEnv interface --

    def step(self, actions):
        # rl_games unpacks as: obs, rewards, dones, infos = vec_env.step(...)
        # so we must return a plain tuple, NOT a dict.
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.task.device, dtype=torch.float32)
        obs, rew, done, info = self.task.step(actions)
        return obs, rew, done, info

    def reset(self):
        obs = self.task.reset()
        return obs

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Called by rl_games' A2C agent once per training iteration with the
        current env-step count.  We use this as the epoch boundary:
          1. Refresh the RSI curriculum from the true frame count.
          2. Flush per-epoch reward stats via task.end_epoch().
          3. Forward epoch stats to TensorBoard under epoch/* so the
             individual reward components (r_dist, r_task, r_act, ...)
             show up next to rl_games' own scalars.

        rl_games checks hasattr(vec_env, "set_train_info") before calling,
        so simply defining it is enough to be wired in.
        """
        # Keep the curriculum in sync with the runner's frame counter so
        # restarts / checkpoints don't restart the RSI schedule from zero.
        self.task.global_step_counter = int(env_frames)
        stats = self.task.end_epoch()

        if self._tb_log_dir is None:
            return

        if self._tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self._tb_log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=self._tb_log_dir)

        if stats.get("num_episodes", 0) > 0:
            step = int(env_frames)
            for k, v in stats.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    self._tb_writer.add_scalar(f"epoch/{k}", float(v), step)
            self._tb_writer.flush()

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        return {
            "observation_space": self._observation_space,
            "action_space": self._action_space,
        }

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        self.task.env.clean_up()


def create_hand_drag_env(task_config, config_name="hand_drag", num_actors=None, **kwargs):
    """Build a HandDragVecEnv for either training or player mode."""
    if num_actors is None:
        num_actors = task_config["env_config"].get("num_envs", 1)
    kwargs["task_config"] = task_config
    return HandDragVecEnv(config_name, num_actors, **kwargs)


# -- Registration function (called by train.py) --

def register_hand_drag_env(task_config):
    """
    Register HandDragVecEnv with rl_games so the Runner can find it.

    Parameters
    ----------
    task_config : dict
        Must contain:
          env_config       : dict  — the configs/env/hand_config.yaml contents
          trajectory_path  : str   — path to expert trajectory.json
        Optional:
          target_joint_idx : int   (default: auto-detect max trajectory motion)
          handle_link_name : str   (default: auto-detect target part link)
          headless         : bool  (default True)
          is_eval_mode     : bool  (default False)
    """
    def create_env(config_name, num_actors, **kwargs):
        return create_hand_drag_env(
            task_config, config_name=config_name, num_actors=num_actors, **kwargs
        )

    vecenv.register("HAND_DRAG", create_env)
    return lambda **kwargs: create_hand_drag_env(task_config, **kwargs)
