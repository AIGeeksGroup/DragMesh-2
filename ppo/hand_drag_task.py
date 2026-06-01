"""
hand_drag_task.py - PPO Baseline RL Wrapper for Dexterous Hand Drag

RL Formulation
--------------
Observation  (Na=1 example, GLA / Phase 4 layout, dim = 1747)
  hand_dof_pos        [51]   3 virtual + 3 wrist + 45 finger positions
  hand_dof_vel        [51]   corresponding joint velocities
  handle_pos          [ 3]   true handle-center world position
  handle_rot          [ 4]   parent rigid-body orientation quaternion (xyzw)
  palm_to_handle_vec  [ 3]   handle - palm in world frame
  palm_to_handle_dist [ 1]   ||handle - palm||
  arti_dof_pos        [Na]   articulated object joint positions
  arti_dof_vel        [Na]   articulated object joint velocities
  history             [16 * 102]  per-step token = [error_t (51), a_{t-1} (51)]
  Flat-history baseline: history collapses to [16 * 51] (error only, 931-D).

Action  (dim = 51)
  Network outputs in [-1, 1]^51, mapped to delta +/-0.05 rad/m.
  New PD target = clamp(current_pos + delta, dof_lower, dof_upper).

Reward  (scalar, per-env, per-step)
  r_dist  = -0.25 * d + 0.10 * exp(-5d)  weak grasp-maintenance shaping
  r_task  = +250.0 * delta(target_joint) task progress
  r_act   = -0.002 * mean(action^2)      smoothness penalty
  r_time  = -0.05                         survival / stall penalty
  Total   = r_dist + r_task + r_act + r_detach + r_success + r_time

RSI (Reference State Initialisation)
  On episode reset:
    non-RSI -> expert drag-start grasp pose, object still closed
    RSI     -> random pre-success frame sampled from expert trajectory.json
  This PPO-from-grasp baseline decouples grasp acquisition from the pull/drag
  skill, making the RL baseline about opening progress rather than reaching.
"""

import json
import math
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse
from scipy.spatial.transform import Rotation as Rot

import sys
import os
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.geometric.hand_object_gym import (
    HandObjectGym,
    N_HAND_DOFS,
    PALM_OFFSET_LOCAL,
    find_handle_for_part,
)


# ---------------------------------------------------------------------------
# PICA v1 -- physical regularization config parsing
# ---------------------------------------------------------------------------
# A single helper returns a fully-defaulted nested dict. Missing keys -> 0.0
# weights / disabled, so an unset config is bit-identical to old behaviour.
# Exposed at module level so tests / smoke scripts can assert defaults.
def _parse_phys_reg_cfg(cfg):
    """Fill in defaults for the physical_regularization config block.

    Schema (top-level "enabled" gates the whole feature; per-term "weight"
    gates each individual reward component). All weights default to 0.0
    so a config with `enabled: true` but no per-term weights still emits
    zero physical penalty -- we never accidentally penalise unless the
    user opts in explicitly.
    """
    cfg = dict(cfg) if cfg else {}
    enabled = bool(cfg.get("enabled", False))

    def _term(name, defaults):
        sub = dict(cfg.get(name, {}) or {})
        merged = dict(defaults)
        merged.update(sub)
        merged["enabled"] = bool(merged.get("enabled", False))
        merged["weight"] = float(merged.get("weight", 0.0))
        # If the parent flag is off OR the term is off, force weight to 0.
        if not (enabled and merged["enabled"]):
            merged["weight"] = 0.0
        return merged

    return {
        "enabled": enabled,
        "action_bound": _term(
            "action_bound", {"threshold": 0.90, "weight": 0.0}
        ),
        "contact_distance": _term(
            "contact_distance", {"safe_dist": 0.08, "weight": 0.0}
        ),
        "slip_action": _term(
            "slip_action", {"weight": 0.0}
        ),
        "action_smoothness": _term(
            "action_smoothness", {"weight": 0.0}
        ),
    }


# ---------------------------------------------------------------------------
# PICA v2a -- dynamics randomization config parsing
# ---------------------------------------------------------------------------
# Returns a fully-defaulted dict so missing keys collapse to disabled. Per-axis
# `enabled` flags AND the top-level `enabled` flag must both be true for a
# given axis to actually randomize. When everything is disabled the apply
# path is a no-op and the env behaviour is bit-identical to pre-v2a.
def _parse_dyn_rand_cfg(cfg):
    cfg = dict(cfg) if cfg else {}
    enabled = bool(cfg.get("enabled", False))

    def _axis(name, default_range=(1.0, 1.0)):
        sub = dict(cfg.get(name, {}) or {})
        sub_enabled = bool(sub.get("enabled", False))
        rng = sub.get("range", list(default_range))
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = default_range
        if hi < lo:
            lo, hi = hi, lo
        # Force disabled axis to a degenerate [1,1] range so any accidental
        # sampling collapses to scale=1.0 (i.e. nominal dynamics).
        if not (enabled and sub_enabled):
            sub_enabled = False
            lo, hi = 1.0, 1.0
        return {"enabled": sub_enabled, "range": [lo, hi]}

    return {
        "enabled": enabled,
        "damping_scale": _axis("damping_scale"),
        "friction_scale": _axis("friction_scale"),
    }


# ---------------------------------------------------------------------------
# PICA v2b -- physical auxiliary target config parsing
# ---------------------------------------------------------------------------
# Auxiliary targets are *transport-only* additions to the obs vector. The
# network slices them off the tail before the actor / critic forward, so the
# actor never sees them. They are used only as supervision for the GLA aux
# head, which trains the temporal encoder to predict observable physical
# response signals from history.
def _parse_phys_aux_cfg(cfg):
    """Parse physical_auxiliary config. Defaults to disabled.

    Supports two modes:
      mode="current"         -- v2b targets [dq_obj, slip_proxy, tracking_stress]
      mode="causal_horizon"  -- v2c targets [q_response_K, max_dist_K,
                                              detach_proxy_K, tracking_stress]

    "current" is the default for backward-compat with the v2b YAML which has
    no "mode" field. New configs that want v2c targets must explicitly set
    mode: causal_horizon.
    """
    cfg = dict(cfg) if cfg else {}
    enabled = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode", "current")).lower()
    horizon = int(cfg.get("horizon", 5))
    targets = dict(cfg.get("targets", {}) or {})
    weights = dict(cfg.get("weights", {}) or {})

    if mode not in ("current", "causal_horizon"):
        # Be loud about typos; defaulting silently leads to hard-to-find bugs.
        raise ValueError(
            f"physical_auxiliary.mode must be 'current' or 'causal_horizon', "
            f"got {mode!r}"
        )

    if mode == "causal_horizon":
        canonical = ["q_response_K", "max_dist_K", "detach_proxy_K", "tracking_stress"]
        default_w = {
            "q_response_K":    1.0,
            "max_dist_K":      1.0,
            "detach_proxy_K":  0.5,
            "tracking_stress": 0.5,
        }
    else:  # "current" -- v2b
        canonical = ["dq_obj", "slip_proxy", "tracking_stress"]
        default_w = {"dq_obj": 1.0, "slip_proxy": 1.0, "tracking_stress": 1.0}

    keys = []
    target_weights = []
    for k in canonical:
        sub = dict(targets.get(k, {}) or {})
        if enabled and bool(sub.get("enabled", True)):
            keys.append(k)
            target_weights.append(float(weights.get(k, default_w[k])))

    gating = dict(cfg.get("gating", {}) or {})
    warmup = dict(cfg.get("warmup", {}) or {})

    final_enabled = bool(enabled and keys)

    return {
        "enabled": final_enabled,
        "mode": mode,
        "horizon": int(max(1, horizon)),
        "target_keys": keys,
        "target_weights": target_weights,
        "aux_dim": len(keys),
        "gating": {
            "enabled":   bool(gating.get("enabled", False)),
            "d_valid":   float(gating.get("d_valid", 0.10)),
            "sharpness": float(gating.get("sharpness", 80.0)),
        },
        "warmup": {
            "enabled":     bool(warmup.get("enabled", False)),
            "start_epoch": int(warmup.get("start_epoch", 10)),
            "end_epoch":   int(warmup.get("end_epoch", 40)),
            "max_weight":  float(warmup.get("max_weight", 0.01)),
        },
    }


# ---------------------------------------------------------------------------
# PICA v2d -- ARAM-lite (adaptive action saturation penalty) config parser
# ---------------------------------------------------------------------------
# lambda_aram is updated end-of-epoch toward target_clip when nominal success
# is above start_after_success. The penalty fires only inside the gate
# (high-effort, low-resistance, in-contact) so the policy is never punished
# for being timid.
def _parse_aram_cfg(cfg):
    cfg = dict(cfg) if cfg else {}
    enabled = bool(cfg.get("enabled", False))
    rg = dict(cfg.get("resistance_gate", {}) or {})
    return {
        "enabled":              enabled,
        "start_after_success":  float(cfg.get("start_after_success", 0.7)),
        "target_clip":          float(cfg.get("target_clip", 0.7)),
        "lambda_lr":            float(cfg.get("lambda_lr", 0.005)),
        "lambda_max":           float(cfg.get("lambda_max", 1.0)),
        "lambda_init":          float(cfg.get("lambda_init", 0.0)),
        "resistance_gate": {
            "enabled":          bool(rg.get("enabled", True)),
            "min_action_l2":    float(rg.get("min_action_l2", 1.0)),
            "min_stall_steps":  int(rg.get("min_stall_steps", 5)),
            "q_response_min":      float(rg.get("q_response_min", 0.005)),
            "q_response_min_frac": (
                None if rg.get("q_response_min_frac", None) is None
                else float(rg.get("q_response_min_frac"))
            ),
            "q_response_min_mode": str(rg.get("q_response_min_mode", "raw")).lower(),
            "contact_dist":        float(rg.get("contact_dist", 0.10)),
        },
    }


# ---------------------------------------------------------------------------
# PICA v2d -- valid reconfiguration reward config parser
# ---------------------------------------------------------------------------
def _parse_reconfig_cfg(cfg):
    cfg = dict(cfg) if cfg else {}
    enabled = bool(cfg.get("enabled", False))
    return {
        "enabled":                  enabled,
        "contact_dist":             float(cfg.get("contact_dist", 0.10)),
        "min_action_l2":            float(cfg.get("min_action_l2", 1.0)),
        "progress_stall_threshold": float(cfg.get("progress_stall_threshold", 0.02)),
        "cooldown_steps":           int(cfg.get("cooldown_steps", 10)),
        "weight_contact_improve":   float(cfg.get("weight_contact_improve", 0.2)),
        "weight_response_improve":  float(cfg.get("weight_response_improve", 1.0)),
        "weight_slip_reduce":       float(cfg.get("weight_slip_reduce", 0.5)),
        "action_delta_threshold":   float(cfg.get("action_delta_threshold", 4.0)),
        "hand_velocity_threshold":  float(cfg.get("hand_velocity_threshold", 8.0)),
    }


class HandDragTask:
    """
    PPO baseline wrapper around HandObjectGym.
    Core methods: reset_idx, compute_observations, compute_reward, pre_physics_step.
    """

    # ==================== Tuneable Hyper-parameters ====================
    max_delta          = 0.05    # max joint displacement per step (rad / m)
    max_episode_length = 300     # steps before forced episode reset
    w_dist_linear      = 0.25    # weak distance shaping; prevents hover hacking
    w_dist_exp         = 0.10    # weak close-range palm-handle maintenance bonus
    dist_sharpness     = 5.0     # k in exp(-k*dist); larger = sharper near goal
    w_task             = 250.0   # task progress dominates the return
    w_act              = 0.002   # light action L2 smoothness penalty
    r_time_penalty     = -0.05   # per-step stall penalty

    # Detachment / early-termination
    detach_dist        = 0.1     # metres: palm_to_handle dist considered "slipped"
    task_done_frac     = 0.5     # task considered done once joint passes this
                                 # fraction of trajectory max motion
    r_detach_penalty   = -50.0   # massive one-shot penalty when detach fires
    r_success_bonus    = 100.0   # one-shot bonus on the step success is reached
                                 # (episode also terminates that step)

    # Optional RSI curriculum schedule. Default is disabled so the baseline is
    # pure PPO-from-grasp: every non-eval reset starts at drag_start_frame_idx.
    # Raise these values if you explicitly want random expert-frame starts.
    rsi_prob_start     = 0.0
    rsi_prob_end       = 0.0
    rsi_decay_end_frac = 0.5     # decay completes at this fraction of max_steps
    decay_mode         = "linear"   # {"linear", "exponential"}
    # Fallback total env-step budget used when update_rsi_prob is never
    # called externally; lets the env self-pace via self.global_step_counter.
    rsi_decay_steps    = 2_000_000

    # Legacy knob retained for older scripts; PPO-from-grasp reset no longer
    # jitters the non-RSI branch.
    split_rsi_radius   = 0.10

    # Post-reset settling: number of physics sub-steps to advance with the
    # PD target frozen at the freshly-placed RSI pose so that contact
    # forces stabilize before PPO actions start perturbing the state.
    n_settle_substeps  = 4

    # Play-mode action smoothing. Training keeps the raw policy actions so
    # checkpoint behavior remains consistent; eval uses an EMA to reduce
    # visible PD target jitter from saturated mu outputs.
    eval_action_ema_alpha = 0.35

    # =====================================================================
    #  Initialisation (setup context -- not boilerplate, but needed for
    #  the core methods to reference the right tensors)
    # =====================================================================

    def __init__(
        self,
        env: HandObjectGym,
        trajectory_path: str,
        target_joint_idx: int = None,
        handle_link_name: str = "link_1",
        include_handle_rot: bool = True,
        is_eval_mode: bool = False,
        epoch_log_path: str = None,
        include_prev_action_in_history: bool = True,
        physical_regularization: dict = None,
        dynamics_randomization: dict = None,
        physical_auxiliary: dict = None,
        aram: dict = None,
        reconfig_reward: dict = None,
    ):
        """
        Parameters
        ----------
        env : HandObjectGym
            Already-initialised base environment (sim, assets, envs loaded).
        trajectory_path : str
            Path to the expert trajectory.json produced by dataset/geometric/run_hand_drag.py.
        target_joint_idx : int
            Which articulated-object DOF is the task joint (0-indexed).
            For a typical GAPartNet drawer this is 1.
        handle_link_name : str
            Rigid-body link name of the target part in the collapsed URDF.
            With collapse_fixed_joints=True the handle's fixed-joint child
            is merged into its parent; pass that parent's name here.
        """
        # -- References to base env --
        self.env      = env
        self.gym      = env.gym
        self.sim      = env.sim
        self.device   = env.device
        self.num_envs = env.num_envs

        # GPU-resident DOF state tensor views (updated in-place by Isaac Gym)
        self.dof_states = env.dof_states   # [total_dofs, 2]  cols: pos, vel
        self.dof_pos    = env.dof_pos      # [N, D_total, 1]  view into col 0
        self.dof_vel    = env.dof_vel      # [N, D_total, 1]  view into col 1
        self.rb_states  = env.rb_states    # [total_rbs, 13]
        self.total_dofs_per_env = env.total_dofs_per_env

        # Articulated-object DOF info
        self.n_arti_dofs = self.total_dofs_per_env - N_HAND_DOFS
        self.include_handle_rot = include_handle_rot
        # target_joint_idx is resolved after trajectory loading (see below)
        self.history_len = 16

        # Phase 4 (GLA) toggle. When True the per-timestep history token is
        # [error_t (51), a_{t-1} (51)] -> 102-D, giving the temporal encoder
        # both the PD tracking error and the control input that produced it.
        # When False the history token is just error_t (51-D), reproducing the
        # flat-history baseline used in `runs/hand_drag_history_*_bounds01/`.
        self.include_prev_action_in_history = bool(include_prev_action_in_history)
        self.history_token_dim = (
            N_HAND_DOFS * 2 if self.include_prev_action_in_history else N_HAND_DOFS
        )

        # -- Observation / action dimensions --
        # obs = hand_pos(51) + hand_vel(51) + handle_pos(3) +
        #       optional handle_rot(4) + palm_to_handle_vec(3) +
        #       palm_to_handle_dist(1) + arti(Na+Na) +
        #       history(history_len * history_token_dim) +
        #       optional aux_targets(aux_dim).
        # Phase 4 GLA obs (Na=1): 115 + 16*102 = 1747-D.
        # Flat-history baseline (Na=1): 115 + 16*51  =  931-D.
        # PICA v2b adds aux_dim (3 by default when enabled) at the very tail.
        handle_pose_dim = 7 if self.include_handle_rot else 3

        # PICA v2b: parse physical_auxiliary config first so we can size
        # obs_buf to include the aux-target tail when enabled.
        self._phys_aux_cfg = _parse_phys_aux_cfg(physical_auxiliary)
        self._phys_aux_enabled = self._phys_aux_cfg["enabled"]
        self._aux_dim = self._phys_aux_cfg["aux_dim"]

        self.obs_dim = (
            N_HAND_DOFS * 2 + handle_pose_dim + 3 + 1 + self.n_arti_dofs * 2
            + self.history_len * self.history_token_dim
            + self._aux_dim
        )
        self.act_dim = N_HAND_DOFS  # 51

        # -- Rigid-body indices (sim-domain, one per env) --
        self.wrist_body_idxs = torch.tensor(
            env.wrist_idxs, dtype=torch.long, device=self.device,
        )
        # Handle body: look up by link name in each env
        self.handle_body_idxs = torch.tensor(
            [
                self.gym.find_actor_rigid_body_index(
                    env.envs[i], env.arti_actors[i],
                    handle_link_name, gymapi.DOMAIN_SIM,
                )
                for i in range(self.num_envs)
            ],
            dtype=torch.long, device=self.device,
        )

        # -- Global actor indices for set_dof_*_tensor_indexed --
        # Actors per env: hand(0), table(1), arti(2)
        n_actors = self.gym.get_actor_count(env.envs[0])
        arange   = torch.arange(self.num_envs, device=self.device)
        self.hand_actor_idxs = (arange * n_actors + 0).to(torch.int32)
        self.arti_actor_idxs = (arange * n_actors + 2).to(torch.int32)

        # -- Joint limits (hand only; arti is DOF_MODE_NONE) --
        self.hand_dof_lower = torch.tensor(
            env.hand_dof_props["lower"][:N_HAND_DOFS],
            dtype=torch.float32, device=self.device,
        )
        self.hand_dof_upper = torch.tensor(
            env.hand_dof_props["upper"][:N_HAND_DOFS],
            dtype=torch.float32, device=self.device,
        )
        # Arti default = lower limits (fully closed position)
        self.arti_dof_default = torch.tensor(
            env.arti_obj_default_dof_state["pos"],
            dtype=torch.float32, device=self.device,
        )

        # -- Persistent GPU buffers --
        self.pos_targets = torch.zeros(
            self.num_envs, self.total_dofs_per_env,
            dtype=torch.float32, device=self.device,
        )
        self.error_history = torch.zeros(
            self.num_envs, self.history_len, N_HAND_DOFS, device=self.device,
        )
        # Phase 4 (GLA) prev-action history -- mirrors error_history layout so
        # the per-timestep token [error_t, a_{t-1}] can be assembled by a
        # single concat along the feature axis. Zeroed on every reset so the
        # GLA encoder never carries an action token from the previous episode.
        self.action_history = torch.zeros(
            self.num_envs, self.history_len, N_HAND_DOFS, device=self.device,
        )
        # Palm offset from wrist origin, pre-expanded for batch quat_rotate
        self.palm_offset = torch.tensor(
            PALM_OFFSET_LOCAL, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)       # [N, 3] constant
        # True grasp target offset from the target rigid-body origin. GAPartNet
        # handles are often fixed children collapsed into the moving parent
        # body, so rb_states[handle_body, :3] is the parent/hinge origin, not
        # the physical handle center.
        self.handle_center_local = self._compute_handle_center_local(
            handle_link_name
        )
        self.true_handle_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # RL episode buffers
        self.obs_buf      = torch.zeros(self.num_envs, self.obs_dim, device=self.device)
        self.rew_buf      = torch.zeros(self.num_envs, device=self.device)
        self.done_buf     = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions      = torch.zeros(self.num_envs, self.act_dim, device=self.device)
        self.filtered_actions = torch.zeros(
            self.num_envs, self.act_dim, device=self.device,
        )
        # Detachment means "slipped after contact", not "started far away".
        # This is armed once the palm first enters detach_dist.
        self.detach_armed_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        # Track previous target-joint position for delta reward
        self.prev_target_joint = torch.zeros(self.num_envs, device=self.device)

        # ------------------------------------------------------------------
        # RSI curriculum state
        # ------------------------------------------------------------------
        # self.rsi_prob is the *live* probability read by reset_idx each call.
        # It starts at rsi_prob_start and is shrunk toward rsi_prob_end by
        # update_rsi_prob(current_step, max_steps), which the PPO runner is
        # expected to call once per training iteration (see method docstring).
        self.rsi_prob            = float(self.rsi_prob_start)
        self.global_step_counter = 0      # total env-steps seen so far
        self.total_env_steps     = 0      # legacy alias kept for compatibility
        self.max_training_steps  = int(self.rsi_decay_steps)  # runner may override

        # Zero-RSI evaluation mode: when True, reset_idx forces rsi_prob=0
        # regardless of the current curriculum value.  Flip before eval rollouts.
        self.is_eval_mode = bool(is_eval_mode)

        # Cached palm state (populated by compute_observations, reused in reward)
        self.palm_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_to_handle_dist = torch.zeros(self.num_envs, device=self.device)

        # ------------------------------------------------------------------
        # Episode-reward logging buffers
        # ------------------------------------------------------------------
        # Running per-env sums of each reward component, zeroed on reset.
        # When an env terminates, reset_idx publishes the mean over the
        # just-finished envs into self.extras["episode"], which rl_games /
        # skrl forward to TensorBoard / WandB automatically.
        self._reward_keys = [
            "reward", "r_dist", "r_task", "r_act", "r_detach", "r_success",
            "r_time",
            # PICA v1 physical regularization components. Always present in
            # the schema so logging columns are stable across runs; weights
            # default to 0 so disabled means literal zero contribution.
            "r_phys_bound", "r_phys_contact", "r_phys_slip",
            "r_phys_smooth", "r_phys_total",
            # PICA v2d ARAM-lite + reconfig reward (always in schema; literal
            # zero when their respective configs are disabled).
            "r_aram", "r_reconfig",
        ]
        self.episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in self._reward_keys
        }
        # Additional per-epoch success tracker (0/1 flag per terminating env)
        self.episode_sums["success"] = torch.zeros(
            self.num_envs, device=self.device,
        )
        self._reward_keys.append("success")  # include in end_epoch() means
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self.episode_final_joint_pos = torch.zeros(self.num_envs, device=self.device)
        self.episode_final_progress = torch.zeros(self.num_envs, device=self.device)
        # Caches for per-step reward components so step() can accumulate them
        self._r_dist_last    = torch.zeros(self.num_envs, device=self.device)
        self._r_task_last    = torch.zeros(self.num_envs, device=self.device)
        self._r_act_last     = torch.zeros(self.num_envs, device=self.device)
        self._r_detach_last  = torch.zeros(self.num_envs, device=self.device)
        self._r_success_last = torch.zeros(self.num_envs, device=self.device)
        self._r_time_last    = torch.zeros(self.num_envs, device=self.device)
        self._success_flag   = torch.zeros(self.num_envs, device=self.device)
        self._normalized_progress_last = torch.zeros(self.num_envs, device=self.device)

        # ------------------------------------------------------------------
        # PICA v1 -- physical regularization state
        # ------------------------------------------------------------------
        # Config (with defaults) is parsed at module level so missing/disabled
        # configs collapse to zero weights and the rew_buf math is unchanged.
        self._phys_reg_cfg = _parse_phys_reg_cfg(physical_regularization)
        self._phys_reg_enabled = self._phys_reg_cfg["enabled"]

        # Dedicated previous-step buffers. Both are intentionally NOT views
        # of self.actions / self.palm_to_handle_dist -- those are overwritten
        # mid-step (pre_physics_step / compute_observations / compute_reward),
        # so a reliable r_phys_smooth and r_phys_slip require shadow copies
        # captured at well-defined points in the step pipeline.
        self.prev_actions_for_phys = torch.zeros(
            self.num_envs, self.act_dim, device=self.device,
        )
        self.prev_palm_to_handle_dist = torch.zeros(
            self.num_envs, device=self.device,
        )

        # Per-step component caches (mirrors _r_<name>_last for the original
        # reward components; consumed by step() to accumulate into episode_sums).
        self._r_phys_bound_last   = torch.zeros(self.num_envs, device=self.device)
        self._r_phys_contact_last = torch.zeros(self.num_envs, device=self.device)
        self._r_phys_slip_last    = torch.zeros(self.num_envs, device=self.device)
        self._r_phys_smooth_last  = torch.zeros(self.num_envs, device=self.device)
        self._r_phys_total_last   = torch.zeros(self.num_envs, device=self.device)

        # ------------------------------------------------------------------
        # PICA v2a -- dynamics randomization state
        # ------------------------------------------------------------------
        # Per-env damping / friction scales (1.0 == nominal). Reset_idx samples
        # a fresh scale per resetting env when randomization is enabled, then
        # calls Isaac Gym's per-actor properties API to apply it. Nominal
        # values are read from the env at init so we never depend on a
        # hardcoded value drifting away from dataset/geometric/hand_object_gym.py.
        self._dyn_rand_cfg = _parse_dyn_rand_cfg(dynamics_randomization)
        self._dyn_rand_enabled = self._dyn_rand_cfg["enabled"]

        # Cache nominal damping (per-DOF) from the env's struct. All arti
        # DOFs share the same damping by construction, so [0] is canonical.
        try:
            self.nominal_obj_damping = float(
                self.env.arti_obj_dof_props["damping"][0]
            )
        except Exception:
            self.nominal_obj_damping = 10.0  # matches dataset/geometric/hand_object_gym default

        # Cache nominal friction by reading the first arti shape on env 0.
        # If the asset has no shapes (extremely defensive), fall back to 5.0.
        try:
            sp0 = self.gym.get_actor_rigid_shape_properties(
                self.env.envs[0], self.env.arti_actors[0]
            )
            self.nominal_obj_friction = float(sp0[0].friction) if len(sp0) > 0 else 5.0
        except Exception:
            self.nominal_obj_friction = 5.0

        # Per-env scale tensors (kept on GPU for cheap reductions in end_epoch).
        self.current_damping_scale = torch.ones(self.num_envs, device=self.device)
        self.current_friction_scale = torch.ones(self.num_envs, device=self.device)

        # Private mutable copy of the env's DOF-properties struct, used as a
        # scratch buffer when we apply per-env damping in reset_idx. We never
        # mutate self.env.arti_obj_dof_props itself so other code paths
        # (e.g. eval-time apply_ood_dynamics_overrides) keep their nominal
        # reference.
        self._dynrand_dof_props_scratch = None
        if self._dyn_rand_enabled:
            try:
                self._dynrand_dof_props_scratch = (
                    self.env.arti_obj_dof_props.copy()
                )
            except Exception:
                # numpy structured arrays support .copy(); if the underlying
                # struct does not, we fall back to the shared one and accept
                # the cross-env coupling (dom-randomization will still work
                # because we re-set damping for every reset env each call).
                self._dynrand_dof_props_scratch = self.env.arti_obj_dof_props

        # ------------------------------------------------------------------
        # PICA v2b -- physical auxiliary targets (computed in compute_obs)
        # ------------------------------------------------------------------
        # Cache last-step aux target tensor for end_epoch logging. The actual
        # transport happens via obs_buf tail; this buffer is just for stats.
        self._aux_targets_last = torch.zeros(
            self.num_envs, max(1, self._aux_dim), device=self.device,
        )
        # Contact-validity gate tracked for debug logging only -- it is
        # NOT used to weight the aux loss in v2b-init (see network code).
        self._phys_aux_gate_last = torch.zeros(self.num_envs, device=self.device)

        # ---- PICA v2c/v2d: causal-horizon ring buffers ----
        # Keep aux-supervision history and v2d reward history separate. v2c
        # aux targets are tied to the observation tick, while ARAM/reconfig
        # rewards need reward-stage state. Sharing one ring silently changes
        # v2c aux target timing when v2d code is merged in.
        self._aux_horizon = self._phys_aux_cfg.get("horizon", 5)
        self._aux_mode = self._phys_aux_cfg.get("mode", "current")
        self._aram_cfg = _parse_aram_cfg(aram)
        self._reconfig_cfg = _parse_reconfig_cfg(reconfig_reward)
        self._aram_enabled = self._aram_cfg["enabled"]
        self._reconfig_enabled = self._reconfig_cfg["enabled"]
        K = max(1, int(self._aux_horizon))

        # v2c physical auxiliary targets: observation-stage window [t-K, t].
        self._aux_ring_buflen = K + 1
        self._aux_idx_tK = 0
        self._aux_palm_dist_hist = torch.zeros(
            self.num_envs, self._aux_ring_buflen, device=self.device,
        )
        self._aux_q_obj_hist = torch.zeros(
            self.num_envs, self._aux_ring_buflen, device=self.device,
        )

        # v2d reward terms: reward-stage history. Reconfig compares two
        # adjacent K-windows, so it needs 2K+1 slots when enabled.
        need_long_hist = bool(self._aram_enabled or self._reconfig_enabled)
        BUFLEN = (2 * K + 1) if need_long_hist else (K + 1)
        self._aux_buflen = BUFLEN
        self._idx_tK = BUFLEN - 1 - K
        self._palm_dist_hist = torch.zeros(
            self.num_envs, BUFLEN, device=self.device,
        )
        self._q_obj_hist = torch.zeros(
            self.num_envs, BUFLEN, device=self.device,
        )

        # ---- PICA v2d: per-env counters and adaptive scalar ----
        # lambda_aram is the only persistent across-epoch scalar -- the rest
        # are reset on episode boundary in reset_idx.
        self.lambda_aram = float(self._aram_cfg["lambda_init"])
        self._stall_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self._reconfig_cooldown = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        # Per-step caches for episode_sums + epoch logging.
        self._r_aram_last     = torch.zeros(self.num_envs, device=self.device)
        self._r_reconfig_last = torch.zeros(self.num_envs, device=self.device)
        self._aram_gate_last     = torch.zeros(self.num_envs, device=self.device)
        self._reconfig_gate_last = torch.zeros(self.num_envs, device=self.device)
        self._action_l2_last     = torch.zeros(self.num_envs, device=self.device)
        self._clip099_last       = torch.zeros(self.num_envs, device=self.device)

        # ---- PICA v2d: epoch-level rate accumulators ----
        # clip099, action_l2, and the two gate-fire rates are best summarized
        # as per-step rates over an epoch (not per-episode), so they live
        # outside _epoch_sum_components.
        self._epoch_clip099_sum     = 0.0
        self._epoch_clip099_n       = 0
        self._epoch_action_l2_sum   = 0.0
        self._epoch_action_l2_n     = 0
        self._epoch_aram_gate_sum   = 0.0
        self._epoch_reconfig_gate_sum = 0.0
        self._epoch_gate_n          = 0

        # ------------------------------------------------------------------
        # Per-epoch accumulators (span one PPO training iteration)
        # ------------------------------------------------------------------
        # reset_idx() can fire many times within a single training epoch,
        # and extras["episode"] would otherwise be overwritten by the last
        # batch of terminations.  These accumulators *sum* over every env
        # that terminated during the current epoch, so end_epoch() can emit
        # an honest epoch-level mean.  The PPO runner owns the boundary:
        # call task.end_epoch() once per training iteration.
        self.epoch_num            = 0
        self._epoch_num_episodes  = 0
        self._epoch_sum_length    = 0
        self._epoch_sum_components = {k: 0.0 for k in self._reward_keys}
        self._epoch_sum_final_joint_pos = 0.0
        self._epoch_sum_normalized_progress = 0.0
        # Running min / max of episodic return seen this epoch
        self._epoch_return_min    = float("inf")
        self._epoch_return_max    = float("-inf")

        # self.extras is read by the PPO runner after every step().  Anything
        # under extras["episode"][<key>] or extras["epoch"][<key>] ends up
        # plotted on TB / WandB.
        self.extras: dict = {}

        # Optional CSV dump of end_epoch() stats -- one row per training
        # iteration.  Pass epoch_log_path="runs/<exp>/epoch_rewards.csv".
        self.epoch_log_path = epoch_log_path
        self._epoch_csv_header_written = False
        self._epoch_csv_warned_schema_change = False
        if self.epoch_log_path:
            os.makedirs(os.path.dirname(self.epoch_log_path) or ".", exist_ok=True)
            self._epoch_csv_header_written = (
                os.path.exists(self.epoch_log_path)
                and os.path.getsize(self.epoch_log_path) > 0
            )

        # -- Load expert trajectory for RSI --
        self._load_trajectory(trajectory_path)

        # Ready-to-pull non-RSI reset state: the first drag frame is where
        # the expert has already grasped the handle but has not opened it yet.
        self.ready_grasp_hand = self.traj_hand[self.drag_start_frame_idx].clone()
        self.ready_grasp_arti = self.traj_arti[self.drag_start_frame_idx].clone()
        self.split_rsi_centre = self.ready_grasp_hand[:3].clone()

        # -- Resolve target joint index --
        # Auto-detect: pick the arti DOF with the largest range of motion
        # in the trajectory (the one that actually moves during the drag).
        # This handles objects with different DOF counts (1-DOF drawers
        # vs 3-DOF cabinets) without manual configuration.
        if target_joint_idx is not None and target_joint_idx < self.n_arti_dofs:
            self.target_joint_idx = target_joint_idx
        else:
            motion_range = self.traj_arti.max(dim=0).values - self.traj_arti.min(dim=0).values
            self.target_joint_idx = int(motion_range.argmax().item())
            print(f"  [auto] target_joint_idx = {self.target_joint_idx} "
                  f"(max motion {motion_range[self.target_joint_idx]:.4f} "
                  f"out of {self.n_arti_dofs} arti DOFs)")

        # Absolute "task done" threshold in joint-coord units, derived from the
        # expert trajectory so it matches each object's range of motion.
        traj_max = float(self.traj_arti[:, self.target_joint_idx].max().item())
        traj_min = float(self.traj_arti[:, self.target_joint_idx].min().item())
        self.task_done_joint = traj_min + self.task_done_frac * (traj_max - traj_min)
        self.task_start_joint = float(
            self.traj_arti[self.drag_start_frame_idx, self.target_joint_idx].item()
        )
        self.task_goal_joint = traj_max
        self.task_progress_denom = max(1e-6, self.task_goal_joint - self.task_start_joint)
        self.target_joint_type = self._infer_target_joint_type()
        print(
            "  [reset] ready-to-pull frame = "
            f"{self.drag_start_frame_idx}, start joint = {self.task_start_joint:.6f}, "
            f"goal joint = {self.task_goal_joint:.6f}, "
            f"joint type = {self.target_joint_type}"
        )
        self._build_rsi_frame_ids()


    def _infer_target_joint_type(self) -> str:
        """Infer target articulated joint type from the loaded GAPartNet URDF.

        The trajectory-selected target_joint_idx indexes non-fixed object
        joints. GAPartNet URDF joint order matches the articulated DOF order
        used by Isaac Gym after fixed-joint collapse, so we map that index to
        the corresponding non-fixed joint type. If anything is missing, fall
        back to unknown and the ARAM gate will use the conservative raw
        threshold.
        """
        try:
            object_id = str(self.env.gapartnet_ids[0])
            candidates = [
                os.path.join(
                    self.env.asset_root, self.env.gapartnet_root,
                    object_id, "mobility.urdf",
                ),
                os.path.join(
                    self.env.asset_root, self.env.gapartnet_root,
                    object_id, "mobility_annotation_gapartnet.urdf",
                ),
            ]
            urdf_path = next((path for path in candidates if os.path.exists(path)), None)
            if urdf_path is None:
                print(f"  [joint-type] URDF not found for object {object_id}; using unknown")
                return "unknown"
            joints = []
            root = ET.parse(urdf_path).getroot()
            for joint in root.findall("joint"):
                joint_type = str(joint.attrib.get("type", "fixed")).lower()
                if joint_type != "fixed":
                    joints.append(joint_type)
            if 0 <= int(self.target_joint_idx) < len(joints):
                return joints[int(self.target_joint_idx)]
            print(
                "  [joint-type] target_joint_idx out of range: "
                f"idx={self.target_joint_idx}, non_fixed_joints={joints}; using unknown"
            )
        except Exception as exc:
            print(f"  [joint-type] failed to infer target joint type: {exc}; using unknown")
        return "unknown"

    def _select_target_part_idx(self, handle_link_name: str):
        """Choose the articulated part whose handle should be tracked."""
        if (
            hasattr(self.env, "gapart_link_names")
            and self.env.gapart_link_names
            and handle_link_name in self.env.gapart_link_names[0]
        ):
            idx = self.env.gapart_link_names[0].index(handle_link_name)
            cat = self.env.gapart_cates[0][idx].lower()
            if "handle" not in cat:
                return idx

        cates = self.env.gapart_cates[0]
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
        return target_idx

    def _compute_handle_center_local(self, handle_link_name: str) -> torch.Tensor:
        """
        Precompute the true handle center in the tracked rigid body's local frame.

        GAPartNet annotations give an OBB for the semantic handle part in the
        closed object pose. Isaac Gym loads fixed handle children collapsed into
        the parent movable rigid body, so the dynamic handle center is:

            body_pos(t) + R_body(t) @ handle_center_local

        where handle_center_local is computed once from env0's initial rigid
        body pose. The same local offset is valid for every parallel env.
        """
        if (
            not hasattr(self.env, "gapart_init_bboxes")
            or not self.env.gapart_init_bboxes
            or len(self.env.gapart_init_bboxes[0]) == 0
        ):
            print("  [handle] no GAPartNet bbox annotations; using rigid-body origin")
            return torch.zeros(self.num_envs, 3, device=self.device)

        target_idx = self._select_target_part_idx(handle_link_name)
        obj_pos = np.asarray(self.env.arti_init_obj_pos_list[0], dtype=np.float64)
        obj_rot = np.asarray(self.env.arti_init_obj_rot_list[0], dtype=np.float64)
        obj_rot_mat = Rot.from_quat(obj_rot).as_matrix()
        scale = float(self.env.cfgs["asset"]["arti_obj_scale"])

        handle_center_world, *_ = find_handle_for_part(
            target_idx,
            self.env.gapart_cates[0],
            self.env.gapart_init_bboxes[0],
            scale,
            obj_pos,
            obj_rot_mat,
        )

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        body_idx0 = self.handle_body_idxs[0]
        body_pos0 = self.rb_states[body_idx0, :3].unsqueeze(0)
        body_rot0 = self.rb_states[body_idx0, 3:7].unsqueeze(0)
        handle_world0 = torch.tensor(
            handle_center_world, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        local0 = quat_rotate_inverse(body_rot0, handle_world0 - body_pos0)
        local = local0.repeat(self.num_envs, 1)

        print(
            "  [handle] true center local offset = "
            f"{local0.squeeze(0).detach().cpu().numpy().round(4).tolist()}"
        )
        return local

    def _true_handle_pose(self):
        """Return dynamic true handle center and parent body rotation tensors."""
        body_pos = self.rb_states[self.handle_body_idxs, :3]
        body_rot = self.rb_states[self.handle_body_idxs, 3:7]
        true_handle_pos = body_pos + quat_rotate(body_rot, self.handle_center_local)
        self.true_handle_pos = true_handle_pos
        return true_handle_pos, body_rot

    def _build_rsi_frame_ids(self):
        """
        Build the trajectory frame pool used by RSI resets.

        Frames that already satisfy the task success threshold create
        one-step "success" episodes at reset time. That teaches the critic a
        shortcut and leaves the actor with almost no control signal, so RSI
        only samples pre-success frames.
        """
        pre_success = (
            self.traj_arti[:, self.target_joint_idx] < self.task_done_joint
        )
        frame_ids = torch.nonzero(pre_success, as_tuple=False).squeeze(-1)
        if frame_ids.numel() == 0:
            frame_ids = torch.arange(self.n_traj, device=self.device)
        self.rsi_frame_ids = frame_ids
        print(
            f"  [rsi] using {int(frame_ids.numel())}/{self.n_traj} "
            "pre-success trajectory frames"
        )

    def _load_trajectory(self, path: str):
        """Parse trajectory.json into GPU tensors for fast RSI sampling."""
        with open(path) as f:
            traj = json.load(f)
        hand_rows, arti_rows, phase_rows = [], [], []
        for fr in traj:
            # Reconstruct the full 51-dim hand DOF vector from saved components
            hand_rows.append(
                fr["virtual_xyz"] + fr["wrist_rpy"] + fr["finger_dofs"]
            )
            arti_rows.append(fr["joint_positions"])
            phase_rows.append(fr.get("phase", ""))
        # [T, 51] and [T, Na_traj] on GPU -- sampled by index during RSI
        self.traj_hand = torch.tensor(
            hand_rows, dtype=torch.float32, device=self.device,
        )
        traj_arti_raw = torch.tensor(
            arti_rows, dtype=torch.float32, device=self.device,
        )
        # Align trajectory arti DOF count with the loaded object.
        # Truncate if trajectory has more DOFs, pad with zeros if fewer.
        na = self.n_arti_dofs
        if traj_arti_raw.shape[1] >= na:
            self.traj_arti = traj_arti_raw[:, :na]
        else:
            pad = torch.zeros(len(traj), na - traj_arti_raw.shape[1],
                              device=self.device)
            self.traj_arti = torch.cat([traj_arti_raw, pad], dim=1)
        self.n_traj = len(traj)
        self.traj_phases = phase_rows
        self.drag_start_frame_idx = next(
            (i for i, phase in enumerate(phase_rows) if phase == "drag"),
            min(40, max(0, self.n_traj - 1)),
        )

    # =====================================================================
    #  PICA v2a:  per-env dynamics randomization
    # =====================================================================
    def _apply_dyn_rand(self, env_ids: torch.Tensor):
        """Sample and apply fresh damping / friction scales for env_ids.

        Per-env Isaac Gym property setters are used because the random
        scales must vary across envs. We mutate a private scratch copy of
        the DOF-props struct so eval-time code that reads
        ``env.arti_obj_dof_props`` still sees the nominal values.
        """
        cfg = self._dyn_rand_cfg
        if (not cfg["enabled"]) or env_ids.numel() == 0:
            return
        damp_on = cfg["damping_scale"]["enabled"]
        fric_on = cfg["friction_scale"]["enabled"]
        if not (damp_on or fric_on):
            return

        n = int(env_ids.numel())
        env_ids_cpu = env_ids.detach().cpu().numpy()

        # ---- Damping (per-DOF property on the arti actor) ----
        if damp_on:
            lo, hi = cfg["damping_scale"]["range"]
            new_scales = torch.empty(n, device=self.device).uniform_(lo, hi)
            self.current_damping_scale[env_ids] = new_scales
            scales_cpu = new_scales.detach().cpu().numpy()
            props = self._dynrand_dof_props_scratch
            for i, scale in zip(env_ids_cpu, scales_cpu):
                # Mutate scratch and apply to this env's arti actor only.
                props["damping"][:] = self.nominal_obj_damping * float(scale)
                self.gym.set_actor_dof_properties(
                    self.env.envs[int(i)],
                    self.env.arti_actors[int(i)],
                    props,
                )

        # ---- Friction (per-shape property on the arti actor) ----
        if fric_on:
            lo, hi = cfg["friction_scale"]["range"]
            new_fric = torch.empty(n, device=self.device).uniform_(lo, hi)
            self.current_friction_scale[env_ids] = new_fric
            fric_cpu = new_fric.detach().cpu().numpy()
            for i, scale in zip(env_ids_cpu, fric_cpu):
                sp = self.gym.get_actor_rigid_shape_properties(
                    self.env.envs[int(i)],
                    self.env.arti_actors[int(i)],
                )
                target_fric = float(self.nominal_obj_friction * float(scale))
                for s in sp:
                    s.friction = target_fric
                self.gym.set_actor_rigid_shape_properties(
                    self.env.envs[int(i)],
                    self.env.arti_actors[int(i)],
                    sp,
                )

    # =====================================================================
    #  RSI CURRICULUM:  update_rsi_prob -- called by the PPO runner
    # =====================================================================

    def update_rsi_prob(self, current_step: int, max_steps: int):
        """
        Recompute self.rsi_prob from a linear (or exponential) schedule.

        How to wire this into the PPO runner
        ------------------------------------
        The PPO runner owns the "epoch / frame" counter.  Call this once
        per training iteration *before* the rollout.  Two integration
        patterns, pick one:

          A. Runner-driven (rl_games).  Subclass A2CAgent and override
             train_epoch(), or call this from the Runner loop:

                 task.update_rsi_prob(
                     current_step=agent.frame,        # env-steps so far
                     max_steps=agent.max_frames,      # from cfg
                 )

          B. Env-driven (zero runner changes).  Set
             task.max_training_steps = total_env_step_budget once at
             startup; step() then auto-calls update_rsi_prob with
             self.global_step_counter every tick (see step()).

        Decay finishes at rsi_decay_end_frac * max_steps (default: halfway
        through training), leaving the second half at pure zero-state
        training so the policy cannot rely on RSI momentum.
        """
        self.max_training_steps = max(1, int(max_steps))
        decay_end = self.rsi_decay_end_frac * self.max_training_steps
        frac      = min(1.0, current_step / max(1.0, decay_end))    # in [0, 1]

        if self.decay_mode == "linear":
            self.rsi_prob = (
                self.rsi_prob_start
                + (self.rsi_prob_end - self.rsi_prob_start) * frac
            )
        else:  # exponential: fast early drop, long tail
            decay = math.exp(-5.0 * frac)
            self.rsi_prob = (
                self.rsi_prob_end
                + (self.rsi_prob_start - self.rsi_prob_end) * decay
            )

        self.rsi_prob = float(max(0.0, min(1.0, self.rsi_prob)))
        return self.rsi_prob

    # =====================================================================
    #  EPOCH LOGGING:  end_epoch -- called by the PPO runner per iteration
    # =====================================================================

    def end_epoch(self) -> dict:
        """
        Aggregate the reward stats for every episode that terminated during
        the current training iteration, publish them to self.extras["epoch"],
        and reset the accumulators for the next iteration.

        How to wire into the PPO runner
        -------------------------------
        rl_games:
            Override A2CAgent.train_epoch() -- call task.end_epoch() after
            the epoch's rollout + update, then forward extras["epoch"] to
            self.writer (TB) / self.experiment_tracker (WandB).

        skrl:
            Hook into Trainer event "epoch_end":
                stats = task.end_epoch()
                for k, v in stats.items():
                    self.writer.add_scalar(f"epoch/{k}", v, self.epoch_num)

        The method is safe to call even if zero episodes finished in the
        epoch (rare, but possible with very long max_episode_length): it
        returns zero-filled stats and increments epoch_num regardless so
        the x-axis stays monotonic.
        """
        n = self._epoch_num_episodes
        if n > 0:
            stats = {
                f"{k}_mean": self._epoch_sum_components[k] / n
                for k in self._reward_keys
            }
            stats["length_mean"] = self._epoch_sum_length / n
            stats["final_joint_pos_mean"] = self._epoch_sum_final_joint_pos / n
            stats["normalized_progress_mean"] = (
                self._epoch_sum_normalized_progress / n
            )
            stats["return_min"]  = self._epoch_return_min
            stats["return_max"]  = self._epoch_return_max
        else:
            stats = {f"{k}_mean": 0.0 for k in self._reward_keys}
            stats["length_mean"] = 0.0
            stats["final_joint_pos_mean"] = 0.0
            stats["normalized_progress_mean"] = 0.0
            stats["return_min"]  = 0.0
            stats["return_max"]  = 0.0

        stats["num_episodes"] = n
        stats["epoch"]        = self.epoch_num
        stats["rsi_prob"]     = self.rsi_prob
        stats["global_step"]  = self.global_step_counter

        # PICA v2a: surface the current per-env dynamics scale distribution
        # so the operator can see at a glance that randomization is firing.
        # When dynamics_randomization is disabled these are 1/1/1.
        stats["damping_scale_mean"]  = float(self.current_damping_scale.mean().item())
        stats["damping_scale_min"]   = float(self.current_damping_scale.min().item())
        stats["damping_scale_max"]   = float(self.current_damping_scale.max().item())
        stats["friction_scale_mean"] = float(self.current_friction_scale.mean().item())

        # PICA v2b: surface the latest aux target distribution + contact gate
        # mean. This is debug-only logging (the loss is computed inside the
        # network, not here) but it lets us sanity-check that aux targets
        # land at reasonable magnitudes during early training.
        if self._phys_aux_enabled:
            keys = self._phys_aux_cfg["target_keys"]
            for ch_idx, key in enumerate(keys):
                stats[f"aux_target_{key}_mean"] = float(
                    self._aux_targets_last[:, ch_idx].mean().item()
                )
            stats["aux_contact_gate_mean"] = float(
                self._phys_aux_gate_last.mean().item()
            )

        # ---- PICA v2d: epoch rate stats + adaptive lambda update ----
        clip099_mean = (
            self._epoch_clip099_sum / max(1, self._epoch_clip099_n)
        )
        action_l2_mean = (
            self._epoch_action_l2_sum / max(1, self._epoch_action_l2_n)
        )
        aram_gate_rate = (
            self._epoch_aram_gate_sum / max(1, self._epoch_gate_n)
        )
        reconfig_gate_rate = (
            self._epoch_reconfig_gate_sum / max(1, self._epoch_gate_n)
        )
        stats["clip099_mean"]       = float(clip099_mean)
        stats["action_l2_mean"]     = float(action_l2_mean)
        stats["aram_gate_rate"]     = float(aram_gate_rate)
        stats["reconfig_gate_rate"] = float(reconfig_gate_rate)
        stats["lambda_aram"]        = float(self.lambda_aram)

        # Adaptive lambda update: nudge lambda toward target_clip when the
        # policy is already mostly succeeding nominally. clamp keeps it in
        # [0, lambda_max]. Only fires when ARAM is enabled.
        if self._aram_enabled:
            success_mean = stats.get("success_mean", 0.0)
            if (
                success_mean > self._aram_cfg["start_after_success"]
                and stats.get("num_episodes", 0) > 0
            ):
                delta = self._aram_cfg["lambda_lr"] * (
                    clip099_mean - self._aram_cfg["target_clip"]
                )
                new_lambda = self.lambda_aram + delta
                self.lambda_aram = float(
                    max(0.0, min(self._aram_cfg["lambda_max"], new_lambda))
                )
                stats["lambda_aram"] = float(self.lambda_aram)

        # Reset per-epoch rate accumulators (per-episode sums are reset
        # below in the existing block).
        self._epoch_clip099_sum   = 0.0
        self._epoch_clip099_n     = 0
        self._epoch_action_l2_sum = 0.0
        self._epoch_action_l2_n   = 0
        self._epoch_aram_gate_sum   = 0.0
        self._epoch_reconfig_gate_sum = 0.0
        self._epoch_gate_n        = 0

        self.extras["epoch"] = stats

        # Optional durable CSV log (one row per training iteration).
        if self.epoch_log_path:
            keys = list(stats.keys())
            if self._epoch_csv_header_written:
                with open(self.epoch_log_path) as f:
                    existing_keys = f.readline().strip().split(",")
                if existing_keys != keys:
                    base, ext = os.path.splitext(self.epoch_log_path)
                    self.epoch_log_path = (
                        f"{base}.schema_{int(time.time())}{ext or '.csv'}"
                    )
                    self._epoch_csv_header_written = False
                    if not self._epoch_csv_warned_schema_change:
                        print(
                            "  [epoch-log] existing CSV header does not match "
                            f"current stats; writing new file: {self.epoch_log_path}"
                        )
                        self._epoch_csv_warned_schema_change = True

            line = ",".join(f"{stats[k]:.6g}" if isinstance(stats[k], float)
                            else str(stats[k]) for k in keys)
            with open(self.epoch_log_path, "a") as f:
                if not self._epoch_csv_header_written:
                    f.write(",".join(keys) + "\n")
                    self._epoch_csv_header_written = True
                f.write(line + "\n")

        # Reset accumulators for the next training iteration
        self._epoch_num_episodes  = 0
        self._epoch_sum_length    = 0
        self._epoch_sum_final_joint_pos = 0.0
        self._epoch_sum_normalized_progress = 0.0
        for k in self._epoch_sum_components:
            self._epoch_sum_components[k] = 0.0
        self._epoch_return_min = float("inf")
        self._epoch_return_max = float("-inf")
        self.epoch_num += 1

        return stats

    # =====================================================================
    #  CORE METHOD 1:  reset_idx  --  RSI (Reference State Initialisation)
    # =====================================================================

    def reset_idx(self, env_ids: torch.Tensor):
        """
        Reset specified environments using the *current* self.rsi_prob.

        RSI logic
        ---------
        For each env in env_ids, draw a uniform random number and compare
        against the effective RSI probability:
          * rand >= rsi_prob -> ready-to-pull state: expert drag-start grasp
          * rand <  rsi_prob -> RSI state:      uniformly sample a random frame
                                from the expert trajectory; set both hand and
                                arti DOFs to match that frame.  Velocities zeroed.

        The coin flip is fully tensorized via torch.rand over env_ids.

        Eval-mode override
        ------------------
        When self.is_eval_mode is True, rsi_prob is forced to 0.0 regardless
        of the curriculum -- eval must not "cheat" with physical momentum
        from expert frames.

        Episode logging
        ---------------
        Before zeroing the per-env reward sums for these terminating envs,
        the mean over env_ids is published into self.extras["episode"] so
        the PPO runner can forward it to TensorBoard / WandB.
        """
        n = len(env_ids)
        if n == 0:
            return

        # ---------- Effective RSI probability ----------
        # Eval hard-disables RSI; otherwise use the curriculum-decayed value.
        effective_rsi_prob = 0.0 if self.is_eval_mode else self.rsi_prob

        # ---------- Tensorized per-env coin flip ----------
        # use_rsi[i] = True  -> that env starts from an expert trajectory frame
        # use_rsi[i] = False -> deterministic expert drag-start grasp pose
        rand_mask = torch.rand(n, device=self.device)
        use_rsi   = rand_mask < effective_rsi_prob                   # [n] bool

        # ---------- PPO-from-grasp default ----------
        # The non-RSI branch is the academic "pull-only" baseline: start
        # exactly at the expert drag-start frame, where the hand is already
        # closed around the handle and the object joint is still near closed.
        hand_pos = self.ready_grasp_hand.unsqueeze(0).expand(n, -1).clone()
        arti_pos = self.ready_grasp_arti.unsqueeze(0).expand(n, -1).clone()

        # ---------- RSI: overwrite selected envs with random trajectory frames ----------
        # Sample only pre-success frames. Sampling the whole trajectory can
        # reset directly into task_success=True and produce one-step fake
        # successes, which was the main failure mode in the old training run.
        n_rsi = int(use_rsi.sum().item())
        if n_rsi > 0:
            pool_idx = torch.randint(
                low=0,
                high=self.rsi_frame_ids.numel(),
                size=(n_rsi,),
                device=self.device,
            )
            fidx = self.rsi_frame_ids[pool_idx]
            hand_pos[use_rsi] = self.traj_hand[fidx]   # [n_rsi, 51]
            arti_pos[use_rsi] = self.traj_arti[fidx]   # [n_rsi, Na]

        # ---------- Write new DOF state into the GPU tensor views ----------
        # dof_pos/dof_vel are views of dof_states, so these writes modify the
        # same underlying memory that set_dof_state_tensor reads.
        self.dof_pos[env_ids, :N_HAND_DOFS, 0] = hand_pos
        self.dof_vel[env_ids, :N_HAND_DOFS, 0] = 0.0
        self.dof_pos[env_ids, N_HAND_DOFS:, 0] = arti_pos
        self.dof_vel[env_ids, N_HAND_DOFS:, 0] = 0.0

        # ---------- Sync PD targets so the controller matches new state ----------
        self.pos_targets[env_ids, :N_HAND_DOFS] = hand_pos

        # ---------- Push to PhysX via indexed API ----------
        hand_ids = self.hand_actor_idxs[env_ids]
        arti_ids = self.arti_actor_idxs[env_ids]
        all_ids  = torch.cat([hand_ids, arti_ids])

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(all_ids),
            len(all_ids),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.pos_targets),
            gymtorch.unwrap_tensor(hand_ids),
            len(hand_ids),
        )

        # ---------- PICA v2a: dynamics randomization (damping / friction) ----
        # Sample fresh per-env scales and push them to PhysX BEFORE the
        # settling substeps, so the post-reset physics already runs under
        # the new dynamics. No-op when dynamics_randomization is disabled.
        if self._dyn_rand_enabled:
            self._apply_dyn_rand(env_ids)

        # ---------- Settling sub-steps ----------
        # Advance PhysX a few ticks with the PD targets frozen at the new
        # pose so contact forces stabilise before the policy's next action
        # perturbs the state.  Non-reset envs hold their last-commanded
        # targets during these sub-steps, so their PD controllers continue
        # converging but no additional policy command is injected.
        if self.n_settle_substeps > 0:
            for _ in range(self.n_settle_substeps):
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # ---------- Publish episode metrics for the terminating envs ----------
        # Two consumers:
        #   extras["episode"]  -- mean over the envs that just terminated (
        #                         rl_games / skrl forward this to TB / WandB).
        #   epoch accumulators -- summed across every reset_idx() call in the
        #                         current PPO iteration; end_epoch() emits the
        #                         iteration-level mean into extras["epoch"].
        episode_info = {}
        for key, buf in self.episode_sums.items():
            comp_sum = buf[env_ids].sum().item()
            episode_info[key] = comp_sum / n                       # batch mean
            self._epoch_sum_components[key] += comp_sum            # epoch sum
            buf[env_ids] = 0.0

        length_sum = self.episode_length_buf[env_ids].sum().item()
        episode_info["episode_length"] = length_sum / n
        self._epoch_sum_length   += length_sum
        self._epoch_num_episodes += n
        self.episode_length_buf[env_ids] = 0

        final_joint_sum = self.episode_final_joint_pos[env_ids].sum().item()
        final_progress_sum = self.episode_final_progress[env_ids].sum().item()
        episode_info["final_joint_pos"] = final_joint_sum / n
        episode_info["normalized_progress"] = final_progress_sum / n
        self._epoch_sum_final_joint_pos += final_joint_sum
        self._epoch_sum_normalized_progress += final_progress_sum

        # Track min / max episodic return seen this epoch (batch mean, cheap).
        batch_ret = episode_info["reward"]
        self._epoch_return_min = min(self._epoch_return_min, batch_ret)
        self._epoch_return_max = max(self._epoch_return_max, batch_ret)

        self.extras["episode"] = episode_info

        # ---------- Reset per-env bookkeeping ----------
        # prev_target_joint is read from the POST-settle state so the
        # first step's delta reward is zero for a stationary handle.
        self.progress_buf[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.error_history[env_ids] = 0.0
        # Zero a_{t-1} history at reset so the first GLA token after reset is
        # [error_0, 0] -- otherwise the encoder would see an action from the
        # previous episode, leaking trajectory information across resets.
        self.action_history[env_ids] = 0.0
        self.prev_target_joint[env_ids] = self.dof_pos[
            env_ids, N_HAND_DOFS + self.target_joint_idx, 0
        ]
        self.episode_final_joint_pos[env_ids] = self.prev_target_joint[env_ids]
        self.episode_final_progress[env_ids] = torch.clamp(
            (self.prev_target_joint[env_ids] - self.task_start_joint)
            / self.task_progress_denom,
            min=0.0,
        )

        # Arm detachment immediately if the reset pose is genuinely on the
        # true handle center. This preserves the "only after contact" gate
        # while making from-grasp resets penalize real first-step slips.
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hdl_pos, _ = self._true_handle_pose()
        wrist_pos = self.rb_states[self.wrist_body_idxs, :3]
        wrist_rot = self.rb_states[self.wrist_body_idxs, 3:7]
        palm_pos = wrist_pos + quat_rotate(wrist_rot, self.palm_offset)
        reset_dist = torch.norm(hdl_pos[env_ids] - palm_pos[env_ids], dim=-1)
        self.detach_armed_buf[env_ids] = reset_dist <= self.detach_dist

        # PICA v1: seed previous-step buffers for the freshly reset envs so
        # the first compute_reward after reset sees zero slip / zero action
        # smoothness contribution.
        self.prev_palm_to_handle_dist[env_ids] = reset_dist
        self.prev_actions_for_phys[env_ids] = 0.0

        # Seed both history rings with the post-reset palm distance and
        # target-joint position. Filling all slots with the reset value keeps
        # q_response_K / max_dist_K / detach_proxy_K at their natural reset
        # values for the first K steps.
        post_reset_q_obj = self.dof_pos[
            env_ids, N_HAND_DOFS + self.target_joint_idx, 0,
        ]
        self._aux_palm_dist_hist[env_ids] = reset_dist.unsqueeze(-1)
        self._aux_q_obj_hist[env_ids] = post_reset_q_obj.unsqueeze(-1)
        self._palm_dist_hist[env_ids] = reset_dist.unsqueeze(-1)
        self._q_obj_hist[env_ids] = post_reset_q_obj.unsqueeze(-1)

        # PICA v2d: reset per-env stall counter and reconfig cooldown. Both
        # accumulate within an episode and must restart cleanly on reset so
        # post-reset frames are not punished / rewarded for state inherited
        # from the previous episode.
        self._stall_steps[env_ids] = 0
        self._reconfig_cooldown[env_ids] = 0

    # =====================================================================
    #  CORE METHOD 2:  compute_observations
    # =====================================================================

    def compute_observations(self) -> torch.Tensor:
        """
        Build the observation tensor from current simulation state.

        Layout per env (GLA / Phase 4 obs, history_token_dim=102):
          [ hand_dof_pos (51) | hand_dof_vel (51) |
            handle_pos   ( 3) | optional handle_rot (4) |
            palm_to_handle_vec (3) | palm_to_handle_dist (1) |
            arti_dof_pos (Na) | arti_dof_vel (Na) |
            history (16 * 102) ]

        The trailing `history_len * history_token_dim` block flattens
        `[N, 16, 102]` in C order, so `obs[:, -16*102:].view(B, 16, 102)` in
        the network recovers the per-timestep tokens [error_t, a_{t-1}] used
        by the GLA encoder. With include_prev_action_in_history=False the
        token collapses to error_t alone for the flat-history baseline.

        All values are raw (un-normalised); rl_games' RunningMeanStd handles
        normalisation downstream of this method.
        """
        # Refresh GPU state tensors from PhysX
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Slice DOF tensor views -- no data copy, just pointer arithmetic
        hand_pos = self.dof_pos[:, :N_HAND_DOFS, 0]            # [N, 51]
        hand_vel = self.dof_vel[:, :N_HAND_DOFS, 0]            # [N, 51]

        # True handle center, reconstructed from the moving parent body pose.
        # hdl_rot remains the parent rigid body orientation, which is the
        # frame used by handle_center_local.
        hdl_pos, hdl_rot = self._true_handle_pose()            # [N, 3], [N, 4]

        # Palm world position: wrist rigid body + R(wrist_quat) @ palm_offset_local
        # Vectorised across envs; no Python loop.
        wrist_pos = self.rb_states[self.wrist_body_idxs, :3]   # [N, 3]
        wrist_rot = self.rb_states[self.wrist_body_idxs, 3:7]  # [N, 4] xyzw
        palm_pos  = wrist_pos + quat_rotate(wrist_rot, self.palm_offset)  # [N, 3]

        # Palm -> handle geometry (the missing direct signal)
        palm_to_handle_vec  = hdl_pos - palm_pos                                   # [N, 3]
        palm_to_handle_dist = torch.norm(palm_to_handle_vec, dim=-1, keepdim=True) # [N, 1]

        # Cache for compute_reward so we don't recompute FK + refresh
        self.palm_pos = palm_pos
        self.palm_to_handle_dist = palm_to_handle_dist.squeeze(-1)

        # Articulated object DOF states
        arti_pos = self.dof_pos[:, N_HAND_DOFS:, 0]            # [N, Na]
        arti_vel = self.dof_vel[:, N_HAND_DOFS:, 0]            # [N, Na]

        current_error = self.pos_targets[:, :N_HAND_DOFS] - hand_pos
        self.error_history = torch.roll(self.error_history, shifts=-1, dims=1)
        self.error_history[:, -1, :] = current_error

        if self.include_prev_action_in_history:
            # self.actions holds the action just applied by pre_physics_step
            # (a_{t-1} relative to the observation we are about to emit).
            # Token layout per timestep: [error_t (51), a_{t-1} (51)] -> 102-D.
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = self.actions
            history_tokens = torch.cat(
                [self.error_history, self.action_history], dim=-1,
            )                                                  # [N, 16, 102]
            flat_history = history_tokens.view(self.num_envs, -1)
        else:
            flat_history = self.error_history.view(self.num_envs, -1)

        obs_parts = [hand_pos, hand_vel, hdl_pos]
        if self.include_handle_rot:
            obs_parts.append(hdl_rot)
        obs_parts.extend([
            palm_to_handle_vec,
            palm_to_handle_dist,
            arti_pos,
            arti_vel,
            flat_history,
        ])

        # ---- PICA v2b/v2c: physical auxiliary targets (obs tail transport) ----
        # The actor / critic / GLA history NEVER read these channels -- the
        # GLA network slices them off the back of obs before splitting into
        # base_obs + history_block. They exist purely to feed the aux head
        # at training time.
        #
        # Two modes are supported:
        #   "current"        v2b: per-step targets [dq_obj, slip_proxy, tracking_stress]
        #   "causal_horizon" v2c: past-window targets [q_response_K, max_dist_K,
        #                          detach_proxy_K, tracking_stress]
        #
        # Order MUST match the order in self._phys_aux_cfg["target_keys"] so
        # the network's per-channel slice / weight stays aligned with the env.
        if self._phys_aux_enabled:
            keys = self._phys_aux_cfg["target_keys"]
            cur_q_obj = arti_pos[:, self.target_joint_idx]
            # v2c aux targets use their own observation-stage ring so they
            # stay aligned with the obs tick even when v2d reward code is on.
            self._aux_palm_dist_hist = torch.roll(
                self._aux_palm_dist_hist, shifts=-1, dims=1,
            )
            self._aux_palm_dist_hist[:, -1] = self.palm_to_handle_dist
            self._aux_q_obj_hist = torch.roll(self._aux_q_obj_hist, shifts=-1, dims=1)
            self._aux_q_obj_hist[:, -1] = cur_q_obj
            idx_tK = self._aux_idx_tK
            cols = []
            for k in keys:
                if k == "dq_obj":
                    col = (cur_q_obj - self.prev_target_joint).unsqueeze(-1)
                elif k == "slip_proxy":
                    col = (
                        self.palm_to_handle_dist - 0.08
                    ).clamp(min=0.0).unsqueeze(-1)
                elif k == "q_response_K":
                    col = (
                        self._aux_q_obj_hist[:, -1] - self._aux_q_obj_hist[:, idx_tK]
                    ).unsqueeze(-1)
                elif k == "max_dist_K":
                    col = self._aux_palm_dist_hist[:, idx_tK:].max(dim=1).values.unsqueeze(-1)
                elif k == "detach_proxy_K":
                    max_d = self._aux_palm_dist_hist[:, idx_tK:].max(dim=1).values
                    col = (max_d > self.detach_dist).float().unsqueeze(-1)
                elif k == "tracking_stress":
                    col = current_error.norm(dim=-1, keepdim=True)
                else:
                    col = torch.zeros(self.num_envs, 1, device=self.device)
                cols.append(col)
            aux_block = torch.cat(cols, dim=-1)
            self._aux_targets_last = aux_block.detach()
            gcfg = self._phys_aux_cfg["gating"]
            self._phys_aux_gate_last = torch.sigmoid(
                gcfg["sharpness"] * (gcfg["d_valid"] - self.palm_to_handle_dist)
            ).detach()
            obs_parts.append(aux_block)

        # Concatenate into a single observation vector
        self.obs_buf = torch.cat(obs_parts, dim=-1)
        return self.obs_buf

    # =====================================================================
    #  CORE METHOD 3:  compute_reward
    # =====================================================================

    def compute_reward(self):
        """
        Vectorised reward computation -- pure PyTorch, no Python loops.

        Components
        ----------
        r_dist    : weak palm-handle proximity shaping        anti-slip only
        r_task    : +w_task * delta(target_joint_position)    dominant progress
        r_act     : -w_act  * mean(action^2)                  smoothness
        r_time    : constant negative per-step reward          anti-stall
        r_detach  : r_detach_penalty  if palm slipped off *before* task done
        r_success : r_success_bonus   on the step task_success first fires

        Termination
        -----------
        done_buf = timeout  OR  detach_fail  OR  task_success

        Success and detachment are mutually exclusive by construction: the
        detach gate is AND-ed with ``task_not_done``, so once the joint
        crosses the success threshold the agent can no longer be penalised
        for letting go of the handle.  The episode also resets *that same
        step*, so there is no "floating around after success" phase.
        """
        # Refresh GPU state tensors from PhysX
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ---- 1. Handle geometry: true dynamic handle center ----
        # Recompute palm/handle distance from the freshly-refreshed rigid-body
        # tensors. Using the cached observation distance makes reward/done one
        # sim step stale, which is especially confusing in play mode.
        hdl_pos, _ = self._true_handle_pose()
        wrist_pos = self.rb_states[self.wrist_body_idxs, :3]
        wrist_rot = self.rb_states[self.wrist_body_idxs, 3:7]
        palm_pos = wrist_pos + quat_rotate(wrist_rot, self.palm_offset)
        palm_to_handle_vec = hdl_pos - palm_pos
        dist = torch.norm(palm_to_handle_vec, dim=-1)
        self.palm_pos = palm_pos
        self.palm_to_handle_dist = dist

        # ---- PICA v2d reward ring: update only for ARAM/reconfig terms. ----
        # v2c physical auxiliary targets use a separate observation-stage ring.
        if self._aram_enabled or self._reconfig_enabled:
            cur_q_obj_for_ring = self.dof_pos[
                :, N_HAND_DOFS + self.target_joint_idx, 0,
            ]
            self._palm_dist_hist = torch.roll(
                self._palm_dist_hist, shifts=-1, dims=1,
            )
            self._palm_dist_hist[:, -1] = dist
            self._q_obj_hist = torch.roll(self._q_obj_hist, shifts=-1, dims=1)
            self._q_obj_hist[:, -1] = cur_q_obj_for_ring

        # ---- 2. Weak distance shaping: enough to keep the grasp, not enough
        #         to make hovering more profitable than opening the object. ----
        r_dist = (
            -self.w_dist_linear * dist
            + self.w_dist_exp * torch.exp(-self.dist_sharpness * dist)
        )

        # ---- Success / not-done masks (shared by r_detach, r_success, done) ----
        cur_joint_abs = self.dof_pos[
            :, N_HAND_DOFS + self.target_joint_idx, 0
        ]                                                       # [N]
        task_success  = cur_joint_abs >= self.task_done_joint   # [N] bool
        task_not_done = ~task_success                           # [N] bool

        # Normalized physical progress for paper/report logging. It is not
        # summed as a reward component; reset_idx publishes the last value of
        # each finished episode into extras["episode"] and extras["epoch"].
        normalized_progress = torch.clamp(
            (cur_joint_abs - self.task_start_joint) / self.task_progress_denom,
            min=0.0,
        )
        self.episode_final_joint_pos.copy_(cur_joint_abs.detach())
        self.episode_final_progress.copy_(normalized_progress.detach())
        self._normalized_progress_last = normalized_progress.detach()

        # ---- 3. Task reward: delta in target joint position ----
        delta_q = cur_joint_abs - self.prev_target_joint
        r_task  = self.w_task * delta_q
        self.prev_target_joint = cur_joint_abs.clone()

        # ---- 4. Action smoothness and time penalties ----
        r_act = -self.w_act * torch.mean(self.actions ** 2, dim=-1)  # [N]
        r_time = torch.full_like(dist, self.r_time_penalty)

        # ---- 5. Detachment gate: slipped off BEFORE the task was done ----
        # Detachment is only meaningful after the palm has first reached the
        # handle. Otherwise zero-RSI/play resets that start far away are
        # incorrectly treated as a slipped grasp on step 1.
        near_handle = dist <= self.detach_dist                  # [N] bool
        detached = dist > self.detach_dist                      # [N] bool
        detach_fail = task_not_done & self.detach_armed_buf & detached
        self.detach_armed_buf |= near_handle
        r_detach    = torch.where(
            detach_fail,
            torch.full_like(dist, self.r_detach_penalty),
            torch.zeros_like(dist),
        )

        # ---- 6. Success bonus: one-shot +r_success_bonus when threshold crossed ----
        # Because task_success also sets done_buf below, the episode resets
        # the very next step -- so this bonus is delivered exactly once.
        r_success = torch.where(
            task_success,
            torch.full_like(dist, self.r_success_bonus),
            torch.zeros_like(dist),
        )

        # ---- Cache per-component rewards for episode logging ----
        self._r_dist_last    = r_dist.detach()
        self._r_task_last    = r_task.detach()
        self._r_act_last     = r_act.detach()
        self._r_detach_last  = r_detach.detach()
        self._r_success_last = r_success.detach()
        self._r_time_last    = r_time.detach()
        self._success_flag   = task_success.float().detach()

        # ---- PICA v1: physical regularization terms ----
        # Each weight defaults to 0.0, so when physical_regularization is
        # missing or disabled, these contribute exactly 0.0 to rew_buf and
        # the math is bit-identical to the pre-PICA reward.
        cfg = self._phys_reg_cfg

        # 1. Action saturation penalty: penalise excursions of |a| past
        #    the bound threshold (e.g. 0.90), squared so the gradient
        #    grows as the policy pushes harder into the clamp.
        bound_excess = (
            self.actions.abs() - cfg["action_bound"]["threshold"]
        ).clamp(min=0.0)
        r_phys_bound = (
            -cfg["action_bound"]["weight"] * bound_excess.pow(2).mean(dim=-1)
        )

        # 2. Contact distance: penalise palm drifting past d_safe from the
        #    handle, squared. Uses the FRESH dist computed above, not the
        #    stale self.palm_to_handle_dist (which compute_observations
        #    overwrites at end of step).
        dist_excess = (dist - cfg["contact_distance"]["safe_dist"]).clamp(min=0.0)
        r_phys_contact = -cfg["contact_distance"]["weight"] * dist_excess.pow(2)

        # 3. Slip-aware action penalty: when the palm slips further from
        #    the handle this step, penalise the action energy that caused
        #    the slip. prev_palm_to_handle_dist is the dedicated previous-
        #    step buffer; it is updated at the END of this method, so at
        #    this point it still holds last step's distance.
        dist_delta = dist - self.prev_palm_to_handle_dist
        slip = dist_delta.clamp(min=0.0)
        action_energy = self.actions.pow(2).mean(dim=-1)
        r_phys_slip = -cfg["slip_action"]["weight"] * slip * action_energy

        # 4. Action smoothness: penalise large step-to-step action changes.
        delta_action = self.actions - self.prev_actions_for_phys
        r_phys_smooth = (
            -cfg["action_smoothness"]["weight"] * delta_action.pow(2).mean(dim=-1)
        )

        r_phys_total = (
            r_phys_bound + r_phys_contact + r_phys_slip + r_phys_smooth
        )

        # Cache for episode logging (mirrors the original _r_<name>_last set).
        self._r_phys_bound_last   = r_phys_bound.detach()
        self._r_phys_contact_last = r_phys_contact.detach()
        self._r_phys_slip_last    = r_phys_slip.detach()
        self._r_phys_smooth_last  = r_phys_smooth.detach()
        self._r_phys_total_last   = r_phys_total.detach()

        # ---- PICA v2d: ARAM-lite + reconfiguration reward ----
        # Both terms read from the (now up-to-date) ring buffers and are
        # additive on top of the standard reward. When their respective
        # configs are disabled, all four cached tensors stay at zero and
        # contribute nothing to rew_buf -- v2c behaviour is preserved.
        action_l2 = self.actions.norm(dim=-1)                    # [N]
        self._action_l2_last = action_l2.detach()
        self._clip099_last = (self.actions.abs().amax(dim=-1) >= 0.99).float().detach()

        idx_tK = self._idx_tK
        # K-window stats over the most recent K+1 entries of the ring.
        max_dist_K_t = self._palm_dist_hist[:, idx_tK:].max(dim=1).values
        q_response_K_t = (
            self._q_obj_hist[:, -1] - self._q_obj_hist[:, idx_tK]
        )

        # ---- ARAM-lite: adaptive saturation penalty inside a strict gate.
        r_aram = torch.zeros_like(dist)
        aram_gate_f = torch.zeros_like(dist)
        if self._aram_enabled:
            acfg = self._aram_cfg
            rg = acfg["resistance_gate"]
            # Stall counter: increments when K-step object response is below
            # threshold. Modes:
            #   raw              -> use the legacy raw joint-unit threshold;
            #   normalized       -> use progress-normalized response;
            #   joint_type_aware -> raw for revolute, normalized for prismatic.
            response_mode = str(rg.get("q_response_min_mode", "raw")).lower()
            use_normalized_response = response_mode == "normalized" or (
                response_mode == "joint_type_aware"
                and getattr(self, "target_joint_type", "unknown") == "prismatic"
            )
            if use_normalized_response and rg.get("q_response_min_frac", None) is not None:
                progress_response_K = q_response_K_t.abs() / max(
                    1e-6, self.task_progress_denom,
                )
                low_resp = progress_response_K < float(rg["q_response_min_frac"])
            else:
                low_resp = q_response_K_t.abs() < rg["q_response_min"]
            self._stall_steps = torch.where(
                low_resp,
                self._stall_steps + 1,
                torch.zeros_like(self._stall_steps),
            )
            high_resistance = self._stall_steps >= int(rg["min_stall_steps"])

            contact_valid = dist < float(rg["contact_dist"])
            not_detached  = dist < self.detach_dist
            high_effort   = action_l2 > float(rg["min_action_l2"])

            aram_gate = contact_valid & not_detached & high_effort & (
                high_resistance if rg["enabled"] else torch.ones_like(contact_valid)
            )
            aram_gate_f = aram_gate.float()
            action_energy = self.actions.pow(2).mean(dim=-1)         # [N]
            r_aram = -float(self.lambda_aram) * aram_gate_f * action_energy

        # ---- Reconfig reward: positive shaping for "useful repositioning"
        # when contact is preserved. Requires the 2K+1 ring (already
        # allocated when reconfig_enabled at __init__ time).
        r_reconfig = torch.zeros_like(dist)
        reconfig_gate_f = torch.zeros_like(dist)
        if self._reconfig_enabled and self._aux_buflen >= (2 * max(1, self._aux_horizon) + 1):
            rcfg = self._reconfig_cfg
            K = int(self._aux_horizon)
            # Older K-window stats use the leading K+1 entries of the ring.
            max_dist_K_tK = self._palm_dist_hist[:, : K + 1].max(dim=1).values
            q_response_K_tK = (
                self._q_obj_hist[:, idx_tK] - self._q_obj_hist[:, 0]
            )
            palm_dist_t  = dist
            palm_dist_tK = self._palm_dist_hist[:, idx_tK]

            contact_improve  = (palm_dist_tK - palm_dist_t).clamp(min=0.0)
            response_improve = (q_response_K_t - q_response_K_tK).clamp(min=0.0)
            slip_reduce      = (max_dist_K_tK - max_dist_K_t).clamp(min=0.0)

            # Hard filters: large action delta or hand-velocity spikes
            # disqualify the sample regardless of geometry.
            delta_action = self.actions - self.prev_actions_for_phys
            action_delta_l2 = delta_action.norm(dim=-1)
            hand_vel_l2 = self.dof_vel[:, :N_HAND_DOFS, 0].norm(dim=-1)
            filter_ok = (
                (action_delta_l2 < float(rcfg["action_delta_threshold"]))
                & (hand_vel_l2 < float(rcfg["hand_velocity_threshold"]))
            )

            # Progress stall: |progress over K steps| < threshold.
            progress_K = q_response_K_t / max(1e-6, self.task_progress_denom)
            progress_stalled = progress_K.abs() < float(rcfg["progress_stall_threshold"])

            cooldown_ok   = self._reconfig_cooldown == 0
            contact_valid = dist < float(rcfg["contact_dist"])
            not_detached  = dist < self.detach_dist
            high_effort   = action_l2 > float(rcfg["min_action_l2"])

            reconfig_gate = (
                contact_valid & not_detached & high_effort
                & progress_stalled & cooldown_ok & filter_ok
            )
            reconfig_gate_f = reconfig_gate.float()

            r_reconfig = reconfig_gate_f * (
                float(rcfg["weight_contact_improve"])  * contact_improve
                + float(rcfg["weight_response_improve"]) * response_improve
                + float(rcfg["weight_slip_reduce"])      * slip_reduce
            )

            # Update cooldown: set to N where reconfig fired, decrement (>=0)
            # otherwise.
            cd_steps = int(rcfg["cooldown_steps"])
            self._reconfig_cooldown = torch.where(
                reconfig_gate,
                torch.full_like(self._reconfig_cooldown, cd_steps),
                (self._reconfig_cooldown - 1).clamp(min=0),
            )

        self._r_aram_last         = r_aram.detach()
        self._r_reconfig_last     = r_reconfig.detach()
        self._aram_gate_last      = aram_gate_f.detach()
        self._reconfig_gate_last  = reconfig_gate_f.detach()

        # ---- Total reward (PICA penalties + v2d shaping; zero by default) ----
        self.rew_buf = (
            r_dist + r_task + r_act + r_detach + r_success + r_time
            + r_phys_total + r_aram + r_reconfig
        )

        # ---- Update prev_palm_to_handle_dist for the NEXT step's slip term ----
        # This must happen AFTER r_phys_slip is computed and BEFORE the next
        # call to compute_reward. End of method is the cleanest spot.
        self.prev_palm_to_handle_dist.copy_(dist.detach())

        # ---- Episode termination: timeout OR detach-fail OR success ----
        # Early-terminating on success prevents the "float aimlessly after
        # opening the drawer" regime that was polluting rollouts.
        self.progress_buf += 1
        timeout       = self.progress_buf >= self.max_episode_length
        self.done_buf = timeout | detach_fail | task_success

        return self.rew_buf, self.done_buf

    # =====================================================================
    #  CORE METHOD 4:  pre_physics_step  --  Delta position control
    # =====================================================================

    def pre_physics_step(self, actions: torch.Tensor):
        """
        Map network output to PD position targets.

        Pipeline
        --------
        1. Clamp raw actions to [-1, 1]  (safety net for the policy output)
        2. In eval, optionally low-pass filter actions with EMA.
        3. Scale: delta = actions * max_delta  (0.05 rad/m)
        4. Add to current DOF positions:  target = cur_pos + delta
        5. Clamp to joint limits:  target = clamp(target, lower, upper)
        6. Push to Isaac Gym PD controller via set_dof_position_target_tensor

        The delta formulation means the policy only needs to output small
        corrections each step, which is much easier to learn than absolute
        position targets for a 51-DOF system.
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        # PICA v1: snapshot a_{t-1} BEFORE the EMA / overwrite below. At entry,
        # self.actions still holds the action that was applied last step (or
        # zeros after reset, since reset_idx zeroes it). compute_reward uses
        # this against the new self.actions to compute r_phys_smooth.
        self.prev_actions_for_phys.copy_(self.actions)

        # Clamp raw network output to [-1, 1]
        raw_actions = actions.clamp(-1.0, 1.0)

        if self.is_eval_mode and self.eval_action_ema_alpha < 1.0:
            alpha = float(self.eval_action_ema_alpha)
            self.filtered_actions.mul_(1.0 - alpha).add_(raw_actions, alpha=alpha)
            self.actions = self.filtered_actions
        else:
            self.actions = raw_actions
            self.filtered_actions.copy_(raw_actions)

        # Scale to physical displacement (max 0.05 rad/m per step)
        delta = self.actions * self.max_delta                   # [N, 51]

        # Read current hand joint positions from the GPU state tensor
        cur_pos = self.dof_pos[:, :N_HAND_DOFS, 0]             # [N, 51]

        # Compute new target = current + delta, respecting joint limits
        new_targets = torch.clamp(
            cur_pos + delta,
            self.hand_dof_lower,   # [51] broadcasts to [N, 51]
            self.hand_dof_upper,   # [51] broadcasts to [N, 51]
        )

        # Write into the target buffer and push to PD controller
        self.pos_targets[:, :N_HAND_DOFS] = new_targets
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pos_targets),
        )

    # =====================================================================
    #  Convenience: full RL step + initial reset
    # =====================================================================

    def reset(self) -> torch.Tensor:
        """Reset ALL environments and return initial observations."""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # One sim step so rigid-body states match the newly-set DOF state
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._render_viewer()
        return self.compute_observations()

    def _render_viewer(self):
        """Update the Isaac Gym viewer when running with --no-headless."""
        if self.env.headless:
            return
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.env.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def step(self, actions: torch.Tensor):
        """
        Full RL step:  action -> simulate -> reward -> auto-reset -> observe.

        Returns
        -------
        obs    : [N, obs_dim]   observations (post-reset for finished envs)
        reward : [N]            per-env scalar reward
        done   : [N]            bool, True if episode ended this step
        info   : dict           self.extras -- contains "episode" dict on
                                env terminations plus rsi_prob / global_step
                                so the PPO runner can log to TB / WandB.
        """
        # 1. Apply delta-position actions to the PD controller
        self.pre_physics_step(actions)

        # Advance curriculum clock (one physics step per env).  If the PPO
        # runner does not explicitly call update_rsi_prob, fall back to
        # letting the env self-pace its own curriculum using this counter
        # and self.max_training_steps (set to rsi_decay_steps by default).
        self.global_step_counter += self.num_envs
        self.total_env_steps      = self.global_step_counter  # legacy alias
        if not self.is_eval_mode:
            self.update_rsi_prob(self.global_step_counter, self.max_training_steps)

        # 2. Advance physics one step (no viewer sync for training speed)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._render_viewer()

        # 3. Compute reward and check termination
        self.compute_reward()

        # 4. Accumulate per-env episode sums *before* reset clears them.
        #    "success" is stored as a 0/1 flag (not a sum) so the per-epoch
        #    mean collapses naturally into a success rate in end_epoch().
        self.episode_sums["reward"]    += self.rew_buf
        self.episode_sums["r_dist"]    += self._r_dist_last
        self.episode_sums["r_task"]    += self._r_task_last
        self.episode_sums["r_act"]     += self._r_act_last
        self.episode_sums["r_detach"]  += self._r_detach_last
        self.episode_sums["r_success"] += self._r_success_last
        self.episode_sums["r_time"]    += self._r_time_last
        # PICA v1: physical regularization components.
        self.episode_sums["r_phys_bound"]   += self._r_phys_bound_last
        self.episode_sums["r_phys_contact"] += self._r_phys_contact_last
        self.episode_sums["r_phys_slip"]    += self._r_phys_slip_last
        self.episode_sums["r_phys_smooth"]  += self._r_phys_smooth_last
        self.episode_sums["r_phys_total"]   += self._r_phys_total_last
        # PICA v2d: ARAM-lite + reconfig reward (always present in schema;
        # literal zero when their respective configs are disabled).
        self.episode_sums["r_aram"]     += self._r_aram_last
        self.episode_sums["r_reconfig"] += self._r_reconfig_last
        # Per-step rate accumulators (per-step, not per-episode).
        self._epoch_clip099_sum   += float(self._clip099_last.sum().item())
        self._epoch_clip099_n     += int(self.num_envs)
        self._epoch_action_l2_sum += float(self._action_l2_last.sum().item())
        self._epoch_action_l2_n   += int(self.num_envs)
        self._epoch_aram_gate_sum     += float(self._aram_gate_last.sum().item())
        self._epoch_reconfig_gate_sum += float(self._reconfig_gate_last.sum().item())
        self._epoch_gate_n            += int(self.num_envs)
        # Latch-style: once an env succeeds, the flag stays 1 for the rest
        # of the episode (the episode terminates that same step anyway).
        self.episode_sums["success"] = torch.maximum(
            self.episode_sums["success"], self._success_flag,
        )
        self.episode_length_buf += 1

        # 5. Auto-reset any finished episodes (reset_idx publishes
        #    self.extras["episode"] with the means over terminating envs)
        done_ids = self.done_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self.reset_idx(done_ids)

        # 6. Build observations for the next policy query
        #    Note: for just-reset envs, DOF positions are correct but
        #    rigid-body positions update after the next simulate() call.
        #    This one-frame staleness is standard in Isaac Gym vec-envs.
        self.compute_observations()

        # 7. Per-step scalars for TB / WandB (PPO runner reads self.extras)
        self.extras["rsi_prob"]     = self.rsi_prob
        self.extras["global_step"]  = self.global_step_counter
        self.extras["is_eval_mode"] = float(self.is_eval_mode)

        return self.obs_buf, self.rew_buf, self.done_buf, self.extras
