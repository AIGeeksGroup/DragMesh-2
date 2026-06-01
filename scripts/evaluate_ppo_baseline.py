"""
Evaluate an rl_games PPO checkpoint with HandDrag task metrics.

This complements ``ppo/train.py --play``: rl_games prints reward/steps only,
while this script writes the task metrics needed for baseline tables.

Example:
    python scripts/evaluate_ppo_baseline.py \
        --object_id 45936 \
        --checkpoint runs/hand_drag_ppo_45936/nn \
        --checkpoint-kind latest \
        --episodes 20
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import yaml

# Isaac Gym MUST be imported before torch.
from isaacgym import gymapi, gymtorch  # noqa: F401
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.geometric.hand_object_gym import HandObjectGym, N_HAND_DOFS  # noqa: E402
from ppo.hand_drag_task import HandDragTask  # noqa: E402
from ppo.rlgames_wrapper import resolve_handle_link_name  # noqa: E402

# Make sure the vendored fla package is importable for GLA checkpoints.
_FLA_ROOT = str((PROJECT_ROOT / "flash-linear-attention").resolve())
if _FLA_ROOT not in sys.path:
    sys.path.insert(0, _FLA_ROOT)


def _is_gla_checkpoint(model_state):
    return any(
        k.startswith("a2c_network.gla.") or k.startswith("a2c_network.token_proj")
        for k in model_state
    )


class RLGamesActor(nn.Module):
    """Actor matching configs/train/mlp/ppo.yaml.

    Supports both deterministic mu and stochastic Gaussian sampling.
    Stochastic mode loads ``a2c_network.sigma`` (logstd, shape [act_dim]),
    which is what rl_games trains with under ``fixed_sigma=True``. The
    deterministic mode is what a hand-engineered controller would do at
    deploy time; stochastic mode is what produced the training reward
    curve, so reporting both makes the sim2eval gap visible.
    """

    def __init__(self, checkpoint, device, stochastic=False):
        super().__init__()
        model = checkpoint["model"]
        obs_dim = int(model["a2c_network.actor_mlp.0.weight"].shape[1])
        hidden0 = int(model["a2c_network.actor_mlp.0.weight"].shape[0])
        hidden1 = int(model["a2c_network.actor_mlp.2.weight"].shape[0])
        hidden2 = int(model["a2c_network.actor_mlp.4.weight"].shape[0])
        act_dim = int(model["a2c_network.mu.weight"].shape[0])
        self.obs_dim = obs_dim
        self.stochastic = bool(stochastic)

        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden0),
            nn.ELU(),
            nn.Linear(hidden0, hidden1),
            nn.ELU(),
            nn.Linear(hidden1, hidden2),
            nn.ELU(),
        )
        self.mu = nn.Linear(hidden2, act_dim)

        self.actor_mlp[0].weight.data.copy_(model["a2c_network.actor_mlp.0.weight"])
        self.actor_mlp[0].bias.data.copy_(model["a2c_network.actor_mlp.0.bias"])
        self.actor_mlp[2].weight.data.copy_(model["a2c_network.actor_mlp.2.weight"])
        self.actor_mlp[2].bias.data.copy_(model["a2c_network.actor_mlp.2.bias"])
        self.actor_mlp[4].weight.data.copy_(model["a2c_network.actor_mlp.4.weight"])
        self.actor_mlp[4].bias.data.copy_(model["a2c_network.actor_mlp.4.bias"])
        self.mu.weight.data.copy_(model["a2c_network.mu.weight"])
        self.mu.bias.data.copy_(model["a2c_network.mu.bias"])

        sigma_param = model.get("a2c_network.sigma")
        if sigma_param is None:
            self.register_buffer("logstd", torch.full((act_dim,), -2.0))
        else:
            self.register_buffer("logstd", sigma_param.float().view(-1))

        rms = checkpoint.get("running_mean_std")
        if rms is None:
            self.register_buffer("running_mean", torch.zeros(obs_dim))
            self.register_buffer("running_var", torch.ones(obs_dim))
        else:
            self.register_buffer("running_mean", rms["running_mean"].float())
            self.register_buffer("running_var", rms["running_var"].float())

        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, obs):
        if obs.shape[-1] > self.obs_dim:
            obs = obs[..., :self.obs_dim]
        elif obs.shape[-1] < self.obs_dim:
            raise ValueError(
                f"Observation dim {obs.shape[-1]} is smaller than checkpoint input {self.obs_dim}"
            )
        obs = (obs - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        obs = torch.clamp(obs, -5.0, 5.0)
        mu = self.mu(self.actor_mlp(obs))
        if self.stochastic:
            std = torch.exp(self.logstd).expand_as(mu)
            mu = mu + torch.randn_like(mu) * std
        return mu


class GLAActor(nn.Module):
    """Actor matching configs/train/gla/ppo.yaml.

    Reconstructs the GLA Network from `backbones/gla/a2c_network.py` and copies
    weights from a checkpoint. Architecture parameters are inferred from
    the state-dict shapes:
      base_dim       = actor_mlp.0.weight.shape[1]
      mlp_units      = [actor_mlp.0.weight.shape[0],
                        actor_mlp.2.weight.shape[0],
                        actor_mlp.4.weight.shape[0]]
      gla_hidden     = token_proj.weight.shape[0]
      token_dim      = token_proj.weight.shape[1]
      gla.num_heads  = inferred indirectly from gla.q_proj.weight (we just
                       reuse the default of 4 since key_dim=hidden*0.5=64,
                       and 64/4 = 16 head_k_dim is the standard config).
    """

    def __init__(self, checkpoint, device, history_length=16, stochastic=False,
                 pool="last", phys_aux=False):
        super().__init__()
        from backbones.gla.a2c_network import GLAA2CBuilder

        model = checkpoint["model"]
        base_dim = int(model["a2c_network.actor_mlp.0.weight"].shape[1])
        h0 = int(model["a2c_network.actor_mlp.0.weight"].shape[0])
        h1 = int(model["a2c_network.actor_mlp.2.weight"].shape[0])
        h2 = int(model["a2c_network.actor_mlp.4.weight"].shape[0])
        token_dim = int(model["a2c_network.token_proj.weight"].shape[1])
        gla_hidden = int(model["a2c_network.token_proj.weight"].shape[0])
        act_dim = int(model["a2c_network.mu.weight"].shape[0])

        # ---- PICA v2b: auto-detect aux head from state_dict ----
        # The presence of aux_head.* keys implies the checkpoint was trained
        # under physical_auxiliary, which means obs_dim = base + history + aux.
        # We honour an explicit `phys_aux` request as well so the caller can
        # force a build (e.g. for shape-mismatch debugging).
        aux_keys_present = any(
            k.startswith("a2c_network.aux_head.") for k in model
        )
        aux_enabled = bool(aux_keys_present or phys_aux)
        if aux_enabled and "a2c_network.aux_head.2.weight" in model:
            aux_pred_dim = int(model["a2c_network.aux_head.2.weight"].shape[0])
            aux_hidden   = int(model["a2c_network.aux_head.2.weight"].shape[1])
        elif aux_enabled:
            # Forced via --phys_aux but no aux_head in state_dict; use defaults
            # so the network exposes the right slot for the env's aux tail.
            aux_pred_dim = 3
            aux_hidden   = 64
        else:
            aux_pred_dim = 0
            aux_hidden   = 64
        # In v2b-init aux_target_dim == aux_pred_dim (no separate gate channel).
        aux_target_dim = aux_pred_dim if aux_enabled else 0

        self.base_dim = base_dim
        self.history_length = int(history_length)
        self.token_dim = token_dim
        self.aux_enabled = aux_enabled
        self.aux_pred_dim = aux_pred_dim
        self.aux_target_dim = aux_target_dim
        self.aux_hidden = aux_hidden
        self.aux_keys_present_in_ckpt = bool(aux_keys_present)
        self.obs_dim = (
            base_dim + self.history_length * token_dim + self.aux_target_dim
        )
        self.stochastic = bool(stochastic)

        params = {
            "separate": False,
            "mlp": {
                "units": [h0, h1, h2],
                "activation": "elu",
                "initializer": {"name": "default"},
                "d2rl": False,
                "norm_only_first_layer": False,
            },
            "space": {"continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": -2.0},
                "fixed_sigma": True,
            }},
            "gla": {
                "history_length": self.history_length,
                "token_dim": self.token_dim,
                "hidden_size": gla_hidden,
                "num_heads": 4,
                "expand_k": 0.5,
                "expand_v": 1.0,
                "mode": "chunk",
                "pool": str(pool).lower(),
            },
            # PICA v2b: build aux head exactly when checkpoint contains one,
            # so load_state_dict produces no unexpected keys. The head is
            # never read at inference because forward sets is_train=False.
            "phys_aux": {
                "enabled":     bool(aux_enabled),
                "pred_dim":    int(aux_pred_dim),
                "target_dim":  int(aux_target_dim),
                "hidden_size": int(aux_hidden),
            },
            "value_activation": "None",
            "normalization": None,
        }

        builder = GLAA2CBuilder()
        builder.load(params)
        self.net = builder.build(
            "a2c",
            actions_num=act_dim,
            input_shape=(self.obs_dim,),
            value_size=1,
            num_seqs=1,
        )

        # Strip the `a2c_network.` prefix from checkpoint keys before loading.
        ckpt_state = {
            k[len("a2c_network."):]: v
            for k, v in model.items()
            if k.startswith("a2c_network.")
        }
        missing, unexpected = self.net.load_state_dict(ckpt_state, strict=False)
        if missing:
            print(f"  [gla-actor] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  [gla-actor] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # Logstd buffer (sigma is loaded as part of the net; expose for sampling).
        sigma_param = model.get("a2c_network.sigma")
        if sigma_param is None:
            self.register_buffer("logstd", torch.full((act_dim,), -2.0))
        else:
            self.register_buffer("logstd", sigma_param.float().view(-1))

        rms = checkpoint.get("running_mean_std")
        if rms is None:
            self.register_buffer("running_mean", torch.zeros(self.obs_dim))
            self.register_buffer("running_var", torch.ones(self.obs_dim))
        else:
            self.register_buffer("running_mean", rms["running_mean"].float())
            self.register_buffer("running_var", rms["running_var"].float())

        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, obs):
        if obs.shape[-1] != self.obs_dim:
            current_history_aux = self.history_length * self.token_dim + self.aux_target_dim
            current_base_dim = obs.shape[-1] - current_history_aux
            pad_dim = self.base_dim - current_base_dim
            if 0 < pad_dim <= 8 and current_base_dim > 0:
                # Eval-only compatibility for older PICA checkpoints whose
                # base_obs had a few extra channels before the history block.
                # Insert zeros between base_obs and history/aux so temporal
                # tokens and aux targets remain aligned.
                base = obs[..., :current_base_dim]
                tail = obs[..., current_base_dim:]
                pad = torch.zeros(*obs.shape[:-1], pad_dim, device=obs.device, dtype=obs.dtype)
                obs = torch.cat([base, pad, tail], dim=-1)
            if obs.shape[-1] != self.obs_dim:
                raise ValueError(
                    f"Observation dim {obs.shape[-1]} != checkpoint obs_dim {self.obs_dim}"
                )
        obs_norm = (obs - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        obs_norm = torch.clamp(obs_norm, -5.0, 5.0)
        # PICA v2b: is_train=False short-circuits the aux head branch in
        # GLAA2CBuilder.Network.forward so no aux compute happens at eval.
        # Action selection therefore depends ONLY on the actor / GLA path,
        # which is identical to v1 / v2a behaviour.
        mu, sigma, _value, _states = self.net({"obs": obs_norm, "is_train": False})
        if self.stochastic:
            std = torch.exp(self.logstd).expand_as(mu)
            mu = mu + torch.randn_like(mu) * std
        return mu


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a HandDrag PPO checkpoint")
    parser.add_argument("--config", default="configs/env/hand_config.yaml")
    parser.add_argument("--object_id", default="45661")
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-kind", choices=("best", "latest"),
                        default="best")
    parser.add_argument("--target_joint_idx", type=int, default=None)
    parser.add_argument("--handle_link_name", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_episode_length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_alpha", type=float, default=1.0,
                        help="Eval action EMA alpha; 1 disables smoothing")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from N(mu, exp(logstd)) instead "
                             "of using deterministic mu. Matches what the "
                             "training reward curve was actually computed on.")
    parser.add_argument("--n_settle_substeps", type=int, default=None,
                        help="Override task.n_settle_substeps (eval only). "
                             "Default in task is 4. Larger values give the "
                             "PD controller more time to stabilize the grasp "
                             "after each reset before the policy starts "
                             "perturbing the state.")
    parser.add_argument("--detach_arm_delay", type=int, default=0,
                        help="Number of post-reset steps during which the "
                             "detach gate is forcibly disarmed. The default "
                             "task arms detach the moment palm-handle "
                             "distance is <= 0.1, which means a saturated "
                             "first-step action can fire the -50 detach "
                             "penalty in one tick. Eval-only override.")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--log_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    # ---- OOD dynamics overrides (eval-only, training defaults preserved) ----
    parser.add_argument("--object_damping_scale", type=float, default=1.0,
                        help="Multiply the articulated-object DOF damping by "
                             "this factor at eval time. Default 1.0 (training "
                             "value 10.0). Larger values make the joint more "
                             "viscous.")
    parser.add_argument("--object_friction_scale", type=float, default=1.0,
                        help="Multiply the articulated-object rigid-shape "
                             "friction by this factor at eval time. Default "
                             "1.0 (training value 5.0). Affects hand <-> "
                             "object contact friction only; hand surface "
                             "friction is unchanged.")
    parser.add_argument("--gla_pool", choices=("last", "mean"), default="last",
                        help="GLA temporal pool used at eval. Must match the "
                             "value the checkpoint was trained with: pool=last "
                             "(default, matches configs/train/gla/ppo.yaml) or "
                             "pool=mean (for checkpoints trained from "
                             "configs/train/gla/ppo_pool_mean.yaml). The state dict "
                             "is identical between the two so this cannot be "
                             "auto-detected from weights alone.")
    parser.add_argument("--phys_aux", "--physical_auxiliary",
                        dest="phys_aux", type=int, default=None,
                        help="0/1: enable physical_auxiliary at eval time so "
                             "HandDragTask appends aux targets to obs and "
                             "GLAActor builds the aux head. When omitted "
                             "(default), it is auto-detected from the "
                             "checkpoint's state_dict (presence of "
                             "a2c_network.aux_head.* keys). Forcing 1 on a "
                             "non-aux checkpoint is a debug knob; forcing 0 "
                             "on an aux checkpoint will raise on load.")
    return parser.parse_known_args()


def apply_ood_dynamics_overrides(env, damping_scale, friction_scale):
    """Eval-only override of articulated-object joint damping and shape friction.

    Re-applies modified DOF properties and shape properties to every parallel
    articulated actor that was already created by HandObjectGym.load_envs().
    Hand DOF properties and hand contact friction are left untouched, and the
    drive_mode (DOF_MODE_NONE) is preserved.

    Returns the (final_damping, final_friction) actually read back from
    Isaac Gym, so the caller can verify the override took effect.
    """
    from isaacgym import gymapi  # noqa: F401  (must be after env init)
    if damping_scale == 1.0 and friction_scale == 1.0:
        return None, None

    # Joint damping (per-DOF, but currently set globally to 10.0 for object).
    if damping_scale != 1.0:
        new_damp = float(env.arti_obj_dof_props["damping"][0]) * damping_scale
        env.arti_obj_dof_props["damping"][:] = new_damp
        for i in range(env.num_envs):
            env.gym.set_actor_dof_properties(
                env.envs[i], env.arti_actors[i], env.arti_obj_dof_props,
            )
    # Verify by reading back env 0's DOF damping.
    if damping_scale != 1.0:
        ck_props = env.gym.get_actor_dof_properties(
            env.envs[0], env.arti_actors[0],
        )
        damp_after = float(ck_props["damping"][0])
    else:
        damp_after = float(env.arti_obj_dof_props["damping"][0])

    # Contact friction (per-shape).
    if friction_scale != 1.0:
        for i in range(env.num_envs):
            sp = env.gym.get_actor_rigid_shape_properties(
                env.envs[i], env.arti_actors[i],
            )
            for s in sp:
                s.friction = float(s.friction) * friction_scale
            env.gym.set_actor_rigid_shape_properties(
                env.envs[i], env.arti_actors[i], sp,
            )
        sp_after = env.gym.get_actor_rigid_shape_properties(
            env.envs[0], env.arti_actors[0],
        )
        fric_after = float(sp_after[0].friction) if len(sp_after) > 0 else None
    else:
        fric_after = 5.0

    return damp_after, fric_after


def resolve_checkpoint_path(checkpoint_arg, checkpoint_kind, run_name="HandDrag"):
    checkpoint_path = Path(checkpoint_arg).resolve()
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")

    candidates = sorted(
        glob.glob(str(checkpoint_path / "*.pth")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found in: {checkpoint_path}")

    if checkpoint_kind == "latest":
        return Path(candidates[0])

    preferred = checkpoint_path / f"{run_name}.pth"
    return preferred if preferred.exists() else Path(candidates[0])


def infer_include_handle_rot(checkpoint, trajectory):
    """Infer (handle_rot, prev_action_in_history) from a checkpoint."""
    if not trajectory:
        raise ValueError("Empty trajectory")
    state = checkpoint["model"]
    input_dim = int(state["a2c_network.actor_mlp.0.weight"].shape[1])
    is_gla_ckpt = _is_gla_checkpoint(state)

    n_arti_dofs = len(trajectory[0]["joint_positions"])
    palm_extra = 3 + 1
    base_with_rot = 51 + 51 + 3 + 4 + palm_extra + n_arti_dofs + n_arti_dofs
    base_without_rot = 51 + 51 + 3 + palm_extra + n_arti_dofs + n_arti_dofs
    history_flat = 16 * 51
    history_gla = 16 * 102

    if is_gla_ckpt:
        # GLA actor_mlp consumes only base_obs.
        if input_dim == base_with_rot:
            print(
                f"  [checkpoint] GLA ckpt: base_dim={input_dim}, "
                "handle_rot=True, prev_action_in_history=True"
            )
            return True, True
        if input_dim == base_with_rot + 3:
            print(
                f"  [checkpoint] legacy GLA ckpt: base_dim={input_dim}, "
                "handle_rot=True, prev_action_in_history=True, eval pads 3 base channels"
            )
            return True, True
        if input_dim == base_without_rot:
            print(
                f"  [checkpoint] GLA ckpt: base_dim={input_dim}, "
                "handle_rot=False, prev_action_in_history=True"
            )
            return False, True
        if input_dim == base_without_rot + 3:
            print(
                f"  [checkpoint] legacy GLA ckpt: base_dim={input_dim}, "
                "handle_rot=False, prev_action_in_history=True, eval pads 3 base channels"
            )
            return False, True
        raise ValueError(
            f"GLA ckpt base_dim {input_dim} does not match "
            f"base_with_rot={base_with_rot} or base_without_rot={base_without_rot}"
        )

    if input_dim in (base_with_rot, base_with_rot + history_flat):
        print(f"  [checkpoint] flat-MLP obs_dim={input_dim} -> handle_rot=True, prev_action=False")
        return True, False
    if input_dim in (base_without_rot, base_without_rot + history_flat):
        print(f"  [checkpoint] flat-MLP obs_dim={input_dim} -> handle_rot=False, prev_action=False")
        return False, False
    if input_dim in (base_with_rot + history_gla, base_without_rot + history_gla):
        # Defensive: unrecognised non-GLA but with 102-D history -- assume
        # prev_action present even though we don't see GLA keys.
        with_rot = (input_dim == base_with_rot + history_gla)
        print(
            f"  [checkpoint] flat-history-with-prev_action obs_dim={input_dim} -> "
            f"handle_rot={with_rot}, prev_action=True"
        )
        return with_rot, True
    raise ValueError(
        "Checkpoint input dim does not match known observation layouts: "
        f"input_dim={input_dim}, base_with_rot={base_with_rot}, "
        f"base_without_rot={base_without_rot}, history_flat={history_flat}, "
        f"history_gla={history_gla}"
    )


def load_config(args):
    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["HEADLESS"] = not args.no_headless
    cfg["num_envs"] = 1
    cfg["cam"]["use_cam"] = False
    cfg["asset"]["arti_gapartnet_ids"] = [int(args.object_id)]
    return cfg


def trajectory_path(args):
    if args.trajectory:
        path = Path(args.trajectory)
        print(f"  [trajectory] using (--trajectory): {path}")
        return path
    manifest_path = Path(os.environ.get(
        "HAND_DRAG_MANIFEST",
        str(PROJECT_ROOT / "data" / "manifest.csv"),
    ))
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
                    return path
    if os.environ.get("HAND_DRAG_ALLOW_DEFAULT_TRAJECTORY") == "1":
        path = PROJECT_ROOT / "output" / "hand_drag" / args.object_id / "trajectory.json"
        print(f"  [trajectory] using (legacy output/hand_drag fallback): {path}")
        return path
    raise FileNotFoundError(
        f"No canonical trajectory for object_id={args.object_id}. "
        "Pass --trajectory or set HAND_DRAG_MANIFEST."
    )


def eval_step(task: HandDragTask, action):
    """Single-env eval step that mirrors ``task.step()`` order.

    rl_games' player goes through ``task.step()``, which runs
    ``pre_physics_step -> simulate -> compute_reward -> reset_idx (if done)
    -> compute_observations``.

    We deliberately skip the internal ``reset_idx`` here so the metric
    capture in ``metric_row`` (which reads ``task.dof_pos`` directly) sees
    the genuine episode-terminal state instead of a freshly reset one.
    The outer evaluation loop calls ``task.reset()`` between episodes, so
    inter-episode state is still reset cleanly via the public reset path.

    The ordering ``compute_reward -> compute_observations`` matches
    ``task.step()`` exactly. For single-env non-done steps it is
    functionally equivalent to the reverse order (compute_reward only
    mutates reward / done buffers; compute_observations only reads
    PhysX state), but matching the order keeps any future state
    coupling consistent with the training-time path.
    """
    task.pre_physics_step(action)
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._render_viewer()
    reward, done = task.compute_reward()
    task.compute_observations()
    return task.obs_buf, reward, done


def metric_row(task, episode, step, reward, done):
    cur_joint = task.dof_pos[0, N_HAND_DOFS + task.target_joint_idx, 0]
    progress = (cur_joint - task.task_start_joint) / task.task_progress_denom
    action_abs_max = task.actions[0].abs().max()
    action_l2 = torch.linalg.vector_norm(task.actions[0])

    return {
        "episode": int(episode),
        "step": int(step),
        "target_joint": float(cur_joint.item()),
        "normalized_progress": float(progress.item()),
        "palm_to_handle_dist": float(task.palm_to_handle_dist[0].item()),
        "reward": float(reward[0].item()),
        "done": int(bool(done[0].item())),
        "success": int(bool(task._success_flag[0].item())),
        "action_abs_max": float(action_abs_max.item()),
        "action_l2": float(action_l2.item()),
    }


def summarize_episode(rows):
    final = rows[-1]
    return {
        "episode": final["episode"],
        "steps": len(rows),
        "return": sum(row["reward"] for row in rows),
        "success": int(any(row["success"] for row in rows)),
        "final_target_joint": final["target_joint"],
        "normalized_progress": final["normalized_progress"],
        "mean_palm_handle_dist": sum(row["palm_to_handle_dist"] for row in rows) / len(rows),
        "mean_action_l2": sum(row["action_l2"] for row in rows) / len(rows),
        "mean_action_abs_max": sum(row["action_abs_max"] for row in rows) / len(rows),
    }


def mean(items, key):
    return sum(float(item[key]) for item in items) / max(1, len(items))


def main():
    args, gym_args = parse_args()
    sys.argv = [sys.argv[0]] + gym_args
    torch.manual_seed(args.seed)

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_kind)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    cfg = load_config(args)
    traj_path = trajectory_path(args)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    with open(traj_path) as f:
        trajectory = json.load(f)
    include_handle_rot, include_prev_action_in_history = infer_include_handle_rot(
        checkpoint, trajectory,
    )
    is_gla = _is_gla_checkpoint(checkpoint["model"])

    # ---- PICA v2b: decide whether the env should append aux targets to obs.
    # Auto-detect from the checkpoint's state_dict (presence of aux_head.*)
    # and let an explicit --phys_aux flag override. Only meaningful for GLA
    # checkpoints; the flat-MLP actor has no aux head pathway.
    aux_keys_in_ckpt = any(
        k.startswith("a2c_network.aux_head.") for k in checkpoint["model"]
    )
    if args.phys_aux is None:
        eval_phys_aux = bool(aux_keys_in_ckpt)
    else:
        eval_phys_aux = bool(int(args.phys_aux))
    if eval_phys_aux and not is_gla:
        print(
            "  [warn] --phys_aux requested but checkpoint is flat-MLP "
            "(no aux head pathway). Disabling aux targets in env."
        )
        eval_phys_aux = False

    eval_phys_aux_cfg = None
    if eval_phys_aux:
        # Infer aux_pred_dim from the checkpoint so the env emits the right
        # number of aux-tail channels. The aux_head's last linear layer's
        # output dim is the regression count; for v2b that is 3 and for v2c
        # that is 4. If the checkpoint has no aux_head (forced --phys_aux 1
        # on a non-aux ckpt), default to v2b's 3 channels.
        ckpt_aux_pred_dim = 0
        if "a2c_network.aux_head.2.weight" in checkpoint["model"]:
            ckpt_aux_pred_dim = int(
                checkpoint["model"]["a2c_network.aux_head.2.weight"].shape[0]
            )

        if ckpt_aux_pred_dim == 4:
            # v2c causal_horizon mode. K is not stored in the checkpoint;
            # default 5 matches the recommended training config. The actor
            # never reads aux targets, so a different K only changes raw
            # values, not behaviour.
            eval_phys_aux_cfg = {
                "enabled": True,
                "mode":    "causal_horizon",
                "horizon": 5,
                "targets": {
                    "q_response_K":    {"enabled": True},
                    "max_dist_K":      {"enabled": True},
                    "detach_proxy_K":  {"enabled": True},
                    "tracking_stress": {"enabled": True},
                },
                "gating": {"enabled": False, "d_valid": 0.10, "sharpness": 80.0},
                "warmup": {"enabled": False, "max_weight": 0.0},
            }
        else:
            # v2b "current" mode (3 targets) or forced-on with no head.
            eval_phys_aux_cfg = {
                "enabled": True,
                "mode":    "current",
                "targets": {
                    "dq_obj":          {"enabled": True},
                    "slip_proxy":      {"enabled": True},
                    "tracking_stress": {"enabled": True},
                },
                "gating": {"enabled": False, "d_valid": 0.10, "sharpness": 80.0},
                "warmup": {"enabled": False, "max_weight": 0.0},
            }

    env = HandObjectGym(cfg)
    env.get_gapartnet_anno()
    env.run_steps(50, refresh_obs=True)
    handle_link_name = resolve_handle_link_name(
        env,
        args.handle_link_name,
        target_joint_idx=args.target_joint_idx,
        trajectory_path=str(traj_path),
        env_config=cfg,
        object_id=args.object_id,
    )

    # ---- Apply OOD dynamics overrides BEFORE the eval loop and BEFORE the
    # task is constructed. The task only reads object DOF state during the
    # episode, not asset properties, so it is safe to swap props here.
    damp_after, fric_after = apply_ood_dynamics_overrides(
        env,
        damping_scale=float(args.object_damping_scale),
        friction_scale=float(args.object_friction_scale),
    )
    if args.object_damping_scale != 1.0 or args.object_friction_scale != 1.0:
        print(
            f"  [ood] object damping x{args.object_damping_scale} "
            f"-> {damp_after} (verified)"
        )
        print(
            f"  [ood] object friction x{args.object_friction_scale} "
            f"-> {fric_after} (verified)"
        )

    task = HandDragTask(
        env,
        trajectory_path=str(traj_path),
        target_joint_idx=args.target_joint_idx,
        handle_link_name=handle_link_name,
        include_handle_rot=include_handle_rot,
        is_eval_mode=True,
        epoch_log_path=None,
        include_prev_action_in_history=include_prev_action_in_history,
        physical_auxiliary=eval_phys_aux_cfg,
    )
    task.max_episode_length = int(args.max_episode_length)
    task.eval_action_ema_alpha = float(args.ema_alpha)
    if args.n_settle_substeps is not None:
        task.n_settle_substeps = int(args.n_settle_substeps)
        print(f"  [override] n_settle_substeps = {task.n_settle_substeps}")
    if args.detach_arm_delay > 0:
        print(f"  [override] detach_arm_delay = {args.detach_arm_delay} steps")
    if is_gla:
        actor = GLAActor(
            checkpoint,
            task.device,
            history_length=task.history_len,
            stochastic=args.stochastic,
            pool=args.gla_pool,
            phys_aux=eval_phys_aux,
        )
        print(f"  [actor] type = GLA  pool={args.gla_pool}")
    else:
        actor = RLGamesActor(checkpoint, task.device, stochastic=args.stochastic)
        print("  [actor] type = flat-MLP")
    print(f"  [actor] mode = {'stochastic' if args.stochastic else 'deterministic'}")

    # ---- PICA v2b: diagnostic startup banner for shape / aux alignment ----
    # Helps catch the train/eval obs-dim mismatch class of bugs at a glance.
    ckpt_obs_dim_inferred = (
        actor.obs_dim if is_gla else getattr(actor, "obs_dim", None)
    )
    aux_size = getattr(actor, "aux_target_dim", 0) if is_gla else 0
    actor_main_obs_dim = (
        getattr(actor, "base_dim", None) if is_gla
        else getattr(actor, "obs_dim", None)
    )
    print()
    print("  [eval-diagnostic]")
    print(f"    checkpoint                = {ckpt_path}")
    print(f"    aux_keys_in_ckpt          = {aux_keys_in_ckpt}")
    print(f"    --phys_aux requested      = {args.phys_aux}")
    print(f"    physical_auxiliary.enabled= {eval_phys_aux}")
    print(f"    inferred ckpt obs_dim     = {ckpt_obs_dim_inferred}")
    print(f"    env obs_dim               = {task.obs_dim}")
    print(f"    aux_size (sliced off)     = {aux_size}")
    print(f"    actor_main_obs_dim        = {actor_main_obs_dim}")
    print(f"    network type              = {'GLA' if is_gla else 'flat-MLP'}")
    print(f"    actor / algo type         = {type(actor).__name__}")
    if is_gla and ckpt_obs_dim_inferred != task.obs_dim:
        print(
            f"  [eval-diagnostic][WARN] obs_dim mismatch: ckpt expects "
            f"{ckpt_obs_dim_inferred} but env emits {task.obs_dim}. "
            "Verify --phys_aux flag and physical_auxiliary YAML config."
        )
    print()

    all_rows = []
    episodes = []
    try:
        for episode in range(args.episodes):
            obs = task.reset()
            rows = []
            for step in range(args.max_episode_length):
                if step < args.detach_arm_delay:
                    task.detach_armed_buf.zero_()
                with torch.no_grad():
                    action = actor(obs).clamp(-1.0, 1.0)
                obs, reward, done = eval_step(task, action)
                row = metric_row(task, episode, step, reward, done)
                rows.append(row)
                all_rows.append(row)
                if bool(done[0].item()):
                    break
            episodes.append(summarize_episode(rows))

        summary = {
            "object_id": str(args.object_id),
            "checkpoint": str(ckpt_path),
            "trajectory": str(traj_path),
            "target_joint_idx": int(task.target_joint_idx),
            "handle_link_name": handle_link_name,
            "episodes": int(args.episodes),
            "success_rate": mean(episodes, "success"),
            "return_mean": mean(episodes, "return"),
            "steps_mean": mean(episodes, "steps"),
            "final_target_joint_mean": mean(episodes, "final_target_joint"),
            "normalized_progress_mean": mean(episodes, "normalized_progress"),
            "mean_palm_handle_dist": mean(episodes, "mean_palm_handle_dist"),
            "mean_action_l2": mean(episodes, "mean_action_l2"),
            "mean_action_abs_max": mean(episodes, "mean_action_abs_max"),
            "stochastic": bool(args.stochastic),
            "episode_metrics": episodes,
            "object_damping_scale": float(args.object_damping_scale),
            "object_friction_scale": float(args.object_friction_scale),
            "object_damping_actual": damp_after,
            "object_friction_actual": fric_after,
        }

        if args.log_csv:
            out = Path(args.log_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)

        if args.summary_json:
            out = Path(args.summary_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)

        print("\nPPO checkpoint evaluation")
        print(f"  checkpoint: {ckpt_path}")
        print(f"  object id: {args.object_id}")
        print(f"  target joint idx: {task.target_joint_idx}")
        print(f"  handle link: {handle_link_name}")
        print(f"  episodes: {args.episodes}")
        print(f"  success rate: {summary['success_rate']:.3f}")
        print(f"  return mean: {summary['return_mean']:.3f}")
        print(f"  steps mean: {summary['steps_mean']:.1f}")
        print(f"  final target joint mean: {summary['final_target_joint_mean']:.6f}")
        print(f"  normalized progress mean: {summary['normalized_progress_mean']:.3f}")
        print(f"  mean palm-handle dist: {summary['mean_palm_handle_dist']:.4f}")
        print(f"  mean action l2: {summary['mean_action_l2']:.3f}")
        if args.log_csv:
            print(f"  wrote metrics: {args.log_csv}")
        if args.summary_json:
            print(f"  wrote summary: {args.summary_json}")
    finally:
        env.clean_up()


if __name__ == "__main__":
    main()
