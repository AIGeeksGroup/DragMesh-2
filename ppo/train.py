"""
train.py — Launch PPO training or play for the dexterous hand drag task.

Usage
-----
    conda activate isaacgym
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    cd /path/to/DragMesh-2

    # Default training (64 envs, headless)
    python ppo/train.py

    # Custom training
    python ppo/train.py --num_envs 128 --object_id 45661 --max_epochs 3000

    # Play a trained checkpoint with the Isaac Gym viewer
    python ppo/train.py --play --checkpoint runs/hand_drag_ppo/nn --no-headless
"""

import argparse
import glob
import json
import os
import sys
import yaml

# Isaac Gym MUST be imported before torch
from isaacgym import gymapi  # noqa: F401
import torch

from rl_games.common import env_configurations
from rl_games.torch_runner import Runner

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppo.rlgames_wrapper import register_hand_drag_env


def resolve_checkpoint_path(checkpoint_arg, checkpoint_kind, run_name):
    """Accept a .pth file or a checkpoint directory and return a concrete file."""
    if checkpoint_arg is None:
        return None

    checkpoint_path = os.path.abspath(checkpoint_arg)
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")

    pattern = os.path.join(checkpoint_path, "*.pth")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found in: {checkpoint_path}")

    if checkpoint_kind == "latest":
        chosen = candidates[0]
    else:
        preferred = os.path.join(checkpoint_path, f"{run_name}.pth")
        chosen = preferred if os.path.exists(preferred) else candidates[0]

    print(f"  [checkpoint] using: {chosen}")
    return chosen


def infer_obs_layout(checkpoint_path, trajectory_path):
    """
    Infer the observation layout (handle-rot + history token width) from a
    checkpoint's first weight matrix.

    Layouts recognised:
      base_with_rot    = 51 + 51 + 3 + 4 + 3 + 1 + Na + Na           (115 for Na=1)
      base_without_rot = 51 + 51 + 3     + 3 + 1 + Na + Na           (111 for Na=1)
      + history block:
          0           -- no history
          16 * 51     -- flat-history baseline (error only)
          16 * 102    -- Phase 4 GLA history (error_t + a_{t-1})

    Returns
    -------
    (include_handle_rot: bool, include_prev_action_in_history: bool)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"]
    is_gla_ckpt = any(
        k.startswith("a2c_network.gla.") or k.startswith("a2c_network.token_proj")
        for k in state
    )
    if "a2c_network.actor_mlp.0.weight" not in state:
        raise ValueError(
            f"Checkpoint missing a2c_network.actor_mlp.0.weight: {checkpoint_path}"
        )
    actor_in_dim = int(state["a2c_network.actor_mlp.0.weight"].shape[1])

    with open(trajectory_path) as f:
        trajectory = json.load(f)
    if not trajectory:
        raise ValueError(f"Empty trajectory file: {trajectory_path}")

    n_arti_dofs = len(trajectory[0]["joint_positions"])
    palm_extra = 3 + 1  # palm->handle vec + dist
    base_with_rot = 51 + 51 + 3 + 4 + palm_extra + n_arti_dofs + n_arti_dofs
    base_without_rot = 51 + 51 + 3 + palm_extra + n_arti_dofs + n_arti_dofs
    history_flat = 16 * 51       # flat-history (error only)
    history_gla = 16 * 102       # Phase 4 GLA (error + prev_action)

    if is_gla_ckpt:
        # GLA actor_mlp consumes only base_obs; history is consumed by the
        # GLA branch and never enters actor_mlp.0.
        if actor_in_dim == base_with_rot:
            print(
                f"  [checkpoint] GLA ckpt: base_dim={actor_in_dim}, "
                "handle_rot=True, prev_action_in_history=True"
            )
            return True, True
        if actor_in_dim == base_without_rot:
            print(
                f"  [checkpoint] GLA ckpt: base_dim={actor_in_dim}, "
                "handle_rot=False, prev_action_in_history=True"
            )
            return False, True
        raise ValueError(
            f"GLA checkpoint base_dim {actor_in_dim} does not match either "
            f"base_with_rot={base_with_rot} or base_without_rot={base_without_rot}"
        )

    candidates = [
        (base_with_rot,    0,            True,  False),
        (base_with_rot,    history_flat, True,  False),
        (base_without_rot, 0,            False, False),
        (base_without_rot, history_flat, False, False),
    ]
    for base, hist, with_rot, with_prev_act in candidates:
        if actor_in_dim == base + hist:
            print(
                f"  [checkpoint] flat-MLP ckpt: obs_dim={actor_in_dim} -> "
                f"handle_rot={with_rot}, prev_action_in_history={with_prev_act}"
            )
            return with_rot, with_prev_act

    raise ValueError(
        f"Checkpoint actor_mlp input dim {actor_in_dim} does not match any "
        f"known layout. base_with_rot={base_with_rot}, "
        f"base_without_rot={base_without_rot}, history_flat={history_flat}, "
        f"history_gla={history_gla}"
    )


def infer_include_handle_rot(checkpoint_path, trajectory_path):
    """Backwards-compatible wrapper retained for older callers."""
    include_rot, _ = infer_obs_layout(checkpoint_path, trajectory_path)
    return include_rot


def main():
    parser = argparse.ArgumentParser(description="PPO train/play for HandDrag")
    parser.add_argument("--config", default="configs/env/hand_config.yaml",
                        help="Base environment config")
    parser.add_argument("--train_config", default="configs/train/mlp/ppo.yaml",
                        help="rl_games PPO training config")
    parser.add_argument("--object_id", default="45661",
                        help="GAPartNet object ID for trajectory")
    parser.add_argument("--trajectory", default=None,
                        help="Explicit trajectory.json path. If omitted, uses "
                             "output/hand_drag/<object_id>/trajectory.json")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Number of parallel environments "
                             "(default: 64 for train, 1 for play)")
    parser.add_argument("--target_joint_idx", type=int, default=None,
                        help="Articulated object DOF index for the task joint "
                             "(default: auto-detect max trajectory motion)")
    parser.add_argument("--handle_link_name", default=None,
                        help="Rigid body link name of the target part "
                             "(default: auto-detect target part link)")
    parser.add_argument("--experiment_name", default=None,
                        help="Override rl_games full_experiment_name. "
                             "Training defaults to <config_name>_<object_id> "
                             "to avoid mixing checkpoints across objects.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max training epochs")
    parser.add_argument("--bounds_loss_coef", type=float, default=None,
                        help="Override PPO bounds_loss_coef in train_config "
                             "(default 0.001). Penalises mu near +/-1; raise "
                             "to discourage mu saturation in long training.")
    # ---- PICA v1: physical_regularization overrides (env-side reward) ----
    parser.add_argument("--phys_reg", type=int, default=None,
                        help="0/1: override physical_regularization.enabled "
                             "from the train config. Useful for A/B parity "
                             "tests on a single YAML.")
    parser.add_argument("--lambda_bound", type=float, default=None,
                        help="Override physical_regularization.action_bound."
                             "weight (env-side action saturation penalty).")
    parser.add_argument("--lambda_contact", type=float, default=None,
                        help="Override physical_regularization.contact_distance."
                             "weight (env-side palm-handle contact penalty).")
    parser.add_argument("--lambda_slip", type=float, default=None,
                        help="Override physical_regularization.slip_action."
                             "weight (env-side slip-aware action penalty).")
    parser.add_argument("--lambda_smooth", type=float, default=None,
                        help="Override physical_regularization.action_smoothness."
                             "weight (env-side step-to-step action delta penalty).")
    parser.add_argument("--play", action="store_true", default=False,
                        help="Run a trained policy instead of training")
    parser.add_argument("--games_num", type=int, default=1,
                        help="Episodes to run in play mode")
    parser.add_argument("--no-headless", action="store_true", default=False,
                        help="Show viewer (slow, for debugging)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint file or checkpoint directory")
    parser.add_argument("--checkpoint-kind", choices=("best", "latest"),
                        default="best",
                        help="When --checkpoint points to a directory, choose "
                             "the named best checkpoint or the latest file")
    args, gym_args = parser.parse_known_args()

    # Replace sys.argv so that gymutil.parse_arguments() inside
    # HandObjectGym.__init__ only sees Isaac Gym-compatible flags
    # (--sim_device, --pipeline, --physx, --headless, etc.)
    # and not our custom ones (--object_id, --num_envs, ...).
    sys.argv = [sys.argv[0]] + gym_args

    # ── Load configs ──
    project_root = os.path.join(os.path.dirname(__file__), "..")
    env_config_path = os.path.join(project_root, args.config)
    with open(env_config_path) as f:
        env_config = yaml.safe_load(f)

    train_config_path = os.path.join(project_root, args.train_config)
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)

    ppo_cfg = train_config["params"]["config"]
    run_name = ppo_cfg["name"]
    base_experiment_name = ppo_cfg.get("full_experiment_name", run_name)
    if args.experiment_name:
        ppo_cfg["full_experiment_name"] = args.experiment_name
    elif not args.play:
        ppo_cfg["full_experiment_name"] = f"{base_experiment_name}_{args.object_id}"
    if args.num_envs is None:
        args.num_envs = 1 if args.play else 64

    # ── Trajectory path ──
    # Clean-data policy: explicit --trajectory wins; otherwise resolve the
    # object through the canonical manifest. The historical output/hand_drag
    # fallback is disabled unless HAND_DRAG_ALLOW_DEFAULT_TRAJECTORY=1 is set.
    if args.trajectory:
        traj_path = os.path.abspath(args.trajectory)
        traj_source = "--trajectory"
    else:
        manifest_path = os.environ.get(
            "HAND_DRAG_MANIFEST",
            os.path.join(project_root, "data", "manifest.csv"),
        )
        traj_path = None
        traj_source = f"manifest:{manifest_path}"
        if os.path.exists(manifest_path):
            import csv
            with open(manifest_path, newline="") as f:
                for row in csv.DictReader(f):
                    if (
                        row.get("object_id") == str(args.object_id)
                        and row.get("enabled", "1").strip() == "1"
                    ):
                        traj_path = row.get("trajectory")
                        break
        if not traj_path and os.environ.get("HAND_DRAG_ALLOW_DEFAULT_TRAJECTORY") == "1":
            traj_path = os.path.join(
                project_root, "output", "hand_drag", args.object_id, "trajectory.json"
            )
            traj_source = "legacy output/hand_drag fallback"
    if not traj_path or not os.path.exists(traj_path):
        print(f"ERROR: Trajectory not found for object_id={args.object_id}: {traj_path}")
        print("Use --trajectory or set HAND_DRAG_MANIFEST to a manifest CSV.")
        sys.exit(1)
    print(f"  [trajectory] using ({traj_source}): {traj_path}")

    # ── Checkpoint resume / play ──
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = resolve_checkpoint_path(
            args.checkpoint, args.checkpoint_kind, run_name
        )
        train_config["params"]["load_checkpoint"] = True
        train_config["params"]["load_path"] = checkpoint_path
    elif args.play:
        default_ckpt_dir = os.path.join(
            project_root,
            "runs",
            train_config["params"]["config"]["full_experiment_name"],
            "nn",
        )
        checkpoint_path = resolve_checkpoint_path(
            default_ckpt_dir, args.checkpoint_kind, run_name
        )
        train_config["params"]["load_checkpoint"] = True
        train_config["params"]["load_path"] = checkpoint_path

    # Phase 4 GLA layout is the new default; the flat-history baseline can
    # still be reproduced by setting params.network.name=actor_critic in the
    # train config, in which case we honor the YAML's history-token choice.
    network_name = train_config["params"]["network"].get("name", "actor_critic")
    include_handle_rot = True
    include_prev_action_in_history = (network_name == "gla_actor_critic")
    if checkpoint_path is not None:
        include_handle_rot, include_prev_action_in_history = infer_obs_layout(
            checkpoint_path, traj_path
        )

    # ── Override num_envs and fix batch sizing ──
    env_config["num_envs"] = args.num_envs
    ppo_cfg["num_actors"] = args.num_envs
    if args.max_epochs is not None:
        ppo_cfg["max_epochs"] = args.max_epochs
    if args.bounds_loss_coef is not None:
        ppo_cfg["bounds_loss_coef"] = float(args.bounds_loss_coef)
        print(f"  [override] bounds_loss_coef = {ppo_cfg['bounds_loss_coef']}")

    if args.play:
        player_cfg = ppo_cfg.setdefault("player", {})
        player_cfg["games_num"] = args.games_num
        player_cfg.setdefault("determenistic", True)

    # rl_games requires: batch_size % minibatch_size == 0
    # batch_size = num_actors * horizon_length
    batch_size = args.num_envs * ppo_cfg["horizon_length"]
    if ppo_cfg["minibatch_size"] > batch_size:
        ppo_cfg["minibatch_size"] = batch_size
        print(f"  [auto] minibatch_size clamped to {batch_size} "
              f"(num_envs={args.num_envs} × horizon={ppo_cfg['horizon_length']})")

    # ── Register custom env with rl_games ──
    # Per-epoch reward CSV (one row per PPO iteration). Training only;
    # play mode shouldn't overwrite the training log.
    epoch_log_path = None
    if not args.play:
        run_dir = os.path.join(
            project_root, "runs",
            ppo_cfg.get("full_experiment_name", run_name),
        )
        epoch_log_path = os.path.join(run_dir, "epoch_rewards.csv")

    # ── PICA v1: read physical_regularization from train config and apply
    # ── any CLI overrides. The schema is fully defaulted env-side, so a
    # ── missing block is equivalent to {"enabled": False}.
    phys_reg = dict(ppo_cfg.get("physical_regularization", {}) or {})
    if args.phys_reg is not None:
        phys_reg["enabled"] = bool(args.phys_reg)
    if args.lambda_bound is not None:
        phys_reg.setdefault("action_bound", {})["weight"] = float(args.lambda_bound)
        phys_reg["action_bound"]["enabled"] = True
    if args.lambda_contact is not None:
        phys_reg.setdefault("contact_distance", {})["weight"] = float(args.lambda_contact)
        phys_reg["contact_distance"]["enabled"] = True
    if args.lambda_slip is not None:
        phys_reg.setdefault("slip_action", {})["weight"] = float(args.lambda_slip)
        phys_reg["slip_action"]["enabled"] = True
    if args.lambda_smooth is not None:
        phys_reg.setdefault("action_smoothness", {})["weight"] = float(args.lambda_smooth)
        phys_reg["action_smoothness"]["enabled"] = True
    if phys_reg.get("enabled"):
        print(f"  [phys_reg] enabled: {phys_reg}")

    # ── PICA v2a: read dynamics_randomization. Unlike phys_reg there are
    # ── no CLI overrides yet -- the YAML knob is the only switch.
    dyn_rand = dict(ppo_cfg.get("dynamics_randomization", {}) or {})
    if dyn_rand.get("enabled"):
        print(f"  [dyn_rand] enabled: {dyn_rand}")

    # ── PICA v2b/v2c: read physical_auxiliary. Mirror the network-relevant
    # ── fields into `params.network.phys_aux` so GLAA2CBuilder.Network can
    # ── size its aux head off the same source of truth as the env. Both
    # ── modes share the same plumbing -- only the canonical target order
    # ── and default weights differ.
    phys_aux = dict(ppo_cfg.get("physical_auxiliary", {}) or {})
    if phys_aux.get("enabled"):
        mode = str(phys_aux.get("mode", "current")).lower()
        if mode == "causal_horizon":
            # v2c: 4 past-window targets.
            canonical = ["q_response_K", "max_dist_K", "detach_proxy_K", "tracking_stress"]
            default_w = {
                "q_response_K":    1.0,
                "max_dist_K":      1.0,
                "detach_proxy_K":  0.5,
                "tracking_stress": 0.5,
            }
        else:
            # v2b: 3 per-step targets. Default mode for backward compat.
            canonical = ["dq_obj", "slip_proxy", "tracking_stress"]
            default_w = {"dq_obj": 1.0, "slip_proxy": 1.0, "tracking_stress": 1.0}

        target_blocks = phys_aux.get("targets", {}) or {}
        weight_blocks = phys_aux.get("weights", {}) or {}

        enabled_keys = [
            k for k in canonical
            if (target_blocks.get(k, {}) or {}).get("enabled", True)
        ]
        enabled_weights = [
            float(weight_blocks.get(k, default_w[k]))
            for k in enabled_keys
        ]

        train_config["params"]["network"]["phys_aux"] = {
            "enabled":        bool(enabled_keys),
            "pred_dim":       int(len(enabled_keys)),
            "target_dim":     int(len(enabled_keys)),
            "hidden_size":    64,
            "mode":           mode,
            "target_keys":    enabled_keys,
            "target_weights": enabled_weights,
        }
        print(
            f"  [phys_aux] enabled: mode={mode} "
            f"keys={enabled_keys} weights={enabled_weights} "
            f"warmup={phys_aux.get('warmup', {})}"
        )

    task_config = {
        "env_config": env_config,
        "trajectory_path": traj_path,
        "object_id": args.object_id,
        "target_joint_idx": args.target_joint_idx,
        "handle_link_name": args.handle_link_name,
        "headless": not args.no_headless,
        "include_handle_rot": include_handle_rot,
        "is_eval_mode": args.play,
        "epoch_log_path": epoch_log_path,
        "include_prev_action_in_history": include_prev_action_in_history,
        "physical_regularization": phys_reg,
        "dynamics_randomization": dyn_rand,
        "physical_auxiliary": phys_aux,
        "aram": ppo_cfg.get("aram"),
        "reconfig_reward": ppo_cfg.get("reconfig_reward"),
    }
    env_creator = register_hand_drag_env(task_config)

    # Tell rl_games: env "hand_drag" → vecenv type "HAND_DRAG"
    env_configurations.configurations["hand_drag"] = {
        "vecenv_type": "HAND_DRAG",
        "env_creator": env_creator,
    }

    # ── Register custom networks before runner.load (which copies the
    # network registry into ModelBuilder's factory) ──
    if network_name == "gla_actor_critic":
        from backbones.gla.a2c_network import register_gla_network
        register_gla_network()
        print("  [network] registered gla_actor_critic builder")

    # ── Launch rl_games Runner ──
    runner = Runner()

    # PICA v2b: register custom A2CAgent subclass when the YAML asks for it.
    # Must happen BEFORE runner.load() because that resolves algo.name.
    if train_config["params"].get("algo", {}).get("name") == "pica_a2c_continuous":
        from pica.a2c_agent import register_pica_a2c_agent
        register_pica_a2c_agent(runner)
        print("  [algo] registered pica_a2c_continuous (PICAA2CAgent)")

    runner.load(train_config)
    runner.reset()
    runner.run({
        "train": not args.play,
        "play": args.play,
        "sigma": None,
        "checkpoint": checkpoint_path,
    })


if __name__ == "__main__":
    main()
