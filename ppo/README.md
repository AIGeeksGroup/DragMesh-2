# PPO

This directory contains PPO-facing runtime code only:

- `train.py`: main training and play entrypoint.
- `hand_drag_task.py`: task observations, rewards, resets, PICA reward terms, and diagnostics.
- `rlgames_wrapper.py`: Isaac Gym environment registration for rl-games.

Related modules are intentionally separated:

- configs: `configs/`
- GLA and other backbone notes: `backbones/`
- PICA agent extension: `pica/`

Run from the repository root:

```bash
python ppo/train.py \
  --train_config configs/train/pica/train_config_gla_pica_drand12_aux_v2c.yaml \
  --object_id 45661 \
  --trajectory output/hand_drag/45661/trajectory.json \
  --num_envs 64 \
  --max_epochs 150 \
  --experiment_name dragmesh2_45661_pica
```
