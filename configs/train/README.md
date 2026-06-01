# Training Configs

- `mlp/`: built-in rl-games MLP actor-critic baselines.
- `gla/`: custom GLA backbone baselines.
- `pica/`: DragMesh-2/PICA method configs.

Pass a config to the trainer with:

```bash
python ppo/train.py --train_config configs/train/pica/train_config_gla_pica_drand12_aux_v2c.yaml
```
