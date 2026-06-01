# PICA Training Configs

These configs are the main DragMesh-2 method family. They use:

- `params.algo.name: pica_a2c_continuous`, registered by `pica/a2c_agent.py`.
- `params.network.name: gla_actor_critic`, registered by `backbones/gla/a2c_network.py`.

Recommended main config:

```text
train_config_gla_pica_drand12_aux_v2c.yaml
```

Fine-tuning and ablation configs are kept with their historical experiment names for traceability.
