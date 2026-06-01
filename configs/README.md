# Configs

All YAML files live under this directory.

```text
configs/
|-- env/
|   `-- hand_config.yaml
`-- train/
    |-- mlp/
    |-- gla/
    `-- pica/
```

- `env/hand_config.yaml`: simulator, hand, camera, asset, and trajectory-generation settings.
- `train/mlp/`: rl-games default MLP baseline configs.
- `train/gla/`: GLA backbone configs without PICA auxiliary agent.
- `train/pica/`: main DragMesh-2/PICA configs using the GLA backbone and PICA agent.
