# Backbones

This directory separates policy backbones from PPO/PICA training logic.

- `gla/`: the implemented Gated Linear Attention temporal backbone used by the main DragMesh-2/PICA policy.
- `mlp/`: the rl-games default MLP actor-critic baseline. It is configured through YAML and does not need a custom Python builder.
- `gru/`: paper ablation slot. No custom GRU builder is included in this release.
- `transformer/`: paper ablation slot. No custom Transformer builder is included in this release.

Only `gla/` registers custom code. MLP uses `params.network.name: actor_critic`.
