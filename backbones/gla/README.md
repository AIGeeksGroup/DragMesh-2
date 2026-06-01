# GLA Backbone

`a2c_network.py` implements the custom rl-games network registered as:

```yaml
params:
  network:
    name: gla_actor_critic
```

It consumes the base proprioceptive observation with an MLP and the 16-step history token block with Gated Linear Attention, then fuses both features before actor/value heads.

Configs using this backbone live in:

- `configs/train/gla/`
- `configs/train/pica/`
