# PICA

PICA is kept as a separate method module instead of being mixed into `ppo/`.

`a2c_agent.py` registers the custom rl-games agent name:

```yaml
params:
  algo:
    name: pica_a2c_continuous
```

PICA reward/auxiliary signals are computed inside `ppo/hand_drag_task.py`, while this agent handles the auxiliary loss and PICA-specific logging/checkpoint behavior.

Main configs live in `configs/train/pica/`.
