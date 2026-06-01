# Geometric Dataset Generator

This folder contains the geometry-only hand-object trajectory generator adapted from the public `neptune-T/pull-push` main branch:

https://github.com/neptune-T/pull-push

The generator produces contact-driven hand and articulated-object trajectories under:

```text
output/hand_drag/<object_id>/trajectory.json
```

Core files:

- `run_hand_drag.py`: main trajectory-generation entrypoint.
- `hand_object_gym.py`: Isaac Gym scene, GAPartNet loading, grasp helpers, and data capture.
- `build_hand_urdf.py`: builds the 51-DoF floating SMPL-X right-hand URDF.
- `utils.py`: point-cloud, camera, and mesh helper utilities.

The generated dataset files are not committed here. A Hugging Face dataset link will be added after release.

Run from the repository root:

```bash
python dataset/geometric/run_hand_drag.py \
  --config configs/env/hand_config.yaml \
  --headless
```
