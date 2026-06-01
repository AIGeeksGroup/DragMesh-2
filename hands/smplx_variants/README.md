# SMPL-X Variants


Files:

- `full_fingers/smplx_full_fingers.xml`
  Exact MJCF copy of the project SMPL-X asset with all finger bodies and all
  finger actuators preserved.
- `full_fingers/smplx_full_fingers.urdf`
  URDF export of the same full-finger articulation. The URDF is mesh-free and
  uses only box / cylinder / sphere primitives so it can be moved into Gym-style
  loaders without extra mesh dependencies.
- `hands_only/smplx_left_hand.xml` and `hands_only/smplx_right_hand.xml`
  Standalone MJCF assets for the complete left and right hand. Each one keeps
  wrist, palm, and all finger bodies.
- `hands_only/smplx_left_hand.urdf` and `hands_only/smplx_right_hand.urdf`
  Gym-friendly URDF exports of the full left and right hand assets.
- `palms_only/smplx_left_palm_wrist.xml` and `palms_only/smplx_right_palm_wrist.xml`
  Standalone MJCF assets that preserve only one wrist and its palm body.
- `palms_only/smplx_left_palm_wrist.urdf` and `palms_only/smplx_right_palm_wrist.urdf`
  URDF exports of the standalone left and right hand assets.
- `full_fingers/metadata.json`, `palms_only/metadata.json`, and
  `hands_only/*.json`, `palms_only/smplx_left_palm_wrist.json`,
  `palms_only/smplx_right_palm_wrist.json`
  Machine-readable summaries of bodies, joints, contact groups, and loading
  hints.
- `overview.json`
  Short index of all generated files.


