# Dexterous Hand Assets

This directory keeps hand assets separate from object assets and training code.

- `floating/`: the 51-DoF floating SMPL-X right hand used by the simulator.
- `smplx_variants/`: the full SMPL-X hand variant collection from the project SMPL-X variant collection, including hands-only, palms-only, and full-finger variants.

The default environment config uses:

```yaml
hand:
  hand_asset_root: hands/floating
  urdf: smplx_right_hand_floating.urdf
```

To rebuild the floating hand URDF:

```bash
python dataset/geometric/build_hand_urdf.py \
  --input hands/smplx_variants/hands_only/smplx_right_hand.urdf \
  --output hands/floating/smplx_right_hand_floating.urdf
```
