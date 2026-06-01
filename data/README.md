# Data Manifests

Training and evaluation scripts can resolve trajectories from `data/manifest.csv` or from the `HAND_DRAG_MANIFEST` environment variable.

Use this schema:

```csv
sample_id,object_id,trajectory,enabled
45661,45661,output/hand_drag/45661/trajectory.json,1
```

Generated trajectories and large datasets are ignored by git. The released dataset package will be linked here after it is uploaded to Hugging Face.
