# Scripts

This directory keeps release-facing entrypoints and reusable analysis scripts.

## Standard Entrypoints

```text
scripts/
|-- train/
|   `-- train_pica.sh
|-- eval/
|   `-- eval_det_stoch_damping.sh
|-- baselines/
|   `-- run_trajectory_tracking.sh
`-- utils/
    `-- check_checkpoint.sh
```

All shell wrappers are configured through environment variables and can be run from any directory.
Default training config paths point into `configs/`.

## Training

```bash
OBJECT_ID=45661 \
TRAJECTORY=output/hand_drag/45661/trajectory.json \
RUN_NAME=dragmesh2_45661_pica \
bash scripts/train/train_pica.sh
```

## Evaluation

```bash
OBJECT_ID=45661 \
TRAJECTORY=output/hand_drag/45661/trajectory.json \
CHECKPOINT=runs/dragmesh2_45661_pica/nn \
bash scripts/eval/eval_det_stoch_damping.sh
```

## Baseline

```bash
OBJECT_ID=45661 \
TRAJECTORY=output/hand_drag/45661/trajectory.json \
bash scripts/baselines/run_trajectory_tracking.sh
```

The Python scripts in this directory can also be called directly for custom experiments.
