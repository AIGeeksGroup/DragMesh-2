# DragMesh-2: Physically Plausible Dexterous Hand-Object Interaction with Articulated Objects

This is the official repository for the paper:

> **DragMesh-2: Physically Plausible Dexterous Hand-Object Interaction with Articulated Objects**
>
> [Tianshan Zhang](https://neptune-t.github.io/)\*, [Yijia Duan](https://github.com/yijiaduan111)\*, [Yanjun Li](https://github.com/yanjun711)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>lead</sup>, and [Hao Tang](https://ha0tang.github.io/)<sup>corr</sup>
>
> School of Computer Science, Peking University
>
> *Equal contribution. <sup>lead</sup>Project lead. <sup>corr</sup>Corresponding author: bjdxtanghao@gmail.com.
>
> ### [Paper](https://arxiv.org/abs/) | [Project Page](https://aigeeksgroup.github.io/DragMesh-2) | [Data]([https://huggingface.co/AIGeeksGroup/DragMesh-2](https://huggingface.co/datasets/AIGeeksGroup/DragMesh-2)) | [Checkpoints](https://huggingface.co/AIGeeksGroup/DragMesh-2)

> [!NOTE]
> The simulation demo and real-robot demo are intentionally left as placeholders in this release. They will be added after the corresponding experiments are finalized.

## Citation

If you find our code or paper useful, please consider citing:

```bibtex

```

## Intro

DragMesh-2 is a contact-driven framework for dexterous hand interaction with articulated objects. It extends articulated interaction from object-centric generation to hand-driven hand-object interaction, where a target part can move only through physical contact between the dexterous hand and the handle.

The repository contains:

- an Isaac Gym articulated-object interaction environment built around a 51-DoF SMPL-X right hand;
- PPO training code with a separated PICA method module and GLA backbone;
- deterministic and stochastic evaluation tools for damping and contact-load robustness;
- trajectory-tracking baseline scripts.

## News

<b>2026/06/01:</b> Initial public-code cleanup for DragMesh-2.

## TODO List

- [x] Release core simulation and PPO/PICA code.
- [x] Add cleaned training, evaluation, and baseline entrypoints.
- [x] Add release README and setup instructions.
- [ ] Release prepared trajectory/data package.
- [ ] Release pretrained checkpoints.
- [ ] Add simulation demo.
- [ ] Add real-robot demo.

## Quick Start

### Environment Setup

The code is designed for Isaac Gym preview environments. Our internal runs use Python 3.8/3.9, CUDA-capable PyTorch, and Isaac Gym.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install external packages that are not distributed through this repository:

```bash
# Isaac Gym
# Download Isaac Gym from NVIDIA, then add it to PYTHONPATH.
export ISAACGYM_ROOT=/path/to/isaacgym
export PYTHONPATH="${ISAACGYM_ROOT}/python:${PYTHONPATH}"

# GLA encoder dependency used by PICA checkpoints.
# Install flash-linear-attention or keep a local checkout at ./flash-linear-attention.
export FLA_ROOT=/path/to/flash-linear-attention
export PYTHONPATH="${FLA_ROOT}:${PYTHONPATH}"
```

If you use a conda environment, also expose its runtime libraries:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
```

### Data Preparation

Prepare GAPartNet assets in this layout:

```text
assets/
`-- gapartnet_example/
    `-- 45661/
        |-- mobility_annotation_gapartnet.urdf
        |-- mobility_v2.json
        `-- link_annotation_gapartnet.json
```

The default config expects:

```yaml
hand.hand_asset_root: hands/floating
asset.asset_root: assets
asset.arti_obj_root: gapartnet_example
```

The repository already includes `smplx_right_hand_floating.urdf`. If you need to rebuild it from a base SMPL-X right-hand URDF:

```bash
python dataset/geometric/build_hand_urdf.py \
  --input hands/smplx_variants/hands_only/smplx_right_hand.urdf \
  --output hands/floating/smplx_right_hand_floating.urdf
```

### Generate Contact Trajectories

Edit `configs/env/hand_config.yaml` to select the GAPartNet object IDs, then run:

```bash
python dataset/geometric/run_hand_drag.py --config configs/env/hand_config.yaml --headless
```

Generated trajectories are written under:

```text
output/hand_drag/<object_id>/trajectory.json
```

For batch training/evaluation, create a manifest at `data/manifest.csv`:

```csv
sample_id,object_id,trajectory,enabled
45661,45661,output/hand_drag/45661/trajectory.json,1
```

You can also pass `--trajectory` directly to all training and evaluation commands.

## Training

Train the main PICA configuration:

```bash
OBJECT_ID=45661 \
TRAJECTORY=output/hand_drag/45661/trajectory.json \
RUN_NAME=dragmesh2_45661_pica \
bash scripts/train/train_pica.sh
```

Equivalent direct command:

```bash
python ppo/train.py \
  --train_config configs/train/pica/train_config_gla_pica_drand12_aux_v2c.yaml \
  --object_id 45661 \
  --trajectory output/hand_drag/45661/trajectory.json \
  --num_envs 64 \
  --max_epochs 150 \
  --experiment_name dragmesh2_45661_pica
```

Checkpoints are saved under:

```text
runs/<experiment_name>/nn/
```

## Evaluation

Run deterministic and stochastic damping evaluation:

```bash
OBJECT_ID=45661 \
TRAJECTORY=output/hand_drag/45661/trajectory.json \
CHECKPOINT=runs/dragmesh2_45661_pica/nn \
OUT_ROOT=output/eval/dragmesh2_45661_pica \
bash scripts/eval/eval_det_stoch_damping.sh
```

Run a single deterministic evaluation:

```bash
python scripts/evaluate_ppo_baseline.py \
  --object_id 45661 \
  --trajectory output/hand_drag/45661/trajectory.json \
  --checkpoint runs/dragmesh2_45661_pica/nn \
  --checkpoint-kind best \
  --episodes 20 \
  --summary_json output/eval/45661_summary.json
```

## Baselines

Run the non-learned trajectory-tracking baseline:

```bash
python scripts/track_trajectory_baseline.py \
  --object_id 45661 \
  --trajectory output/hand_drag/45661/trajectory.json \
  --phase drag \
  --summary_json output/baselines/45661_tracking.json
```

Batch baseline evaluation over `data/manifest.csv`:

```bash
python scripts/run_trajectory_baselines.py \
  --manifest data/manifest.csv \
  --phase drag
```

## Repository Layout

```text
.
|-- dataset/geometric/              # Geometry-only dataset generator
|-- hands/                          # Dexterous hand assets
|-- configs/                        # Environment and training YAML files
|-- backbones/                      # GLA backbone plus MLP/GRU/Transformer notes
|-- pica/                           # PICA agent extension and method notes
|-- ppo/                            # PPO trainer, task, and rl-games environment wrapper
|-- scripts/
|   |-- train/                      # Standard training wrappers
|   |-- eval/                       # Standard evaluation wrappers
|   `-- baselines/                  # Baseline wrappers
|-- assets/                         # External articulated assets live here
|-- data/                           # Manifest templates and local data pointers
`-- demos/                          # Simulation and real-robot demos, coming soon
```

## Acknowledgement

We thank the authors of Isaac Gym, GAPartNet, SMPL-X, rl-games, and flash-linear-attention for their open-source code and datasets.
