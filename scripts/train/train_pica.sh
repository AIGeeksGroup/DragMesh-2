#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYBIN="${PYBIN:-python}"

OBJECT_ID="${OBJECT_ID:-45661}"
CONFIG="${CONFIG:-configs/train/pica/train_config_gla_pica_drand12_aux_v2c.yaml}"
TRAJECTORY="${TRAJECTORY:-}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
RUN_NAME="${RUN_NAME:-dragmesh2_${OBJECT_ID}_pica}"

cd "${PROJECT_ROOT}"

args=(
  ppo/train.py
  --train_config "${CONFIG}"
  --object_id "${OBJECT_ID}"
  --num_envs "${NUM_ENVS}"
  --max_epochs "${MAX_EPOCHS}"
  --experiment_name "${RUN_NAME}"
)

if [ -n "${TRAJECTORY}" ]; then
  args+=(--trajectory "${TRAJECTORY}")
fi

"${PYBIN}" "${args[@]}"
