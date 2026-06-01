#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYBIN="${PYBIN:-python}"

OBJECT_ID="${OBJECT_ID:-45661}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a .pth file or checkpoint directory}"
TRAJECTORY="${TRAJECTORY:-}"
OUT_ROOT="${OUT_ROOT:-output/eval/${OBJECT_ID}}"
EPISODES="${EPISODES:-20}"
SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-300}"
DAMPS="${DAMPS:-1.0 2.0 4.0}"
GLA_POOL="${GLA_POOL:-last}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}"

for mode in det stoch; do
  for damp in ${DAMPS}; do
    prefix="${OUT_ROOT}/obj${OBJECT_ID}_${mode}_damp${damp}"
    args=(
      scripts/evaluate_ppo_baseline.py
      --object_id "${OBJECT_ID}"
      --checkpoint "${CHECKPOINT}"
      --checkpoint-kind "${CHECKPOINT_KIND}"
      --episodes "${EPISODES}"
      --seed "${SEED}"
      --max_episode_length "${MAX_LEN}"
      --object_damping_scale "${damp}"
      --gla_pool "${GLA_POOL}"
      --log_csv "${prefix}_metrics.csv"
      --summary_json "${prefix}_summary.json"
    )
    if [ -n "${TRAJECTORY}" ]; then
      args+=(--trajectory "${TRAJECTORY}")
    fi
    if [ "${mode}" = "stoch" ]; then
      args+=(--stochastic)
    fi
    "${PYBIN}" "${args[@]}"
  done
done
