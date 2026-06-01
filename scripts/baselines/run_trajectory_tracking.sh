#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYBIN="${PYBIN:-python}"

OBJECT_ID="${OBJECT_ID:-45661}"
TRAJECTORY="${TRAJECTORY:?Set TRAJECTORY to a trajectory.json path}"
PHASE="${PHASE:-drag}"
OUT_DIR="${OUT_DIR:-output/baselines/trajectory_tracking}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

"${PYBIN}" scripts/track_trajectory_baseline.py \
  --object_id "${OBJECT_ID}" \
  --trajectory "${TRAJECTORY}" \
  --phase "${PHASE}" \
  --log_csv "${OUT_DIR}/${OBJECT_ID}_metrics.csv" \
  --summary_json "${OUT_DIR}/${OBJECT_ID}_summary.json"
