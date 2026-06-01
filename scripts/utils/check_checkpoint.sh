#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
if [ -z "${TARGET}" ]; then
  echo "usage: bash scripts/utils/check_checkpoint.sh <run-name|checkpoint-dir>" >&2
  exit 2
fi

if [ -d "${TARGET}" ]; then
  CKPT_DIR="${TARGET}"
elif [ -d "runs/${TARGET}/nn" ]; then
  CKPT_DIR="runs/${TARGET}/nn"
else
  echo "checkpoint directory not found: ${TARGET}" >&2
  exit 1
fi

find "${CKPT_DIR}" -maxdepth 1 -type f -name '*.pth' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
