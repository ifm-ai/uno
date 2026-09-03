#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: bash training/run_fixed_b8_slurm.sh [epochs]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_EPOCHS="${1:-${TRAIN_EPOCHS:-1}}"

: "${DATASET_PATH:?Set DATASET_PATH to the prepared OpenThoughts dataset.}"
: "${TRAIN_STORAGE_ROOT:?Set TRAIN_STORAGE_ROOT to shared training storage.}"

TRAIN_EPOCHS="$(${PYTHON_BIN} -c \
  'import sys; from training.create_fixed_curriculum import parse_epochs; print(format(parse_epochs(sys.argv[1]), "f"))' \
  "${TRAIN_EPOCHS}")"
epoch_tag="${TRAIN_EPOCHS//./p}"
RUN_NAME="${RUN_NAME:-uno-qwen3-8b-b8-${epoch_tag}epoch-$(date -u +%Y%m%dT%H%M%SZ)}"
CURRICULUM_PATH="${CURRICULUM_PATH:-${TRAIN_STORAGE_ROOT}/curricula/${RUN_NAME}.yaml}"

"${PYTHON_BIN}" -m training.create_fixed_curriculum \
  --output "${CURRICULUM_PATH}" \
  --epochs "${TRAIN_EPOCHS}" \
  --block-size 8 \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"

max_steps="$(${PYTHON_BIN} -c \
  'import sys; from training.curriculum import BlockCurriculumPlan; print(BlockCurriculumPlan.from_yaml(sys.argv[1]).max_steps)' \
  "${CURRICULUM_PATH}")"
default_warmup=$((max_steps * 2 / 100))
if (( default_warmup >= max_steps )); then
  default_warmup=$((max_steps - 1))
fi

export RUN_NAME CURRICULUM_PATH
export WARMUP_STEPS="${WARMUP_STEPS:-${default_warmup}}"

echo "Fixed B=8 schedule: requested_epochs=${TRAIN_EPOCHS}, steps=${max_steps}, warmup_steps=${WARMUP_STEPS}"
exec bash "${REPO_ROOT}/training/run_slurm.sh"
