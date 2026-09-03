#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_NAME="${RUN_NAME:-uno-qwen3-8b-3epoch-$(date -u +%Y%m%dT%H%M%SZ)}"
export CURRICULUM_PATH="${CURRICULUM_PATH:-${REPO_ROOT}/configs/training/uno_3epoch_curriculum.yaml}"

exec bash "${REPO_ROOT}/scripts/train_uno_slurm.sh"
