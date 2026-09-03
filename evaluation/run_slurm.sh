#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: MODEL_EXAMPLE=uno_qwen3_8B $0 BENCHMARK" >&2
  exit 2
fi

if [[ -n "${NANO_VLLM_UNO_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${NANO_VLLM_UNO_REPO_ROOT}"
else
  REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
MODEL_EXAMPLE="${MODEL_EXAMPLE:-uno_qwen3_8B}"
RUNNER="${REPO_ROOT}/examples/${MODEL_EXAMPLE}/run_eval.sh"
if [[ ! -x "${RUNNER}" ]]; then
  echo "Could not find the benchmark runner at ${RUNNER}" >&2
  echo "Set MODEL_EXAMPLE to uno_qwen3_8B, uno_8B, or uno_1B." >&2
  exit 127
fi
exec "${RUNNER}" "$1"
