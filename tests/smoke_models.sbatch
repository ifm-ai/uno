#!/usr/bin/env bash
#SBATCH --job-name=uno-refactor-smoke
#SBATCH --partition=main
#SBATCH --constraint=nvidia_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

set -euo pipefail

: "${UNO_REPO_ROOT:?Set UNO_REPO_ROOT to the refactored checkout}"
: "${SMOKE_ROOT:?Set SMOKE_ROOT to a shared output directory}"

models=(uno_qwen3_8B uno_8B uno_1B)
index="${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
model="${models[$((index % 3))]}"
workflow=$((index / 3))

export PYTHON="${PYTHON:-/mnt/weka/home/lingjie.chen/miniconda3/envs/nano-fa3-torch28/bin/python}"
export DATA_PARALLEL_SIZE=1
export MAX_NUM_SEQS=1
export MAX_MODEL_LEN=4096
export MAX_NUM_BATCHED_TOKENS=4096
export ATTENTION_BACKEND=fa2
export RESULTS_ROOT="${SMOKE_ROOT}/evaluation"
export RUN_NAME="${model}"
export HF_HOME="${HF_HOME:-/mnt/weka/shrd/k2m/lingjie.chen/hf_cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export PYTHONDONTWRITEBYTECODE=1

cd "${UNO_REPO_ROOT}"
if (( workflow == 0 )); then
  exec bash "examples/${model}/run_inference.sh" \
    --prompt "Return only the number 4." \
    --max-tokens 8 \
    --no-progress
fi

exec bash "examples/${model}/run_eval.sh" gsm8k \
  --limit 1 \
  --max-tokens 8 \
  --no-progress
