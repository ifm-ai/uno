#!/usr/bin/env bash
#SBATCH --job-name=k2h7b-tpf
#SBATCH --partition=main
#SBATCH --account=arch
#SBATCH --qos=arch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4

set -euo pipefail

REPO_ROOT="${NANO_VLLM_UNO_REPO_ROOT:-/mnt/weka/home/lingjie.chen/uno}"
BASE_ROOT="${K2_HORIZON_ARTIFACT_ROOT:-/mnt/weka/shrd/k2m/lingjie.chen/hf_remote_smoke_20260903}"
benchmarks=(gsm8k math500 aime24 aime25 aime26)
benchmark="${benchmarks[${SLURM_ARRAY_TASK_ID}]}"

export PYTHON="${PYTHON:-/mnt/weka/home/lingjie.chen/miniconda3/envs/nano-fa3-torch28/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MODEL="${MODEL:-${BASE_ROOT}/K2-Horizon-7B}"
export GATED_LORA_PATH="${GATED_LORA_PATH:-${BASE_ROOT}/K2-Horizon-7B-Uno}"
unset MODEL_REVISION GATED_LORA_REVISION
unset DATA OUTPUT GRADES SCORES GENERATION_SUMMARY LIMIT
unset MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_TOKENS MAX_NUM_SEQS
unset TEMPERATURE TOP_K TOP_P ATTENTION_BACKEND DIFFUSION_BLOCK_SIZE
unset CUDA_GRAPH_BLOCK_SIZES CUDA_GRAPH_BATCH_SIZES MASK_TOKEN_ID
unset STOP_TOKEN_IDS NOISE_MODE INSTRUCTION
export RESULTS_ROOT="${RESULTS_ROOT:?Set RESULTS_ROOT to a unique shared output directory}"
export RUN_NAME="${RUN_NAME:-official-k2-horizon-7b-uno-linear-b8-n1}"
export DATA_PARALLEL_SIZE=8
export TENSOR_PARALLEL_SIZE=1
export NUM_SAMPLES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_DISTRIBUTED_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 31) % 20000))
export SKIP_GRADING=0

exec "${REPO_ROOT}/scripts/k2_horizon/run_${benchmark}_linear_b8.sh"
