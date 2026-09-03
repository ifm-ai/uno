#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATA="${DATA:-/mnt/weka/shrd/k2m/bbq_diff_eval/data/gsm8k_cot_zeroshot.jsonl}"
export INSTRUCTION="${INSTRUCTION-}"
export NUM_SAMPLES=1
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_K="${TOP_K:-50}"
export TOP_P="${TOP_P:-0.95}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-262144}"
export MAX_TOKENS="${MAX_TOKENS:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
export DIFFUSION_BLOCK_SIZE=8
export CUDA_GRAPH_BLOCK_SIZES="${CUDA_GRAPH_BLOCK_SIZES:-1,8}"
export CUDA_GRAPH_BATCH_SIZES="${CUDA_GRAPH_BATCH_SIZES:-1,2,4}"
export MASK_TOKEN_ID="${MASK_TOKEN_ID:-250624}"
export STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-250019,1}"
export NOISE_MODE="${NOISE_MODE:-random_uniform}"

exec "${SCRIPT_DIR}/run_benchmark.sh" gsm8k
