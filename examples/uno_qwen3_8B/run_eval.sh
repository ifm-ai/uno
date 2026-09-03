#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BENCHMARK [EVALUATION_ARGS...]" >&2
  exit 2
fi
BENCHMARK="$1"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${RESULTS_ROOT:-${REPO_ROOT}/results/uno_qwen3_8B}/${RUN_NAME:-release}/${BENCHMARK}"

exec "${PYTHON:-python}" -m evaluation.run \
  --benchmark "${BENCHMARK}" \
  --model "${MODEL:-s-sahoo/uno-qwen3-8B}" \
  --model-revision "${MODEL_REVISION:-b8a7577b3223bdcf2b3af0f2fc6e95258b3bbc29}" \
  --gated-lora-subfolder "${GATED_LORA_SUBFOLDER:-adapter}" \
  --output "${OUTPUT_DIR}/generations.jsonl" \
  --summary-output "${OUTPUT_DIR}/generation_summary.json" \
  --grades "${OUTPUT_DIR}/grades.jsonl" \
  --scores "${OUTPUT_DIR}/scores.json" \
  --data-parallel-size "${DATA_PARALLEL_SIZE:-8}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --attention-backend "${ATTENTION_BACKEND:-fa2}" \
  --num-samples "${NUM_SAMPLES:-1}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top-k "${TOP_K:-50}" \
  --top-p "${TOP_P:-0.95}" \
  --diffusion-block-size "${DIFFUSION_BLOCK_SIZE:-16}" \
  "$@"
