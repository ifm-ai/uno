#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BENCHMARK [EVALUATION_ARGS...]" >&2
  exit 2
fi
BENCHMARK="$1"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${RESULTS_ROOT:-${REPO_ROOT}/results/uno_1B}/${RUN_NAME:-release}/${BENCHMARK}"

exec "${PYTHON:-python}" -m evaluation.run \
  --benchmark "${BENCHMARK}" \
  --model "${MODEL:-IFM/K2-Horizon-0.9B}" \
  --model-revision "${MODEL_REVISION:-ee770e713760cf6350e4322cdbbff91a163b7d70}" \
  --gated-lora-path "${GATED_LORA_PATH:-IFM/K2-Horizon-0.9B-Uno}" \
  --gated-lora-revision "${GATED_LORA_REVISION:-b0d8896a301a2f4bc755538b1234a35100da50d0}" \
  --output "${OUTPUT_DIR}/generations.jsonl" \
  --summary-output "${OUTPUT_DIR}/generation_summary.json" \
  --grades "${OUTPUT_DIR}/grades.jsonl" \
  --scores "${OUTPUT_DIR}/scores.json" \
  --data-parallel-size "${DATA_PARALLEL_SIZE:-8}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-131072}" \
  --max-tokens "${MAX_TOKENS:-131072}" \
  --attention-backend "${ATTENTION_BACKEND:-fa3}" \
  --num-samples "${NUM_SAMPLES:-1}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top-k "${TOP_K:-50}" \
  --top-p "${TOP_P:-0.95}" \
  --diffusion-block-size "${DIFFUSION_BLOCK_SIZE:-8}" \
  --mask-token-id "${MASK_TOKEN_ID:-64256}" \
  --stop-token-ids "${STOP_TOKEN_IDS:-64019,1}" \
  --instruction "${INSTRUCTION-}" \
  "$@"
