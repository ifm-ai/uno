#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BENCHMARK [EVALUATION_ARGS...]" >&2
  exit 2
fi
BENCHMARK="$1"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${RESULTS_ROOT:-${REPO_ROOT}/results/uno_8B}/${RUN_NAME:-release}/${BENCHMARK}"
case "${BENCHMARK}" in
  aime25|aime26) DEFAULT_CONTEXT=500000; DEFAULT_MAX_TOKENS=500000 ;;
  *) DEFAULT_CONTEXT=262144; DEFAULT_MAX_TOKENS=131072 ;;
esac

exec "${PYTHON:-python}" -m evaluation.run \
  --benchmark "${BENCHMARK}" \
  --model "${MODEL:-IFM/K2-Horizon-7B}" \
  --model-revision "${MODEL_REVISION:-586b03f0fd1fbbf2f13eeafc33749e95ae34dd10}" \
  --gated-lora-path "${GATED_LORA_PATH:-IFM/K2-Horizon-7B-Uno}" \
  --gated-lora-revision "${GATED_LORA_REVISION:-ec92bbd768f4a404319625204544782e3377bcd7}" \
  --output "${OUTPUT_DIR}/generations.jsonl" \
  --summary-output "${OUTPUT_DIR}/generation_summary.json" \
  --grades "${OUTPUT_DIR}/grades.jsonl" \
  --scores "${OUTPUT_DIR}/scores.json" \
  --data-parallel-size "${DATA_PARALLEL_SIZE:-8}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-model-len "${MAX_MODEL_LEN:-${DEFAULT_CONTEXT}}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-${DEFAULT_CONTEXT}}" \
  --max-tokens "${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}" \
  --attention-backend "${ATTENTION_BACKEND:-fa3}" \
  --num-samples "${NUM_SAMPLES:-1}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top-k "${TOP_K:-50}" \
  --top-p "${TOP_P:-0.95}" \
  --diffusion-block-size "${DIFFUSION_BLOCK_SIZE:-8}" \
  --mask-token-id "${MASK_TOKEN_ID:-250624}" \
  --stop-token-ids "${STOP_TOKEN_IDS:-250019,1}" \
  --instruction "${INSTRUCTION-}" \
  "$@"
