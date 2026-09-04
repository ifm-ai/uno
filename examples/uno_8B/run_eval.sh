#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BENCHMARK [EVALUATION_ARGS...]" >&2
  exit 2
fi
BENCHMARK="$1"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUT_DIR="${RESULTS_ROOT:-${REPO_ROOT}/results/uno_8B}/${RUN_NAME:-release}/${BENCHMARK}"
GRADE_ARGS=()
if [[ "${SKIP_GRADING:-0}" == "1" ]]; then
  GRADE_ARGS+=(--skip-grading)
elif [[ "${BENCHMARK}" =~ ^(hle|lcr|aa-lcr|omniscience|aa-omniscience)$ ]]; then
  : "${JUDGE_MODEL:?Set JUDGE_MODEL for ${BENCHMARK}, or SKIP_GRADING=1}"
  GRADE_ARGS+=(--judge-model "${JUDGE_MODEL}")
  [[ -z "${JUDGE_BASE_URL:-}" ]] || GRADE_ARGS+=(--judge-base-url "${JUDGE_BASE_URL}")
  [[ -z "${JUDGE_API_KEY:-}" ]] || GRADE_ARGS+=(--judge-api-key "${JUDGE_API_KEY}")
fi

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
  --attention-backend "${ATTENTION_BACKEND:-fa3}" \
  --diffusion-block-size "${DIFFUSION_BLOCK_SIZE:-8}" \
  --mask-token-id "${MASK_TOKEN_ID:-250624}" \
  --stop-token-ids "${STOP_TOKEN_IDS:-250019,1}" \
  "${GRADE_ARGS[@]}" \
  "$@"
