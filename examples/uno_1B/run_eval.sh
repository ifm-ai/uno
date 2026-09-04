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
OUTPUT_DIR="${RESULTS_ROOT:-${REPO_ROOT}/results/uno_1B}/${RUN_NAME:-release}/${BENCHMARK}"
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
  "$@" \
  --benchmark "${BENCHMARK}" \
  --model IFM/K2-Horizon-0.9B \
  --model-revision ee770e713760cf6350e4322cdbbff91a163b7d70 \
  --gated-lora-path IFM/K2-Horizon-0.9B-Uno \
  --gated-lora-revision b0d8896a301a2f4bc755538b1234a35100da50d0 \
  --output "${OUTPUT_DIR}/generations.jsonl" \
  --summary-output "${OUTPUT_DIR}/generation_summary.json" \
  --grades "${OUTPUT_DIR}/grades.jsonl" \
  --scores "${OUTPUT_DIR}/scores.json" \
  --data-parallel-size "${DATA_PARALLEL_SIZE:-8}" \
  --attention-backend fa3 \
  --diffusion-block-size 8 \
  --mask-token-id 64256 \
  --stop-token-ids 64019,1 \
  "${GRADE_ARGS[@]}"
