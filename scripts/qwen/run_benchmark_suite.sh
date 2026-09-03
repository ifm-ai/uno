#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/release_model_defaults.sh"
BENCHMARKS=(
  gsm8k
  math500
  aime24
  aime25
  aime26
  humaneval
  mbpp
  lcbv6
  gpqa
  gpqa_diamond
  mmlu_pro
  ifeval
)

for benchmark in "${BENCHMARKS[@]}"; do
  echo "===== ${benchmark} ====="
  OUTPUT_DIR="${RESULTS_ROOT:-${SCRIPT_DIR}/../../results/qwen/${RUN_NAME:-$(basename "${GATED_LORA_PATH}")}}/${benchmark}" \
    "${SCRIPT_DIR}/run_benchmark.sh" "${benchmark}"
done
