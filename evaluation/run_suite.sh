#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_EXAMPLE="${MODEL_EXAMPLE:-uno_qwen3_8B}"
RUN_NAME="${RUN_NAME:-release}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/${MODEL_EXAMPLE}}"
RUNNER="${REPO_ROOT}/examples/${MODEL_EXAMPLE}/run_eval.sh"

default_benchmarks=(
  aime24
  aime25
  aime26
  arc_challenge
  gpqa_diamond
  gsm8k
  hle
  humaneval
  ifeval
  lcr
  math500
  mbpp
  omniscience
)
if [[ -n "${BENCHMARKS:-}" ]]; then
  read -r -a benchmarks <<< "${BENCHMARKS}"
else
  benchmarks=("${default_benchmarks[@]}")
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "Could not find the benchmark runner at ${RUNNER}" >&2
  echo "Set MODEL_EXAMPLE to uno_qwen3_8B, uno_8B, or uno_1B." >&2
  exit 127
fi

export RESULTS_ROOT RUN_NAME
failures=()
for benchmark in "${benchmarks[@]}"; do
  command=(bash "${RUNNER}" "${benchmark}")
  printf 'Running:'
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi
  if ! "${command[@]}"; then
    failures+=("${benchmark}")
  fi
done

if (( ${#failures[@]} > 0 )); then
  printf 'Failed benchmarks:' >&2
  printf ' %s' "${failures[@]}" >&2
  printf '\n' >&2
  exit 1
fi
