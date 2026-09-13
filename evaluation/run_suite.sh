#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_EXAMPLE="${MODEL_EXAMPLE:-uno_qwen3_8B}"
RUN_NAME="${RUN_NAME:-release}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/${MODEL_EXAMPLE}}"
RUNNER="${REPO_ROOT}/examples/${MODEL_EXAMPLE}/run_eval.sh"

# Canonical external judges. Model names are fixed here so a full-suite run
# cannot accidentally grade HLE and the Artificial Analysis tasks with the
# same model. Endpoints and credentials remain environment-specific.
HLE_JUDGE_MODEL="${HLE_JUDGE_MODEL:-gpt-5.5}"
HLE_JUDGE_BASE_URL="${HLE_JUDGE_BASE_URL:-${OPENAI_BASE_URL:-}}"
HLE_JUDGE_API_KEY="${HLE_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
GLM_JUDGE_MODEL="${GLM_JUDGE_MODEL:-GLM-5.2-FP8}"
GLM_JUDGE_BASE_URL="${GLM_JUDGE_BASE_URL:-}"
GLM_JUDGE_API_KEY="${GLM_JUDGE_API_KEY:-}"

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

if [[ "${SKIP_GRADING:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  for benchmark in "${benchmarks[@]}"; do
    case "${benchmark}" in
      hle)
        if [[ "${HLE_JUDGE_MODEL}" != "gpt-5.5" ]]; then
          echo "HLE requires HLE_JUDGE_MODEL=gpt-5.5." >&2
          exit 2
        fi
        if [[ -z "${HLE_JUDGE_BASE_URL}" && -z "${HLE_JUDGE_API_KEY}" ]]; then
          echo "Set HLE_JUDGE_API_KEY (or OPENAI_API_KEY), or configure HLE_JUDGE_BASE_URL." >&2
          exit 2
        fi
        ;;
      lcr|aa-lcr|omniscience|aa-omniscience)
        if [[ "${GLM_JUDGE_MODEL}" != *"GLM-5.2-FP8"* ]]; then
          echo "${benchmark} requires GLM_JUDGE_MODEL containing GLM-5.2-FP8." >&2
          exit 2
        fi
        if [[ -z "${GLM_JUDGE_BASE_URL}" ]]; then
          echo "Set GLM_JUDGE_BASE_URL to the OpenAI-compatible GLM-5.2-FP8 endpoint." >&2
          exit 2
        fi
        ;;
    esac
  done
fi

run_benchmark() {
  local benchmark="$1"
  shift
  case "${benchmark}" in
    hle)
      JUDGE_MODEL="${HLE_JUDGE_MODEL}" \
      JUDGE_BASE_URL="${HLE_JUDGE_BASE_URL}" \
      JUDGE_API_KEY="${HLE_JUDGE_API_KEY}" \
        "$@"
      ;;
    lcr|aa-lcr|omniscience|aa-omniscience)
      JUDGE_MODEL="${GLM_JUDGE_MODEL}" \
      JUDGE_BASE_URL="${GLM_JUDGE_BASE_URL}" \
      JUDGE_API_KEY="${GLM_JUDGE_API_KEY}" \
        "$@"
      ;;
    *)
      "$@"
      ;;
  esac
}

export RESULTS_ROOT RUN_NAME
failures=()
for benchmark in "${benchmarks[@]}"; do
  command=(bash "${RUNNER}" "${benchmark}")
  printf 'Running:'
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    case "${benchmark}" in
      hle)
        echo "Judge: ${HLE_JUDGE_MODEL} (HLE)"
        ;;
      lcr|aa-lcr|omniscience|aa-omniscience)
        echo "Judge: ${GLM_JUDGE_MODEL} (Artificial Analysis)"
        ;;
    esac
    continue
  fi
  if ! run_benchmark "${benchmark}" "${command[@]}"; then
    failures+=("${benchmark}")
  fi
done

if (( ${#failures[@]} > 0 )); then
  printf 'Failed benchmarks:' >&2
  printf ' %s' "${failures[@]}" >&2
  printf '\n' >&2
  exit 1
fi
