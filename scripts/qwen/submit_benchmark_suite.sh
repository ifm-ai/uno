#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/release_model_defaults.sh"

RUN_NAME="${RUN_NAME:-$(basename "${GATED_LORA_PATH}")}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/qwen}"
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/${RUN_NAME}/slurm}"
SLURM_CPUS_PER_JOB="${SLURM_CPUS_PER_JOB:-64}"
SLURM_TIME="${SLURM_TIME:-2-00:00:00}"

default_benchmarks=(
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
if [[ -n "${BENCHMARKS:-}" ]]; then
  read -r -a benchmarks <<< "${BENCHMARKS}"
else
  benchmarks=("${default_benchmarks[@]}")
fi
read -r -a extra_sbatch_args <<< "${SBATCH_ARGS:-}"

mkdir -p "${LOG_DIR}"
export RESULTS_ROOT RUN_NAME
export NANO_VLLM_UNO_REPO_ROOT="${REPO_ROOT}"

for benchmark in "${benchmarks[@]}"; do
  command=(
    sbatch
    --parsable
    --nodes=1
    --ntasks=1
    --gres=gpu:8
    --cpus-per-task="${SLURM_CPUS_PER_JOB}"
    --exclusive
    --time="${SLURM_TIME}"
    --job-name="nano-uno-${benchmark}"
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
    "${extra_sbatch_args[@]}"
    "${SCRIPT_DIR}/run_benchmark_slurm.sh"
    "${benchmark}"
  )
  printf 'Submitting:'
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi
  job_id="$("${command[@]}")"
  echo "${benchmark}: ${job_id}"
done
