#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_EXAMPLE="${MODEL_EXAMPLE:-uno_qwen3_8B}"
RUN_NAME="${RUN_NAME:-${MODEL_EXAMPLE}-pull}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/${MODEL_EXAMPLE}}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SLURM_TIME="${SLURM_TIME:-2-00:00:00}"
LOG_ROOT="${RESULTS_ROOT}/${RUN_NAME}/slurm-submit"
read -r -a extra_sbatch_args <<< "${SBATCH_ARGS:-}"

default_benchmarks=(
  aime24 aime25 aime26 arc_challenge gpqa_diamond gsm8k hle
  humaneval ifeval lcr math500 mbpp omniscience
)
if [[ -n "${BENCHMARKS:-}" ]]; then
  read -r -a benchmarks <<< "${BENCHMARKS}"
else
  benchmarks=("${default_benchmarks[@]}")
fi

mkdir -p "${LOG_ROOT}"
export RESULTS_ROOT MODEL_EXAMPLE
export NANO_VLLM_UNO_REPO_ROOT="${REPO_ROOT}"

# One benchmark per allocation is intentional: canonical sample count,
# temperature, context, and output budget differ across the suite.
for benchmark in "${benchmarks[@]}"; do
  benchmark_run_name="${RUN_NAME}/${benchmark}"
  job_name="${JOB_NAME:-nano-uno-pull}-${benchmark}"
  command=(
    sbatch
    --parsable
    --nodes="${NODES}"
    --ntasks="$((NODES * GPUS_PER_NODE))"
    --ntasks-per-node="${GPUS_PER_NODE}"
    --gres="gpu:${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --exclusive
    --time="${SLURM_TIME}"
    --job-name="${job_name}"
    --output="${LOG_ROOT}/%x-%j.out"
    --error="${LOG_ROOT}/%x-%j.err"
    --export="ALL,BENCHMARKS=${benchmark},RUN_NAME=${benchmark_run_name}"
    "${extra_sbatch_args[@]}"
    "${SCRIPT_DIR}/run_pull_suite_slurm.sh"
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
