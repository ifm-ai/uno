#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_EXAMPLE="${MODEL_EXAMPLE:-uno_qwen3_8B}"
RUN_NAME="${RUN_NAME:-${MODEL_EXAMPLE}-pull-b${DIFFUSION_BLOCK_SIZE:-16}}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/${MODEL_EXAMPLE}}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SLURM_TIME="${SLURM_TIME:-2-00:00:00}"
JOB_NAME="${JOB_NAME:-nano-uno-pull}"
LOG_ROOT="${RESULTS_ROOT}/${RUN_NAME}/slurm-submit"
read -r -a extra_sbatch_args <<< "${SBATCH_ARGS:-}"

mkdir -p "${LOG_ROOT}"
export RUN_NAME RESULTS_ROOT
export MODEL_EXAMPLE
export NANO_VLLM_UNO_REPO_ROOT="${REPO_ROOT}"

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
  --job-name="${JOB_NAME}"
  --output="${LOG_ROOT}/%x-%j.out"
  --error="${LOG_ROOT}/%x-%j.err"
  "${extra_sbatch_args[@]}"
  "${SCRIPT_DIR}/run_pull_suite_slurm.sh"
)

printf 'Submitting:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
job_id="$("${command[@]}")"
echo "Submitted ${job_id}"
echo "Submit log: ${LOG_ROOT}/${JOB_NAME}-${job_id}.out"
echo "Results: ${RESULTS_ROOT}/${RUN_NAME}"
