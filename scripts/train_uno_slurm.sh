#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${UNO_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${UNO_REPO_ROOT}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
LAUNCHER_PATH="${REPO_ROOT}/scripts/train_uno_slurm.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
TIME_LIMIT="${TIME_LIMIT:-7-00:00:00}"
RUN_NAME="${RUN_NAME:-uno-qwen3-8b-$(date -u +%Y%m%dT%H%M%SZ)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-uno-training}"
LORA_TARGET="${LORA_TARGET:-all}"
LORA_RANK="${LORA_RANK:-128}"
CE_TARGET="${CE_TARGET:-teacher}"
CE_ALPHA="${CE_ALPHA:-0.0}"
KL_BETA="${KL_BETA:-0.0}"
TV_GAMMA="${TV_GAMMA:-1.0}"
DEEPSPEED_PATH="${DEEPSPEED_PATH:-${REPO_ROOT}/configs/training/deepspeed_zero2.json}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-562}"

: "${DATASET_PATH:?Set DATASET_PATH to the prepared OpenThoughts dataset.}"
: "${TRAIN_STORAGE_ROOT:?Set TRAIN_STORAGE_ROOT to shared training storage.}"
: "${CURRICULUM_PATH:?Set CURRICULUM_PATH to a Uno curriculum YAML file.}"

denominator=$((NODES * GPUS_PER_NODE * PER_DEVICE_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % denominator != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by nodes*gpus*per-device=${denominator}." >&2
  exit 2
fi
computed_accumulation=$((GLOBAL_BATCH_SIZE / denominator))
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${computed_accumulation}}"
if (( denominator * GRADIENT_ACCUMULATION_STEPS != GLOBAL_BATCH_SIZE )); then
  echo "Hardware and accumulation resolve to global batch $((denominator * GRADIENT_ACCUMULATION_STEPS)), expected ${GLOBAL_BATCH_SIZE}." >&2
  exit 2
fi

HF_CACHE_DIR="${HF_CACHE_DIR:-${TRAIN_STORAGE_ROOT}/hf-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_STORAGE_ROOT}/runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${TRAIN_STORAGE_ROOT}/logs/${RUN_NAME}}"
MASTER_PORT="${MASTER_PORT:-}"

export UNO_REPO_ROOT="${REPO_ROOT}"
export REPO_ROOT PYTHON_BIN NODES GPUS_PER_NODE PER_DEVICE_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS RUN_NAME WANDB_MODE WANDB_PROJECT
export LORA_TARGET LORA_RANK LORA_ALPHA CE_TARGET CE_ALPHA KL_BETA TV_GAMMA
export CURRICULUM_PATH DEEPSPEED_PATH
export LEARNING_RATE WARMUP_STEPS
export DATASET_PATH TRAIN_STORAGE_ROOT HF_CACHE_DIR OUTPUT_DIR LOG_DIR MASTER_PORT
export GLOBAL_BATCH_SIZE CPUS_PER_TASK TIME_LIMIT
export RESUME_FROM_CHECKPOINT
export HF_HOME="${HF_HOME:-${HF_CACHE_DIR}}"

if [[ "${UNO_NODE_WORKER:-0}" == "1" ]]; then
  if [[ -n "${UNO_LOCAL_TMPDIR:-}" ]]; then
    worker_tmpdir="${UNO_LOCAL_TMPDIR}"
  elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
    worker_tmpdir="${SLURM_TMPDIR}/uno-${SLURM_PROCID}"
  else
    worker_tmpdir="/tmp/uno-${USER:-user}-${SLURM_JOB_ID}-${SLURM_PROCID}"
  fi
  mkdir -p "${worker_tmpdir}"
  export TMPDIR="${worker_tmpdir}"

  command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    "--nnodes=${SLURM_NNODES}"
    "--nproc-per-node=${GPUS_PER_NODE}"
    "--node-rank=${SLURM_PROCID}"
    "--master-addr=${MASTER_ADDR}"
    "--master-port=${MASTER_PORT}"
    -m nano_vllm_uno.training.train
    --dataset-path "${DATASET_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --curriculum "${CURRICULUM_PATH}"
    --deepspeed "${DEEPSPEED_PATH}"
    --model-cache-dir "${HF_CACHE_DIR}"
    --local-files-only
    --ce-target "${CE_TARGET}"
    --ce-alpha "${CE_ALPHA}"
    --kl-beta "${KL_BETA}"
    --tv-gamma "${TV_GAMMA}"
    --lora-target "${LORA_TARGET}"
    --lora-rank "${LORA_RANK}"
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning-rate "${LEARNING_RATE}"
    --warmup-steps "${WARMUP_STEPS}"
    --run-name "${RUN_NAME}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-mode "${WANDB_MODE}"
  )
  if [[ -n "${LORA_ALPHA:-}" ]]; then
    command+=(--lora-alpha "${LORA_ALPHA}")
  fi
  if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    command+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi
  printf 'Node %s command:' "${SLURM_PROCID}"
  printf ' %q' "${command[@]}"
  printf '\n'
  echo "Node ${SLURM_PROCID} IPC temp: ${TMPDIR}"
  exec "${command[@]}"
fi

mkdir -p "${LOG_DIR}" "${TRAIN_STORAGE_ROOT}/runs" "${HF_CACHE_DIR}"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Prepared dataset directory not found: ${DATASET_PATH}" >&2
  exit 2
fi
if [[ ! -f "${DATASET_PATH}/uno_dataset_manifest.json" ]]; then
  echo "Prepared dataset manifest not found: ${DATASET_PATH}/uno_dataset_manifest.json" >&2
  exit 2
fi
if [[ ! -f "${CURRICULUM_PATH}" ]]; then
  echo "Curriculum file not found: ${CURRICULUM_PATH}" >&2
  exit 2
fi
if [[ ! -f "${DEEPSPEED_PATH}" ]]; then
  echo "DeepSpeed configuration not found: ${DEEPSPEED_PATH}" >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  sbatch_args=(
    --parsable
    "--job-name=${RUN_NAME}"
    "--nodes=${NODES}"
    --ntasks-per-node=1
    "--gres=gpu:${GPUS_PER_NODE}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--output=${LOG_DIR}/slurm-%j.out"
    "--error=${LOG_DIR}/slurm-%j.err"
    --export=ALL
  )
  [[ -n "${SLURM_PARTITION:-}" ]] && sbatch_args+=("--partition=${SLURM_PARTITION}")
  [[ -n "${SLURM_ACCOUNT:-}" ]] && sbatch_args+=("--account=${SLURM_ACCOUNT}")
  [[ -n "${SLURM_QOS:-}" ]] && sbatch_args+=("--qos=${SLURM_QOS}")
  job_id="$(sbatch "${sbatch_args[@]}" "${LAUNCHER_PATH}")"
  echo "Submitted ${job_id}"
  echo "Logs: ${LOG_DIR}/slurm-${job_id}.out and ${LOG_DIR}/slurm-${job_id}.err"
  echo "Results: ${OUTPUT_DIR}"
  exit 0
fi

echo "Prefetching IFM/uno-qwen3-8b-base into ${HF_CACHE_DIR}"
srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m nano_vllm_uno.training.prefetch_model \
  --cache-dir "${HF_CACHE_DIR}"

readarray -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR="${MASTER_ADDR:-${allocated_hosts[0]}}"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export MASTER_ADDR MASTER_PORT
export UNO_NODE_WORKER=1

echo "Launching ${NODES} nodes x ${GPUS_PER_NODE} GPUs; global batch ${GLOBAL_BATCH_SIZE}."
echo "Logs: ${LOG_DIR}/slurm-${SLURM_JOB_ID}.out and ${LOG_DIR}/slurm-${SLURM_JOB_ID}.err"
echo "Results: ${OUTPUT_DIR}"
srun --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 \
  bash "${LAUNCHER_PATH}"
