#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

set -euo pipefail

if [[ -n "${NANO_VLLM_UNO_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${NANO_VLLM_UNO_REPO_ROOT}"
else
  REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
SCRIPT_DIR="${REPO_ROOT}/evaluation"
PYTHON="${PYTHON:-python}"

MODEL_EXAMPLE="${MODEL_EXAMPLE:-uno_qwen3_8B}"
if [[ -z "${MODEL:-}" ]]; then
  case "${MODEL_EXAMPLE}" in
    uno_qwen3_8B)
      MODEL_REVISION="${MODEL_REVISION:-b8a7577b3223bdcf2b3af0f2fc6e95258b3bbc29}"
      MODEL="$("${PYTHON}" -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download("s-sahoo/uno-qwen3-8B", revision=sys.argv[1]))' "${MODEL_REVISION}")"
      GATED_LORA_PATH="${MODEL}/adapter"
      ;;
    uno_8B)
      MODEL=IFM/K2-Horizon-7B
      MODEL_REVISION="${MODEL_REVISION:-586b03f0fd1fbbf2f13eeafc33749e95ae34dd10}"
      GATED_LORA_PATH=IFM/K2-Horizon-7B-Uno
      GATED_LORA_REVISION="${GATED_LORA_REVISION:-ec92bbd768f4a404319625204544782e3377bcd7}"
      ;;
    uno_1B)
      MODEL=IFM/K2-Horizon-0.9B
      MODEL_REVISION="${MODEL_REVISION:-ee770e713760cf6350e4322cdbbff91a163b7d70}"
      GATED_LORA_PATH=IFM/K2-Horizon-0.9B-Uno
      GATED_LORA_REVISION="${GATED_LORA_REVISION:-b0d8896a301a2f4bc755538b1234a35100da50d0}"
      ;;
    *) echo "Unknown MODEL_EXAMPLE: ${MODEL_EXAMPLE}" >&2; exit 2 ;;
  esac
fi

uno_cuda_graph_batch_sizes() {
  local maximum="$1" batch=1 values=()
  while (( batch <= maximum )); do values+=("${batch}"); batch=$((batch * 2)); done
  [[ "${values[-1]}" == "${maximum}" ]] || values+=("${maximum}")
  local IFS=,; echo "${values[*]}"
}

adapter_or_model="${GATED_LORA_PATH:-${MODEL}}"
RUN_NAME="${RUN_NAME:-$(basename "${adapter_or_model}")-pull-b${DIFFUSION_BLOCK_SIZE:-16}}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/qwen}"
RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
JOB_LOG_DIR="${RUN_DIR}/slurm-${SLURM_JOB_ID}"
mkdir -p "${JOB_LOG_DIR}"
export NLTK_DATA="${NLTK_DATA:-${REPO_ROOT}/.cache/nltk_data}"
mkdir -p "${NLTK_DATA}"

export NANO_MODEL="${MODEL}"
export NANO_MODEL_REVISION="${MODEL_REVISION:-}"
export NANO_TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
export NANO_TOKENIZER_REVISION="${TOKENIZER_REVISION:-${MODEL_REVISION:-}}"
export NANO_GATED_LORA_PATH="${GATED_LORA_PATH:-}"
export NANO_GATED_LORA_REVISION="${GATED_LORA_REVISION:-}"
export NANO_HF_CACHE_DIR="${HF_CACHE_DIR:-}"
export NANO_HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-0}"
export NANO_RESULTS_ROOT="${RESULTS_ROOT}"
export NANO_RUN_NAME="${RUN_NAME}"
export NANO_BENCHMARKS="${BENCHMARKS:-}"
export NANO_DATA_ROOT="${DATA_ROOT:-}"
export NANO_NUM_SAMPLES="${NUM_SAMPLES:-1}"
export NANO_LIMIT="${LIMIT:-}"
export NANO_MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export NANO_CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export NANO_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export NANO_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export NANO_ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
export NANO_MAX_TOKENS="${MAX_TOKENS:-}"
export NANO_TEMPERATURE="${TEMPERATURE:-1.0}"
export NANO_TOP_K="${TOP_K:-50}"
export NANO_TOP_P="${TOP_P:-0.95}"
export NANO_DIFFUSION_BLOCK_SIZE="${DIFFUSION_BLOCK_SIZE:-16}"
export NANO_TREE_VERIFY_SIZE="${TREE_VERIFY_SIZE:-}"
export NANO_TREE_CANDIDATE_TOP_K="${TREE_CANDIDATE_TOP_K:-16}"
export NANO_TORCH_COMPILE="${TORCH_COMPILE:-0}"
export NANO_NOISE_MODE="${NOISE_MODE:-random_uniform}"
export NANO_NOISE_SALT="${NOISE_SALT:-}"
export NANO_IGNORE_EOS="${IGNORE_EOS:-0}"
export NANO_MASK_TOKEN_ID="${MASK_TOKEN_ID:-}"
export NANO_STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-}"
export NANO_CUDA_GRAPH_BLOCK_SIZES="${CUDA_GRAPH_BLOCK_SIZES:-1,${NANO_DIFFUSION_BLOCK_SIZE}}"
if [[ -n "${CUDA_GRAPH_BATCH_SIZES:-}" ]]; then
  NANO_CUDA_GRAPH_BATCH_SIZES="${CUDA_GRAPH_BATCH_SIZES}"
else
  NANO_CUDA_GRAPH_BATCH_SIZES="$(
    uno_cuda_graph_batch_sizes "${NANO_MAX_NUM_SEQS}"
  )"
fi
export NANO_CUDA_GRAPH_BATCH_SIZES
export NANO_SAVE_TOKEN_IDS="${SAVE_TOKEN_IDS:-0}"

NANO_WORKERS="${SLURM_NTASKS:-$((SLURM_NNODES * 8))}"
NANO_PULL_COORDINATOR_HOST="$(
  scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p'
)"
NANO_PULL_COORDINATOR_PORT="${NANO_PULL_COORDINATOR_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
NANO_PULL_AUTH_KEY="${NANO_PULL_AUTH_KEY:-${SLURM_JOB_ID}}"
export NANO_PULL_COORDINATOR_HOST NANO_PULL_COORDINATOR_PORT NANO_PULL_AUTH_KEY

generation_command=("${PYTHON}" -m evaluation.pull_suite)
cd "${REPO_ROOT}"
printf 'Generation command:'
printf ' %q' "${generation_command[@]}"
printf '\nResolved environment and resume contract will be saved to %s\n' \
  "${RUN_DIR}/run_manifest.json"
printf 'Coordinator/worker logs: %s\n' "${JOB_LOG_DIR}"

srun \
  --unbuffered \
  --overlap \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${NANO_WORKERS}" \
  --ntasks-per-node="$((NANO_WORKERS / SLURM_NNODES))" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  --output="${JOB_LOG_DIR}/worker-%t.out" \
  --error="${JOB_LOG_DIR}/worker-%t.err" \
  "${generation_command[@]}"

if [[ "${SKIP_GRADING:-0}" == "1" ]]; then
  echo "Generation complete; grading skipped."
  exit 0
fi

GRADER_NUM_PROCESSES="${GRADER_NUM_PROCESSES:-64}"
benchmark_data_tsv="${JOB_LOG_DIR}/benchmark-data.tsv"
"${PYTHON}" -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for benchmark in manifest["benchmarks"]:
    print("{}\t{}".format(benchmark["name"], benchmark["data"]))
' "${RUN_DIR}/run_manifest.json" >"${benchmark_data_tsv}"

while IFS=$'\t' read -r benchmark data_path; do
  benchmark_dir="${RUN_DIR}/${benchmark}"
  grading_command=(
    "${PYTHON}"
    -m evaluation.score_generations
    --benchmark "${benchmark}"
    --data "${data_path}"
    --generations "${benchmark_dir}/generations.jsonl"
    --grades "${benchmark_dir}/grades.jsonl"
    --scores "${benchmark_dir}/scores.json"
    --generation-summary "${benchmark_dir}/generation_summary.json"
    --grader-num-processes "${GRADER_NUM_PROCESSES}"
  )
  printf 'Grading command:'
  printf ' %q' "${grading_command[@]}"
  printf '\n'
  "${grading_command[@]}" \
    >"${JOB_LOG_DIR}/grade-${benchmark}.out" \
    2>"${JOB_LOG_DIR}/grade-${benchmark}.err"
done <"${benchmark_data_tsv}"

echo "Suite results: ${RUN_DIR}"
echo "Suite generation metrics: ${RUN_DIR}/suite_generation_summary.json"
echo "Slurm attempt logs: ${JOB_LOG_DIR}"
