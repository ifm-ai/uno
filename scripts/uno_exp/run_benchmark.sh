#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 BENCHMARK" >&2
  exit 2
fi

BENCHMARK="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

source "${SCRIPT_DIR}/release_model_defaults.sh"
source "${SCRIPT_DIR}/eval_defaults.sh"

RUN_NAME="${RUN_NAME:-$(basename "${GATED_LORA_PATH}")}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/uno_exp}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${RUN_NAME}/${BENCHMARK}}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/generations.jsonl}"
GRADES="${GRADES:-${OUTPUT_DIR}/grades.jsonl}"
SCORES="${SCORES:-${OUTPUT_DIR}/scores.json}"
GENERATION_SUMMARY="${GENERATION_SUMMARY:-${OUTPUT_DIR}/generation_summary.json}"

DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa2}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.95}"
DIFFUSION_BLOCK_SIZE="${DIFFUSION_BLOCK_SIZE:-16}"
NOISE_MODE="${NOISE_MODE:-random_uniform}"
GRADER_NUM_PROCESSES="${GRADER_NUM_PROCESSES:-${SLURM_CPUS_PER_TASK:-64}}"

if [[ -z "${CUDA_GRAPH_BATCH_SIZES:-}" ]]; then
  CUDA_GRAPH_BATCH_SIZES="$(uno_cuda_graph_batch_sizes "${MAX_NUM_SEQS}")"
fi
CUDA_GRAPH_BLOCK_SIZES="${CUDA_GRAPH_BLOCK_SIZES:-1,${DIFFUSION_BLOCK_SIZE}}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

generation_command=(
  "${PYTHON}"
  -m nano_vllm_uno.eval.generate_benchmark
  --benchmark "${BENCHMARK}"
  --model "${MODEL}"
  --tokenizer-path "${TOKENIZER_PATH:-${MODEL}}"
  --gated-lora-path "${GATED_LORA_PATH}"
  --output "${OUTPUT}"
  --summary-output "${GENERATION_SUMMARY}"
  --data-parallel-size "${DATA_PARALLEL_SIZE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --attention-backend "${ATTENTION_BACKEND}"
  --num-samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-k "${TOP_K}"
  --top-p "${TOP_P}"
  --diffusion-block-size "${DIFFUSION_BLOCK_SIZE}"
  --cuda-graph-block-sizes "${CUDA_GRAPH_BLOCK_SIZES}"
  --cuda-graph-batch-sizes "${CUDA_GRAPH_BATCH_SIZES}"
  --noise-mode "${NOISE_MODE}"
)

if [[ -n "${MODEL_REVISION:-}" ]]; then
  generation_command+=(--model-revision "${MODEL_REVISION}")
fi
if [[ -n "${GATED_LORA_REVISION:-}" ]]; then
  generation_command+=(--gated-lora-revision "${GATED_LORA_REVISION}")
fi
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  generation_command+=(--hf-cache-dir "${HF_CACHE_DIR}")
fi
if [[ "${HF_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  generation_command+=(--hf-local-files-only)
fi

if [[ -n "${DATA:-}" ]]; then
  generation_command+=(--data "${DATA}")
fi
if [[ -n "${MAX_TOKENS:-}" ]]; then
  generation_command+=(--max-tokens "${MAX_TOKENS}")
fi
if [[ -n "${TREE_VERIFY_SIZE:-}" ]]; then
  generation_command+=(--tree-verify-size "${TREE_VERIFY_SIZE}")
fi
if [[ -n "${TREE_CANDIDATE_TOP_K:-}" ]]; then
  generation_command+=(--tree-candidate-top-k "${TREE_CANDIDATE_TOP_K}")
fi
if [[ "${TORCH_COMPILE:-0}" == "1" ]]; then
  generation_command+=(--torch-compile)
fi
if [[ ${INSTRUCTION+x} ]]; then
  generation_command+=(--instruction "${INSTRUCTION}")
fi
if [[ -n "${CHAT_TEMPLATE_KWARGS_JSON:-}" ]]; then
  generation_command+=(
    --chat-template-kwargs-json "${CHAT_TEMPLATE_KWARGS_JSON}"
  )
fi
if [[ -n "${LIMIT:-}" ]]; then
  generation_command+=(--limit "${LIMIT}")
fi
if [[ -n "${MASK_TOKEN_ID:-}" ]]; then
  generation_command+=(--mask-token-id "${MASK_TOKEN_ID}")
fi
if [[ -n "${STOP_TOKEN_IDS:-}" ]]; then
  generation_command+=(--stop-token-ids "${STOP_TOKEN_IDS}")
fi
if [[ "${IGNORE_EOS:-0}" == "1" ]]; then
  generation_command+=(--ignore-eos)
fi
if [[ "${SAVE_TOKEN_IDS:-0}" == "1" ]]; then
  generation_command+=(--save-token-ids)
fi
if [[ "${NO_PROGRESS:-0}" == "1" ]]; then
  generation_command+=(--no-progress)
fi

printf 'Generation command:'
printf ' %q' "${generation_command[@]}"
printf '\n'
"${generation_command[@]}"

if [[ "${SKIP_GRADING:-0}" == "1" ]]; then
  echo "Generation complete; grading skipped."
  exit 0
fi

grading_command=(
  "${PYTHON}"
  -m nano_vllm_uno.eval.score_generations
  --benchmark "${BENCHMARK}"
  --generations "${OUTPUT}"
  --grades "${GRADES}"
  --scores "${SCORES}"
  --generation-summary "${GENERATION_SUMMARY}"
  --grader-num-processes "${GRADER_NUM_PROCESSES}"
)
if [[ -n "${DATA:-}" ]]; then
  grading_command+=(--data "${DATA}")
fi
if [[ -n "${PARSER:-}" ]]; then
  grading_command+=(--parser "${PARSER}")
fi
if [[ -n "${GRADER_TIMEOUT:-}" ]]; then
  grading_command+=(--timeout "${GRADER_TIMEOUT}")
fi

printf 'Grading command:'
printf ' %q' "${grading_command[@]}"
printf '\n'
"${grading_command[@]}"

echo "Generations: ${OUTPUT}"
echo "Generation metrics: ${GENERATION_SUMMARY}"
echo "Grades: ${GRADES}"
echo "Scores: ${SCORES}"
