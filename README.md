# [Uno](https://github.com/drproduck/nano-vllm-uno)

[![Uno](https://img.shields.io/badge/Hugging%20Face-Uno-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/IFM/K2-Horizon-7B-Uno)
[![Uno-Qwen](https://img.shields.io/badge/Hugging%20Face-Uno%20Qwen-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/s-sahoo/uno-qwen3-8B)
[![License](https://img.shields.io/badge/License-Apache--2.0-2F80ED.svg)](LICENSE)

We present UNO, a diffusion-augmented LLM with two pathways in a unified architecture:
- The AR pathway uses the base AR weights.
- The diffusion pathway combines these weights with LoRA-based diffusion adapters.

In this repository, we provide:
- `Uno` Training Pipeline
   - AR weights initialized from `Qwen3-8B`,
   - Diffusion weights trained on `OpenThoughts` using a progressive block-size curriculum.
- $\Psi$-spec samplers
  - `Linear` Sampler
  - `Tree` Sampler
# Getting Started

## Installation

```bash
conda create -n nano-vllm-uno python=3.10 pip -y
conda activate nano-vllm-uno
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  'https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
python -m pip install -e '.[eval,train]'
```

The installation above uses FlashAttention-2 (FA2) and supports linear decoding.
Tree verification additionally requires FlashAttention-3 (FA3), installed in
the same environment:

```bash
python -m pip install ninja==1.13.0
git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=16 python -m pip install --no-build-isolation .
```

## Preparation

Accept the terms for the gated
[GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa), then
authenticate with Hugging Face:

```bash
hf auth login
```

The evaluation launcher automatically downloads and prepares any missing data
from pinned public sources. To prepare all 12 benchmarks in advance, run:

```bash
export HF_HOME=/path/to/shared/hf-cache
export UNO_EVAL_DATA_DIR=/path/to/shared/uno-eval-data
python scripts/qwen/prepare_benchmark_data.py
```

The suite contains GSM8K, MATH500, AIME 2024--2026, HumanEval, MBPP,
LiveCodeBench v6, GPQA, GPQA-Diamond, MMLU-Pro, and IFEval.

> [!WARNING]
> HumanEval, MBPP, and LiveCodeBench grading execute model-generated Python.
> Run coding evaluations only in an isolated, secure sandbox.

Training requires an explicitly prepared copy of the pinned
OpenThoughts3-1.2M corpus:

```bash
python -m nano_vllm_uno.training.prepare_openthoughts \
  --output /path/to/shared/openthoughts-uno-4095 \
  --cache-dir /path/to/shared/hf-cache \
  --num-proc 32
```

## Checkpoints

- Base model: [IFM/uno-qwen3-8b-base](https://huggingface.co/IFM/uno-qwen3-8b-base),
  revision `4ccfeed3fba497e40495fe6dc5c15c89f7f1e2cd`.
- LoRA adapter:
  [s-sahoo/uno-qwen3-8B](https://huggingface.co/s-sahoo/uno-qwen3-8B),
  revision `79536cf8c70aa48b9badc2532ffef208947463e3`.

The launchers use these pinned revisions by default; see
[`scripts/qwen/release_model_defaults.sh`](scripts/qwen/release_model_defaults.sh).

Model-specific launchers are organized under `scripts/qwen` and
`scripts/k2_horizon`. The Qwen directory contains the complete generic
evaluation suite; the K2 Horizon directory contains the K2 model defaults and
linear-throughput wrappers while reusing the same benchmark runner.

# Inference

## Linear Sampler (System Throughput optimal)

The default pull-based evaluation runs all 12 benchmarks using one persistent
TP=1 worker per GPU. It uses one eight-GPU node, block size `B=16`, a
32,768-token request-level context budget, temperature 1, top-p 0.95, top-k 50,
and up to 64 active sequences per GPU:

```bash
RESULTS_ROOT=/path/to/shared/results \
HF_CACHE_DIR=/path/to/shared/hf-cache \
ATTENTION_BACKEND=fa2 \
SBATCH_ARGS="--account=my-account --partition=my-partition" \
bash scripts/qwen/submit_pull_suite.sh
```

The launcher prepares missing data, grades every benchmark, and writes
generations, scores, resolved settings, per-sequence forward counts, and
aggregate TPF under `${RESULTS_ROOT}/${RUN_NAME}`. Useful overrides include
`RUN_NAME`, `BENCHMARKS`, `LIMIT`, `NUM_SAMPLES`, `DIFFUSION_BLOCK_SIZE`,
`MAX_NUM_SEQS`, `MODEL`, and `GATED_LORA_PATH`.

To submit benchmarks independently instead, use the per-benchmark script:

```bash
RESULTS_ROOT=/path/to/shared/results \
HF_CACHE_DIR=/path/to/shared/hf-cache \
BENCHMARKS="aime24 humaneval" \
SBATCH_ARGS="--account=my-account --partition=my-partition" \
bash scripts/qwen/submit_benchmark_suite.sh
```

## Tree Sampler (Per-request Throughput Optimal)

Install FA3 as described above, then launch the paper tree configuration with
block size `B=16`, verification budget `V=60`, and candidate top-k 32:

```bash
RESULTS_ROOT=/path/to/shared/results \
HF_CACHE_DIR=/path/to/shared/hf-cache \
ATTENTION_BACKEND=fa3 \
DIFFUSION_BLOCK_SIZE=16 \
TREE_VERIFY_SIZE=60 \
TREE_CANDIDATE_TOP_K=32 \
TORCH_COMPILE=0 \
SBATCH_ARGS="--account=my-account --partition=my-partition" \
bash scripts/qwen/submit_pull_suite.sh
```

# Training 

Training is LoRA-only and starts from the pinned Uno base model above. W&B logging is enabled by default, so run
`wandb login` first or set `WANDB_MODE=disabled`.

After preparing OpenThoughts (as outlined above), launch the default two-node, 16-GPU, three-epoch block-curriculum training:

```bash
DATASET_PATH=/path/to/shared/openthoughts-uno-4095 \
TRAIN_STORAGE_ROOT=/path/to/shared/uno-training \
SLURM_ACCOUNT=my-account \
SLURM_PARTITION=my-partition \
bash scripts/train_uno_3epoch_slurm.sh
```

The training schedule runs for half an epoch at each value of `B`: `2`, `4`, `6`, `8`, `12`, and `16`.

### Training configuration

- Global batch size: `128`
- Learning rate: `1e-5`
- Warmup: `562` steps, or 2% of training
- Learning-rate decay: disabled to simplify continued fine-tuning

### LoRA configuration

A rank-128 LoRA adapter is applied to the Q, K, V, O, and MLP gate, up, and down projections, with an alpha of `2048`.

The following overrides are available:

- `LORA_RANK`
- `LORA_ALPHA`
- `LORA_TARGET`

If `LORA_ALPHA` is omitted, it defaults to `16 * LORA_RANK`.

`LORA_TARGET` accepts:

- `all`, the default
- Attention projection combinations such as `o`, `q`, `qk`, or `qkvo`
- Explicit comma-separated projection names

### Training objective

The default objective is TV-only:

- CE weight: `CE_ALPHA=0`
- Reverse-KL weight: `KL_BETA=0`
- TV weight: `TV_GAMMA=1`

When `CE_ALPHA` is positive, `CE_TARGET` selects the labels used for the CE objective. For example, set `CE_TARGET=ground_truth` to use ground-truth labels.

### Additional overrides

Other useful overrides include `WANDB_MODE` and the Slurm hardware variables.

To resume from a curriculum boundary, set `RESUME_FROM_CHECKPOINT` and retain the original `RUN_NAME`.

## Acknowledgements

Uno's runtime builds on
[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), and its
training utilities are adapted from
[LlamaFactory](https://github.com/hiyouga/LlamaFactory). See [NOTICE](NOTICE)
for the corresponding attribution and license notices.

## Citation

```bibtex
@article{uno2026,
  title   = {},
  author  = {},
  journal = {},
  year    = {2026}
}
```
