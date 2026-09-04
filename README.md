# [Uno](https://github.com/ifm-ai/uno)

[![K2 Horizon 7B Uno](https://img.shields.io/badge/Hugging%20Face-K2%20Horizon%207B%20Uno-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/IFM/K2-Horizon-7B-Uno)
[![K2 Horizon 0.9B Uno](https://img.shields.io/badge/Hugging%20Face-K2%20Horizon%200.9B%20Uno-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/IFM/K2-Horizon-0.9B-Uno)
[![Uno Qwen3 8B](https://img.shields.io/badge/Hugging%20Face-Uno%20Qwen3%208B-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/s-sahoo/uno-qwen3-8B)
[![License](https://img.shields.io/badge/License-Apache--2.0-2F80ED.svg)](LICENSE)

Uno is a diffusion-augmented language model with two pathways in a unified
architecture:

- The autoregressive pathway uses the base autoregressive weights.
- The diffusion pathway augments the base model with a conditional LoRA
  adapter.

Uno supports lossless speculative decoding through two $\Psi$-spec samplers:

- **Linear sampling** for high system throughput.
- **Tree sampling** for high per-request throughput.

## What This Repository Provides

- A Nano-vLLM-based inference engine for Uno models.
- Linear and tree speculative samplers.
- Model support for Uno Qwen3 8B and K2 Horizon Uno models.
- A LoRA-based diffusion training pipeline.
- Shared benchmark generation, grading, and throughput measurement utilities.
- Model-specific launchers for reproducible evaluation.

## Code Organization

The runtime remains under `nano_vllm_uno`. Training and evaluation build on the
same engine but are kept in their own modules and launchers:

```text
uno/
├── nano_vllm_uno/
│   ├── engine/                 # Scheduling, KV cache, linear and tree decoding
│   ├── layers/                 # Attention, normalization, sampling, and kernels
│   ├── models/                 # Qwen3 and K2/XLLM model implementations
│   ├── eval/                   # Dataset loading, generation, graders, and scoring
│   ├── training/               # Data, objectives, LoRA, checkpoints, and trainer
│   ├── llm.py                  # User-facing generation API
│   └── sampling_params.py      # Sampling configuration
├── configs/
│   └── training/               # Training and DeepSpeed configurations
├── scripts/
│   ├── common/                 # Shared benchmark entry point
│   ├── qwen/                   # Uno Qwen3 8B launchers and model defaults
│   ├── k2_horizon/             # K2 Horizon Uno launchers and model defaults
│   └── train_uno_*.sh          # Training launchers
├── tests/                      # Runtime, model, evaluation, and training tests
├── pyproject.toml
└── README.md
```

The model-specific benchmark scripts only resolve model defaults and then
delegate generation and grading to `scripts/common/run_benchmark.sh`. This
keeps the evaluation logic shared across model families.

## Getting Started

### Installation

Create a Python 3.10 environment and install Uno with its evaluation and
training dependencies:

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

This installs FlashAttention-2 (FA2), which is sufficient for linear
decoding. Tree verification additionally requires FlashAttention-3 (FA3):

```bash
python -m pip install ninja==1.13.0
git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=16 python -m pip install --no-build-isolation .
```

### Checkpoints

| Model | Uno checkpoint | Loading layout |
| --- | --- | --- |
| K2 Horizon 7B Uno | [IFM/K2-Horizon-7B-Uno](https://huggingface.co/IFM/K2-Horizon-7B-Uno) | Base model: `IFM/K2-Horizon-7B`; Uno repository contains the conditional LoRA adapter |
| K2 Horizon 0.9B Uno | [IFM/K2-Horizon-0.9B-Uno](https://huggingface.co/IFM/K2-Horizon-0.9B-Uno) | Base model: `IFM/K2-Horizon-0.9B`; Uno repository contains the conditional LoRA adapter |
| Uno Qwen3 8B | [s-sahoo/uno-qwen3-8B](https://huggingface.co/s-sahoo/uno-qwen3-8B) | Base weights are at the repository root; the conditional LoRA adapter is under `adapter/` |

Public checkpoints can be downloaded without a Hugging Face token. A token is
still required for gated datasets such as GPQA and for any private or gated
model repository.

## Reproducing Experiments

### Generate Samples with the Linear Sampler

The Qwen launchers use `s-sahoo/uno-qwen3-8B` by default. They resolve the
repository through the standard Hugging Face cache and load the base weights
from the snapshot root and the adapter from `adapter/`:

```bash
RESULTS_ROOT=/path/to/results \
DATA_PARALLEL_SIZE=1 \
DIFFUSION_BLOCK_SIZE=8 \
  bash scripts/qwen/run_gsm8k_eval.sh
```

To pin a reproducible Hugging Face snapshot, set
`UNO_BUNDLE_REVISION=<commit-sha>`. Otherwise, the launcher uses the current
`main` revision.

K2 Horizon 7B wrappers use `IFM/K2-Horizon-7B` with the conditional adapter
from `IFM/K2-Horizon-7B-Uno` by default:

```bash
RESULTS_ROOT=/path/to/results \
DATA_PARALLEL_SIZE=1 \
  bash scripts/k2_horizon/run_gsm8k_linear_b8.sh
```

K2 Horizon 0.9B uses the same shared runner with its own model, adapter, token
IDs, and context limit:

```bash
MODEL=IFM/K2-Horizon-0.9B \
GATED_LORA_PATH=IFM/K2-Horizon-0.9B-Uno \
MASK_TOKEN_ID=64256 \
STOP_TOKEN_IDS=64019,1 \
MAX_MODEL_LEN=131072 \
MAX_NUM_BATCHED_TOKENS=131072 \
MAX_TOKENS=131072 \
RESULTS_ROOT=/path/to/results \
DATA_PARALLEL_SIZE=1 \
  bash scripts/k2_horizon/run_gsm8k_linear_b8.sh
```

Set `DATA_PARALLEL_SIZE` to the number of independent model replicas. With
tensor parallelism disabled, each replica uses one GPU.

### Generate Samples with the Tree Sampler

After installing FA3, enable tree verification through the shared Qwen suite.
For example:

```bash
RESULTS_ROOT=/path/to/results \
ATTENTION_BACKEND=fa3 \
DIFFUSION_BLOCK_SIZE=16 \
TREE_VERIFY_SIZE=60 \
TREE_CANDIDATE_TOP_K=32 \
TORCH_COMPILE=0 \
SBATCH_ARGS="--account=my-account --partition=my-partition" \
  bash scripts/qwen/submit_pull_suite.sh
```

`DIFFUSION_BLOCK_SIZE` controls the draft block length,
`TREE_CANDIDATE_TOP_K` controls candidate expansion, and
`TREE_VERIFY_SIZE` controls the number of tree nodes verified together.

### Evaluate

Evaluation has two stages:

1. `nano_vllm_uno.eval.generate_benchmark` generates answers and records
   per-sequence decoding statistics.
2. `nano_vllm_uno.eval.score_generations` applies the benchmark-specific
   grader and writes the final scores.

The model-specific shell wrappers run both stages through the shared entry
point. To run one Qwen benchmark:

```bash
RESULTS_ROOT=/path/to/results \
  bash scripts/qwen/run_math500_eval.sh
```

To submit selected benchmarks as independent Slurm jobs:

```bash
RESULTS_ROOT=/path/to/results \
BENCHMARKS="gsm8k math500 aime24 aime25 aime26" \
SBATCH_ARGS="--account=my-account --partition=my-partition" \
  bash scripts/qwen/submit_benchmark_suite.sh
```

To run the persistent pull-based suite on one eight-GPU node:

```bash
RESULTS_ROOT=/path/to/results \
SBATCH_ARGS="--account=my-account --partition=my-partition" \
  bash scripts/qwen/submit_pull_suite.sh
```

Each completed benchmark directory contains:

```text
generations.jsonl          # Prompts, completions, token counts, and decode stats
generation_summary.json   # Aggregate generation, TPF, and throughput metrics
grades.jsonl               # Per-sample grading records
scores.json                # Aggregate benchmark scores
```

Useful overrides include `MODEL`, `MODEL_REVISION`, `GATED_LORA_PATH`,
`GATED_LORA_REVISION`, `RUN_NAME`, `RESULTS_ROOT`, `BENCHMARKS`, `LIMIT`,
`NUM_SAMPLES`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MAX_TOKENS`,
`DIFFUSION_BLOCK_SIZE`, and `MAX_NUM_SEQS`.

#### Prepare Evaluation Data

Evaluation launchers download missing datasets from pinned public sources. To
prepare the complete Qwen benchmark suite in advance:

```bash
export UNO_EVAL_DATA_DIR=/path/to/uno-eval-data
python scripts/qwen/prepare_benchmark_data.py
```

Accept the terms for the gated
[GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa) and run
`hf auth login` before preparing GPQA.

> [!WARNING]
> HumanEval, MBPP, and LiveCodeBench grading execute model-generated Python.
> Run coding evaluations only in an isolated, secure sandbox.

### Train

Training is LoRA-only. First prepare the pinned OpenThoughts3-1.2M corpus:

```bash
python -m nano_vllm_uno.training.prepare_openthoughts \
  --output /path/to/openthoughts-uno-4095 \
  --cache-dir /path/to/hf-cache \
  --num-proc 32
```

Then launch the default two-node, 16-GPU, three-epoch curriculum:

```bash
DATASET_PATH=/path/to/openthoughts-uno-4095 \
TRAIN_STORAGE_ROOT=/path/to/uno-training \
SLURM_ACCOUNT=my-account \
SLURM_PARTITION=my-partition \
  bash scripts/train_uno_3epoch_slurm.sh
```

The default curriculum spends half an epoch at each diffusion block size:
`2`, `4`, `6`, `8`, `12`, and `16`.

#### Default Training Configuration

| Setting | Value |
| --- | --- |
| Global batch size | `128` |
| Learning rate | `1e-5` |
| Warmup | `562` steps, or 2% of training |
| Learning-rate decay | Disabled |
| LoRA rank | `128` |
| LoRA alpha | `2048` |
| LoRA targets | Q, K, V, O, gate, up, and down projections |
| CE weight | `0` |
| Reverse-KL weight | `0` |
| TV weight | `1` |

W&B logging is enabled by default. Run `wandb login` before training or set
`WANDB_MODE=disabled`. To resume training, set `RESUME_FROM_CHECKPOINT` and
reuse the original `RUN_NAME`.

## Acknowledgements

Uno's runtime builds on
[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), and its training
utilities are adapted from
[LlamaFactory](https://github.com/hiyouga/LlamaFactory). See [NOTICE](NOTICE)
for attribution and license notices.

## Citation

```bibtex
@article{uno2026,
  title   = {},
  author  = {},
  journal = {},
  year    = {2026}
}
```
