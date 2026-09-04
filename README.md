<div align="center">
  <h1><a href="https://s-sahoo.com/uno/">Unlocking Lossless Speedups in LLMs via Discrete Diffusion</a></h1>
  <p>
    <a href="https://s-sahoo.github.io">Subham Sekhar Sahoo</a>,
    <a href="https://lingjiechen2.github.io">Lingjie Chen</a>,
    <a href="https://drproduck.github.io">Khiem Pham</a>,
    <a href="https://j-geuter.github.io">Jonathan Geuter</a>,
    <a href="https://www.linkedin.com/in/junlin110/">Junlin Chen</a>,<br>
    <a href="https://scholar.google.com/citations?user=ghpn6JkAAAAJ&amp;hl=en">Chaitanya Dwivedi</a>,
    <a href="https://nightlessbaron.github.io">Varad Pimpalkhute</a>,
    <a href="https://akhauriyash.github.io">Yash Akhauri</a>,
    <a href="https://www.linkedin.com/in/alexander-moreno-ab151542/">Alexander Moreno</a>,
    <a href="https://moonfolk.github.io">Mikhail Yurochkin</a>,<br>
    <a href="https://zhentingwang.github.io">Zhenting Wang</a>,
    <a href="https://scholar.google.com/citations?user=y_cwSKAAAAAJ&amp;hl=en">Mostafa Elhoushi</a>,
    <a href="https://scholar.google.com/citations?user=JHUfMr0AAAAJ&amp;hl=en">Nolan Dey</a>,
    <a href="https://sites.google.com/site/shaneabergsma/">Shane Bergsma</a>,
    <a href="https://scholar.google.com/citations?user=wkbvCf0AAAAJ&amp;hl=en">Joel Hestness</a>,<br>
    <a href="https://hwang595.github.io">Hongyi Wang</a>,
    <a href="https://johnthickstun.com">John Thickstun</a>,
    <a href="https://mbzuai.ac.ae/study/faculty/professor-eric-xing/">Eric Xing</a>,
    <a href="https://hunterhector.github.io">Zhengzhong Liu</a>
  </p>
  <p>
    <a href="https://arxiv.org/abs/2609.04010"><img src="https://img.shields.io/badge/arXiv-2609.04010-B31B1B.svg" alt="arXiv"></a>
    <a href="https://s-sahoo.com/uno/"><img src="https://img.shields.io/badge/Project-Page-4B5563.svg" alt="Project Page"></a>
    <a href="https://huggingface.co/IFM/K2-Horizon-7B-Uno"><img src="https://img.shields.io/badge/Hugging%20Face-Uno%208B-FFD21E?logo=huggingface&amp;logoColor=000" alt="Uno 8B"></a>
    <a href="https://huggingface.co/IFM/K2-Horizon-0.9B-Uno"><img src="https://img.shields.io/badge/Hugging%20Face-Uno%201B-FFD21E?logo=huggingface&amp;logoColor=000" alt="Uno 1B"></a>
    <a href="https://huggingface.co/s-sahoo/uno-qwen3-8B"><img src="https://img.shields.io/badge/Hugging%20Face-Uno%20Qwen3%208B-FFD21E?logo=huggingface&amp;logoColor=000" alt="Uno Qwen3 8B"></a>
  </p>
  <a href="https://s-sahoo.com/uno/"><img src="uno_results.png" alt="Uno training pipeline and evaluation results"></a>
</div>

<br>

**Uno** is a diffusion-augmented language model that preserves the distribution
and quality of an autoregressive (AR) model while using diffusion to generate
multiple tokens in parallel. Its parameters are decoupled into standard AR
weights, trained with next-token prediction, and lightweight diffusion weights,
learned through a low-overhead Diffusion Distillation stage and implemented
here as a conditional LoRA adapter. This design can augment existing
open-weight AR models and requires no separate draft model. At inference time,
the $\Psi$-Spec family of samplers provides lossless acceleration: linear
sampling targets high system throughput, while tree sampling targets high
per-request throughput.

In this repo, we release:

- **Inference**
  - A Nano-vLLM-based inference engine shared by all Uno models.
  - Linear sampling for high system throughput.
  - Tree sampling for high per-request throughput.
  - Ready-to-run recipes for
    [Uno Qwen3 8B](https://huggingface.co/s-sahoo/uno-qwen3-8B),
    [Uno 8B](https://huggingface.co/IFM/K2-Horizon-7B-Uno), and
    [Uno 1B](https://huggingface.co/IFM/K2-Horizon-0.9B-Uno).
- **Training**
  - A conditional-LoRA diffusion training pipeline for Uno Qwen3 8B.
  - OpenThoughts data preparation and progressive block-size curricula.
  - Multi-node Slurm launchers with checkpoint resume support.
- **Evaluation**
  - One shared generation and scoring pipeline across model families.
  - Benchmark-specific data loaders and graders.
  - Accuracy, tokens per forward (TPF), and tokens per second (TPS) metrics.
  - Single-benchmark, full-suite, and persistent pull-based launchers.

## Table of Contents

- [Code Organization](#code-organization)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Checkpoints](#checkpoints)
- [Reproducing Experiments](#reproducing-experiments)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Code Organization

The repository is organized around the workflows users run:

- [`generation.py`](generation.py) contains the shared model loading,
  conditional-LoRA attachment, prompt formatting, generation, and TPF/TPS
  accounting used by both inference and evaluation.
- [`inference.py`](inference.py) is the command-line entry point for generating
  from free-form prompts with linear or tree sampling.
- [`evaluation/`](evaluation) defines the benchmark protocols, dataset loaders,
  generation pipeline, parsers, and benchmark-specific graders.
- [`training/`](training) contains data preparation, conditional-LoRA training,
  diffusion objectives, curriculum configuration, and checkpointing.
- [`examples/`](examples) provides directly executable training, inference, and
  evaluation recipes for Uno Qwen3 8B, Uno 8B, and Uno 1B. Each model directory
  supplies only its model-specific defaults and calls the shared workflows.

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

| Model | Base model | Uno weights |
| --- | --- | --- |
| Uno 8B | [IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) | [IFM/K2-Horizon-7B-Uno](https://huggingface.co/IFM/K2-Horizon-7B-Uno) |
| Uno 1B | [IFM/K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) | [IFM/K2-Horizon-0.9B-Uno](https://huggingface.co/IFM/K2-Horizon-0.9B-Uno) |
| Uno Qwen3 8B | [s-sahoo/uno-qwen3-8B](https://huggingface.co/s-sahoo/uno-qwen3-8B) | [`adapter/`](https://huggingface.co/s-sahoo/uno-qwen3-8B/tree/main/adapter) |

Public checkpoints can be downloaded without a Hugging Face token. A token is
still required for gated datasets such as GPQA and for any private or gated
model repository.

## Reproducing Experiments

All public model recipes live under `examples/`. Each recipe supplies the
model-specific defaults and delegates to the same shared training, inference,
or evaluation workflow.

### Training

To train Uno Qwen3 8B, prepare the training data and run the shared training
entry point as follows.

#### 1. Prepare the Training Data

Prepare the pinned OpenThoughts3-1.2M corpus once in shared storage:

```bash
python -m training.prepare_openthoughts \
  --output /path/to/openthoughts-uno-4095 \
  --cache-dir /path/to/hf-cache \
  --num-proc 32
```

#### 2. Train Uno Qwen3 8B

The following is a single-GPU command. Gradient accumulation preserves the
released global batch size of 128:

```bash
python -m training.train \
  --dataset-path /path/to/openthoughts-uno-4095 \
  --output-dir /path/to/uno-training \
  --curriculum training/configs/uno_3epoch_curriculum.yaml \
  --deepspeed training/configs/deepspeed_zero2.json \
  --per-device-batch-size 8 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-5 \
  --warmup-steps 562 \
  --lora-target all \
  --lora-rank 128 \
  --lora-alpha 2048 \
  --ce-alpha 0 \
  --kl-beta 0 \
  --tv-gamma 1
```

See [`training/train.py`](training/train.py) and
[`training/configs/uno_3epoch_curriculum.yaml`](training/configs/uno_3epoch_curriculum.yaml)
for the complete training configuration. Resume an interrupted run with
`--resume-from-checkpoint /path/to/checkpoint` and the same `--output-dir`.

### Inference

Use the model recipe that matches the checkpoint. All three recipes call the
same shared `inference.py` workflow:

```bash
# Uno Qwen3 8B
bash examples/uno_qwen3_8B/run_inference.sh \
  --prompt "Solve 2 + 2 and explain your reasoning."

# Uno 8B
bash examples/uno_8B/run_inference.sh \
  --prompt "Solve 2 + 2 and explain your reasoning."

# Uno 1B
bash examples/uno_1B/run_inference.sh \
  --prompt "Solve 2 + 2 and explain your reasoning."
```

The example launchers use linear sampling by default. To enable tree sampling,
install FA3 and pass the tree parameters to the same entry point:

```bash
ATTENTION_BACKEND=fa3 \
  bash examples/uno_qwen3_8B/run_inference.sh \
    --prompt "Solve 2 + 2 and explain your reasoning." \
    --diffusion-block-size 16 \
    --tree-candidate-top-k 32 \
    --tree-verify-size 60
```

The command prints generated text together with output-token count, elapsed
time, TPS, decoder statistics, and TPF. Common controls include
`--temperature`, `--top-p`, `--top-k`, `--max-tokens`,
`--diffusion-block-size`, and `--max-num-seqs`.

### Evaluation

The evaluation workflow uses canonical benchmark settings from
`evaluation/benchmarks.py`. Generation and grading are separate internally,
while each model recipe runs both stages through one command. See the
[Uno project page](https://s-sahoo.com/uno/) for the reported evaluation
results.

#### 1. Prepare Evaluation Data

Missing public datasets are prepared automatically. They can also be prepared
in advance:

```bash
export UNO_EVAL_DATA_DIR=/path/to/uno-eval-data

python -m evaluation.prepare_data \
  --output-dir "${UNO_EVAL_DATA_DIR}"
```

Keep `UNO_EVAL_DATA_DIR` set when running `run_eval.sh`. The evaluation runner
will reuse the prepared JSONL files from this directory and will prepare any
missing benchmark there automatically.

Accept the terms for the gated
[GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa) and run
`hf auth login` before preparing GPQA.

> [!WARNING]
> HumanEval and MBPP grading execute model-generated Python. Run coding
> evaluations only in an isolated, secure sandbox.

#### 2. Evaluate One Benchmark

Set the shared output and parallelism options, then choose the model recipe:

```bash
export RESULTS_ROOT=/path/to/results
export DATA_PARALLEL_SIZE=1

# Uno Qwen3 8B
bash examples/uno_qwen3_8B/run_eval.sh gsm8k

# Uno 8B
bash examples/uno_8B/run_eval.sh gsm8k

# Uno 1B
bash examples/uno_1B/run_eval.sh gsm8k
```

The available benchmark names are `aime24`, `aime25`, `aime26`,
`arc_challenge`, `gpqa_diamond`, `gsm8k`, `hle`, `humaneval`,
`ifeval`, `lcr`, `math500`, `mbpp`, and `omniscience`.

#### 3. Evaluate the Full Suite

Submit one independent Slurm job per benchmark:

```bash
MODEL_EXAMPLE=uno_qwen3_8B \
RESULTS_ROOT=/path/to/results \
SBATCH_ARGS="--account=my-account --partition=my-partition" \
  bash evaluation/submit_suite.sh
```

Set `MODEL_EXAMPLE` to `uno_qwen3_8B`, `uno_8B`, or `uno_1B`. A
persistent pull-based launcher is also available at
`evaluation/submit_pull_suite.sh`.

Canonical sampling, context, output-budget, and prompt settings are resolved by
the benchmark registry. Deliberate overrides must include
`--protocol-arm <label>`, which prevents an experimental arm from being
mistaken for a protocol-comparable result. HLE, AA-LCR, and AA-Omniscience
additionally require an external judge configured through `JUDGE_MODEL` and,
when needed, `JUDGE_BASE_URL` and `JUDGE_API_KEY`.

Each completed benchmark directory contains:

```text
generations.jsonl          # Prompts, completions, token counts, and decode stats
generation_summary.json   # Aggregate generation, TPF, and TPS metrics
grades.jsonl               # Per-sample grading records
scores.json                # Aggregate benchmark scores
```

## Acknowledgements

Uno's runtime builds on
[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), and its training
utilities are adapted from
[LlamaFactory](https://github.com/hiyouga/LlamaFactory). See [NOTICE](NOTICE)
for attribution and license notices.

## Citation

```bibtex
@misc{sahoo2026unlockinglosslessspeedupsllms,
      title={Unlocking Lossless Speedups in LLMs via Discrete Diffusion},
      author={Subham Sekhar Sahoo and Lingjie Chen and Khiem Pham and Jonathan Geuter and Chaitanya Dwivedi and Varad Pimpalkhute and Yash Akhauri and Alexander Moreno and Mikhail Yurochkin and Zhenting Wang and Mostafa Elhoushi and Nolan Dey and Shane Bergsma and Joel Hestness and John Thickstun and Eric Xing and Zhengzhong Liu},
      year={2026},
      eprint={2609.04010},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.04010},
}
```
