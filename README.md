# Uno

Uno combines an autoregressive base model with a conditional LoRA diffusion
path and provides linear and tree speculative samplers.

## Repository layout

```text
nano_vllm_uno/       Inference engine
generation.py        Generation shared by free inference and evaluation
inference.py         Free-form prompt inference
evaluation/          Benchmark data, generation, grading, and Slurm launchers
training/            Qwen Uno LoRA training and training configuration
examples/            Model-specific, directly executable recipes
tests/               Engine and workflow tests
```

The engine is model-agnostic at its public boundary. Model-specific defaults
live only in the three example directories:

- `examples/uno_qwen3_8B`: `s-sahoo/uno-qwen3-8B`
- `examples/uno_8B`: `IFM/K2-Horizon-7B-Uno`
- `examples/uno_1B`: `IFM/K2-Horizon-0.9B-Uno`

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

Linear decoding uses FlashAttention-2. Tree verification additionally requires
FlashAttention-3 from `flash-attention/hopper`.

## Inference

Each model recipe calls the same top-level `inference.py` implementation:

```bash
bash examples/uno_qwen3_8B/run_inference.sh --prompt "Solve 2 + 2."
bash examples/uno_8B/run_inference.sh --prompt "Solve 2 + 2."
bash examples/uno_1B/run_inference.sh --prompt "Solve 2 + 2."
```

Sampling parameters can be overridden at the command line:

```bash
bash examples/uno_8B/run_inference.sh \
  --prompt "Solve 2 + 2." \
  --diffusion-block-size 16 \
  --tree-candidate-top-k 32 \
  --tree-verify-size 32 \
  --attention-backend fa3
```

## Evaluation

Prepare public benchmark data when desired; individual evaluation runs also
prepare missing data lazily:

```bash
python -m evaluation.prepare_data gsm8k math500
```

Run the same benchmark workflow for any supported release:

```bash
bash examples/uno_qwen3_8B/run_eval.sh gsm8k
bash examples/uno_8B/run_eval.sh gsm8k
bash examples/uno_1B/run_eval.sh gsm8k
```

Every run writes `generations.jsonl`, `generation_summary.json`, `grades.jsonl`,
and `scores.json`. The summary includes output tokens per second and decoder
tokens per sequence forward (TPF).

The built-in suite includes GSM8K, MATH500, AIME 2024--2026, HumanEval, MBPP,
LiveCodeBench v6, GPQA, GPQA-Diamond, MMLU-Pro, IFEval, and AA-LCR. AA-LCR uses
an OpenAI-compatible external judge and caller-supplied JSONL data via `--data`.
It requires:

```bash
export UNO_LCR_JUDGE_URL=http://judge-host:8000/v1
export UNO_LCR_JUDGE_MODEL=judge-model-name
export UNO_LCR_JUDGE_API_KEY=optional-token
```

Code benchmark graders execute model-generated Python. Run them only in an
isolated environment.

For Slurm, `MODEL_EXAMPLE` selects one of the three example directories:

```bash
MODEL_EXAMPLE=uno_8B BENCHMARKS="gsm8k math500" \
  bash evaluation/submit_suite.sh
```

## Training

The published training implementation is the Qwen Uno LoRA recipe. K2 releases
currently provide inference and evaluation only.

Prepare the pinned OpenThoughts corpus:

```bash
python -m training.prepare_openthoughts \
  --output /path/to/openthoughts-uno-4095 \
  --cache-dir /path/to/hf-cache \
  --num-proc 32
```

Launch the default curriculum on Slurm:

```bash
DATASET_PATH=/path/to/openthoughts-uno-4095 \
TRAIN_STORAGE_ROOT=/path/to/training-storage \
SLURM_ACCOUNT=my-account \
SLURM_PARTITION=my-partition \
  bash examples/uno_qwen3_8B/run_train.sh
```

Training configuration is colocated under `training/configs/`. The default
recipe uses LoRA rank 128, alpha 2048, global batch size 128, learning rate
`1e-5`, and a B2/B4/B6/B8/B12/B16 curriculum.

## Acknowledgements

The runtime builds on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).
Training utilities are adapted from
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). See `NOTICE` for
license and attribution details.
