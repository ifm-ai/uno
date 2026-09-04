# Uno Qwen3 8B

Uno Qwen3 8B adds diffusion drafting weights to Qwen3-8B while preserving the
base autoregressive model as the verifier distribution. The public
[s-sahoo/uno-qwen3-8B](https://huggingface.co/s-sahoo/uno-qwen3-8B) repository
is a self-contained bundle: the base model is stored at the repository root
and the conditional LoRA adapter is stored under `adapter/`.

See the [Uno paper](https://arxiv.org/abs/2609.04010) for the method, training
recipe, and comparison with speculative and diffusion decoding baselines.

## Files

```text
examples/uno_qwen3_8B/
├── run_inference.sh    # Free-form generation with the released bundle
├── run_eval.sh         # Benchmark generation and grading
├── run_train.sh        # Published Uno Qwen3 8B training launcher
└── README.md           # This guide
```

## Inference

Run free-form inference directly from the public Hugging Face bundle:

```bash
bash examples/uno_qwen3_8B/run_inference.sh \
  --prompt "Solve 2 + 2 and explain your reasoning."
```

The launcher resolves the adapter from the bundle's `adapter/` subfolder and
uses linear sampling by default. Public checkpoints do not require an
`HF_TOKEN`. To use a local copy of the same bundle:

```bash
MODEL=/path/to/uno-qwen3-8B \
  bash examples/uno_qwen3_8B/run_inference.sh \
    --prompt "Solve 2 + 2 and explain your reasoning."
```

Additional inference arguments, such as `--max-tokens`, `--temperature`,
`--top-p`, and `--diffusion-block-size`, are forwarded to the shared
`inference.py` entry point.

## Evaluation

Run one supported benchmark by name:

```bash
bash examples/uno_qwen3_8B/run_eval.sh gsm8k
```

The launcher supplies the Qwen-specific model bundle and inference backend.
Dataset-specific sampling and scoring settings come from
`evaluation/benchmarks.py`.

Missing public evaluation data is prepared automatically. To reuse a custom
data directory and choose a result root:

```bash
UNO_EVAL_DATA_DIR=/path/to/uno-eval-data \
RESULTS_ROOT=/path/to/results \
  bash examples/uno_qwen3_8B/run_eval.sh math500
```

Each completed run writes:

```text
generations.jsonl
generation_summary.json
grades.jsonl
scores.json
```

Benchmarks that use an external model judge additionally require
`JUDGE_MODEL`; `JUDGE_BASE_URL` and `JUDGE_API_KEY` can be supplied for an
OpenAI-compatible endpoint. To generate without grading, set
`SKIP_GRADING=1`.

## Evaluation Results

The following Uno Qwen3 8B results are reported in Table 3 of the
[Uno paper](https://arxiv.org/abs/2609.04010). The reported TPF uses greedy
decoding and the Tree sampler with `(B, K, V) = (16, 32, 60)`. These numbers
therefore describe that paper arm rather than the default linear command shown
above.

| Benchmark | Accuracy (%) | TPF |
| --- | ---: | ---: |
| GSM8K | 96.1 | 3.56 |
| MATH500 | 96.4 | 3.92 |
| AIME-24 | 76.7 | 4.01 |
| AIME-25 | 76.7 | 3.96 |
| AIME-26 | 73.3 | 4.05 |
| HumanEval | 94.8 | 3.67 |
| MBPP | 89.0 | 4.21 |
| LiveCodeBench v6 | 51.4 | 3.75 |
| GPQA | 58.0 | 4.10 |
| GPQA-Diamond | 60.9 | 4.01 |
| MMLU-Pro | 74.8 | 3.61 |
| IFEval | 86.5 | 2.49 |

For the paper's lossless throughput comparison at temperature 1, Uno Qwen3
8B reaches 5,733 tokens/s at the system-throughput-optimal Linear `B=4`
setting and 445 tokens/s at the per-request-optimal Tree `B=16, V=32`
setting on the fixed 1K-input/8K-output workload using one H200 GPU. Accuracy,
TPF, and throughput rows use the sampler settings stated in their respective
paper tables and should not be mixed without preserving those settings.

## Training

Prepare OpenThoughts once using the shared training workflow:

```bash
python -m training.prepare_openthoughts \
  --output /path/to/openthoughts-uno-4095 \
  --cache-dir /path/to/hf-cache \
  --num-proc 32
```

Then launch the published Slurm recipe:

```bash
DATASET_PATH=/path/to/openthoughts-uno-4095 \
TRAIN_STORAGE_ROOT=/path/to/shared-storage \
  bash examples/uno_qwen3_8B/run_train.sh
```

The launcher uses `training/configs/uno_3epoch_curriculum.yaml` by default.
Cluster-specific Slurm settings can be supplied through `SLURM_PARTITION`,
`SLURM_ACCOUNT`, `SLURM_QOS`, and `SLURM_CONSTRAINT`. See
`training/run_slurm.sh` for the complete set of supported environment
variables.

## Notes

- The released Hugging Face repository contains both the base weights and the
  `adapter/` subfolder; a separate adapter path is not required.
- Linear sampling can use FA2. Tree verification requires FA3 or FA4.
- Model-judge benchmarks require the judge configuration described above.
- Always record the sampler and decoding protocol alongside reported TPF/TPS.
