# Uno 8B

Uno 8B augments the frozen autoregressive weights of
[K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) with the lightweight
conditional LoRA adapter released as
[K2-Horizon-7B-Uno](https://huggingface.co/IFM/K2-Horizon-7B-Uno). The base
model remains the verifier distribution, while the adapter drafts multiple
tokens in parallel for lossless speculative decoding.

See the [Uno paper](https://arxiv.org/abs/2609.04010) for the method, training
details, and full evaluation protocol.

## Files

```text
examples/uno_8B/
|-- run_inference.sh    # Free-form generation with the released checkpoint
|-- run_eval.sh         # Benchmark generation and grading
`-- README.md           # This guide
```

This release provides inference and evaluation for Uno 8B. A corresponding
training launcher is not included in this directory.

## Inference

Run free-form inference with the public base model and adapter:

```bash
bash examples/uno_8B/run_inference.sh \
  --prompt "Solve 2 + 2 and explain your reasoning."
```

The launcher downloads the public checkpoints through Hugging Face and uses
FA3 linear sampling by default. Public checkpoints do not require an
`HF_TOKEN`. To use local artifacts, override the two model sources:

```bash
MODEL=/path/to/K2-Horizon-7B \
GATED_LORA_PATH=/path/to/K2-Horizon-7B-Uno \
  bash examples/uno_8B/run_inference.sh \
    --prompt "Solve 2 + 2 and explain your reasoning."
```

Additional inference arguments, such as `--max-tokens`, `--temperature`,
`--top-p`, and `--diffusion-block-size`, are forwarded to the shared
`inference.py` entry point.

## Evaluation

Run one supported benchmark by name:

```bash
bash examples/uno_8B/run_eval.sh gsm8k
```

The launcher supplies the model-specific base, adapter, token IDs, and
attention backend. Dataset-specific sampling and scoring settings come from
`evaluation/benchmarks.py`.

Missing public evaluation data is prepared automatically. To keep prepared
data and results in custom locations, set:

```bash
UNO_EVAL_DATA_DIR=/path/to/uno-eval-data \
RESULTS_ROOT=/path/to/results \
  bash examples/uno_8B/run_eval.sh math500
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

The following results are reported in Table 1 of the
[Uno paper](https://arxiv.org/abs/2609.04010). Accuracy is average pass@1.
`TPF 1` uses the system-throughput-optimal Linear sampler with block size 4;
`TPF 2` uses the per-request-throughput-optimal Tree sampler with
`(B, K, V) = (16, 32, 32)`.

| Benchmark | Accuracy (%) | TPF 1 | TPF 2 |
| --- | ---: | ---: | ---: |
| tau3 Banking | 25.8 | 1.8 | 2.7 |
| tau2 Telecom | 90.1 | 1.7 | 2.1 |
| tau2 Retail | 67.1 | 1.8 | 2.4 |
| Terminal-Bench v2.1 | 39.6 | 2.1 | 2.7 |
| SWE-bench Verified | 68.4 | 2.2 | 3.1 |
| AA-LCR | 68.0 | 1.8 | 2.6 |
| Humanity's Last Exam | 18.6 | 1.8 | 2.8 |
| GPQA-Diamond | 77.1 | 2.0 | 4.2 |
| AA-Omniscience | 14.3 | 1.7 | 3.1 |
| GSM8K | 95.4 | 1.9 | 2.7 |
| MATH500 | 98.9 | 1.9 | 2.4 |
| AIME-24 | 93.0 | 1.9 | 2.5 |
| AIME-25 | 90.7 | 1.8 | 2.4 |
| AIME-26 | 86.3 | 1.8 | 2.4 |
| MBPP | 84.1 | 1.8 | 2.4 |
| HumanEval | 95.2 | 1.9 | 3.4 |
| **Average TPF** | -- | **1.9** | **2.7** |

On the paper's fixed 1K-input/8K-output throughput test using one H200 GPU,
Uno 8B reaches 5,255 tokens/s maximum system throughput and 405 tokens/s
maximum per-request throughput. These are paper-reported measurements; when
comparing a new run, keep the sampler, batch size, workload length, hardware,
and benchmark protocol fixed.

## Training

The paper trains the diffusion weights as rank-128 conditional LoRA adapters
while keeping the AR weights frozen. The public repository currently provides
the released Uno 8B adapter and its inference/evaluation workflows, but not a
model-specific `run_train.sh` recipe in this directory.

## Notes

- Linear sampling supports FA2, FA3, and FA4.
- Tree verification requires FA3 or FA4; FA2 tree verification is rejected.
- Model-judge benchmarks require the judge configuration described above.
- Reported accuracy and speed are comparable only when their evaluation and
  sampler protocols match.
