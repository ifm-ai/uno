# Uno 1B

Uno 1B augments the frozen autoregressive weights of
[K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) with the
conditional LoRA adapter released as
[K2-Horizon-0.9B-Uno](https://huggingface.co/IFM/K2-Horizon-0.9B-Uno). The
example name follows the rounded model-size name used by the project; the
released base checkpoint is named `0.9B`.

See the [Uno paper](https://arxiv.org/abs/2609.04010) for the method and the
evaluation methodology used for the released Uno family.

## Files

```text
examples/uno_1B/
├── run_inference.sh    # Free-form generation with the released checkpoint
├── run_eval.sh         # Benchmark generation and grading
└── README.md           # This guide
```

This release provides inference and evaluation for Uno 1B. A corresponding
training launcher is not included in this directory.

## Inference

Run free-form inference with the public base model and adapter:

```bash
bash examples/uno_1B/run_inference.sh \
  --prompt "Solve 2 + 2 and explain your reasoning."
```

The launcher uses the model's 131,072-token YaRN context configuration and K2
token IDs. Public checkpoints do not require an `HF_TOKEN`. The inference
launcher accepts `MODEL` and `GATED_LORA_PATH` overrides for local artifacts.

Additional inference arguments, such as `--max-tokens`, `--temperature`,
`--top-p`, and `--diffusion-block-size`, are forwarded to the shared
`inference.py` entry point.

## Evaluation

Run one supported benchmark by name:

```bash
bash examples/uno_1B/run_eval.sh gsm8k
```

The evaluation launcher pins the released base and adapter revisions, FA3,
diffusion block size 8, and the K2 token IDs. Dataset-specific sampling and
scoring settings come from `evaluation/benchmarks.py`.

Missing public evaluation data is prepared automatically. To keep prepared
data and results in custom locations, set:

```bash
UNO_EVAL_DATA_DIR=/path/to/uno-eval-data \
RESULTS_ROOT=/path/to/results \
  bash examples/uno_1B/run_eval.sh math500
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

Evaluation results for Uno 1B will be available soon.

## Training

Training code for Uno 1B will be released soon.

## Notes

- Linear sampling supports FA2, FA3, and FA4.
- Tree verification requires FA3 or FA4; FA2 tree verification is rejected.
- Model-judge benchmarks require the judge configuration described above.
- Reported accuracy and speed are comparable only when their evaluation and
  sampler protocols match.
