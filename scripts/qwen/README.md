# Qwen evaluation scripts

This directory contains the Qwen3 model defaults and model-specific benchmark
entry points. They delegate generation and scoring to
`scripts/common/run_benchmark.sh`. Override `MODEL`, `GATED_LORA_PATH`, and the
sampling environment variables to evaluate another compatible Qwen checkpoint.

To use a revision-pinned Hugging Face bundle with base weights at the repository
root and gated LoRA files under `adapter/`, set the bundle once at the Qwen
entrypoint:

```bash
UNO_BUNDLE_REPO=s-sahoo/uno-qwen3-8B \
UNO_BUNDLE_REVISION=3666d661986f1b13d75231a9b5f85831ccfc45d4 \
  bash scripts/qwen/run_gsm8k_eval.sh
```

The entrypoint resolves the snapshot through the standard Hugging Face cache,
then uses the snapshot root as `MODEL` and `<snapshot>/adapter` as
`GATED_LORA_PATH`. All Qwen benchmark wrappers inherit this behavior. Private or
gated repositories use credentials from `hf auth login`; tokens must not be
embedded in launch scripts.
