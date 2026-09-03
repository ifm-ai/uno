# Qwen evaluation scripts

This directory contains the Qwen3 model defaults and model-specific benchmark
entry points. They delegate generation and scoring to
`scripts/common/run_benchmark.sh`. Override `MODEL`, `GATED_LORA_PATH`, and the
sampling environment variables to evaluate another compatible Qwen checkpoint.
