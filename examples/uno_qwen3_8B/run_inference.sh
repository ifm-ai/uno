#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python}" "${REPO_ROOT}/inference.py" \
  --model "${MODEL:-s-sahoo/uno-qwen3-8B}" \
  --model-revision "${MODEL_REVISION:-b8a7577b3223bdcf2b3af0f2fc6e95258b3bbc29}" \
  --gated-lora-subfolder "${GATED_LORA_SUBFOLDER:-adapter}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --attention-backend "${ATTENTION_BACKEND:-fa2}" \
  "$@"
