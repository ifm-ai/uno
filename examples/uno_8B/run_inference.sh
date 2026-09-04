#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python}" "${REPO_ROOT}/inference.py" \
  --model "${MODEL:-IFM/K2-Horizon-7B}" \
  --model-revision "${MODEL_REVISION:-586b03f0fd1fbbf2f13eeafc33749e95ae34dd10}" \
  --gated-lora-path "${GATED_LORA_PATH:-IFM/K2-Horizon-7B-Uno}" \
  --gated-lora-revision "${GATED_LORA_REVISION:-ec92bbd768f4a404319625204544782e3377bcd7}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-262144}" \
  --mask-token-id "${MASK_TOKEN_ID:-250624}" \
  --stop-token-ids "${STOP_TOKEN_IDS:-250019,1}" \
  --attention-backend "${ATTENTION_BACKEND:-fa3}" \
  "$@"
