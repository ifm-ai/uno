#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python}" "${REPO_ROOT}/inference.py" \
  --model "${MODEL:-IFM/K2-Horizon-0.9B}" \
  --model-revision "${MODEL_REVISION:-ee770e713760cf6350e4322cdbbff91a163b7d70}" \
  --gated-lora-path "${GATED_LORA_PATH:-IFM/K2-Horizon-0.9B-Uno}" \
  --gated-lora-revision "${GATED_LORA_REVISION:-b0d8896a301a2f4bc755538b1234a35100da50d0}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-131072}" \
  --mask-token-id "${MASK_TOKEN_ID:-64256}" \
  --stop-token-ids "${STOP_TOKEN_IDS:-64019,1}" \
  --attention-backend "${ATTENTION_BACKEND:-fa3}" \
  "$@"
