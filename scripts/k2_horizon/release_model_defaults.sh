#!/usr/bin/env bash

# Public K2 Horizon artifacts. Explicit environment variables take priority.
if [[ -z "${MODEL:-}" ]]; then
  MODEL="IFM/K2-Horizon-7B"
  MODEL_REVISION="${MODEL_REVISION:-cf704725b7424dc655db05fd1650d1dae6803972}"
fi
if [[ -z "${GATED_LORA_PATH:-}" ]]; then
  GATED_LORA_PATH="IFM/K2-Horizon-7B-Uno"
  GATED_LORA_REVISION="${GATED_LORA_REVISION:-0919053130c65a69612b12713ba4df8087845f76}"
fi

export MODEL MODEL_REVISION GATED_LORA_PATH GATED_LORA_REVISION
