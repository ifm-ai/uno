#!/usr/bin/env bash

# Public release artifacts. Explicit environment variables always take priority.
if [[ -z "${MODEL:-}" ]]; then
  MODEL="IFM/uno-qwen3-8b-base"
  MODEL_REVISION="${MODEL_REVISION:-4ccfeed3fba497e40495fe6dc5c15c89f7f1e2cd}"
fi
if [[ -z "${GATED_LORA_PATH:-}" ]]; then
  GATED_LORA_PATH="IFM/uno-qwen3-8b-lora-r128"
  GATED_LORA_REVISION="${GATED_LORA_REVISION:-fc17223b353f31be0eda939cc1e26423e05f54c0}"
fi

export MODEL MODEL_REVISION GATED_LORA_PATH GATED_LORA_REVISION
