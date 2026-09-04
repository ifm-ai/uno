"""Shared prompt formatting and generation for inference and evaluation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from nano_vllm_uno import LLM, SamplingParams
from nano_vllm_uno.utils.hub import resolve_hf_snapshot


def resolve_model_sources(
    model: str,
    gated_lora_path: str | None,
    *,
    model_revision: str | None = None,
    gated_lora_subfolder: str | None = None,
    hf_cache_dir: str | None = None,
    hf_local_files_only: bool = False,
) -> tuple[str, str | None]:
    """Resolve a model bundle when its adapter lives in a subfolder."""

    if gated_lora_subfolder is None:
        return model, gated_lora_path
    if gated_lora_path is not None:
        raise ValueError(
            "Use either --gated-lora-path or --gated-lora-subfolder, not both"
        )
    model_dir = resolve_hf_snapshot(
        model,
        revision=model_revision,
        cache_dir=hf_cache_dir,
        local_files_only=hf_local_files_only,
        artifact_name="model bundle",
    )
    adapter_dir = Path(model_dir) / gated_lora_subfolder
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(f"Adapter subfolder is invalid: {adapter_dir}")
    return model_dir, str(adapter_dir)


def format_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    instruction: str = "",
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], str, list[dict[str, Any]]]:
    """Render messages once as text and once as token IDs."""

    messages = [dict(message) for message in messages]
    instruction = instruction.strip()
    system_message = next(
        (message for message in messages if message["role"] == "system"),
        None,
    )
    if instruction:
        if system_message is None:
            messages.insert(0, {"role": "system", "content": instruction})
        elif instruction not in system_message["content"]:
            existing = system_message["content"].strip()
            system_message["content"] = (
                f"{instruction}\n\n{existing}" if existing else instruction
            )

    if tokenizer.chat_template is None:
        raise ValueError("The tokenizer needs a chat_template to render messages")
    template_kwargs = dict(chat_template_kwargs or {})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **template_kwargs,
    )
    return token_ids, rendered, messages


def generate(
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    *,
    model: str,
    tokenizer_path: str | None = None,
    request_max_tokens: list[int] | None = None,
    use_tqdm: bool = True,
    **engine_kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate tokenized prompts and return outputs plus throughput metrics."""

    llm = LLM(model=model, tokenizer_path=tokenizer_path, **engine_kwargs)
    start = perf_counter()
    try:
        outputs = llm.generate(
            prompts,
            sampling_params,
            request_max_tokens=request_max_tokens,
            use_tqdm=use_tqdm,
        )
        elapsed = perf_counter() - start
        stats = dict(getattr(llm, "last_generate_stats", {}))
    finally:
        llm.exit()

    total_output_tokens = sum(len(output["token_ids"]) for output in outputs)
    forwards = int(stats.get("forwards", 0))
    metrics = {
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": (
            total_output_tokens / elapsed if elapsed else 0.0
        ),
        "decoder_stats": stats,
        "decoder_tokens_per_sequence_forward": (
            int(stats.get("accepts", 0)) / forwards if forwards else 0.0
        ),
    }
    return outputs, metrics
