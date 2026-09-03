#!/usr/bin/env python3
"""Run free-form Uno inference with the shared nano-vllm-uno engine."""

from __future__ import annotations

import argparse
import json

from generation import format_chat_prompt, generate, resolve_model_sources
from nano_vllm_uno import SamplingParams
from nano_vllm_uno.utils.hf_compat import load_tokenizer
from nano_vllm_uno.utils.model_tokens import resolve_model_token_ids


def _int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-path")
    adapter = parser.add_mutually_exclusive_group()
    adapter.add_argument("--gated-lora-path")
    adapter.add_argument("--gated-lora-subfolder")
    parser.add_argument("--gated-lora-revision")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--diffusion-block-size", type=int, default=8)
    parser.add_argument("--tree-verify-size", type=int)
    parser.add_argument("--tree-candidate-top-k", type=int, default=32)
    parser.add_argument("--attention-backend", choices=("fa2", "fa3", "fa4"), default="fa2")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--stop-token-ids", type=_int_list)
    parser.add_argument(
        "--noise-mode",
        choices=("random_uniform", "deterministic_uniform", "mask"),
        default="random_uniform",
    )
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    requested_model = args.model
    model, adapter = resolve_model_sources(
        args.model,
        args.gated_lora_path,
        model_revision=args.model_revision,
        gated_lora_subfolder=args.gated_lora_subfolder,
        hf_cache_dir=args.hf_cache_dir,
        hf_local_files_only=args.hf_local_files_only,
    )
    bundled = args.gated_lora_subfolder is not None
    model_revision = None if bundled else args.model_revision
    adapter_revision = None if bundled else args.gated_lora_revision
    tokenizer_path = args.tokenizer_path or model
    tokenizer = load_tokenizer(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
        revision=model_revision,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.hf_local_files_only,
    )
    mask_token_id, stop_token_ids, _ = resolve_model_token_ids(
        model,
        tokenizer,
        mask_token_id=args.mask_token_id,
        stop_token_ids=args.stop_token_ids,
        revision=model_revision,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.hf_local_files_only,
        noise_mode=args.noise_mode,
    )
    prompts = [
        format_chat_prompt(
            tokenizer,
            [{"role": "user", "content": prompt}],
            args.system_prompt,
        )[0]
        for prompt in args.prompt
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids,
        mask_token_id=mask_token_id,
        noise_mode=args.noise_mode,
        diffusion_block_size=args.diffusion_block_size,
    )
    outputs, metrics = generate(
        prompts,
        sampling_params,
        model=model,
        tokenizer_path=tokenizer_path,
        request_max_tokens=[args.max_tokens] * len(prompts),
        use_tqdm=not args.no_progress,
        model_revision=model_revision,
        hf_cache_dir=args.hf_cache_dir,
        hf_local_files_only=args.hf_local_files_only,
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=(
            args.max_num_batched_tokens or args.max_model_len
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        max_diffusion_block_size=args.diffusion_block_size,
        cuda_graph_block_sizes=sorted({1, args.diffusion_block_size}),
        tree_verify_size=args.tree_verify_size,
        tree_candidate_top_k=args.tree_candidate_top_k,
        gated_lora_path=adapter,
        gated_lora_revision=adapter_revision,
    )
    result = {
        "model": requested_model,
        "gated_lora_path": adapter,
        "generations": [output["text"] for output in outputs],
        **metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
