#!/usr/bin/env python3
"""Generate responses for a benchmark with nano-vllm-uno data parallelism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from transformers import AutoTokenizer

from nano_vllm_uno import LLM, SamplingParams
from nano_vllm_uno.engine.sequence import DECODE_STAT_KEYS
from nano_vllm_uno.eval.benchmarks import get_benchmark, list_benchmarks
from nano_vllm_uno.eval.context_budget import (
    DEFAULT_CONTEXT_LENGTH,
    active_forward_reserve,
    resolve_completion_budget,
)
from nano_vllm_uno.eval.model_tokens import resolve_model_token_ids


ID_FIELDS = ("id", "problem_id", "index", "row")
ANSWER_FIELDS = ("ground_truth", "answer")


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def first_field(record: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return None


def format_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    instruction: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], str, list[dict[str, Any]]]:
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
        raise ValueError(
            "The tokenizer needs a chat_template to render chat_input"
        )
    template_kwargs = dict(chat_template_kwargs or {})
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    prompt_token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **template_kwargs,
    )
    return prompt_token_ids, rendered_prompt, messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        help=(
            "Use canonical suite settings for a benchmark. Available: "
            + ", ".join(list_benchmarks())
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Generation summary JSON; defaults next to --output.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--num-samples",
        "--repetitions",
        dest="num_samples",
        type=int,
        default=None,
        help="Generations per problem; --repetitions is a compatibility alias.",
    )
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--attention-backend",
        choices=("fa2", "fa3", "fa4"),
        default="fa3",
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--stop-token-ids", type=parse_int_list)
    parser.add_argument("--diffusion-block-size", type=int, default=16)
    parser.add_argument("--tree-verify-size", type=int)
    parser.add_argument("--tree-candidate-top-k", type=int, default=16)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument(
        "--noise-mode",
        choices=("random_uniform", "deterministic_uniform", "mask"),
        default="random_uniform",
    )
    parser.add_argument("--noise-salt", type=int)
    parser.add_argument("--gated-lora-path")
    parser.add_argument("--gated-lora-revision")
    parser.add_argument(
        "--cuda-graph-block-sizes",
        type=parse_int_list,
        default=None,
        help="Comma-separated forward lengths; defaults to the active block size.",
    )
    parser.add_argument(
        "--cuda-graph-batch-sizes",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated CUDA graph batch sizes.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="System instruction added to chat_input; pass an empty string to disable.",
    )
    parser.add_argument(
        "--chat-template-kwargs-json",
        default=None,
        help=(
            "JSON object forwarded to tokenizer.apply_chat_template. Values "
            "override canonical benchmark defaults."
        ),
    )
    parser.add_argument(
        "--skip-row-count-check",
        action="store_true",
        help=(
            "Allow a full, non-limited dataset whose row count differs from "
            "the canonical suite."
        ),
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = get_benchmark(args.benchmark) if args.benchmark else None
    if args.data is None:
        if benchmark is None:
            raise ValueError("--data is required when --benchmark is not set")
        from nano_vllm_uno.eval.data import prepare_benchmark_data

        args.data = prepare_benchmark_data(benchmark.name)
    if args.output is None:
        output_name = (
            f"{benchmark.name}-generations.jsonl"
            if benchmark is not None
            else "aime25-generations.jsonl"
        )
        args.output = Path(output_name)
    if args.summary_output is None:
        args.summary_output = args.output.with_name(
            f"{args.output.stem.removesuffix('-generations')}-generation-summary.json"
        )
    if args.num_samples is None:
        args.num_samples = 1
    if args.instruction is None:
        args.instruction = (
            benchmark.instruction
            if benchmark is not None
            else "Please reason step by step and put your final answer in \\boxed{}."
        )
    chat_template_kwargs = (
        dict(benchmark.chat_template_kwargs) if benchmark is not None else {}
    )
    if args.chat_template_kwargs_json is not None:
        overrides = json.loads(args.chat_template_kwargs_json)
        if not isinstance(overrides, dict):
            raise ValueError("--chat-template-kwargs-json must be a JSON object")
        chat_template_kwargs.update(overrides)

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    records = load_records(args.data, args.limit)
    if (
        benchmark is not None
        and args.limit is None
        and not args.skip_row_count_check
        and len(records) != benchmark.expected_rows
    ):
        raise ValueError(
            f"{benchmark.name} expects {benchmark.expected_rows} rows, "
            f"but {args.data} contains {len(records)}. Pass the canonical "
            "dataset or use --skip-row-count-check intentionally."
        )
    tokenizer_path = args.tokenizer_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
        revision=args.model_revision,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.hf_local_files_only,
    )
    resolved_mask_id, resolved_stop_ids, vocab_size = resolve_model_token_ids(
        args.model,
        tokenizer,
        mask_token_id=args.mask_token_id,
        stop_token_ids=args.stop_token_ids,
        revision=args.model_revision,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.hf_local_files_only,
    )

    prompts: list[list[int]] = []
    problems: list[str] = []
    chat_inputs: list[list[dict[str, Any]]] = []
    completion_inputs: list[Any] = []
    record_ids: list[Any] = []
    ground_truths: list[Any] = []
    for index, record in enumerate(records):
        messages = record.get("chat_input")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Record {index} needs a nonempty chat_input")
        prompt_token_ids, rendered_prompt, effective_messages = format_prompt(
            tokenizer,
            messages,
            args.instruction,
            chat_template_kwargs,
        )
        if not rendered_prompt.strip():
            raise ValueError(f"Record {index} rendered prompt is empty")
        problems.append(rendered_prompt)
        chat_inputs.append(effective_messages)
        completion_inputs.append(record.get("completion_input"))
        record_id = first_field(record, ID_FIELDS)
        if record_id is None:
            record_id = index
        record_ids.append(record_id)
        ground_truths.append(first_field(record, ANSWER_FIELDS))
        prompts.append(prompt_token_ids)

    # Question-major ordering: q0 sample0..N-1, q1 sample0..N-1, ...
    prompts = [
        prompt
        for prompt in prompts
        for _ in range(args.num_samples)
    ]
    forward_reserve = active_forward_reserve(
        args.diffusion_block_size,
        args.tree_verify_size,
    )
    prompt_token_counts = [len(prompt) for prompt in prompts]
    request_max_tokens = [
        resolve_completion_budget(
            prompt_tokens=prompt_token_count,
            context_length=args.max_model_len,
            reserve_tokens=forward_reserve,
            global_max_tokens=args.max_tokens,
        )
        for prompt_token_count in prompt_token_counts
    ]
    valid_indices = [
        index
        for index, max_tokens in enumerate(request_max_tokens)
        if max_tokens > 0
    ]

    graph_block_sizes = args.cuda_graph_block_sizes or [args.diffusion_block_size]
    llm = LLM(
        model=args.model,
        model_revision=args.model_revision,
        hf_cache_dir=args.hf_cache_dir,
        hf_local_files_only=args.hf_local_files_only,
        tokenizer_path=tokenizer_path,
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        max_diffusion_block_size=args.diffusion_block_size,
        tree_verify_size=args.tree_verify_size,
        tree_candidate_top_k=args.tree_candidate_top_k,
        cuda_graph_block_sizes=graph_block_sizes,
        cuda_graph_batch_sizes=args.cuda_graph_batch_sizes,
        torch_compile=args.torch_compile,
        gated_lora_path=args.gated_lora_path,
        gated_lora_revision=args.gated_lora_revision,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens or args.max_model_len,
        ignore_eos=args.ignore_eos,
        stop_token_ids=resolved_stop_ids,
        mask_token_id=resolved_mask_id,
        noise_mode=args.noise_mode,
        noise_salt=args.noise_salt,
        diffusion_block_size=args.diffusion_block_size,
    )

    start = perf_counter()
    try:
        valid_outputs = (
            llm.generate(
                [prompts[index] for index in valid_indices],
                sampling_params,
                use_tqdm=not args.no_progress,
                request_max_tokens=[
                    request_max_tokens[index]
                    for index in valid_indices
                ],
            )
            if valid_indices
            else []
        )
        elapsed = perf_counter() - start
        stats = getattr(llm, "last_generate_stats", {})
    finally:
        llm.exit()

    outputs: list[dict[str, Any]] = [
        {
            "text": "",
            "token_ids": [],
            "stats": {key: 0 for key in DECODE_STAT_KEYS},
            "error_type": "context_length_exceeded",
        }
        for _ in prompts
    ]
    for index, output in zip(valid_indices, valid_outputs):
        outputs[index] = output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_output_tokens = 0
    with args.output.open("w", encoding="utf-8") as handle:
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Expected {len(prompts)} outputs, got {len(outputs)}"
            )
        for flat_index, output in enumerate(outputs):
            record_index, sample_index = divmod(flat_index, args.num_samples)
            record_id = record_ids[record_index]
            problem = problems[record_index]
            chat_input = chat_inputs[record_index]
            completion_input = completion_inputs[record_index]
            ground_truth = ground_truths[record_index]
            token_ids = output["token_ids"]
            total_output_tokens += len(token_ids)
            row = {
                "id": f"{record_id}:sample{sample_index}",
                "source_row": record_id,
                "sample_index": sample_index,
                "problem": problem,
                "rendered_prompt": problem,
                "chat_input": chat_input,
                "ground_truth": ground_truth,
                "generation": output["text"],
                "output_token_count": len(token_ids),
                "stats": output["stats"],
                "prompt_token_count": prompt_token_counts[flat_index],
                "context_length": args.max_model_len,
                "active_forward_reserve": forward_reserve,
                "resolved_max_tokens": request_max_tokens[flat_index],
            }
            if output.get("error_type") is not None:
                row["error_type"] = output["error_type"]
            if completion_input is not None:
                row["completion_input"] = completion_input
            if args.save_token_ids:
                row["token_ids"] = token_ids
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": benchmark.name if benchmark is not None else None,
        "task": benchmark.task if benchmark is not None else None,
        "model": args.model,
        "tokenizer_path": tokenizer_path,
        "gated_lora_path": args.gated_lora_path,
        "data": str(args.data),
        "output": str(args.output),
        "num_problems": len(records),
        "num_samples_per_problem": args.num_samples,
        "num_generations": len(outputs),
        "num_context_length_failures": len(outputs) - len(valid_outputs),
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": total_output_tokens / elapsed,
        "decoder_stats": stats,
        "decoder_tokens_per_sequence_forward": (
            stats.get("accepts", 0) / stats["forwards"]
            if stats.get("forwards", 0)
            else 0.0
        ),
        "resolved_settings": {
            "global_max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs_per_replica": args.max_num_seqs,
            "attention_backend": args.attention_backend,
            "data_parallel_size": args.data_parallel_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "diffusion_block_size": args.diffusion_block_size,
            "tree_verify_size": args.tree_verify_size,
            "tree_candidate_top_k": args.tree_candidate_top_k,
            "torch_compile": args.torch_compile,
            "active_forward_reserve": forward_reserve,
            "resolved_max_tokens_min": min(request_max_tokens),
            "resolved_max_tokens_max": max(request_max_tokens),
            "noise_mode": args.noise_mode,
            "mask_token_id": resolved_mask_id,
            "stop_token_ids": resolved_stop_ids,
            "vocab_size": vocab_size,
            "instruction": args.instruction,
            "chat_template_kwargs": chat_template_kwargs,
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
