import atexit
import logging
import os
from time import perf_counter, perf_counter_ns
from typing import Callable
from tqdm.auto import tqdm
import torch.multiprocessing as mp

from nano_vllm_uno.config import Config
from nano_vllm_uno.sampling_params import SamplingParams
from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.engine.scheduler import Scheduler
from nano_vllm_uno.engine.model_runner import ModelRunner
from nano_vllm_uno.utils.hf_compat import load_tokenizer

logger = logging.getLogger(__name__)


def _trim_terminal_stop_tokens(
    token_ids: list[int],
    stop_token_ids: set[int],
) -> list[int]:
    """Remove stop tokens consumed by the decoder from returned completions."""
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] in stop_token_ids:
        end -= 1
    return token_ids[:end]


class LLMEngine:

    def __init__(self, model, tokenizer_path=None, **kwargs):
        if "gated_lora" in kwargs:
            raise TypeError(
                "gated_lora is derived from gated_lora_path and cannot be set directly"
            )
        removed_lora_args = {"lora_path", "lora_mode"}.intersection(kwargs)
        if removed_lora_args:
            names = ", ".join(sorted(removed_lora_args))
            raise TypeError(
                f"Removed LoRA argument(s): {names}. "
                "Use gated_lora_path to enable draft-noise-only LoRA."
            )
        if "diffusion_block_size" in kwargs:
            if "max_diffusion_block_size" in kwargs:
                raise TypeError(
                    "Pass only max_diffusion_block_size or its compatibility alias "
                    "diffusion_block_size, not both"
                )
            kwargs["max_diffusion_block_size"] = kwargs.pop(
                "diffusion_block_size"
            )
        config = Config(model, **kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size

        tokenizer_path = tokenizer_path or config.model
        self.tokenizer = load_tokenizer(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
            revision=(
                config.model_revision
                if tokenizer_path == config.model_source
                else None
            ),
            cache_dir=config.hf_cache_dir,
            local_files_only=config.hf_local_files_only,
        )
        config.eos = self.tokenizer.eos_token_id
        config.pad = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        self.ps = []
        self.events = []
        self._exited = False  # Guard against double cleanup
        ctx = mp.get_context("spawn")

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)
        self.sampling_params: SamplingParams | None = None
        self._decode_step_wall_ns: list[int] = []
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True

        worker_id = os.environ.get("SLURM_PROCID")
        worker = f" worker={worker_id}" if worker_id is not None else ""
        print(
            f"Engine stats{worker}: "
            f"preemptions={self.scheduler.num_preemptions}",
            flush=True,
        )

        if self.model_runner is not None:
            self.model_runner.call("exit")
            self.model_runner = None

        for rank, process in enumerate(self.ps, start=1):
            process.join(timeout=10)
            if process.is_alive():
                logger.warning("Worker %d did not exit; terminating it", rank)
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    logger.warning("Worker %d did not terminate; killing it", rank)
                    process.kill()
        self.ps.clear()

    def get_stats(self) -> dict[str, int]:
        return {"preemptions": self.scheduler.num_preemptions}

    def _validate_sampling_params(self, sampling_params: SamplingParams) -> None:
        if (
            sampling_params.diffusion_block_size
            > self.config.max_diffusion_block_size
        ):
            raise ValueError(
                "SamplingParams.diffusion_block_size="
                f"{sampling_params.diffusion_block_size} exceeds engine capacity "
                "max_diffusion_block_size="
                f"{self.config.max_diffusion_block_size}"
            )

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        max_tokens: int | None = None,
    ) -> int:
        self._validate_sampling_params(sampling_params)
        if self.scheduler.is_finished():
            self.sampling_params = sampling_params
        elif self.sampling_params is not sampling_params:
            raise ValueError(
                "All active requests must share one SamplingParams object"
            )
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if max_tokens is not None and int(max_tokens) < 1:
            raise ValueError("per-request max_tokens must be positive")
        seq = Sequence(prompt, max_completion_tokens=max_tokens)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        step_start_ns = perf_counter_ns() if self.config.decode_timing else 0
        sampling_params = self.sampling_params
        if sampling_params is None:
            raise RuntimeError("Cannot step without active sampling parameters")
        seqs, is_prefill = self.scheduler.schedule(sampling_params)
        num_prefill_tokens = (
            sum(len(seq) - seq.num_cached_tokens for seq in seqs)
            if is_prefill
            else 0
        )
        result = self.model_runner.call(
            "run",
            seqs,
            is_prefill,
            sampling_params,
        )

        # Make only KV-complete full pages visible to subsequent scheduler batches.
        for seq in seqs:
            self.scheduler.block_manager.publish_computed_prefix_blocks(seq)

        self.scheduler.postprocess(seqs, result, sampling_params)
        num_new_tokens = sum(len(token_ids) for token_ids in result)

        outputs = [
            (seq.seq_id, seq.completion_token_ids, dict(seq.stats))
            for seq in seqs
            if seq.is_finished
        ]
        if self.config.decode_timing and not is_prefill:
            self._decode_step_wall_ns.append(perf_counter_ns() - step_start_ns)
        num_tokens = num_prefill_tokens if is_prefill else -num_new_tokens
        return outputs, num_tokens

    def reset_decode_timing(self) -> None:
        self._decode_step_wall_ns.clear()
        self.model_runner.call("reset_decode_timing")

    def get_decode_timing(self) -> dict[str, object]:
        model_timing = self.model_runner.call("get_decode_timing")
        forward_records = model_timing["forward_records"]
        gpu_by_step: dict[int, float] = {}
        for record in forward_records:
            step_index = int(record["step_index"])
            gpu_by_step[step_index] = (
                gpu_by_step.get(step_index, 0.0) + float(record["gpu_ms"])
            )
        step_wall_ms = [value / 1e6 for value in self._decode_step_wall_ns]
        model_gpu_ms = [
            gpu_by_step.get(step_index, 0.0)
            for step_index in range(1, len(step_wall_ms) + 1)
        ]
        return {
            "enabled": bool(model_timing["enabled"]),
            "timing_scope": "decode_steps_only",
            "model_gpu_scope": "backbone_and_lm_head",
            "step_wall_ms": step_wall_ms,
            "model_gpu_ms": model_gpu_ms,
            "exposed_non_model_ms": [
                wall_ms - gpu_ms
                for wall_ms, gpu_ms in zip(step_wall_ms, model_gpu_ms)
            ],
            "forward_records": forward_records,
        }

    def is_finished(self):
        return self.scheduler.is_finished()

    def finalize_output(
        self,
        token_ids: list[int],
        stats: dict[str, int],
        sampling_params: SamplingParams,
        detokenize: bool = True,
    ) -> dict[str, object]:
        terminal_stop_ids = {
            self.scheduler.eos,
            *(sampling_params.stop_token_ids or []),
        }
        token_ids = _trim_terminal_stop_tokens(token_ids, terminal_stop_ids)
        output = {
            "token_ids": token_ids,
            "stats": stats,
        }
        if detokenize:
            output["text"] = self.tokenizer.decode(token_ids)
        return output

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
        detokenize: bool = True,
        progress_callback: Callable[[list[dict[str, int]]], None] | None = None,
        request_max_tokens: list[int] | None = None,
    ) -> list[dict[str, object]]:
        if isinstance(sampling_params, list):
            raise ValueError(
                "generate() now accepts exactly one SamplingParams object per call. "
                "All prompts in the call share that sampling configuration."
            )
        self._validate_sampling_params(sampling_params)
        if request_max_tokens is not None and len(request_max_tokens) != len(prompts):
            raise ValueError(
                "request_max_tokens must contain one value per prompt"
            )
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        seq_ids = []
        for index, prompt in enumerate(prompts):
            max_tokens = (
                None
                if request_max_tokens is None
                else int(request_max_tokens[index])
            )
            seq_ids.append(
                self.add_request(
                    prompt,
                    sampling_params,
                    max_tokens=max_tokens,
                )
            )
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids, stats in output:
                outputs[seq_id] = (token_ids, stats)
            if output:
                completed = len(output)
                if use_tqdm:
                    pbar.update(completed)
                if progress_callback is not None:
                    progress_callback([stats for _, _, stats in output])
        outputs = [outputs[seq_id] for seq_id in seq_ids]

        outputs = [
            self.finalize_output(
                token_ids,
                stats,
                sampling_params,
                detokenize=detokenize,
            )
            for token_ids, stats in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
