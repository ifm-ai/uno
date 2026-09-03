# nano_vllm_uno/llm.py

from __future__ import annotations

import atexit
import os
import subprocess
import sys
from dataclasses import replace
from multiprocessing import Pipe
from multiprocessing.connection import wait
from typing import Any, Optional

from tqdm.auto import tqdm

from nano_vllm_uno.engine.llm_engine import LLMEngine
from nano_vllm_uno.engine.sequence import DECODE_STAT_KEYS
from nano_vllm_uno.sampling_params import SamplingParams


class LLM(LLMEngine):
    """
    User-facing API for nano-vllm.

    Extends LLMEngine with:
      - `greedy` flag on generate()
      - independent data-parallel replicas
    """

    def __init__(
        self,
        model,
        tokenizer_path=None,
        data_parallel_size: int = 1,
        **kwargs: Any,
    ):
        self.data_parallel_size = int(data_parallel_size)
        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        if self.data_parallel_size == 1:
            super().__init__(model, tokenizer_path, **kwargs)
            return

        self._exited = False
        self._dp_workers = []
        tp = int(kwargs.get("tensor_parallel_size", 1))
        base_port = int(os.environ.get("TORCH_DISTRIBUTED_PORT", 20000 + os.getpid() % 20000))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = (
            [gpu.strip() for gpu in visible.split(",")]
            if visible
            else [str(i) for i in range(self.data_parallel_size * tp)]
        )
        if len(gpu_ids) < self.data_parallel_size * tp:
            raise ValueError(
                f"data_parallel_size={self.data_parallel_size} and tensor_parallel_size={tp} "
                f"require {self.data_parallel_size * tp} visible GPUs, got {len(gpu_ids)}"
            )

        try:
            for rank in range(self.data_parallel_size):
                parent_conn, child_conn = Pipe()
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[rank * tp : (rank + 1) * tp])
                env["TORCH_DISTRIBUTED_PORT"] = str(base_port + rank)
                process = subprocess.Popen(
                    [sys.executable, "-m", "nano_vllm_uno.dp_worker", str(child_conn.fileno())],
                    env=env,
                    pass_fds=(child_conn.fileno(),),
                )
                child_conn.close()
                self._dp_workers.append((process, parent_conn))
                parent_conn.send((model, tokenizer_path, kwargs))
            for rank, (_, conn) in enumerate(self._dp_workers):
                ok, result = conn.recv()
                if not ok:
                    raise RuntimeError(f"data-parallel replica {rank} failed to start:\n{result}")
        except BaseException:
            self.exit()
            raise
        atexit.register(self.exit)

    def generate(
        self,
        prompts,
        sampling_params: Optional[SamplingParams] = None,
        *,
        # convenience flag for pure greedy / argmax
        greedy: Optional[bool] = None,
        request_max_tokens: list[int] | None = None,
        use_tqdm: bool = True,
        detokenize: bool = True,
        **kwargs: Any,
    ):
        """
        Extra flags:

        - greedy:
            * None  -> keep whatever is in `sampling_params`.
            * True  -> force pure greedy (temperature=0, disable top_k/top_p).
            * False -> ensure non-greedy (temperature>0 if it was 0).
        - use_tqdm: display generation progress.
        - detokenize: include decoded text alongside output token IDs.

        """

        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, list):
            raise ValueError(
                "LLM.generate() accepts exactly one SamplingParams object per call. "
                "All prompts in the call share that sampling configuration."
            )

        sp = sampling_params

        # ------------------------------------------------------------
        # 1) Greedy handling
        # ------------------------------------------------------------
        if greedy is not None:
            if greedy:
                # Pure argmax decoding: temperature=0, full-vocab.
                sp = replace(
                    sp,
                    temperature=0.0,
                    top_k=None,
                    top_p=None,
                )
            else:
                # Explicitly non-greedy: if temperature was 0, bump it.
                if getattr(sp, "temperature", 1.0) == 0.0:
                    sp = replace(sp, temperature=1.0)

        if self.data_parallel_size == 1:
            outputs = super().generate(
                prompts,
                sp,
                request_max_tokens=request_max_tokens,
                use_tqdm=use_tqdm,
                detokenize=detokenize,
                **kwargs,
            )
            self.last_generate_stats = {
                key: sum(int(output["stats"][key]) for output in outputs)
                for key in DECODE_STAT_KEYS
            }
            return outputs
        if not prompts:
            self.last_generate_stats = {key: 0 for key in DECODE_STAT_KEYS}
            return []
        if request_max_tokens is not None and len(request_max_tokens) != len(prompts):
            raise ValueError(
                "request_max_tokens must contain one value per prompt"
            )

        shards = [[] for _ in self._dp_workers]
        for index, prompt in enumerate(prompts):
            per_request_limit = (
                None
                if request_max_tokens is None
                else int(request_max_tokens[index])
            )
            shards[index % len(shards)].append(
                (index, prompt, per_request_limit)
            )
        active = {}
        worker_kwargs = {
            **kwargs,
            "use_tqdm": False,
            "detokenize": detokenize,
        }
        progress_stats = {key: 0 for key in DECODE_STAT_KEYS}
        for rank, shard in enumerate(shards):
            if not shard:
                continue
            indices, worker_prompts, worker_limits = zip(*shard)
            conn = self._dp_workers[rank][1]
            rank_worker_kwargs = dict(worker_kwargs)
            if request_max_tokens is not None:
                rank_worker_kwargs["request_max_tokens"] = list(worker_limits)
            conn.send(
                (
                    "generate",
                    list(worker_prompts),
                    sp,
                    rank_worker_kwargs,
                    use_tqdm,
                )
            )
            active[conn] = (rank, indices)

        ordered = [None] * len(prompts)
        aggregate_stats = {}
        pbar = (
            tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        try:
            while active:
                for conn in wait(active):
                    rank, indices = active[conn]
                    message = conn.recv()
                    if message[0] == "progress":
                        completed_stats = message[1]
                        for stats in completed_stats:
                            for key in DECODE_STAT_KEYS:
                                progress_stats[key] += int(stats[key])
                        if pbar is not None:
                            pbar.update(len(completed_stats))
                            forwards = progress_stats["forwards"]
                            tpf = (
                                progress_stats["accepts"] / forwards
                                if forwards
                                else 0.0
                            )
                            pbar.set_postfix({"TPF": f"{tpf:.2f}"})
                        continue

                    ok, result = message
                    if not ok:
                        raise RuntimeError(
                            f"data-parallel replica {rank} failed during generate:\n{result}"
                        )
                    result, stats = result
                    if len(result) != len(indices):
                        raise RuntimeError(
                            f"data-parallel replica {rank} returned an unexpected output count"
                        )
                    for index, output in zip(indices, result):
                        ordered[index] = output
                    for key in DECODE_STAT_KEYS:
                        aggregate_stats[key] = (
                            aggregate_stats.get(key, 0) + int(stats[key])
                        )
                    del active[conn]
        finally:
            if pbar is not None:
                pbar.close()
        self.last_generate_stats = aggregate_stats
        return ordered

    def exit(self):
        if self.data_parallel_size == 1:
            return super().exit()
        if getattr(self, "_exited", False):
            return
        self._exited = True
        for process, conn in getattr(self, "_dp_workers", []):
            try:
                conn.send(("exit",))
                conn.close()
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self._dp_workers = []
