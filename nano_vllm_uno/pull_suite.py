"""Pull-based, multi-GPU evaluator for the canonical 12-task suite."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

from nano_vllm_uno.engine.async_engine import AsyncLLMEngine
from nano_vllm_uno.engine.sequence import DECODE_STAT_KEYS
from nano_vllm_uno.eval.benchmarks import (
    BENCHMARKS,
    BenchmarkConfig,
    get_benchmark,
    list_benchmarks,
)
from nano_vllm_uno.eval.context_budget import (
    DEFAULT_CONTEXT_LENGTH,
    active_forward_reserve,
    resolve_completion_budget,
)
from nano_vllm_uno.eval.model_tokens import resolve_model_token_ids
from nano_vllm_uno.utils.hf_compat import load_tokenizer
from nano_vllm_uno.sampling_params import SamplingParams


ID_FIELDS = ("id", "problem_id", "index", "row")
ANSWER_FIELDS = ("ground_truth", "answer")
MANIFEST_VERSION = 1

# Give every Slurm job a fixed-width block of localhost TCPStore ports so
# adjacent jobs cannot select overlapping worker ranges.
_TORCH_PORT_MIN = 40000
_TORCH_PORT_MAX_EXCLUSIVE = 60000
_TORCH_PORT_STRIDE = 64
_TORCH_PORT_SLOTS = (
    (_TORCH_PORT_MAX_EXCLUSIVE - _TORCH_PORT_MIN) // _TORCH_PORT_STRIDE
)


def _torch_distributed_port(job_id: int, local_rank: int) -> int:
    """Return a worker-local TCPStore port from a non-overlapping job block."""
    job_id = int(job_id)
    local_rank = int(local_rank)
    if job_id < 0:
        raise ValueError("job_id must be non-negative")
    if not 0 <= local_rank < _TORCH_PORT_STRIDE:
        raise ValueError(
            f"local_rank must be in [0, {_TORCH_PORT_STRIDE}), got {local_rank}"
        )
    job_slot = job_id % _TORCH_PORT_SLOTS
    return _TORCH_PORT_MIN + job_slot * _TORCH_PORT_STRIDE + local_rank


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value is None or not value.strip() else int(value)


def _cuda_graph_batch_ladder(max_num_seqs: int) -> list[int]:
    """Return power-of-two CUDA graph batches through the concurrency cap."""
    if max_num_seqs < 1:
        return []
    batch_sizes = []
    batch_size = 1
    while batch_size <= max_num_seqs:
        batch_sizes.append(batch_size)
        batch_size *= 2
    if batch_sizes[-1] != max_num_seqs:
        batch_sizes.append(max_num_seqs)
    return batch_sizes


def _int_list(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise ValueError(f"Expected positive comma-separated integers, got {value!r}")
    return values


def _selected_benchmarks() -> tuple[str, ...]:
    raw = os.environ.get("NANO_BENCHMARKS", "").replace(",", " ")
    if not raw.strip():
        return list_benchmarks()
    names = tuple(get_benchmark(name).name for name in raw.split())
    if len(names) != len(set(names)):
        raise ValueError(f"NANO_BENCHMARKS contains duplicates: {names}")
    return names


@dataclass(frozen=True)
class SuiteSettings:
    model: str
    model_revision: str | None
    tokenizer_path: str
    tokenizer_revision: str | None
    gated_lora_path: str | None
    gated_lora_revision: str | None
    hf_cache_dir: str | None
    hf_local_files_only: bool
    results_root: Path
    run_name: str
    benchmarks: tuple[str, ...]
    data_root: Path | None
    num_samples: int
    limit: int | None
    max_num_seqs: int
    context_length: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    attention_backend: str
    global_max_tokens: int | None
    temperature: float
    top_k: int | None
    top_p: float | None
    diffusion_block_size: int
    tree_verify_size: int | None
    tree_candidate_top_k: int
    torch_compile: bool
    noise_mode: str
    noise_salt: int | None
    ignore_eos: bool
    mask_token_id: int | None
    stop_token_ids: tuple[int, ...] | None
    cuda_graph_block_sizes: tuple[int, ...]
    cuda_graph_batch_sizes: tuple[int, ...]
    save_token_ids: bool

    @classmethod
    def from_env(cls) -> "SuiteSettings":
        model = os.environ["NANO_MODEL"]
        model_revision = os.environ.get("NANO_MODEL_REVISION") or None
        tokenizer_path = os.environ.get("NANO_TOKENIZER_PATH", model)
        block_size = int(os.environ.get("NANO_DIFFUSION_BLOCK_SIZE", "16"))
        max_num_seqs = int(os.environ.get("NANO_MAX_NUM_SEQS", "64"))
        default_graph_batches = _cuda_graph_batch_ladder(max_num_seqs)
        settings = cls(
            model=model,
            model_revision=model_revision,
            tokenizer_path=tokenizer_path,
            tokenizer_revision=(
                os.environ.get("NANO_TOKENIZER_REVISION")
                or (model_revision if tokenizer_path == model else None)
            ),
            gated_lora_path=os.environ.get("NANO_GATED_LORA_PATH") or None,
            gated_lora_revision=(
                os.environ.get("NANO_GATED_LORA_REVISION") or None
            ),
            hf_cache_dir=os.environ.get("NANO_HF_CACHE_DIR") or None,
            hf_local_files_only=_env_bool("NANO_HF_LOCAL_FILES_ONLY"),
            results_root=Path(os.environ["NANO_RESULTS_ROOT"]),
            run_name=os.environ["NANO_RUN_NAME"],
            benchmarks=_selected_benchmarks(),
            data_root=(
                Path(os.environ["NANO_DATA_ROOT"])
                if os.environ.get("NANO_DATA_ROOT")
                else None
            ),
            num_samples=int(os.environ.get("NANO_NUM_SAMPLES", "1")),
            limit=_optional_int("NANO_LIMIT"),
            max_num_seqs=max_num_seqs,
            context_length=int(
                os.environ.get("NANO_CONTEXT_LENGTH", str(DEFAULT_CONTEXT_LENGTH))
            ),
            max_num_batched_tokens=int(
                os.environ.get(
                    "NANO_MAX_NUM_BATCHED_TOKENS",
                    str(DEFAULT_CONTEXT_LENGTH),
                )
            ),
            gpu_memory_utilization=float(
                os.environ.get("NANO_GPU_MEMORY_UTILIZATION", "0.90")
            ),
            attention_backend=os.environ.get("NANO_ATTENTION_BACKEND", "fa3"),
            global_max_tokens=_optional_int("NANO_MAX_TOKENS"),
            temperature=float(os.environ.get("NANO_TEMPERATURE", "1.0")),
            top_k=_optional_int("NANO_TOP_K") if "NANO_TOP_K" in os.environ else 50,
            top_p=(
                float(os.environ["NANO_TOP_P"])
                if os.environ.get("NANO_TOP_P")
                else 0.95
            ),
            diffusion_block_size=block_size,
            tree_verify_size=_optional_int("NANO_TREE_VERIFY_SIZE"),
            tree_candidate_top_k=int(
                os.environ.get("NANO_TREE_CANDIDATE_TOP_K", "16")
            ),
            torch_compile=_env_bool("NANO_TORCH_COMPILE"),
            noise_mode=os.environ.get("NANO_NOISE_MODE", "random_uniform"),
            noise_salt=_optional_int("NANO_NOISE_SALT"),
            ignore_eos=_env_bool("NANO_IGNORE_EOS"),
            mask_token_id=_optional_int("NANO_MASK_TOKEN_ID"),
            stop_token_ids=(
                tuple(_int_list(os.environ["NANO_STOP_TOKEN_IDS"]))
                if os.environ.get("NANO_STOP_TOKEN_IDS")
                else None
            ),
            cuda_graph_block_sizes=tuple(
                _int_list(
                    os.environ.get(
                        "NANO_CUDA_GRAPH_BLOCK_SIZES",
                        f"1,{block_size}",
                    )
                )
            ),
            cuda_graph_batch_sizes=tuple(
                _int_list(
                    os.environ.get(
                        "NANO_CUDA_GRAPH_BATCH_SIZES",
                        ",".join(map(str, default_graph_batches)),
                    )
                )
            ),
            save_token_ids=_env_bool("NANO_SAVE_TOKEN_IDS"),
        )
        settings.validate()
        return settings

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.run_name

    @property
    def forward_reserve(self) -> int:
        return active_forward_reserve(
            self.diffusion_block_size,
            self.tree_verify_size,
        )

    def validate(self) -> None:
        if self.num_samples < 1:
            raise ValueError("NANO_NUM_SAMPLES must be positive")
        if self.limit is not None and self.limit < 1:
            raise ValueError("NANO_LIMIT must be positive")
        if self.max_num_seqs < 1:
            raise ValueError("NANO_MAX_NUM_SEQS must be positive")
        if self.context_length < 1:
            raise ValueError("NANO_CONTEXT_LENGTH must be positive")
        if self.max_num_batched_tokens < 1:
            raise ValueError("NANO_MAX_NUM_BATCHED_TOKENS must be positive")
        if self.attention_backend not in {"fa2", "fa3", "fa4"}:
            raise ValueError("NANO_ATTENTION_BACKEND must be fa2, fa3, or fa4")
        if self.tree_verify_size is not None and self.attention_backend == "fa2":
            raise ValueError(
                "Tree verification requires the fa3 or fa4 attention backend"
            )
        if self.torch_compile and self.tree_verify_size is not None:
            raise ValueError("Tree verification requires NANO_TORCH_COMPILE=0")


@dataclass
class PreparedSuite:
    jobs: list[dict[str, Any]]
    immediate_results: list[dict[str, Any]]
    output_by_benchmark: dict[str, Path]
    total_by_benchmark: dict[str, int]
    completed_by_benchmark: dict[str, int]
    data_by_benchmark: dict[str, Path]

    @property
    def missing_count(self) -> int:
        return len(self.jobs) + len(self.immediate_results)


@dataclass
class WriteStats:
    num_results: int = 0
    output_tokens: int = 0


_MANAGER_JOB_QUEUE: queue.Queue | None = None
_MANAGER_RESULT_QUEUE: queue.Queue | None = None


def _job_queue() -> queue.Queue:
    if _MANAGER_JOB_QUEUE is None:
        raise RuntimeError("coordinator job queue is not initialized")
    return _MANAGER_JOB_QUEUE


def _result_queue() -> queue.Queue:
    if _MANAGER_RESULT_QUEUE is None:
        raise RuntimeError("coordinator result queue is not initialized")
    return _MANAGER_RESULT_QUEUE


class PullQueueManager(BaseManager):
    pass


PullQueueManager.register("job_queue", callable=_job_queue)
PullQueueManager.register("result_queue", callable=_result_queue)


def _first_field(record: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def _load_records(
    path: Path,
    limit: int | None = None,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    if not records and not allow_empty:
        raise ValueError(f"No records found in {path}")
    return records


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in _load_records(path, allow_empty=True):
        output_id = str(row["id"])
        if output_id in ids:
            raise ValueError(f"Duplicate generation id {output_id!r} in {path}")
        ids.add(output_id)
    return ids


def _data_path(config: BenchmarkConfig, data_root: Path | None) -> Path:
    path = config.data_path if data_root is None else data_root / config.data_path.name
    if path.is_file():
        return path
    from nano_vllm_uno.eval.data import prepare_benchmark_data

    return prepare_benchmark_data(config.name, output_dir=path.parent)


def format_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    config: BenchmarkConfig,
) -> tuple[list[int], str, list[dict[str, Any]]]:
    messages = [dict(message) for message in messages]
    system_message = next(
        (message for message in messages if message.get("role") == "system"),
        None,
    )
    if config.instruction:
        if system_message is None:
            messages.insert(0, {"role": "system", "content": config.instruction})
        elif config.instruction not in system_message["content"]:
            existing = system_message["content"].strip()
            system_message["content"] = (
                f"{config.instruction}\n\n{existing}"
                if existing
                else config.instruction
            )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **config.chat_template_kwargs,
    )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **config.chat_template_kwargs,
    )
    return list(token_ids), str(rendered), messages


def prepare_suite(settings: SuiteSettings, tokenizer: Any) -> PreparedSuite:
    jobs: list[dict[str, Any]] = []
    immediate_results: list[dict[str, Any]] = []
    output_by_benchmark = {}
    total_by_benchmark = {}
    completed_by_benchmark = {}
    data_by_benchmark = {}

    for benchmark_name in settings.benchmarks:
        config = BENCHMARKS[benchmark_name]
        data_path = _data_path(config, settings.data_root)
        records = _load_records(data_path, settings.limit)
        if settings.limit is None and len(records) != config.expected_rows:
            raise ValueError(
                f"{config.name} expects {config.expected_rows} rows, but "
                f"{data_path} contains {len(records)}"
            )
        output = settings.run_dir / config.name / "generations.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        completed_ids = _completed_ids(output)
        expected_ids: set[str] = set()
        output_by_benchmark[config.name] = output
        data_by_benchmark[config.name] = data_path
        total_by_benchmark[config.name] = len(records) * settings.num_samples

        for record_index, record in enumerate(records):
            messages = record.get("chat_input")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{data_path}: row {record_index} needs chat_input")
            prompt_ids, rendered, chat_input = format_prompt(tokenizer, messages, config)
            source_row = _first_field(record, ID_FIELDS)
            if source_row is None:
                source_row = record_index
            ground_truth = _first_field(record, ANSWER_FIELDS)
            resolved_max_tokens = resolve_completion_budget(
                prompt_tokens=len(prompt_ids),
                context_length=settings.context_length,
                reserve_tokens=settings.forward_reserve,
                global_max_tokens=settings.global_max_tokens,
            )
            for sample_index in range(settings.num_samples):
                output_id = f"{source_row}:sample{sample_index}"
                if output_id in expected_ids:
                    raise ValueError(f"Duplicate expected id {output_id!r} in {data_path}")
                expected_ids.add(output_id)
                if output_id in completed_ids:
                    continue
                row = {
                    "id": output_id,
                    "source_row": source_row,
                    "sample_index": sample_index,
                    "problem": rendered,
                    "rendered_prompt": rendered,
                    "chat_input": chat_input,
                    "ground_truth": ground_truth,
                    "prompt_token_count": len(prompt_ids),
                    "context_length": settings.context_length,
                    "active_forward_reserve": settings.forward_reserve,
                    "resolved_max_tokens": resolved_max_tokens,
                }
                if record.get("completion_input") is not None:
                    row["completion_input"] = record["completion_input"]
                if resolved_max_tokens == 0:
                    immediate_results.append(
                        {
                            "benchmark": config.name,
                            "row": {
                                **row,
                                "generation": "",
                                "output_token_count": 0,
                                "stats": {key: 0 for key in DECODE_STAT_KEYS},
                                "error_type": "context_length_exceeded",
                            },
                        }
                    )
                else:
                    jobs.append(
                        {
                            "benchmark": config.name,
                            "prompt_token_ids": prompt_ids,
                            "max_tokens": resolved_max_tokens,
                            "row": row,
                        }
                    )

        unexpected = completed_ids - expected_ids
        if unexpected:
            example = sorted(unexpected)[:3]
            raise ValueError(f"{output} contains IDs outside this run: {example}")
        completed_by_benchmark[config.name] = len(completed_ids)

    return PreparedSuite(
        jobs=jobs,
        immediate_results=immediate_results,
        output_by_benchmark=output_by_benchmark,
        total_by_benchmark=total_by_benchmark,
        completed_by_benchmark=completed_by_benchmark,
        data_by_benchmark=data_by_benchmark,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    settings: SuiteSettings,
    suite: PreparedSuite,
    *,
    worker_count: int,
    mask_token_id: int,
    stop_token_ids: list[int],
    vocab_size: int,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "backend": "nano_vllm_uno_pull_suite",
        "model": settings.model,
        "model_revision": settings.model_revision,
        "tokenizer_path": settings.tokenizer_path,
        "tokenizer_revision": settings.tokenizer_revision,
        "gated_lora_path": settings.gated_lora_path,
        "gated_lora_revision": settings.gated_lora_revision,
        "worker_count": worker_count,
        "benchmarks": [
            {
                "name": name,
                "task": BENCHMARKS[name].task,
                "data": str(suite.data_by_benchmark[name]),
                "data_sha256": _sha256(suite.data_by_benchmark[name]),
                "rows": suite.total_by_benchmark[name] // settings.num_samples,
                "instruction": BENCHMARKS[name].instruction,
                "chat_template_kwargs": BENCHMARKS[name].chat_template_kwargs,
                "parser": BENCHMARKS[name].parser,
            }
            for name in settings.benchmarks
        ],
        "num_samples": settings.num_samples,
        "limit": settings.limit,
        "context": {
            "length": settings.context_length,
            "active_forward_reserve": settings.forward_reserve,
            "global_max_tokens": settings.global_max_tokens,
        },
        "engine": {
            "tensor_parallel_size": 1,
            "max_num_seqs_per_gpu": settings.max_num_seqs,
            "max_num_batched_tokens": settings.max_num_batched_tokens,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
            "attention_backend": settings.attention_backend,
            "diffusion_block_size": settings.diffusion_block_size,
            "tree_verify_size": settings.tree_verify_size,
            "tree_candidate_top_k": settings.tree_candidate_top_k,
            "torch_compile": settings.torch_compile,
            "cuda_graph_block_sizes": list(settings.cuda_graph_block_sizes),
            "cuda_graph_batch_sizes": list(settings.cuda_graph_batch_sizes),
        },
        "sampling": {
            "temperature": settings.temperature,
            "top_k": settings.top_k,
            "top_p": settings.top_p,
            "ignore_eos": settings.ignore_eos,
            "noise_mode": settings.noise_mode,
            "noise_salt": settings.noise_salt,
            "mask_token_id": mask_token_id,
            "stop_token_ids": stop_token_ids,
            "vocab_size": vocab_size,
        },
        "save_token_ids": settings.save_token_ids,
    }


def write_or_validate_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / "run_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"Refusing incompatible resume in {run_dir}; resolved settings "
                "differ from run_manifest.json"
            )
        return path
    if any(run_dir.glob("*/generations.jsonl")):
        raise ValueError(
            f"Refusing to resume {run_dir}: generations exist without a run manifest"
        )
    _write_json(path, manifest)
    return path


def write_results(
    result_queue: Any,
    suite: PreparedSuite,
    stop_event: threading.Event,
) -> WriteStats:
    handles = {
        name: path.open("a", encoding="utf-8", buffering=1)
        for name, path in suite.output_by_benchmark.items()
    }
    bars = {
        name: tqdm(
            total=suite.total_by_benchmark[name],
            initial=suite.completed_by_benchmark[name],
            desc=name,
            unit="gen",
            position=position,
            leave=True,
        )
        for position, name in enumerate(suite.output_by_benchmark)
    }
    written = WriteStats()
    try:
        while written.num_results < suite.missing_count:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                if stop_event.is_set():
                    raise RuntimeError(
                        "Generation stopped before all queued results were written: "
                        f"{written.num_results}/{suite.missing_count}"
                    )
                continue
            benchmark = result["benchmark"]
            row = result["row"]
            handles[benchmark].write(json.dumps(row, ensure_ascii=False) + "\n")
            written.num_results += 1
            written.output_tokens += int(row.get("output_token_count", 0))
            bars[benchmark].update(1)
    finally:
        for handle in handles.values():
            handle.close()
        for bar in bars.values():
            bar.close()
    return written


def _resolve_worker_model_token_ids(
    engine: Any,
    settings: SuiteSettings,
) -> tuple[int, list[int], int]:
    config = engine.config
    return resolve_model_token_ids(
        config.model_source,
        engine.tokenizer,
        mask_token_id=settings.mask_token_id,
        stop_token_ids=(
            list(settings.stop_token_ids)
            if settings.stop_token_ids is not None
            else None
        ),
        revision=config.model_revision,
        cache_dir=config.hf_cache_dir,
        local_files_only=config.hf_local_files_only,
        noise_mode=settings.noise_mode,
    )


def run_generation_worker(
    *,
    job_queue: Any,
    result_queue: Any,
    settings: SuiteSettings,
) -> None:
    first_job = job_queue.get()
    if first_job is None:
        return

    from nano_vllm_uno.engine.llm_engine import LLMEngine

    engine = LLMEngine(
        settings.model,
        model_revision=settings.model_revision,
        hf_cache_dir=settings.hf_cache_dir,
        hf_local_files_only=settings.hf_local_files_only,
        tokenizer_path=settings.tokenizer_path,
        tensor_parallel_size=1,
        max_num_seqs=settings.max_num_seqs,
        max_model_len=settings.context_length,
        max_num_batched_tokens=settings.max_num_batched_tokens,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        attention_backend=settings.attention_backend,
        gated_lora_path=settings.gated_lora_path,
        gated_lora_revision=settings.gated_lora_revision,
        max_diffusion_block_size=settings.diffusion_block_size,
        tree_verify_size=settings.tree_verify_size,
        tree_candidate_top_k=settings.tree_candidate_top_k,
        cuda_graph_block_sizes=list(settings.cuda_graph_block_sizes),
        cuda_graph_batch_sizes=list(settings.cuda_graph_batch_sizes),
        torch_compile=settings.torch_compile,
    )
    mask_id, stop_ids, _ = _resolve_worker_model_token_ids(engine, settings)
    sampling_params = SamplingParams(
        max_tokens=settings.global_max_tokens or settings.context_length,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        ignore_eos=settings.ignore_eos,
        stop_token_ids=stop_ids,
        mask_token_id=mask_id,
        noise_mode=settings.noise_mode,
        noise_salt=settings.noise_salt,
        diffusion_block_size=settings.diffusion_block_size,
    )
    async_engine = AsyncLLMEngine(engine, sampling_params)
    async_engine.start()
    pending: dict[
        concurrent.futures.Future,
        tuple[dict[str, Any], float],
    ] = {}
    next_job: dict[str, Any] | None = first_job
    no_more_jobs = False

    def submit(job: dict[str, Any]) -> None:
        future = async_engine.submit(
            job["prompt_token_ids"],
            request_id=job["row"]["id"],
            max_tokens=job["max_tokens"],
        )
        pending[future] = (job, perf_counter())

    try:
        while pending or not no_more_jobs:
            while not no_more_jobs and len(pending) < settings.max_num_seqs:
                if next_job is not None:
                    job = next_job
                    next_job = None
                else:
                    try:
                        job = job_queue.get(block=not pending)
                    except queue.Empty:
                        break
                if job is None:
                    no_more_jobs = True
                    break
                submit(job)

            if not pending:
                continue
            finished, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in finished:
                job, submitted_at = pending.pop(future)
                result = future.result()
                row = {
                    **job["row"],
                    "generation": result.text,
                    "output_token_count": len(result.token_ids),
                    "stats": dict(result.stats),
                    "request_elapsed_seconds": perf_counter() - submitted_at,
                }
                if settings.save_token_ids:
                    row["token_ids"] = result.token_ids
                result_queue.put({"benchmark": job["benchmark"], "row": row})
    finally:
        async_engine.shutdown()


def _connect_manager(host: str, port: int, auth_key: bytes) -> PullQueueManager:
    manager = PullQueueManager(address=(host, port), authkey=auth_key)
    deadline = time.monotonic() + float(
        os.environ.get("NANO_PULL_CONNECT_TIMEOUT", "600")
    )
    while True:
        try:
            manager.connect()
            return manager
        except (ConnectionRefusedError, EOFError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out connecting to coordinator {host}:{port}")
            time.sleep(1)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {key: 0 for key in DECODE_STAT_KEYS}
    sequence_tpfs = []
    total_output_tokens = 0
    context_failures = 0
    request_seconds = 0.0
    for row in rows:
        row_stats = row.get("stats") or {}
        for key in DECODE_STAT_KEYS:
            stats[key] += int(row_stats.get(key, 0))
        forwards = int(row_stats.get("forwards", 0))
        if forwards:
            sequence_tpfs.append(int(row_stats.get("accepts", 0)) / forwards)
        total_output_tokens += int(row.get("output_token_count", 0))
        context_failures += row.get("error_type") == "context_length_exceeded"
        request_seconds += float(row.get("request_elapsed_seconds", 0.0))
    return {
        "num_generations": len(rows),
        "num_context_length_failures": context_failures,
        "total_output_tokens": total_output_tokens,
        "decoder_stats": stats,
        "decoder_tokens_per_sequence_forward": (
            stats["accepts"] / stats["forwards"] if stats["forwards"] else 0.0
        ),
        "forward_weighted_mean_sequence_tpf": (
            stats["accepts"] / stats["forwards"] if stats["forwards"] else 0.0
        ),
        "unweighted_mean_sequence_tpf": (
            sum(sequence_tpfs) / len(sequence_tpfs) if sequence_tpfs else 0.0
        ),
        "num_sequences_with_decoder_forwards": len(sequence_tpfs),
        "sum_request_elapsed_seconds": request_seconds,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_generation_summaries(
    settings: SuiteSettings,
    suite: PreparedSuite,
    manifest: dict[str, Any],
    *,
    elapsed_seconds: float,
    invocation_stats: WriteStats,
    worker_count: int,
) -> None:
    all_rows: list[dict[str, Any]] = []
    benchmark_summaries = {}
    for name, path in suite.output_by_benchmark.items():
        rows = _load_records(path)
        aggregate = _aggregate_rows(rows)
        summary = {
            "benchmark": name,
            "task": BENCHMARKS[name].task,
            "model": settings.model,
            "tokenizer_path": settings.tokenizer_path,
            "gated_lora_path": settings.gated_lora_path,
            "data": str(suite.data_by_benchmark[name]),
            "output": str(path),
            "num_problems": suite.total_by_benchmark[name] // settings.num_samples,
            "num_samples_per_problem": settings.num_samples,
            **aggregate,
            "resolved_settings": {
                **manifest["context"],
                **manifest["engine"],
                **manifest["sampling"],
            },
        }
        _write_json(path.with_name("generation_summary.json"), summary)
        benchmark_summaries[name] = summary
        all_rows.extend(rows)

    aggregate = _aggregate_rows(all_rows)
    suite_summary_path = settings.run_dir / "suite_generation_summary.json"
    previous_invocations: list[dict[str, Any]] = []
    if suite_summary_path.exists():
        previous = json.loads(suite_summary_path.read_text(encoding="utf-8"))
        previous_invocations = list(previous.get("invocations", []))
        if not previous_invocations and "invocation_elapsed_seconds" in previous:
            previous_invocations.append(
                {
                    "slurm_job_id": previous.get("slurm_job_id"),
                    "num_generations": previous.get("invocation_num_generations", 0),
                    "output_tokens": previous.get("invocation_output_tokens", 0),
                    "elapsed_seconds": previous["invocation_elapsed_seconds"],
                }
            )
    current_invocation = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "num_generations": invocation_stats.num_results,
        "output_tokens": invocation_stats.output_tokens,
        "elapsed_seconds": elapsed_seconds,
    }
    invocations = [*previous_invocations, current_invocation]
    generation_invocations = [
        invocation
        for invocation in invocations
        if int(invocation.get("num_generations", 0)) > 0
    ]
    generation_elapsed = sum(
        float(invocation["elapsed_seconds"])
        for invocation in generation_invocations
    )
    generation_output_tokens = sum(
        int(invocation["output_tokens"])
        for invocation in generation_invocations
    )
    suite_summary = {
        "model": settings.model,
        "gated_lora_path": settings.gated_lora_path,
        "benchmarks": list(settings.benchmarks),
        "worker_count": worker_count,
        **aggregate,
        "invocation_num_generations": invocation_stats.num_results,
        "invocation_output_tokens": invocation_stats.output_tokens,
        "invocation_elapsed_seconds": elapsed_seconds,
        "invocation_output_tokens_per_second": (
            invocation_stats.output_tokens / elapsed_seconds
            if elapsed_seconds
            else 0.0
        ),
        "invocation_output_tokens_per_second_per_gpu": (
            invocation_stats.output_tokens / elapsed_seconds / worker_count
            if elapsed_seconds and worker_count
            else 0.0
        ),
        "invocations": invocations,
        "generation_invocation_elapsed_seconds": generation_elapsed,
        "generation_invocation_output_tokens": generation_output_tokens,
        "generation_invocation_output_tokens_per_second": (
            generation_output_tokens / generation_elapsed
            if generation_elapsed
            else 0.0
        ),
        "generation_invocation_output_tokens_per_second_per_gpu": (
            generation_output_tokens / generation_elapsed / worker_count
            if generation_elapsed and worker_count
            else 0.0
        ),
        "manifest": str(settings.run_dir / "run_manifest.json"),
        "benchmark_summaries": {
            name: str(settings.run_dir / name / "generation_summary.json")
            for name in benchmark_summaries
        },
    }
    _write_json(suite_summary_path, suite_summary)


def run_coordinator(
    *,
    worker_count: int,
    settings: SuiteSettings,
    port: int,
    auth_key: bytes,
) -> None:
    job_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    global _MANAGER_JOB_QUEUE, _MANAGER_RESULT_QUEUE
    _MANAGER_JOB_QUEUE = job_queue
    _MANAGER_RESULT_QUEUE = result_queue

    manager = PullQueueManager(address=("", port), authkey=auth_key)
    server = manager.get_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()

    tokenizer = load_tokenizer(
        settings.tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
        revision=settings.tokenizer_revision,
        cache_dir=settings.hf_cache_dir,
        local_files_only=settings.hf_local_files_only,
    )
    mask_id, stop_ids, vocab_size = resolve_model_token_ids(
        settings.model,
        tokenizer,
        mask_token_id=settings.mask_token_id,
        stop_token_ids=(
            list(settings.stop_token_ids)
            if settings.stop_token_ids is not None
            else None
        ),
        revision=settings.model_revision,
        cache_dir=settings.hf_cache_dir,
        local_files_only=settings.hf_local_files_only,
        noise_mode=settings.noise_mode,
    )
    suite = prepare_suite(settings, tokenizer)
    manifest = build_manifest(
        settings,
        suite,
        worker_count=worker_count,
        mask_token_id=mask_id,
        stop_token_ids=stop_ids,
        vocab_size=vocab_size,
    )
    manifest_path = write_or_validate_manifest(settings.run_dir, manifest)
    print(f"Resolved run manifest: {manifest_path}", flush=True)
    for name in suite.output_by_benchmark:
        print(
            f"{name}: {suite.completed_by_benchmark[name]}/"
            f"{suite.total_by_benchmark[name]} complete",
            flush=True,
        )
    print(
        f"Queued {len(suite.jobs)} model generations and "
        f"{len(suite.immediate_results)} context failures",
        flush=True,
    )

    for result in suite.immediate_results:
        result_queue.put(result)
    for job in suite.jobs:
        job_queue.put(job)
    for _ in range(worker_count):
        job_queue.put(None)

    stop_event = threading.Event()
    start = perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(write_results, result_queue, suite, stop_event)
        try:
            run_generation_worker(
                job_queue=job_queue,
                result_queue=result_queue,
                settings=settings,
            )
            write_stats = writer.result()
        except BaseException:
            stop_event.set()
            try:
                writer.result(timeout=5)
            except BaseException:
                pass
            raise
    elapsed = perf_counter() - start
    write_generation_summaries(
        settings,
        suite,
        manifest,
        elapsed_seconds=elapsed,
        invocation_stats=write_stats,
        worker_count=worker_count,
    )


def main() -> None:
    settings = SuiteSettings.from_env()
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", str(rank)))
    worker_count = int(os.environ.get("SLURM_NTASKS", "1"))
    port = int(os.environ.get("NANO_PULL_COORDINATOR_PORT", "19191"))
    auth_key = os.environ.get("NANO_PULL_AUTH_KEY", "nano-vllm-uno").encode()
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    port_base = os.environ.get("NANO_TORCH_DISTRIBUTED_PORT_BASE")
    torch_port = (
        int(port_base) + local_rank
        if port_base is not None
        else _torch_distributed_port(job_id, local_rank)
    )
    if not 1 <= torch_port <= 65535:
        raise ValueError(f"Resolved TORCH_DISTRIBUTED_PORT is invalid: {torch_port}")
    os.environ["TORCH_DISTRIBUTED_PORT"] = str(torch_port)

    if rank == 0:
        run_coordinator(
            worker_count=worker_count,
            settings=settings,
            port=port,
            auth_key=auth_key,
        )
        return

    manager = _connect_manager(
        os.environ["NANO_PULL_COORDINATOR_HOST"],
        port,
        auth_key,
    )
    run_generation_worker(
        job_queue=manager.job_queue(),
        result_queue=manager.result_queue(),
        settings=settings,
    )


if __name__ == "__main__":
    main()
