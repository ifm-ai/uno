import os
import logging
from dataclasses import dataclass, field

os.environ.setdefault(
    "HF_MODULES_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "huggingface", "modules"),
)

import torch
from transformers import AutoConfig
import transformers.dynamic_module_utils as transformers_dynamic_module_utils
import transformers.utils as transformers_utils

from nano_vllm_uno.utils.hub import ADAPTER_ALLOW_PATTERNS, resolve_hf_snapshot
from nano_vllm_uno.utils.hf_compat import is_native_k2_model, load_model_config

logger = logging.getLogger(__name__)

transformers_dynamic_module_utils.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
transformers_utils.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]


@dataclass
class Config:
    model: str
    model_revision: str | None = None
    hf_cache_dir: str | None = None
    hf_local_files_only: bool = False
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    dtype: torch.dtype = field(init=False)
    eos: int = -1
    pad: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    fail_on_preemption: bool = False
    attention_backend: str = "fa3"

    # Supplying an adapter enables gated LoRA only on draft noise
    # rows. Seed, verify, prefill, and AR rows always use the base model.
    gated_lora_path: str | None = None
    gated_lora_revision: str | None = None
    model_source: str = field(init=False)
    gated_lora_source: str | None = field(init=False, default=None)

    # Maximum runtime draft width. SamplingParams.diffusion_block_size selects
    # the actual width for each generate call.
    max_diffusion_block_size: int = 16
    # None captures every width through max_diffusion_block_size.
    cuda_graph_block_sizes: list[int] | None = None
    # Optional fixed-size verifier tree. None preserves linear verification.
    tree_verify_size: int | None = None
    tree_candidate_top_k: int = 16
    # CUDA-graph batch shapes to capture. None uses SGLang's speculative
    # decoding ladder through max_num_seqs; an explicit list captures only
    # those batch sizes.
    cuda_graph_batch_sizes: list[int] | None = None
    # Compile block-decode backbones before CUDA-graph capture.
    torch_compile: bool = False
    # Opt-in decode timing; disabled serving paths only evaluate guard branches.
    decode_timing: bool = False

    @property
    def gated_lora(self) -> bool:
        return self.gated_lora_path is not None

    def __post_init__(self):
        self.model_source = self.model
        model_source_is_local = os.path.isdir(
            os.path.expanduser(os.fspath(self.model_source))
        )
        self.model = resolve_hf_snapshot(
            self.model,
            revision=self.model_revision,
            cache_dir=self.hf_cache_dir,
            local_files_only=self.hf_local_files_only,
            artifact_name="model",
        )
        self.attention_backend = str(self.attention_backend).strip().lower()
        if self.attention_backend not in {"fa2", "fa3", "fa4"}:
            raise ValueError(
                "attention_backend must be 'fa2', 'fa3', or 'fa4', got "
                f"{self.attention_backend!r}"
            )
        # Auto-detect DeepSpeed checkpoint directory
        model_path = self.model
        if os.path.isdir(model_path):
            # Check if this is a DeepSpeed checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(model_path)
                              if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_path, d))]
            if checkpoint_dirs:
                # Use the latest checkpoint directory
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
                model_path = os.path.join(model_path, latest_checkpoint)
                logger.info("Using DeepSpeed checkpoint: %s", model_path)

        assert os.path.isdir(model_path), f"Model path does not exist: {model_path}"
        if self.attention_backend == "fa4":
            if self.kvcache_block_size == 256:
                self.kvcache_block_size = 128
            if self.kvcache_block_size != 128:
                raise ValueError("FA4 requires kvcache_block_size=128")
        elif self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256")
        assert 1 <= self.tensor_parallel_size <= 8
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        self.max_diffusion_block_size = int(self.max_diffusion_block_size)
        if self.max_diffusion_block_size < 1:
            raise ValueError("max_diffusion_block_size must be >= 1")
        self.tree_candidate_top_k = int(self.tree_candidate_top_k)
        if self.tree_candidate_top_k < 1:
            raise ValueError("tree_candidate_top_k must be >= 1")
        if self.tree_verify_size is not None:
            self.tree_verify_size = int(self.tree_verify_size)
            if self.tree_verify_size < 1:
                raise ValueError("tree_verify_size must include at least the root")
            if self.tree_candidate_top_k == 1:
                if self.tree_verify_size != self.max_diffusion_block_size:
                    raise ValueError(
                        "tree_candidate_top_k=1 selects the linear fast path; "
                        "tree_verify_size must equal max_diffusion_block_size"
                    )
                # (B, K=1, V=B) is the linear verifier written in tree
                # notation. Canonicalize it before allocating tree workspaces
                # or capturing tree CUDA graphs.
                self.tree_verify_size = None
            if (
                self.tree_verify_size is not None
                and self.attention_backend == "fa2"
            ):
                raise ValueError("tree verification requires FA3 or FA4")
        self.cuda_graph_block_sizes = (
            list(range(1, self.max_diffusion_block_size + 1))
            if self.cuda_graph_block_sizes is None
            else sorted({int(size) for size in self.cuda_graph_block_sizes})
        )
        if not self.cuda_graph_block_sizes:
            raise ValueError("cuda_graph_block_sizes must contain at least one size")
        if self.cuda_graph_block_sizes[0] < 1:
            raise ValueError("cuda_graph_block_sizes must contain only positive sizes")
        if self.cuda_graph_block_sizes[-1] > self.max_diffusion_block_size:
            raise ValueError(
                "cuda_graph_block_sizes cannot exceed "
                f"max_diffusion_block_size={self.max_diffusion_block_size}"
            )
        if self.cuda_graph_batch_sizes is not None:
            self.cuda_graph_batch_sizes = sorted(
                {int(batch_size) for batch_size in self.cuda_graph_batch_sizes}
            )
            if not self.cuda_graph_batch_sizes:
                raise ValueError("cuda_graph_batch_sizes must contain at least one size")
            if self.cuda_graph_batch_sizes[0] < 1:
                raise ValueError("cuda_graph_batch_sizes must contain only positive sizes")
            if self.cuda_graph_batch_sizes[-1] > self.max_num_seqs:
                raise ValueError(
                    "cuda_graph_batch_sizes cannot exceed max_num_seqs="
                    f"{self.max_num_seqs}: {self.cuda_graph_batch_sizes}"
                )
        if self.torch_compile and self.enforce_eager:
            raise ValueError("torch_compile requires CUDA graphs")
        if self.torch_compile and self.tree_verify_size is not None:
            raise ValueError(
                "torch_compile is currently supported only by linear verification"
            )
        if self.gated_lora:
            self.gated_lora_source = self.gated_lora_path
            self.gated_lora_path = resolve_hf_snapshot(
                self.gated_lora_path,
                revision=self.gated_lora_revision,
                cache_dir=self.hf_cache_dir,
                local_files_only=self.hf_local_files_only,
                allow_patterns=ADAPTER_ALLOW_PATTERNS,
                artifact_name="gated LoRA adapter",
            )
        if is_native_k2_model(model_path):
            self.hf_config = load_model_config(
                self.model_source,
                resolved_path=model_path,
                revision=self.model_revision,
                cache_dir=self.hf_cache_dir,
                local_files_only=self.hf_local_files_only,
            )
        elif model_source_is_local:
            self.hf_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        else:
            # Keep the Hub repository namespace for trusted remote code. A
            # resolved snapshot directory is named after its revision SHA,
            # which is not a valid Python package name when it starts with a
            # digit. Model weights still load from the pinned local snapshot.
            self.hf_config = AutoConfig.from_pretrained(
                self.model_source,
                trust_remote_code=True,
                revision=self.model_revision,
                cache_dir=self.hf_cache_dir,
                local_files_only=self.hf_local_files_only,
            )
        force_float32 = os.environ.get("NANO_VLLM_FORCE_FLOAT32", "0") == "1"
        if force_float32:
            self.dtype = torch.float32
        else:
            # Transformers <4.56 stores this as torch_dtype; newer versions
            # expose dtype and retain torch_dtype only as a deprecated alias.
            dtype = getattr(self.hf_config, "dtype", None)
            if dtype is None:
                dtype = getattr(self.hf_config, "torch_dtype", None)
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype.removeprefix("torch."))
            if dtype is None:
                dtype = torch.float16
            elif dtype == torch.float32:
                dtype = torch.bfloat16
            if not isinstance(dtype, torch.dtype):
                raise TypeError(f"Unsupported model dtype: {dtype!r}")
            self.dtype = dtype
        if self.max_model_len > self.hf_config.max_position_embeddings:
            self.hf_config.max_position_embeddings = self.max_model_len
        else:
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
