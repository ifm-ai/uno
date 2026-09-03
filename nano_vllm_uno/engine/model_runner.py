import pickle
import os
import logging
import tempfile
import uuid
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nano_vllm_uno.config import Config
from nano_vllm_uno.engine.block_cuda_graph_runner import BlockCudaGraphRunner
from nano_vllm_uno.engine.block_forward_workspace import BlockForwardWorkspace
from nano_vllm_uno.engine.draft_tree import ancestor_mask_from_parents
from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.engine.tree_page_table import build_tree_page_table
from nano_vllm_uno.engine.tree_kv_copy import (
    build_tree_kv_slots,
    copy_tree_kv,
)
from nano_vllm_uno.models.qwen3 import Qwen3ForCausalLM
from nano_vllm_uno.layers.sampler import Sampler
from nano_vllm_uno.utils.context import (
    ContextMode,
    reset_context,
    set_context,
)
from nano_vllm_uno.utils.loader import load_model
from nano_vllm_uno.sampling_params import SamplingParams
from nano_vllm_uno.utils.lora import load_lora_adapter

from typing import Optional
from nano_vllm_uno.engine.two_pass_decoding import TwoPassDecoder

from datetime import timedelta

logger = logging.getLogger(__name__)


def copy_pinned(dst: torch.Tensor, values) -> None:
    """Copy Python values to a tensor through pinned CPU memory."""
    src = torch.tensor(
        values,
        dtype=dst.dtype,
        device="cpu",
        pin_memory=True,
    )
    dst.copy_(src, non_blocking=True)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self._use_flash_tree_attention = (
            config.attention_backend in {"fa3", "fa4"}
            and config.dtype != torch.float32
        )

        self.block_graph_runner: Optional[BlockCudaGraphRunner] = None
        self.cuda_graph_hits = 0
        self.cuda_graph_misses = 0
        self.cuda_graph_memory: dict[str, int] = {}
        self._decode_timing_step = 0
        self._decode_forward_events: list[dict[str, object]] = []
        
        timeout_minutes = int(os.environ.get('TORCH_DISTRIBUTED_TIMEOUT_MINUTES', '3'))
        master_port = int(os.environ.get('TORCH_DISTRIBUTED_PORT', '2333'))
        self._dist_store_path = None
        if self.world_size == 1:
            self._dist_store_path = os.path.join(
                tempfile.gettempdir(),
                f"nano_vllm_uno_dist_{os.getpid()}_{uuid.uuid4().hex}",
            )
            init_method = f"file://{self._dist_store_path}"
        else:
            init_method = f"tcp://localhost:{master_port}"
        self.shm_name = f"nano_vllm_uno_{master_port}"
        try:
            dist.init_process_group(
                "nccl",
                init_method,
                world_size=self.world_size,
                rank=rank,
                timeout=timedelta(minutes=timeout_minutes),
            )
        except BaseException:
            self._remove_dist_store()
            raise
        torch.cuda.set_device(rank)
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        torch.set_default_device("cuda")
        model_type = getattr(hf_config, "model_type", "").lower()
        if model_type in {"qwen3", "sdar"}:
            self.model = Qwen3ForCausalLM(hf_config, config.attention_backend)
        else:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. Supported: qwen3, sdar"
            )
        load_model(self.model, config.model)
        if config.gated_lora:
            load_lora_adapter(self.model, config.gated_lora_path)

        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()

        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # Unlike nano-vLLM's one-forward AR step, a two-pass TP step makes
        # rank-0 sampling decisions between its draft and verify forwards.
        # Followers reuse these small receive tensors for those broadcasts.
        self._decoder_sync_buffers: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        self.two_pass_decoder = TwoPassDecoder(
            run_block=self._run_block,
            eos_token_id=self.config.eos,
            pad_token_id=self.config.pad,
            vocab_size=self.config.hf_config.vocab_size,
            sampler=self.sampler,
            is_driver=(self.rank == 0 or self.world_size == 1),
            sync_decision=self._sync_decoder_decision,
            gated_lora=self.config.gated_lora,
            run_tree=(
                self._run_tree
                if self.config.tree_verify_size is not None
                else None
            ),
            compact_tree_kv=(
                self._compact_tree_kv
                if self.config.tree_verify_size is not None
                else None
            ),
            tree_verify_size=self.config.tree_verify_size,
            tree_candidate_top_k=self.config.tree_candidate_top_k,
        )

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # Rank 0 sends scheduler batches to TP workers through this
                # command buffer. Decode uses compact sequence state, while
                # prefill still carries the prompt token IDs and may exceed
                # nano-vLLM's original 1 MiB buffer for large prompt batches.
                shm_size = int(os.environ.get("INFERENCE_ENGINE_SHM_SIZE", str(2**26)))
                try:
                    self.shm = SharedMemory(
                        name=self.shm_name,
                        create=True,
                        size=shm_size,
                    )
                except FileExistsError:
                    # Recover a named segment left behind by a crashed run.
                    stale_shm = SharedMemory(name=self.shm_name, create=False)
                    stale_shm.close()
                    stale_shm.unlink()
                    self.shm = SharedMemory(
                        name=self.shm_name,
                        create=True,
                        size=shm_size,
                    )
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=self.shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.block_graph_runner is not None:
            self.block_graph_runner.close()
            self.block_graph_runner = None
        torch.cuda.synchronize()
        dist.destroy_process_group()
        self._remove_dist_store()

    def _remove_dist_store(self):
        if self._dist_store_path is None:
            return
        try:
            os.unlink(self._dist_store_path)
        except FileNotFoundError:
            pass
        self._dist_store_path = None

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        if method_name == "run":
            states, is_prefill, sampling_params = args
            args = [
                [Sequence.from_worker_state(state) for state in states],
                is_prefill,
                sampling_params,
            ]
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        payload_args = args
        if method_name == "run":
            seqs, is_prefill, sampling_params = args
            payload_args = (
                [seq.to_worker_state(include_token_ids=is_prefill) for seq in seqs],
                is_prefill,
                sampling_params,
            )
        data = pickle.dumps([method_name, *payload_args])
        n = len(data)
        capacity = len(self.shm.buf)
        if n + 4 > capacity:
            raise RuntimeError(
                "TP worker command is too large for shared memory: "
                f"payload={n} bytes, capacity={capacity - 4} bytes. "
                "Increase INFERENCE_ENGINE_SHM_SIZE or lower max_num_seqs."
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        seq_len = min(max_num_batched_tokens, self.config.max_model_len)
        num_seqs = min(
            max_num_batched_tokens // seq_len,
            self.config.max_num_seqs,
        )
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        self.run(seqs, True, SamplingParams())
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # Allocate the required two-pass workspace first so its exact footprint,
        # rather than a conservative estimate, is included in device usage.
        memory_stats = torch.cuda.memory_stats()
        current_before_workspace = memory_stats["allocated_bytes.all.current"]
        warmup_peak = memory_stats["allocated_bytes.all.peak"]
        runtime_reserve = max(0, warmup_peak - current_before_workspace)
        self.block_workspace = BlockForwardWorkspace(config, self.block_size)

        if config.num_kvcache_blocks <= 0:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            block_bytes = (
                2
                * hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * head_dim
                * config.dtype.itemsize
            )

            available_memory = int(
                total * config.gpu_memory_utilization
                - used
                - runtime_reserve
            )

            config.num_kvcache_blocks = available_memory // block_bytes

            if config.num_kvcache_blocks <= 0:
                raise RuntimeError(
                    "Insufficient GPU memory for KV cache allocation. "
                    f"Available: {available_memory / 1e9:.2f} GB, "
                    f"Required per block: {block_bytes / 1e6:.2f} MB, "
                    f"Would need at least {block_bytes / 1e9:.2f} GB for 1 block. "
                    "Try freeing GPU memory, increasing gpu_memory_utilization "
                    f"(current: {config.gpu_memory_utilization}), or reducing "
                    "the configured batch/graph shapes."
                )

        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def stage_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(
            ContextMode.PREFILL,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            lora_enabled=False,
        )
        return input_ids, positions

    @torch.inference_mode()
    def _run_prefill(
        self,
        seqs: list[Sequence],
        sampling_params: SamplingParams,
    ) -> list[list[int]] | None:
        """Prefill prompt KV and emit its first completion token.

        The final prefill logit is the same causal next-token distribution
        that draft position zero would otherwise recompute. Append that token
        as the new uncached tail for every block size, so the first draft input
        starts with it instead of re-forwarding the final prompt token.
        """
        input_ids, positions = self.stage_prefill(seqs)
        logits = self.model.compute_logits(self.model(input_ids, positions))
        if self.rank == 0 or self.world_size == 1:
            sampled_tokens, _ = self.sampler(logits, sampling_params)
            token_ids = sampled_tokens.tolist()
        else:
            token_ids = None
        reset_context()
        if token_ids is None:
            return None
        for seq, token_id in zip(seqs, token_ids):
            seq.extend_tokens([token_id])
            seq.num_cached_tokens = len(seq) - 1
        return [[int(token_id)] for token_id in token_ids]

    def _execute_staged_block(
        self,
        *,
        batch_size: int,
        block_len: int,
        lora_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Execute one rank's staged block through a graph or eager fallback."""
        total_tokens = batch_size * block_len
        workspace = self.block_workspace
        input_ids = workspace.input_ids[:total_tokens]
        positions = workspace.positions[:total_tokens]
        slot_mapping = workspace.slot_mapping[:total_tokens]
        kv_seqlens = workspace.kv_seqlens[:batch_size]
        block_tables = workspace.block_tables[:batch_size]
        lora_enabled = lora_mask is not None

        timing_start = None
        if self.config.decode_timing:
            timing_start = torch.cuda.Event(enable_timing=True)
            timing_start.record()

        graph_output = None
        if self.block_graph_runner is not None:
            graph_output = self.block_graph_runner.replay(
                batch_size=batch_size,
                block_len=block_len,
                lora_enabled=lora_enabled,
            )

        graph_hit = graph_output is not None
        if graph_hit:
            self.cuda_graph_hits += 1
            hidden_states = graph_output
        else:
            set_context(
                ContextMode.BLOCK_DECODE,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                kv_seqlens=kv_seqlens,
                seqlen_q=block_len,
                lora_mask=lora_mask,
                lora_enabled=lora_enabled,
            )
            hidden_states = self.model(input_ids, positions)
            self.cuda_graph_misses += 1

        logits = self.model.compute_logits(hidden_states)
        if timing_start is not None:
            self._record_decode_forward_timing(
                timing_start,
                batch_size=batch_size,
                block_len=block_len,
                graph_hit=graph_hit,
            )
        return logits

    def _record_decode_forward_timing(
        self,
        timing_start,
        *,
        batch_size: int,
        block_len: int,
        graph_hit: bool,
    ) -> None:
        timing_end = torch.cuda.Event(enable_timing=True)
        timing_end.record()
        self._decode_forward_events.append(
            {
                "start": timing_start,
                "end": timing_end,
                "step_index": self._decode_timing_step,
                "batch_size": batch_size,
                "block_len": block_len,
                "cuda_graph_hit": graph_hit,
            }
        )

    def reset_decode_timing(self) -> None:
        self._decode_timing_step = 0
        self._decode_forward_events.clear()

    def get_decode_timing(self) -> dict[str, object]:
        if not self.config.decode_timing:
            return {"enabled": False, "forward_records": []}
        torch.cuda.synchronize()
        records = []
        for raw in self._decode_forward_events:
            record = {
                key: value
                for key, value in raw.items()
                if key not in ("start", "end")
            }
            record["gpu_ms"] = raw["start"].elapsed_time(raw["end"])
            records.append(record)
        return {"enabled": True, "forward_records": records}

    def _sync_decoder_decision(
        self,
        tensor: torch.Tensor | None,
        shape: tuple[int, ...],
        key: str,
    ) -> torch.Tensor:
        """Share one compact rank-0 decoder decision with all TP ranks."""
        if self.world_size == 1:
            return tensor

        if self.rank == 0:
            decision = tensor.contiguous()
        else:
            # Mixed block lengths and changing active batches produce several
            # small decision shapes; cache each follower receive allocation.
            buffer_key = (key, tuple(shape))
            decision = self._decoder_sync_buffers.get(buffer_key)
            if decision is None:
                decision = torch.empty(shape, dtype=torch.long, device="cuda")
                self._decoder_sync_buffers[buffer_key] = decision
        dist.broadcast(decision, src=0)
        return decision


    def stage_block(
        self,
        seqs: list[Sequence],
        tokens_batch: torch.Tensor,
        lora_mask_batch: torch.Tensor | None = None,
        position_offsets_batch: torch.Tensor | None = None,
    ) -> tuple[int, int, list[int], torch.Tensor | None]:
        """Stage one token block at each sequence's current KV frontier."""
        B = len(seqs)
        if tokens_batch.ndim != 2 or tokens_batch.size(0) != B:
            raise ValueError(
                "two-pass tokens must have shape [B, L], "
                f"got {tuple(tokens_batch.shape)} for B={B}"
            )
        L = int(tokens_batch.size(1))
        if L < 1:
            raise ValueError("two-pass forward requires at least one token")
        if (
            position_offsets_batch is not None
            and position_offsets_batch.shape != tokens_batch.shape
        ):
            raise ValueError(
                "position offsets must match the staged token shape: "
                f"{tuple(position_offsets_batch.shape)} != {tuple(tokens_batch.shape)}"
            )
        self.block_workspace.validate_shape(B, L)

        write_starts = [int(seq.num_cached_tokens) for seq in seqs]
        write_ends = [write_start + L for write_start in write_starts]
        total_tokens = B * L
        workspace = self.block_workspace
        input_ids = workspace.input_ids[:total_tokens]
        slot_mapping = workspace.slot_mapping[:total_tokens]
        kv_seqlens = workspace.kv_seqlens[:B]
        block_tables = workspace.block_tables[:B]
        lora_mask = None
        if self.config.gated_lora and lora_mask_batch is not None:
            lora_mask = workspace.lora_mask[:total_tokens]
            lora_mask.copy_(
                lora_mask_batch.reshape(-1).to(
                    device=lora_mask.device,
                    dtype=lora_mask.dtype,
                )
            )

        use_device_tree_metadata = position_offsets_batch is not None
        positions_host: list[int] = []
        slot_mapping_host: list[int] = []
        max_active_blocks = max(len(seq.block_table) for seq in seqs)
        max_block_cols = block_tables.shape[1]
        padded_block_tables: list[list[int]] = []
        for i, (seq, write_start, write_end) in enumerate(
            zip(seqs, write_starts, write_ends)
        ):
            seq_len = len(seq)
            if seq_len < 1:
                raise ValueError(f"Sequence {i} must contain at least one token")
            if not 0 <= write_start <= seq_len:
                raise ValueError(
                    f"Sequence {i} has invalid KV frontier {write_start} "
                    f"for sequence length {seq_len}"
                )

            num_blocks = len(seq.block_table)
            # The scheduler reserves the complete verify span before dispatch.
            blocks_needed = (write_end + self.block_size - 1) // self.block_size
            if num_blocks < blocks_needed:
                raise RuntimeError(
                    "Scheduler did not reserve enough KV blocks for block forward: "
                    f"seq_id={seq.seq_id}, blocks={num_blocks}, "
                    f"required={blocks_needed}, write_start={write_start}, "
                    f"token_len={L}"
                )
            if num_blocks > max_block_cols:
                raise RuntimeError(
                    f"Sequence {i} needs {num_blocks} blocks but the block "
                    f"buffer only has {max_block_cols}; max_model_len="
                    f"{self.config.max_model_len}, write_start={write_start}, L={L}"
                )

            if not use_device_tree_metadata:
                positions_host.extend(range(write_start, write_end))
                slot_mapping_host.extend(
                    seq.block_table[position // self.block_size]
                    * self.block_size
                    + position % self.block_size
                    for position in range(write_start, write_end)
                )
            padded_block_tables.append(
                seq.block_table + [-1] * (max_active_blocks - num_blocks)
            )

        input_ids.copy_(tokens_batch.reshape(-1), non_blocking=True)
        if not use_device_tree_metadata:
            copy_pinned(workspace.positions[:total_tokens], positions_host)
            copy_pinned(slot_mapping, slot_mapping_host)
        # FlashAttention expects the total valid KV length after this block.
        copy_pinned(kv_seqlens, write_ends)
        block_tables.fill_(-1)
        copy_pinned(
            block_tables[:, :max_active_blocks],
            padded_block_tables,
        )

        return B, L, write_ends, lora_mask

    def _run_block(
        self,
        seqs: list[Sequence],
        tokens_batch: torch.Tensor,
        lora_mask_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Stage and execute one block, then advance each sequence's KV frontier."""
        B, L, write_ends, lora_mask = self.stage_block(
            seqs,
            tokens_batch,
            lora_mask_batch,
        )
        logits = self._execute_staged_block(
            batch_size=B,
            block_len=L,
            lora_mask=lora_mask,
        )
        reset_context()
        for seq, write_end in zip(seqs, write_ends):
            seq.num_cached_tokens = write_end

        if self.rank == 0 or self.world_size == 1:
            return logits.view(B, L, logits.size(-1))
        return None

    def _build_tree_page_table(
        self,
        parent_indices: torch.Tensor,
        depths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ancestor/self slot rows for the suffix half of FA3 cascade."""
        batch_size, tree_size = parent_indices.shape
        workspace = self.block_workspace
        if (
            workspace.tree_prefix_seqlens is None
            or workspace.tree_page_table is None
            or workspace.tree_cache_seqlens is None
        ):
            raise RuntimeError("tree FA3 workspace is not configured")
        if tree_size != workspace.tree_size:
            raise ValueError(
                f"tree size {tree_size} does not match workspace "
                f"tree size {workspace.tree_size}"
            )

        page_table = workspace.tree_page_table[
            : batch_size * tree_size
        ].view(batch_size, tree_size, workspace.max_tree_pages)
        cache_seqlens = workspace.tree_cache_seqlens[
            : batch_size * tree_size
        ]
        build_tree_page_table(
            parent_indices,
            depths,
            workspace.block_tables[:batch_size],
            workspace.kv_seqlens[:batch_size],
            workspace.tree_prefix_seqlens[:batch_size],
            workspace.positions[: batch_size * tree_size],
            workspace.slot_mapping[: batch_size * tree_size],
            page_table.view(
                batch_size * tree_size,
                workspace.max_tree_pages,
            ),
            cache_seqlens,
            self.block_size,
        )
        return (
            workspace.tree_prefix_seqlens[:batch_size],
            page_table.view(batch_size * tree_size, workspace.max_tree_pages),
            cache_seqlens,
        )

    @torch.inference_mode()
    def _run_tree(
        self,
        seqs: list[Sequence],
        tokens_batch: torch.Tensor,
        depths_batch: torch.Tensor,
        parent_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run target tree verification with exact visibility metadata."""
        B, Q, write_ends, _ = self.stage_block(
            seqs,
            tokens_batch,
            position_offsets_batch=depths_batch,
        )
        if parent_indices.shape != (B, Q):
            raise ValueError(
                f"parent_indices must have shape {(B, Q)}, "
                f"got {parent_indices.shape}"
            )

        total_tokens = B * Q
        workspace = self.block_workspace
        if not self._use_flash_tree_attention:
            workspace.tree_mask[:B].copy_(
                ancestor_mask_from_parents(parent_indices)
            )
        (
            tree_prefix_seqlens,
            tree_page_table,
            tree_cache_seqlens,
        ) = self._build_tree_page_table(
            parent_indices,
            depths_batch,
        )

        timing_start = None
        if self.config.decode_timing:
            timing_start = torch.cuda.Event(enable_timing=True)
            timing_start.record()

        graph_output = None
        if self.block_graph_runner is not None:
            graph_output = self.block_graph_runner.replay_tree(
                batch_size=B,
                tree_size=Q,
            )
        graph_hit = graph_output is not None
        if graph_hit:
            self.cuda_graph_hits += 1
            logits = self.model.compute_logits(graph_output)
        else:
            set_context(
                ContextMode.TREE_VERIFY,
                slot_mapping=workspace.slot_mapping[:total_tokens],
                block_tables=workspace.block_tables[:B],
                seqlen_q=Q,
                tree_mask=workspace.tree_mask[:B],
                tree_kv_seqlens=tuple(write_ends),
                tree_prefix_seqlens=tree_prefix_seqlens,
                tree_page_table=tree_page_table,
                tree_cache_seqlens=tree_cache_seqlens,
                lora_enabled=False,
            )
            hidden_states = self.model(
                workspace.input_ids[:total_tokens],
                workspace.positions[:total_tokens],
            )
            self.cuda_graph_misses += 1
            logits = self.model.compute_logits(hidden_states)
        if timing_start is not None:
            self._record_decode_forward_timing(
                timing_start,
                batch_size=B,
                block_len=Q,
                graph_hit=graph_hit,
            )
        reset_context()

        for seq, write_end in zip(seqs, write_ends):
            seq.num_cached_tokens = write_end
        if self.rank == 0 or self.world_size == 1:
            return logits.view(B, Q, logits.size(-1))
        return None

    @torch.inference_mode()
    def _compact_tree_kv(
        self,
        seqs: list[Sequence],
        accepted_node_indices: torch.Tensor,
        cache_lengths: torch.Tensor,
        tree_size: int,
    ) -> None:
        """Move the accepted tree path without synchronizing its indices to host."""
        if accepted_node_indices.ndim != 2:
            raise ValueError("accepted tree node indices must have shape [B, L]")
        if accepted_node_indices.size(0) != len(seqs):
            raise ValueError("accepted tree node batch does not match sequences")
        if cache_lengths.shape != (len(seqs),):
            raise ValueError("tree cache lengths must have shape [B]")

        batch_size, max_path_len = accepted_node_indices.shape
        device = accepted_node_indices.device
        workspace = getattr(self, "block_workspace", None)
        if (
            workspace is not None
            and workspace.tree_prefix_seqlens is not None
            and workspace.block_tables.device == device
            and workspace.tree_kv_source is not None
        ):
            num_locations = batch_size * max_path_len
            source = workspace.tree_kv_source[:num_locations]
            destination = workspace.tree_kv_destination[:num_locations]
            build_tree_kv_slots(
                accepted_node_indices,
                cache_lengths,
                workspace.tree_prefix_seqlens[:batch_size],
                workspace.block_tables[:batch_size],
                source,
                destination,
                self.block_size,
            )
        else:
            path_offsets = torch.arange(
                max_path_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            valid = path_offsets < cache_lengths.to(torch.long).unsqueeze(1)
            safe_nodes = torch.where(
                valid,
                accepted_node_indices.to(torch.long),
                path_offsets,
            )
            prefix_host = []
            block_tables_host = []
            max_blocks = max(len(seq.block_table) for seq in seqs)
            for seq in seqs:
                prefix_len = int(seq.num_cached_tokens) - int(tree_size)
                if prefix_len < 0:
                    raise RuntimeError(
                        f"invalid tree KV frontier for seq_id={seq.seq_id}: "
                        f"cached={seq.num_cached_tokens}, tree_size={tree_size}"
                    )
                prefix_host.append(prefix_len)
                block_tables_host.append(
                    seq.block_table + [0] * (max_blocks - len(seq.block_table))
                )
            prefix_lengths = torch.tensor(
                prefix_host,
                dtype=torch.long,
                device=device,
            )
            block_tables = torch.tensor(
                block_tables_host,
                dtype=torch.int32,
                device=device,
            )
            source_positions = prefix_lengths.unsqueeze(1) + safe_nodes
            destination_positions = prefix_lengths.unsqueeze(1) + path_offsets

            def physical_slots(logical_positions: torch.Tensor) -> torch.Tensor:
                block_indices = logical_positions.div(
                    self.block_size,
                    rounding_mode="floor",
                )
                block_offsets = logical_positions.remainder(self.block_size)
                physical_blocks = block_tables.gather(1, block_indices)
                return (
                    physical_blocks.to(torch.long) * self.block_size
                    + block_offsets
                )

            source = physical_slots(source_positions).reshape(-1)
            destination = physical_slots(destination_positions).reshape(-1)
        flat_cache = self.kv_cache.view(
            self.kv_cache.size(0),
            self.kv_cache.size(1),
            -1,
            self.kv_cache.size(-2),
            self.kv_cache.size(-1),
        )
        copy_tree_kv(
            flat_cache,
            destination,
            source,
            path_length=max_path_len,
        )

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        sampling_params: SamplingParams,
    ) -> list[list[int]] | None:
        if is_prefill:
            return self._run_prefill(seqs, sampling_params)

        if self.config.decode_timing:
            self._decode_timing_step += 1
        token_ids_batch = self.two_pass_decoder.run_cycle(
            seqs,
            sampling_params,
        )
        return token_ids_batch if self.rank == 0 or self.world_size == 1 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Capture the block-forward graphs used by both AR and diffusion."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_before, _ = torch.cuda.mem_get_info()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        self.block_graph_runner = BlockCudaGraphRunner(
            self.config,
            self.block_size,
            self.model,
            self.block_workspace,
        )
        self.block_graph_runner.capture()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info()
        self.cuda_graph_memory = {
            "device_footprint_bytes": max(0, int(free_before - free_after)),
            "allocated_delta_bytes": max(
                0, int(torch.cuda.memory_allocated() - allocated_before)
            ),
            "reserved_delta_bytes": max(
                0, int(torch.cuda.memory_reserved() - reserved_before)
            ),
            "num_graphs": (
                len(self.block_graph_runner.graphs)
                + len(self.block_graph_runner.tree_graphs)
            ),
        }
        logger.info("CUDA graph capture footprint: %s", self.cuda_graph_memory)
