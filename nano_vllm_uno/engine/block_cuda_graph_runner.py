from collections.abc import Callable
from contextlib import nullcontext

import torch

from nano_vllm_uno.config import Config
from nano_vllm_uno.engine.block_forward_workspace import BlockForwardWorkspace
from nano_vllm_uno.utils.context import (
    ContextMode,
    reset_context,
    set_context,
)


class BlockCudaGraphRunner:
    """Capture and replay fixed-shape CUDA graphs for block forwards."""

    def __init__(
        self,
        config: Config,
        block_size: int,
        model,
        workspace: BlockForwardWorkspace,
    ):
        self.config = config
        self.block_size = block_size
        self.model = model
        self.workspace = workspace
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.tree_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool = None
        self.outputs: torch.Tensor | None = None
        self.hidden_outputs: dict[tuple, torch.Tensor] = {}
        self.compiled_forward: Callable | None = None
        self.batch_sizes: list[int] = []
        self.tree_batch_sizes: list[int] = []

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def _prepare_compiled_forward(self) -> None:
        if not self.config.torch_compile:
            return
        self.compiled_forward = torch.compile(
            torch.no_grad()(self._forward_hidden),
            # KV stores and fused Q/K normalization are intentional graph
            # boundaries: both mutate external storage through custom kernels.
            # Compiling the pure tensor regions between them is faster than one
            # monolithic graph and avoids functionalizing those side effects.
            fullgraph=False,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )

    def graph_key(self, batch_size: int, block_len: int, lora_enabled: bool) -> tuple:
        if self.config.gated_lora:
            return (batch_size, block_len, bool(lora_enabled))
        return (batch_size, block_len)

    @staticmethod
    def capture_batch_sizes(max_batch_size: int) -> list[int]:
        """Return SGLang's speculative-decoding batch buckets."""
        batch_sizes = set(range(1, 9))
        batch_sizes.update(range(10, 33, 2))
        batch_sizes.update(range(40, 65, 4))
        batch_sizes.update(range(72, 257, 8))
        batch_sizes.update(range(272, max_batch_size + 1, 16))
        batch_sizes.add(max_batch_size)
        return sorted(bs for bs in batch_sizes if bs <= max_batch_size)

    @classmethod
    def resolve_capture_batch_sizes(
        cls,
        max_batch_size: int,
        configured_batch_sizes: list[int] | None,
    ) -> list[int]:
        if configured_batch_sizes is None:
            return cls.capture_batch_sizes(max_batch_size)
        return list(configured_batch_sizes)

    def select_graph(
        self,
        batch_size: int,
        block_len: int,
        lora_enabled: bool,
    ) -> tuple[int | None, tuple | None]:
        for graph_batch_size in self.batch_sizes:
            if graph_batch_size < batch_size:
                continue
            key = self.graph_key(graph_batch_size, block_len, lora_enabled)
            if key in self.graphs:
                return graph_batch_size, key
        return None, None

    def _pad_workspace(
        self,
        *,
        batch_size: int,
        graph_batch_size: int,
        block_len: int,
        lora_enabled: bool,
    ) -> tuple[int, int]:
        real_tokens = batch_size * block_len
        graph_tokens = graph_batch_size * block_len
        workspace = self.workspace

        if graph_tokens > real_tokens:
            workspace.input_ids[real_tokens:graph_tokens].fill_(0)
            workspace.positions[real_tokens:graph_tokens].fill_(0)
            workspace.slot_mapping[real_tokens:graph_tokens].fill_(-1)
        if graph_batch_size > batch_size:
            # Dummy rows have an empty prefix plus the current staged block.
            workspace.kv_seqlens[batch_size:graph_batch_size].fill_(block_len)
            # Dummy rows are discarded, but attention kernels still need valid
            # page ids for their static graph shape.
            workspace.block_tables[batch_size:graph_batch_size].zero_()

        if lora_enabled and graph_tokens > real_tokens:
            workspace.lora_mask[real_tokens:graph_tokens].zero_()

        return real_tokens, graph_tokens

    def replay(
        self,
        *,
        batch_size: int,
        block_len: int,
        lora_enabled: bool,
    ) -> torch.Tensor | None:
        graph_batch_size, graph_key = self.select_graph(
            batch_size,
            block_len,
            lora_enabled,
        )
        if graph_batch_size is None or graph_key is None:
            return None

        real_tokens, graph_tokens = self._pad_workspace(
            batch_size=batch_size,
            graph_batch_size=graph_batch_size,
            block_len=block_len,
            lora_enabled=lora_enabled,
        )
        workspace = self.workspace
        set_context(
            ContextMode.BLOCK_DECODE,
            slot_mapping=workspace.slot_mapping[:graph_tokens],
            block_tables=workspace.block_tables[:graph_batch_size],
            kv_seqlens=workspace.kv_seqlens[:graph_batch_size],
            seqlen_q=block_len,
            lora_mask=(
                workspace.lora_mask[:graph_tokens]
                if lora_enabled
                else None
            ),
            lora_enabled=lora_enabled,
        )
        self.graphs[graph_key].replay()
        if self.config.torch_compile:
            return self.hidden_outputs[graph_key][:real_tokens]
        return self.outputs[:real_tokens]

    def replay_tree(
        self,
        *,
        batch_size: int,
        tree_size: int,
    ) -> torch.Tensor | None:
        graph_batch_size = next(
            (
                candidate
                for candidate in self.tree_batch_sizes
                if candidate >= batch_size and candidate in self.tree_graphs
            ),
            None,
        )
        if graph_batch_size is None or batch_size > 512:
            return None

        real_tokens = batch_size * tree_size
        graph_tokens = graph_batch_size * tree_size
        workspace = self.workspace
        if graph_tokens > real_tokens:
            workspace.input_ids[real_tokens:graph_tokens].zero_()
            workspace.positions[real_tokens:graph_tokens].zero_()
            workspace.slot_mapping[real_tokens:graph_tokens].fill_(-1)
            workspace.tree_page_table[real_tokens:graph_tokens, 0].zero_()
            workspace.tree_cache_seqlens[real_tokens:graph_tokens].fill_(1)
        if graph_batch_size > batch_size:
            workspace.block_tables[batch_size:graph_batch_size].zero_()
            workspace.tree_mask[batch_size:graph_batch_size].zero_()
            workspace.tree_prefix_seqlens[batch_size:graph_batch_size].fill_(1)

        set_context(
            ContextMode.TREE_VERIFY,
            slot_mapping=workspace.slot_mapping[:graph_tokens],
            block_tables=workspace.block_tables[:graph_batch_size],
            seqlen_q=tree_size,
            tree_mask=workspace.tree_mask[:graph_batch_size],
            tree_kv_seqlens=(tree_size + 1,) * graph_batch_size,
            tree_prefix_seqlens=workspace.tree_prefix_seqlens[:graph_batch_size],
            tree_page_table=workspace.tree_page_table[:graph_tokens],
            tree_cache_seqlens=workspace.tree_cache_seqlens[:graph_tokens],
            lora_enabled=False,
        )
        self.tree_graphs[graph_batch_size].replay()
        return self.outputs[:real_tokens]

    @torch.inference_mode()
    def capture(self) -> None:
        config = self.config
        graph_block_lens = list(config.cuda_graph_block_sizes)
        max_graph_batch_size = config.max_num_seqs
        graph_batch_sizes = self.resolve_capture_batch_sizes(
            max_graph_batch_size,
            config.cuda_graph_batch_sizes,
        )
        graph_bs_by_len = {
            block_len: graph_batch_sizes
            for block_len in graph_block_lens
        }
        self.batch_sizes = sorted(
            {
                batch_size
                for sizes in graph_bs_by_len.values()
                for batch_size in sizes
            }
        )
        max_batch_size = max(self.batch_sizes)
        max_total_tokens = max(
            max(graph_bs_by_len[block_len]) * block_len
            for block_len in graph_block_lens
        )
        tree_size = int(config.tree_verify_size or 0)
        if tree_size:
            max_total_tokens = max(
                max_total_tokens,
                max(graph_batch_sizes) * tree_size,
            )
        self.workspace.validate_shape(max_batch_size, max(graph_block_lens))
        if tree_size:
            self.workspace.validate_shape(max_batch_size, tree_size)
        if not config.torch_compile:
            hidden_size = config.hf_config.hidden_size
            self.outputs = torch.zeros(max_total_tokens, hidden_size)
        workspace = self.workspace
        # The former graph-only inputs were all initialized to zero. Preserve
        # that capture precondition now that graphs share the general workspace;
        # its block-table sentinel is otherwise -1 for normal runtime staging.
        workspace.input_ids[:max_total_tokens].zero_()
        workspace.positions[:max_total_tokens].zero_()
        workspace.slot_mapping[:max_total_tokens].zero_()
        workspace.lora_mask[:max_total_tokens].zero_()
        workspace.kv_seqlens[:max_batch_size].zero_()
        workspace.block_tables[:max_batch_size].zero_()
        self._prepare_compiled_forward()

        graph_specs = []
        for block_len, batch_sizes in graph_bs_by_len.items():
            lora_variants = (
                (False, True)
                if config.gated_lora and block_len > 1
                else (False,)
            )
            graph_specs.extend(
                (batch_size, block_len, lora_enabled)
                for batch_size in batch_sizes
                for lora_enabled in lora_variants
            )

        compile_context = nullcontext()
        if config.torch_compile:
            import torch._dynamo.config as dynamo_config

            # A separate static graph is expected for every captured shape and
            # LoRA state. Raise Dynamo's per-code-object limit only for capture;
            # no global compiler settings leak into the serving process.
            cache_limit = max(
                dynamo_config.cache_size_limit,
                len(graph_specs) + 4,
            )
            accumulated_limit = max(
                dynamo_config.accumulated_cache_size_limit,
                cache_limit + 4,
            )
            compile_context = dynamo_config.patch(
                cache_size_limit=cache_limit,
                accumulated_cache_size_limit=accumulated_limit,
            )

        with compile_context:
            for batch_size, block_len, lora_enabled in sorted(
                graph_specs,
                key=lambda spec: spec[0] * spec[1],
                reverse=True,
            ):
                total_tokens = batch_size * block_len
                key = self.graph_key(batch_size, block_len, lora_enabled)
                graph = torch.cuda.CUDAGraph()
                # Capture uses an empty prefix, so the post-store length is L.
                workspace.kv_seqlens[:batch_size].fill_(block_len)

                set_context(
                    ContextMode.BLOCK_DECODE,
                    slot_mapping=workspace.slot_mapping[:total_tokens],
                    block_tables=workspace.block_tables[:batch_size],
                    kv_seqlens=workspace.kv_seqlens[:batch_size],
                    seqlen_q=block_len,
                    lora_mask=(
                        workspace.lora_mask[:total_tokens]
                        if lora_enabled and config.gated_lora
                        else None
                    ),
                    lora_enabled=lora_enabled,
                )

                if config.torch_compile:
                    assert self.compiled_forward is not None
                    # Load lazy custom kernels and initialize any LoRA overlap
                    # streams before Dynamo tracing and CUDA capture.
                    compiled_output = self._forward_hidden(
                        workspace.input_ids[:total_tokens],
                        workspace.positions[:total_tokens],
                    )
                    torch.cuda.synchronize()
                    # The first call compiles/autotunes this static context; the
                    # second confirms it is stable before outer graph capture.
                    for _ in range(2):
                        compiled_output = self.compiled_forward(
                            workspace.input_ids[:total_tokens],
                            workspace.positions[:total_tokens],
                        )
                        torch.cuda.synchronize()
                else:
                    self.outputs[:total_tokens] = self.model(
                        workspace.input_ids[:total_tokens],
                        workspace.positions[:total_tokens],
                    )

                with torch.cuda.graph(graph, self.graph_pool):
                    if config.torch_compile:
                        compiled_output = self.compiled_forward(
                            workspace.input_ids[:total_tokens],
                            workspace.positions[:total_tokens],
                        )
                    else:
                        self.outputs[:total_tokens] = self.model(
                            workspace.input_ids[:total_tokens],
                            workspace.positions[:total_tokens],
                        )

                if config.torch_compile:
                    self.hidden_outputs[key] = compiled_output

                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[key] = graph
                torch.cuda.synchronize()
                reset_context()

        if tree_size:
            self.tree_batch_sizes = list(graph_batch_sizes)
            total_tree_rows = max_batch_size * tree_size
            workspace.tree_page_table[:total_tree_rows, 0].zero_()
            workspace.tree_cache_seqlens[:total_tree_rows].fill_(1)
            workspace.tree_prefix_seqlens[:max_batch_size].fill_(1)
            workspace.tree_mask[:max_batch_size].zero_()
            for batch_size in sorted(graph_batch_sizes, reverse=True):
                total_tokens = batch_size * tree_size
                graph = torch.cuda.CUDAGraph()
                set_context(
                    ContextMode.TREE_VERIFY,
                    slot_mapping=workspace.slot_mapping[:total_tokens],
                    block_tables=workspace.block_tables[:batch_size],
                    seqlen_q=tree_size,
                    tree_mask=workspace.tree_mask[:batch_size],
                    tree_kv_seqlens=(tree_size + 1,) * batch_size,
                    tree_prefix_seqlens=workspace.tree_prefix_seqlens[:batch_size],
                    tree_page_table=workspace.tree_page_table[:total_tokens],
                    tree_cache_seqlens=workspace.tree_cache_seqlens[:total_tokens],
                    lora_enabled=False,
                )
                self.outputs[:total_tokens] = self.model(
                    workspace.input_ids[:total_tokens],
                    workspace.positions[:total_tokens],
                )
                with torch.cuda.graph(graph, self.graph_pool):
                    self.outputs[:total_tokens] = self.model(
                        workspace.input_ids[:total_tokens],
                        workspace.positions[:total_tokens],
                    )
                self.tree_graphs[batch_size] = graph
                torch.cuda.synchronize()
                reset_context()

    def close(self) -> None:
        self.graphs.clear()
        self.tree_graphs.clear()
        self.hidden_outputs.clear()
        self.compiled_forward = None
        self.outputs = None
        self.graph_pool = None
