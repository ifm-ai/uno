import torch

from nano_vllm_uno.config import Config


class BlockForwardWorkspace:
    """Reusable tensors used to prepare eager and captured block forwards."""

    def __init__(self, config: Config, block_size: int):
        self.block_size = block_size
        self.max_batch_size = config.max_num_seqs
        self.max_block_len = max(
            config.max_diffusion_block_size,
            config.tree_verify_size or 0,
            max(config.cuda_graph_block_sizes),
        )
        # Verify may temporarily stage one complete block beyond the largest
        # committed sequence before rejection rolls the KV frontier back.
        self.max_blocks_per_seq = (
            config.max_model_len
            + self.max_block_len
            + block_size
            - 1
        ) // block_size
        max_total_tokens = self.max_batch_size * self.max_block_len

        self.input_ids = torch.zeros(
            max_total_tokens,
            dtype=torch.int64,
            device="cuda",
        )
        self.positions = torch.zeros(
            max_total_tokens,
            dtype=torch.int64,
            device="cuda",
        )
        self.slot_mapping = torch.zeros(
            max_total_tokens,
            dtype=torch.int32,
            device="cuda",
        )
        self.lora_mask = torch.zeros(
            max_total_tokens,
            dtype=torch.float32,
            device="cuda",
        )
        # Total valid KV length after the staged block has been stored.
        self.kv_seqlens = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device="cuda",
        )
        self.block_tables = torch.full(
            (self.max_batch_size, self.max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self.tree_size = int(config.tree_verify_size or 0)
        if self.tree_size:
            self.max_tree_pages = self.tree_size
            self.tree_prefix_seqlens = torch.ones(
                self.max_batch_size,
                dtype=torch.int32,
                device="cuda",
            )
            self.tree_page_table = torch.zeros(
                (self.max_batch_size * self.tree_size, self.max_tree_pages),
                dtype=torch.int32,
                device="cuda",
            )
            self.tree_cache_seqlens = torch.ones(
                self.max_batch_size * self.tree_size,
                dtype=torch.int32,
                device="cuda",
            )
            self.tree_mask = torch.zeros(
                (self.max_batch_size, self.tree_size, self.tree_size),
                dtype=torch.bool,
                device="cuda",
            )
            max_path_locations = (
                self.max_batch_size * config.max_diffusion_block_size
            )
            self.tree_kv_source = torch.empty(
                max_path_locations,
                dtype=torch.long,
                device="cuda",
            )
            self.tree_kv_destination = torch.empty_like(self.tree_kv_source)
        else:
            self.max_tree_pages = 0
            self.tree_prefix_seqlens = None
            self.tree_page_table = None
            self.tree_cache_seqlens = None
            self.tree_mask = None
            self.tree_kv_source = None
            self.tree_kv_destination = None

    def validate_shape(self, batch_size: int, block_len: int) -> None:
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"two-pass batch size {batch_size} exceeds the engine buffer "
                f"capacity {self.max_batch_size}"
            )
        if block_len > self.max_block_len:
            raise ValueError(
                f"two-pass block size {block_len} exceeds the engine buffer "
                f"capacity {self.max_block_len}; increase max_diffusion_block_size "
                f"or tree_verify_size, or include {block_len} in "
                "cuda_graph_block_sizes when "
                "constructing LLM"
            )
