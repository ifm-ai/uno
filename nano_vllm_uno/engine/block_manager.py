from collections import deque
import xxhash
import numpy as np

from nano_vllm_uno.engine.sequence import Sequence

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _cacheable_len(self, seq: Sequence) -> int:
        """Return tokens that may be treated as persistent KV/prefix cache."""
        return max(0, len(seq) - 1)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> Block:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # Remove retained prefix-cache metadata before repurposing this KV page.
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def plan_prefix_cache_and_check_capacity(self, seq: Sequence) -> int | None:
        """Return reusable prefix blocks if the request fits, otherwise None.

        This is the cache-aware equivalent of can_allocate() in nano-vLLM.
        """
        h = -1
        num_cached_blocks = 0
        num_required_blocks = seq.num_blocks
        cacheable_len = self._cacheable_len(seq)

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            block_end = min((i + 1) * self.block_size, len(seq))
            if len(token_ids) != self.block_size or block_end > cacheable_len:
                break
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # Only used hits reduce demand; free hits leave the pool when reactivated.
            if block_id in self.used_block_ids:
                num_required_blocks -= 1

        if len(self.free_block_ids) < num_required_blocks:
            return None
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """Attach the planned cached prefix, then allocate its uncached suffix."""
        assert not seq.block_table
        assert 0 <= num_cached_blocks <= seq.num_blocks
        h = -1

        # Existing prefix pages already contain valid KV; only acquire references.
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            assert block.token_ids == token_ids
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # Reactivate the matching free page without resetting its cached KV.
                assert block.ref_count == 0 and block.hash != -1
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        for _ in range(num_cached_blocks, seq.num_blocks):
            block = self._allocate_block()
            seq.block_table.append(block.block_id)

        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def publish_computed_prefix_blocks(self, seq: Sequence) -> None:
        """Publish completed prefix-cache pages after a forward.

        Equivalent to hash_blocks() in original nano-vLLM, but uses the final
        num_cached_tokens because multi-token draft/verify may roll back KV.
        """
        # Unlike one-token AR, the stable token count is unknown until verify
        # and rollback finish, so the decoder's final cached length is authoritative.
        end = min(seq.num_cached_tokens // self.block_size, seq.num_blocks)
        start = end
        while start > 0 and self.blocks[seq.block_table[start - 1]].hash == -1:
            start -= 1
        if start == end:
            return

        # Published pages form a contiguous prefix, so only hash the new suffix.
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block = self.blocks[seq.block_table[i]]
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def num_blocks_needed_for_forward(
        self,
        seq: Sequence,
        num_forward_tokens: int = 1,
    ) -> int:
        """Return additional KV blocks needed for the complete forward span."""
        if num_forward_tokens <= 0:
            return 0
        assert seq.block_table, "Forward sizing requires an admitted sequence"
        total_tokens = len(seq) + num_forward_tokens
        blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
        return max(0, blocks_needed - len(seq.block_table))

    def has_enough_blocks_for_forward(
        self,
        seq: Sequence,
        num_forward_tokens: int = 1,
    ) -> bool:
        return len(self.free_block_ids) >= self.num_blocks_needed_for_forward(
            seq,
            num_forward_tokens,
        )

    def reserve_blocks_for_forward(
        self,
        seq: Sequence,
        num_forward_tokens: int,
    ) -> int:
        """Reserve additional KV pages for the complete draft/verify span.

        Reserved pages remain attached until the sequence is preempted or ends.
        """
        additional = self.num_blocks_needed_for_forward(
            seq,
            num_forward_tokens,
        )
        if additional > len(self.free_block_ids):
            raise RuntimeError(
                f"Cannot reserve {additional} KV blocks for seq_id={seq.seq_id}; "
                f"only {len(self.free_block_ids)} are free"
            )
        for _ in range(additional):
            block = self._allocate_block()
            seq.block_table.append(block.block_id)
        return additional
