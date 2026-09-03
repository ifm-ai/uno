from collections import deque

from nano_vllm_uno.config import Config
from nano_vllm_uno.engine.sequence import Sequence, SequenceStatus
from nano_vllm_uno.engine.block_manager import BlockManager
from nano_vllm_uno.sampling_params import SamplingParams


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.fail_on_preemption = config.fail_on_preemption
        self.tree_verify_size = config.tree_verify_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0
        self.num_prefill_batches = 0
        self.prefill_batch_sizes: list[int] = []
        self.decode_batch_sizes: list[int] = []
    
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(
        self,
        sampling_params: SamplingParams,
    ) -> tuple[list[Sequence], bool]:
        # Prefill is intentionally unchunked.
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        available_prefill_slots = max(0, self.max_num_seqs - len(self.running))
        while self.waiting and num_seqs < available_prefill_slots:
            seq = self.waiting[0]
            num_cached_blocks = (
                self.block_manager.plan_prefix_cache_and_check_capacity(seq)
            )
            if num_cached_blocks is None:
                break
            # Prefix-cache hits do not contribute query tokens to this prefill.
            num_uncached_tokens = len(seq) - num_cached_blocks * self.block_manager.block_size
            if num_uncached_tokens > self.max_num_batched_tokens:
                raise ValueError(
                    "Prompt exceeds the non-chunked prefill budget: "
                    f"{num_uncached_tokens} uncached tokens > "
                    f"max_num_batched_tokens={self.max_num_batched_tokens}"
                )
            if num_batched_tokens + num_uncached_tokens > self.max_num_batched_tokens:
                break
            num_seqs += 1
            self.block_manager.allocate(seq, num_cached_blocks)
            num_batched_tokens += num_uncached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.num_prefill_batches += 1
            self.prefill_batch_sizes.append(len(scheduled_seqs))
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            block_len = sampling_params.diffusion_block_size
            # max scratch capacity for 1 decode step,
            #  block_len for draft, tree_verify_size optionally for verify with tree
            forward_tokens = max(block_len, self.tree_verify_size or 0)
            while not self.block_manager.has_enough_blocks_for_forward(
                seq,
                forward_tokens,
            ):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.reserve_blocks_for_forward(seq, forward_tokens)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.decode_batch_sizes.append(len(scheduled_seqs))
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        if self.fail_on_preemption:
            raise RuntimeError(
                "Scheduler preemption disabled for this run: "
                f"seq_id={seq.seq_id}, len={len(seq)}, "
                f"free_blocks={len(self.block_manager.free_block_ids)}"
            )
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def reset_stats(self):
        self.num_preemptions = 0
        self.num_prefill_batches = 0
        self.prefill_batch_sizes.clear()
        self.decode_batch_sizes.clear()

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids_batch: list[list[int]],
        sampling_params: SamplingParams,
    ) -> None:
        """Finish sequences after a decoder has appended multiple tokens."""
        stop_token_ids = set(sampling_params.stop_token_ids or [])
        for seq, token_ids in zip(seqs, token_ids_batch):
            # Check if EOS was generated (in any of the new tokens)
            if not sampling_params.ignore_eos and (
                self.eos in token_ids or any(tok in stop_token_ids for tok in token_ids)
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            # Check if max_tokens reached
            elif seq.num_completion_tokens >= (
                seq.max_completion_tokens
                if seq.max_completion_tokens is not None
                else sampling_params.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
