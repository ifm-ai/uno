from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Tuple

import torch
from torch import Tensor

from nano_vllm_uno.engine.draft_tree import (
    build_draft_tree_batch,
    walk_tree,
)
from nano_vllm_uno.engine.fused_verify_kernel import fused_spec_verify_from_logits
from nano_vllm_uno.engine.noise import build_draft_batch
from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.layers.sampler import (
    Sampler,
    build_sparse_top_k_probs,
    sample_from_probs,
)
from nano_vllm_uno.sampling_params import SamplingParams


class BlockForwardFn(Protocol):
    def __call__(
        self,
        seqs: List[Sequence],
        tokens_batch: Tensor,
        lora_mask_batch: Optional[Tensor] = None,
    ) -> Optional[Tensor]: ...


class TreeForwardFn(Protocol):
    def __call__(
        self,
        seqs: List[Sequence],
        tokens_batch: Tensor,
        depths_batch: Tensor,
        parent_indices: Tensor,
    ) -> Optional[Tensor]: ...


class CompactTreeKvFn(Protocol):
    def __call__(
        self,
        seqs: List[Sequence],
        accepted_node_indices: Tensor,
        cache_lengths: Tensor,
        tree_size: int,
    ) -> None: ...


SyncDecisionFn = Callable[[Optional[Tensor], Tuple[int, ...], str], Tensor]


@torch.compile(dynamic=True)
def _sparse_verify_tensors(
    clean_ids: Tensor,
    clean_probs: Tensor,
    draft_ids: Tensor,
    draft_probs: Tensor,
    spec_vals: Tensor,
    accept_random: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return accept flags and correction probabilities on clean support."""
    p_spec = torch.where(
        clean_ids.eq(spec_vals.unsqueeze(1)),
        clean_probs,
        torch.zeros_like(clean_probs),
    ).sum(dim=1)
    q_spec = torch.where(
        draft_ids.eq(spec_vals.unsqueeze(1)),
        draft_probs,
        torch.zeros_like(draft_probs),
    ).sum(dim=1)
    ratio = torch.where(q_spec > 0, p_spec / q_spec, torch.zeros_like(p_spec))
    accepted = accept_random.lt(torch.clamp(ratio, max=1.0))

    q_on_clean = torch.where(
        clean_ids.unsqueeze(2).eq(draft_ids.unsqueeze(1)),
        draft_probs.unsqueeze(1),
        torch.zeros((), dtype=draft_probs.dtype, device=draft_probs.device),
    ).sum(dim=2)
    correction_probs = torch.clamp(clean_probs - q_on_clean, min=0.0)
    correction_sum = correction_probs.sum(dim=1, keepdim=True)
    correction_probs = torch.where(
        correction_sum > 0,
        correction_probs / correction_sum.clamp_min(1e-12),
        clean_probs,
    )
    return accepted, correction_probs


class TwoPassDecoder:
    """Two-pass decoder with separate draft and verify forwards.

    For block length L:
      draft:  [seed, noise_1, ..., noise_{L-1}] -> sample clean + draft specs
      verify: [clean, spec_1, ..., spec_{L-1}] -> accept/reject left-to-right

    The draft forward is seeded because it needs p(. | last committed token).
    The verify forward appends clean/spec tokens after the current committed KV.
    """

    def __init__(
        self,
        run_block: BlockForwardFn,
        eos_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        device: Optional[torch.device] = None,
        sampler: Optional[Sampler] = None,
        is_driver: bool = True,
        sync_decision: Optional[SyncDecisionFn] = None,
        gated_lora: bool = False,
        run_tree: Optional[TreeForwardFn] = None,
        compact_tree_kv: Optional[CompactTreeKvFn] = None,
        tree_verify_size: Optional[int] = None,
        tree_candidate_top_k: int = 16,
    ) -> None:
        """Store model callbacks, token ids, and device state."""
        self.run_block = run_block
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.vocab_size = int(vocab_size)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = sampler or Sampler()
        self.is_driver = bool(is_driver)
        self.sync_decision = sync_decision
        self.gated_lora = bool(gated_lora)
        self.run_tree = run_tree
        self.compact_tree_kv = compact_tree_kv
        self.tree_verify_size = tree_verify_size
        self.tree_candidate_top_k = tree_candidate_top_k
        if self.tree_verify_size is not None and (
            self.run_tree is None or self.compact_tree_kv is None
        ):
            raise ValueError(
                "tree verification requires run_tree and compact_tree_kv callbacks"
            )
        self._fused_verify_seed_counter = 0
        self._tree_buffers: dict[str, Tensor] = {}

    @torch.inference_mode()
    def run_cycle(
        self,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
    ) -> List[List[int]]:
        """Run one draft, verify, and commit cycle for the active sequences."""
        if not seqs:
            return []

        accepted: List[List[int]] = [[] for _ in seqs]
        block_len = int(sampling_params.diffusion_block_size)
        proposal_batch, draft_logits, draft_distribution = self.draft_stage(
            seqs,
            block_len,
            sampling_params,
        )
        if self.tree_verify_size is None:
            committed_driver = self.verify_stage(
                seqs,
                proposal_batch,
                draft_logits,
                draft_distribution,
                sampling_params,
            )
        else:
            committed_driver = self.tree_verify_stage(
                seqs,
                proposal_batch[:, 0],
                draft_logits,
                block_len,
                sampling_params,
            )

        # Both paths publish committed tokens here. Tree construction and
        # accepted-path KV compaction stay on GPU until this transfer.
        if self.is_driver:
            committed_rows = self._copy_committed_rows_to_host(
                committed_driver
            )
            self._apply_committed_rows(
                seqs,
                accepted,
                committed_rows,
                block_len,
            )
        return accepted

    @staticmethod
    def _copy_committed_rows_to_host(
        committed_driver: Tensor,
    ) -> List[List[int]]:
        """Synchronize the compact committed payload to the driver host."""
        return committed_driver.detach().cpu().tolist()

    @staticmethod
    def _apply_committed_rows(
        seqs: List[Sequence],
        accepted: List[List[int]],
        committed_rows: List[List[int]],
        block_len: int,
    ) -> None:
        """Publish one synchronized committed payload into Python sequences."""
        for seq, accepted_tokens, committed_row in zip(
            seqs,
            accepted,
            committed_rows,
        ):
            commit_length = int(committed_row[-1])
            committed = committed_row[:commit_length]
            if seq.max_completion_tokens is not None:
                remaining = max(
                    0,
                    seq.max_completion_tokens - seq.num_completion_tokens,
                )
                committed = committed[:remaining]
                commit_length = len(committed)
            lookahead_survived = commit_length > block_len
            if committed:
                seq.extend_tokens(committed)
                accepted_tokens.extend(committed)
            stats = seq.stats
            stats["forwards"] += 2
            stats["accepts"] += len(committed)
            stats["lookaheads"] += int(lookahead_survived)
            target_cached = max(0, len(seq) - 1)
            seq.rollback_kv_to(target_cached)
            if seq.num_cached_tokens != target_cached:
                raise RuntimeError(
                    f"Verify invariant violated: len={len(seq)} "
                    f"num_cached={seq.num_cached_tokens}"
                )

    def draft_stage(
        self,
        seqs: List[Sequence],
        block_len: int,
        sampling_params: SamplingParams,
    ) -> Tuple[
        Tensor,
        Optional[Tensor],
        Optional[Tuple[Tensor, Tensor]],
    ]:
        """Draft one proposal block and restore the committed KV frontier.

        Input:  [uncached seed, noise_1, ..., noise_(L-1)]
        Output: [clean, proposal_1, ..., proposal_(L-1)]
        """
        B = len(seqs)
        draft_batch = self._sync_driver_decision(
            build_draft_batch(
                seqs,
                block_len,
                sampling_params,
                vocab_size=self.vocab_size,
                device=self.device,
            )
            if self.is_driver
            else None,
            (B, block_len),
            "draft_input",
        )

        # Gated LoRA applies only to the noise rows, never to the causal seed.
        lora_mask_batch = None
        if self.gated_lora and block_len > 1:
            lora_mask_batch = torch.ones_like(draft_batch, dtype=torch.float32)
            lora_mask_batch[:, 0] = 0.0
        logits = self.run_block(
            seqs,
            draft_batch,
            lora_mask_batch=lora_mask_batch,
        )

        # proposal = 1 clean + L-1 draft
        if self.is_driver:
            if self.tree_verify_size is None:
                proposal_batch, draft_distribution = (
                    self._sample_proposal_tokens(
                        seqs,
                        logits,
                        sampling_params,
                    )
                )
            else:
                # The root is target-sampled from the base-model seed row.
                # Other proposal entries are unused: the tree is selected
                # directly from the future-position draft logits.
                proposal_batch = self._tree_buffer(
                    "proposal_batch",
                    (B, block_len),
                    torch.long,
                    logits.device,
                )
                proposal_batch.zero_()
                root_tokens, _ = self.sampler(logits[:, 0, :], sampling_params)
                proposal_batch[:, 0] = root_tokens
                draft_distribution = None
            draft_logits = logits[:, 1:, :]
        else:
            proposal_batch = None
            draft_logits = None
            draft_distribution = None
        proposal_batch = self._sync_driver_decision(
            proposal_batch,
            (B, block_len),
            "proposal_tokens",
        )

        # Discard the noise KV. The seed remains cached, so the logical sequence
        # and KV frontier are equal before verification.
        for seq in seqs:
            seq.rollback_kv_to(len(seq))
            if len(seq) != seq.num_cached_tokens:
                raise RuntimeError(
                    f"Draft invariant violated: len={len(seq)} "
                    f"num_cached={seq.num_cached_tokens}"
                )
        return proposal_batch, draft_logits, draft_distribution

    def verify_stage(
        self,
        seqs: List[Sequence],
        proposal_batch: Tensor,
        draft_logits: Optional[Tensor],
        draft_distribution: Optional[Tuple[Tensor, Tensor]],
        sampling_params: SamplingParams,
    ) -> Optional[Tensor]:
        """Verify proposals and return each sequence's committed-token payload.

        Input:  [clean, proposal_1, ..., proposal_(L-1)]
        Output: [accepted prefix, correction or lookahead, commit length]
        """
        _, block_len = proposal_batch.shape
        logits = self.run_block(seqs, proposal_batch)
        if not self.is_driver:
            return None

        verify_logits = logits[:, : block_len - 1, :]
        lookahead_tokens, _ = self.sampler(
            logits[:, -1, :],
            sampling_params,
        )
        return self._verify_batch(
            proposal_batch,
            verify_logits,
            draft_logits,
            sampling_params,
            lookahead_tokens,
            draft_distribution=draft_distribution,
        )

    def tree_verify_stage(
        self,
        seqs: List[Sequence],
        root_tokens: Tensor,
        draft_logits: Optional[Tensor],
        block_len: int,
        sampling_params: SamplingParams,
    ) -> Optional[Tensor]:
        """Build, verify, walk, and compact one fixed-size target draft tree."""
        batch_size = len(seqs)
        tree_size = self.tree_verify_size
        if self.is_driver:
            if draft_logits is None:
                raise RuntimeError("driver draft logits are unavailable")
            tree_batch = build_draft_tree_batch(
                root_tokens,
                draft_logits,
                max_nodes=tree_size,
                candidate_top_k=self.tree_candidate_top_k,
                temperature=sampling_params.temperature,
                workspace=self._tree_buffers,
            )
            tree_token_ids = tree_batch.token_ids
            parent_indices = tree_batch.parent_indices
            depths = tree_batch.depths
        else:
            tree_token_ids = None
            parent_indices = None
            depths = None

        tree_token_ids = self._sync_driver_decision(
            tree_token_ids,
            (batch_size, tree_size),
            "tree_tokens",
        )
        depths = self._sync_driver_decision(
            depths,
            (batch_size, tree_size),
            "tree_depths",
        )
        parent_indices = self._sync_driver_decision(
            parent_indices,
            (batch_size, tree_size),
            "tree_parents",
        )
        # get the target distributions for proposal nodes
        logits = self.run_tree(
            seqs,
            tree_token_ids,
            depths,
            parent_indices,
        )

        if self.is_driver:
            target_tokens, _ = self.sampler(
                logits.reshape(-1, logits.size(-1)),
                sampling_params,
            )
            target_tokens = target_tokens.view(batch_size, tree_size)
            walk_out = (
                self._tree_buffer(
                    "walk_committed",
                    (batch_size, block_len + 1),
                    tree_token_ids.dtype,
                    tree_token_ids.device,
                ),
                self._tree_buffer(
                    "walk_accepted_nodes",
                    (batch_size, block_len),
                    torch.long,
                    tree_token_ids.device,
                ),
                self._tree_buffer(
                    "walk_lengths",
                    (batch_size,),
                    torch.long,
                    tree_token_ids.device,
                ),
            )
            committed, accepted_nodes, lengths = walk_tree(
                tree_token_ids,
                parent_indices,
                target_tokens,
                max_depth=block_len - 1,
                pad_token_id=self.pad_token_id,
                out=walk_out,
            )
            lengths = self._truncate_committed_batch(
                committed,
                lengths,
                sampling_params,
            )
            cache_lengths = self._tree_buffer(
                "cache_lengths",
                (batch_size,),
                torch.long,
                committed.device,
            )
            torch.sub(lengths, 1, out=cache_lengths)
            cache_lengths.clamp_(min=0, max=block_len)
            payload = self._tree_buffer(
                "payload",
                (batch_size, committed.size(1) + 1),
                committed.dtype,
                committed.device,
            )
            payload[:, :-1] = committed
            payload[:, -1] = lengths
        else:
            accepted_nodes = None
            cache_lengths = None
            payload = None

        accepted_nodes = self._sync_driver_decision(
            accepted_nodes,
            (batch_size, block_len),
            "tree_accepted_nodes",
        )
        cache_lengths = self._sync_driver_decision(
            cache_lengths,
            (batch_size,),
            "tree_cache_lengths",
        )
        self.compact_tree_kv(
            seqs,
            accepted_nodes,
            cache_lengths,
            tree_size,
        )
        return payload

    def _tree_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        tensor = self._tree_buffers.get(name)
        if (
            tensor is None
            or tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self._tree_buffers[name] = tensor
        return tensor

    def _sync_driver_decision(
        self,
        tensor: Optional[Tensor],
        shape: Tuple[int, ...],
        key: str,
    ) -> Tensor:
        """Broadcast one compact rank-0 decoder decision to every TP rank."""
        if self.sync_decision is None:
            if tensor is None:
                raise RuntimeError(f"Driver decision {key!r} is unavailable")
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"Driver decision {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(shape)}"
                )
            return tensor
        return self.sync_decision(tensor, tuple(shape), key)

    def _truncate_committed_batch(
        self,
        committed_batch: Tensor,
        lengths: Tensor,
        sampling_params: SamplingParams,
    ) -> Tensor:
        """Vectorized EOS/stop-token truncation for committed-token rows.

        We truncate tokens after EOS. we don't truncate tokens after length limit.
        """
        if sampling_params.ignore_eos:
            return lengths

        positions = torch.arange(
            committed_batch.size(1),
            device=committed_batch.device,
        ).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        stop_mask = committed_batch.eq(self.eos_token_id)
        for stop_token_id in sampling_params.stop_token_ids or []:
            stop_mask |= committed_batch.eq(int(stop_token_id))
        stop_mask &= valid

        eos_hit = stop_mask.any(dim=1)
        first_stop = torch.argmax(stop_mask.to(torch.int64), dim=1)
        return torch.where(eos_hit, first_stop + 1, lengths)

    @torch.inference_mode()
    def _sample_proposal_tokens(
        self,
        seqs: List[Sequence],
        logits: Tensor,
        sampling_params: SamplingParams,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Sample proposals and retain the sparse draft q used for verification."""
        B, L, _ = logits.shape
        proposal_tokens = torch.empty(
            (B, L),
            dtype=torch.long,
            device=logits.device,
        )
        draft_distribution = None
        # first clean token
        clean_logits = logits[:, 0, :]
        clean_tokens, _ = self.sampler(clean_logits, sampling_params)
        proposal_tokens[:, 0] = clean_tokens
        if L > 1:  # Draft tokens; block-one AR skips to return.
            spec_logits = logits[:, 1:, :].reshape(-1, logits.size(-1))
            spec_tokens, draft_distribution = self.sampler(
                spec_logits,
                sampling_params,
            )
            proposal_tokens[:, 1:] = spec_tokens.reshape(B, L - 1)
        return proposal_tokens, draft_distribution

    def _build_committed_payload(
        self,
        proposal_batch: Tensor,
        accepted_flags: Tensor,
        correction_tokens: Tensor,
        sampling_params: SamplingParams,
        lookahead_tokens: Tensor,
    ) -> Tensor:
        """Build [committed tokens..., length] rows for one host transfer."""
        B, L = proposal_batch.shape
        device = proposal_batch.device
        n_spec = L - 1
        max_commit_length = L + 1
        accepted_flags = accepted_flags.to(device=device, dtype=torch.bool)
        correction_tokens = correction_tokens.to(device=device, dtype=proposal_batch.dtype)
        reject_mask = ~accepted_flags
        rejected = reject_mask.any(dim=1)
        first_reject = torch.zeros((B,), dtype=torch.long, device=device)
        payload = torch.empty(
            (B, max_commit_length + 1),
            dtype=proposal_batch.dtype,
            device=device,
        )
        committed = payload[:, :max_commit_length]
        committed[:, :L] = proposal_batch

        if n_spec > 0:
            first_reject = torch.argmax(reject_mask.to(torch.int64), dim=1)
            reject_cols = first_reject + 1
            replacement = correction_tokens.gather(
                1,
                first_reject.unsqueeze(1),
            ).squeeze(1)
            original = committed.gather(1, reject_cols.unsqueeze(1)).squeeze(1)
            committed.scatter_(
                1,
                reject_cols.unsqueeze(1),
                torch.where(rejected, replacement, original).unsqueeze(1),
            )

        full_length = torch.full((B,), L, dtype=torch.long, device=device)
        lengths = torch.where(rejected, first_reject + 2, full_length)

        lookahead_tokens = lookahead_tokens.to(
            device=device,
            dtype=proposal_batch.dtype,
        ).reshape(B)
        committed[:, L] = lookahead_tokens
        lengths = lengths + (~rejected).to(lengths.dtype)

        lengths = self._truncate_committed_batch(
            committed,
            lengths,
            sampling_params,
        )
        payload[:, -1] = lengths
        return payload

    def _verify_filtered(
        self,
        proposal_batch: Tensor,
        verify_logits: Tensor,
        sampling_params: SamplingParams,
        lookahead_tokens: Tensor,
        draft_distribution: Tuple[Tensor, Tensor],
    ) -> Tensor:
        """Apply top-k followed by top-p to verifier p/q distributions."""
        B = int(proposal_batch.size(0))
        L = int(proposal_batch.size(1))
        if L <= 1:
            return self._build_committed_payload(
                proposal_batch,
                proposal_batch[:, 1:].bool(),
                proposal_batch[:, 1:],
                sampling_params,
                lookahead_tokens,
            )

        num_spec = L - 1
        clean_flat = verify_logits[:, :num_spec, :].reshape(-1, verify_logits.size(-1))
        spec_vals = proposal_batch[:, 1:L].reshape(-1).to(verify_logits.device)

        temperature = sampling_params.temperature
        top_k = int(sampling_params.top_k)
        top_p = sampling_params.top_p
        clean_ids, clean_probs = build_sparse_top_k_probs(clean_flat, temperature, top_k, top_p)
        draft_ids, draft_probs = draft_distribution

        accept_random = torch.rand(
            clean_probs.size(0),
            dtype=clean_probs.dtype,
            device=clean_probs.device,
        )
        verify_tensors = _sparse_verify_tensors
        accepted_flat, correction_probs = verify_tensors(
            clean_ids,
            clean_probs,
            draft_ids,
            draft_probs,
            spec_vals,
            accept_random,
        )
        correction_offsets = sample_from_probs(correction_probs).unsqueeze(1)
        corr_flat = clean_ids.gather(1, correction_offsets).squeeze(1)

        return self._build_committed_payload(
            proposal_batch,
            accepted_flat.reshape(B, num_spec),
            corr_flat.reshape(B, num_spec),
            sampling_params,
            lookahead_tokens,
        )

    def _verify_batch(
        self,
        proposal_batch: Tensor,
        verify_logits: Tensor,
        draft_logits: Tensor,
        sampling_params: SamplingParams,
        lookahead_tokens: Tensor,
        draft_distribution: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Dispatch the batch to its greedy, filtered, or fused verifier."""
        L = int(proposal_batch.size(1))
        if L <= 1: # AR
            return self._build_committed_payload(
                proposal_batch,
                proposal_batch[:, 1:].bool(),
                proposal_batch[:, 1:],
                sampling_params,
                lookahead_tokens,
            )
        temperature = sampling_params.temperature
        if temperature <= 0.0:
            return self._verify_greedy(
                proposal_batch,
                verify_logits,
                sampling_params,
                lookahead_tokens,
            )
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        use_top_k = top_k is not None and 0 < int(top_k) < verify_logits.size(-1)
        use_top_p = top_p is not None and 0.0 < float(top_p) < 1.0
        if use_top_k:
            if draft_distribution is None:
                raise RuntimeError(
                    "Filtered draft verification requires the sampled sparse q"
                )
            return self._verify_filtered(
                proposal_batch,
                verify_logits,
                sampling_params,
                lookahead_tokens,
                draft_distribution,
            )
        if use_top_p:
            raise ValueError("top_k must be smaller than the model vocabulary")
        return self._verify_unfiltered_fused(
            proposal_batch,
            verify_logits,
            draft_logits,
            sampling_params,
            lookahead_tokens,
        )

    def _verify_unfiltered_fused(
        self,
        proposal_batch: Tensor,
        verify_logits: Tensor,
        draft_logits: Tensor,
        sampling_params: SamplingParams,
        lookahead_tokens: Tensor,
    ) -> Tensor:
        """Verify unfiltered stochastic proposals with the fused kernel."""
        B = int(proposal_batch.size(0))
        L = int(proposal_batch.size(1))
        clean_flat = verify_logits[:, : L - 1, :].reshape(-1, verify_logits.size(-1))
        draft_flat = draft_logits[:, : L - 1, :].reshape(
            -1,
            verify_logits.size(-1),
        )
        spec_vals = proposal_batch[:, 1:L].reshape(-1).to(verify_logits.device)
        self._fused_verify_seed_counter += 1
        accepted_flat, corr_flat = fused_spec_verify_from_logits(
            clean_flat,
            draft_flat,
            spec_vals,
            temperature=sampling_params.temperature,
            alpha=1.0,
            gumbel_seed=self._fused_verify_seed_counter,
        )
        return self._build_committed_payload(
            proposal_batch,
            accepted_flat.reshape(B, L - 1).bool(),
            corr_flat.reshape(B, L - 1),
            sampling_params,
            lookahead_tokens,
        )

    def _verify_greedy(
        self,
        proposal_batch: Tensor,
        verify_logits: Tensor,
        sampling_params: SamplingParams,
        lookahead_tokens: Tensor,
    ) -> Tensor:
        """Run greedy verification by matching proposals against verifier argmax."""
        L = int(proposal_batch.size(1))
        target_tokens = (
            torch.argmax(verify_logits[:, : L - 1, :], dim=-1)
            if L > 1
            else proposal_batch[:, 1:]
        )
        accepted_flags = proposal_batch[:, 1:L].to(target_tokens.device) == target_tokens
        return self._build_committed_payload(
            proposal_batch,
            accepted_flags,
            target_tokens,
            sampling_params,
            lookahead_tokens,
        )
