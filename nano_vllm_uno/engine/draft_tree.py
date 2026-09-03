from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from nano_vllm_uno.engine.tree_builder import (
    build_candidate_log_probs_cuda,
    build_tree_from_candidates_cuda,
)
from nano_vllm_uno.engine.tree_walk import walk_tree as walk_tree_kernel


@dataclass
class _TreeNode:
    token_id: int
    depth: int
    parent: int
    log_mass: float
    children: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftTree:
    """One prefix-closed draft tree in parent-before-child order."""

    token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]
    depths: tuple[int, ...]
    log_masses: tuple[float, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class DraftTreeBatch:
    """Fixed-shape tensors consumed by one target tree forward."""

    token_ids: Tensor
    parent_indices: Tensor
    depths: Tensor
    log_masses: Tensor


def build_tree_from_candidates(
    root_token: int,
    top_tokens: Sequence[Sequence[int]],
    top_log_probs: Sequence[Sequence[float]],
    max_nodes: int,
) -> DraftTree:
    """Build a fixed-budget best-first tree."""
    num_depths = len(top_tokens)
    if len(top_log_probs) != num_depths:
        raise ValueError("top_tokens and top_log_probs must have the same depth")
    if max_nodes < 1:
        raise ValueError("max_nodes must include at least the root")
    for depth, (tokens, log_probs) in enumerate(zip(top_tokens, top_log_probs)):
        if not tokens or len(tokens) != len(log_probs):
            raise ValueError(
                f"candidate row {depth} must contain equally sized non-empty lists"
            )

    nodes = [
        _TreeNode(
            token_id=int(root_token),
            depth=0,
            parent=-1,
            log_mass=0.0,
        )
    ]

    def add_child(
        parent: int,
        token_id: int,
        depth: int,
        log_mass: float,
    ) -> int:
        existing = nodes[parent].children.get(token_id)
        if existing is not None:
            return existing
        node_index = len(nodes)
        nodes.append(
            _TreeNode(
                token_id=token_id,
                depth=depth,
                parent=parent,
                log_mass=log_mass,
            )
        )
        nodes[parent].children[token_id] = node_index
        return node_index

    # heapq is a min-heap. This tuple implements the deterministic ordering:
    # higher mass, lower depth, lower rank, lower token id, lower parent index.
    candidates: list[tuple[float, int, int, int, int]] = []

    def expose_missing_children(parent_index: int) -> None:
        parent_node = nodes[parent_index]
        position = parent_node.depth
        if position >= num_depths:
            return
        for rank, (token, log_prob) in enumerate(
            zip(top_tokens[position], top_log_probs[position])
        ):
            token_id = int(token)
            if token_id in parent_node.children:
                continue
            candidate_mass = parent_node.log_mass + float(log_prob)
            if math.isnan(candidate_mass):
                raise ValueError("draft candidate log probabilities must not be NaN")
            heapq.heappush(
                candidates,
                (
                    -candidate_mass,
                    position + 1,
                    rank,
                    token_id,
                    parent_index,
                ),
            )

    expose_missing_children(0)

    while len(nodes) < max_nodes and candidates:
        neg_log_mass, depth, rank, token_id, parent_index = heapq.heappop(
            candidates
        )
        del rank  # rank participates in heap ordering only.
        if token_id in nodes[parent_index].children:
            continue
        node_index = add_child(
            parent_index,
            token_id,
            depth,
            -neg_log_mass,
        )
        expose_missing_children(node_index)

    return DraftTree(
        token_ids=tuple(node.token_id for node in nodes),
        parent_indices=tuple(node.parent for node in nodes),
        depths=tuple(node.depth for node in nodes),
        log_masses=tuple(node.log_mass for node in nodes),
    )


def ancestor_mask_from_parents(parent_indices: Tensor) -> Tensor:
    """Return [B, Q, Q], where each row sees itself and its ancestors."""
    if parent_indices.ndim != 2:
        raise ValueError(
            f"parent_indices must have shape [B, Q], got {tuple(parent_indices.shape)}"
        )
    batch_size, num_nodes = parent_indices.shape
    if num_nodes < 1:
        raise ValueError("a draft tree must contain its root")

    node_ids = torch.arange(
        num_nodes,
        dtype=parent_indices.dtype,
        device=parent_indices.device,
    )
    ancestors = node_ids.unsqueeze(0).expand(batch_size, -1)
    columns = node_ids.view(1, 1, num_nodes)
    mask = torch.zeros(
        (batch_size, num_nodes, num_nodes),
        dtype=torch.bool,
        device=parent_indices.device,
    )
    for _ in range(num_nodes):
        valid = ancestors.ge(0)
        mask |= valid.unsqueeze(2) & columns.eq(ancestors.unsqueeze(2))
        ancestors = parent_indices.gather(1, ancestors.clamp_min(0))
        ancestors = torch.where(valid, ancestors, -1)
    return mask


@torch.inference_mode()
def build_draft_tree_batch(
    root_tokens: Tensor,
    draft_logits: Tensor,
    *,
    max_nodes: int,
    candidate_top_k: int,
    temperature: float,
    workspace: dict[str, Tensor] | None = None,
) -> DraftTreeBatch:
    """Select compact candidates and build a fixed-budget best-first tree."""
    if root_tokens.ndim != 1:
        raise ValueError(f"root_tokens must have shape [B], got {root_tokens.shape}")
    if draft_logits.ndim != 3 or draft_logits.size(0) != root_tokens.size(0):
        raise ValueError(
            "draft_logits must have shape [B, depth, vocab] with the same B "
            f"as root_tokens; got {tuple(draft_logits.shape)}"
        )
    if candidate_top_k < 1:
        raise ValueError("candidate_top_k must be >= 1")
    num_depths = int(draft_logits.size(1))

    vocab_size = int(draft_logits.size(-1))
    candidate_top_k = min(int(candidate_top_k), vocab_size)
    def buffer(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        if workspace is None:
            return torch.empty(shape, dtype=dtype, device=draft_logits.device)
        value = workspace.get(name)
        if (
            value is None
            or value.shape != shape
            or value.dtype != dtype
            or value.device != draft_logits.device
        ):
            value = torch.empty(shape, dtype=dtype, device=draft_logits.device)
            workspace[name] = value
        return value

    if root_tokens.is_cuda:
        batch_size, num_depths, _ = draft_logits.shape
        candidate_shape = (batch_size, num_depths, candidate_top_k)
        top_values = buffer("top_values", candidate_shape, draft_logits.dtype)
        top_token_ids = buffer("top_token_ids", candidate_shape, torch.long)
        torch.topk(
            draft_logits,
            k=candidate_top_k,
            dim=-1,
            sorted=True,
            out=(top_values, top_token_ids),
        )
        top_log_probs = buffer(
            "top_log_probs",
            candidate_shape,
            torch.float32,
        )
        num_partial_blocks = (vocab_size + 8191) // 8192
        partial_shape = (batch_size * num_depths, num_partial_blocks)
        build_candidate_log_probs_cuda(
            draft_logits,
            top_values,
            top_log_probs,
            buffer("partial_lse_max", partial_shape, torch.float32),
            buffer("partial_lse_sum", partial_shape, torch.float32),
            temperature,
        )
        tree_shape = (batch_size, max_nodes)
        token_ids, parent_indices, depths, log_masses = (
            build_tree_from_candidates_cuda(
                root_tokens,
                top_token_ids,
                top_log_probs,
                max_nodes,
                out=(
                    buffer("tree_token_ids", tree_shape, torch.long),
                    buffer("tree_parent_indices", tree_shape, torch.long),
                    buffer("tree_depths", tree_shape, torch.long),
                    buffer("tree_log_masses", tree_shape, torch.float32),
                ),
            )
        )
        return DraftTreeBatch(
            token_ids=token_ids,
            parent_indices=parent_indices,
            depths=depths,
            log_masses=log_masses,
        )

    scaled_logits = draft_logits.float()
    if temperature > 0.0:
        scaled_logits = scaled_logits / float(temperature)
    top_values, top_token_ids = torch.topk(
        scaled_logits,
        k=candidate_top_k,
        dim=-1,
        sorted=True,
    )
    top_log_probs = top_values - torch.logsumexp(
        scaled_logits,
        dim=-1,
        keepdim=True,
    )

    roots_host = root_tokens.detach().cpu().tolist()
    tokens_host = top_token_ids.detach().cpu().tolist()
    log_probs_host = top_log_probs.detach().cpu().tolist()
    trees = [
        build_tree_from_candidates(
            root,
            tokens,
            log_probs,
            max_nodes,
        )
        for root, tokens, log_probs in zip(
            roots_host,
            tokens_host,
            log_probs_host,
        )
    ]
    for tree in trees:
        if tree.num_nodes != max_nodes:
            raise ValueError(
                f"candidate set produced only {tree.num_nodes} tree nodes, "
                f"but fixed tree verification requires {max_nodes}; increase "
                "tree_candidate_top_k or lower tree_verify_size"
            )

    device = root_tokens.device
    token_ids = torch.tensor(
        [tree.token_ids for tree in trees],
        dtype=torch.long,
        device=device,
    )
    parent_indices = torch.tensor(
        [tree.parent_indices for tree in trees],
        dtype=torch.long,
        device=device,
    )
    depths = torch.tensor(
        [tree.depths for tree in trees],
        dtype=torch.long,
        device=device,
    )
    log_masses = torch.tensor(
        [tree.log_masses for tree in trees],
        dtype=torch.float32,
        device=device,
    )
    return DraftTreeBatch(
        token_ids=token_ids,
        parent_indices=parent_indices,
        depths=depths,
        log_masses=log_masses,
    )


def walk_tree(
    tree_token_ids: Tensor,
    parent_indices: Tensor,
    target_tokens: Tensor,
    *,
    max_depth: int,
    pad_token_id: int,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Walk target samples through a tree without host synchronization.

    Returns committed candidate tokens, accepted node indices, and untruncated
    committed lengths. The final committed token is intentionally uncached.
    """
    if (
        tree_token_ids.ndim != 2
        or parent_indices.shape != tree_token_ids.shape
        or target_tokens.shape != tree_token_ids.shape
    ):
        raise ValueError("tree tokens, parents, and target tokens must share [B, Q]")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if tree_token_ids.is_cuda:
        return walk_tree_kernel(
            tree_token_ids,
            parent_indices,
            target_tokens,
            max_depth=max_depth,
            pad_token_id=pad_token_id,
            out=out,
        )

    batch_size, num_nodes = tree_token_ids.shape
    max_cached_tokens = max_depth + 1
    max_committed_tokens = max_cached_tokens + 1
    device = tree_token_ids.device
    dtype = tree_token_ids.dtype

    committed = torch.full(
        (batch_size, max_committed_tokens),
        int(pad_token_id),
        dtype=dtype,
        device=device,
    )
    accepted_nodes = torch.full(
        (batch_size, max_cached_tokens),
        -1,
        dtype=torch.long,
        device=device,
    )
    committed[:, 0] = tree_token_ids[:, 0]
    accepted_nodes[:, 0] = 0

    current = torch.zeros(batch_size, dtype=torch.long, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    lengths = torch.ones(batch_size, dtype=torch.long, device=device)

    for step in range(1, max_committed_tokens):
        sampled = target_tokens.gather(1, current.unsqueeze(1)).squeeze(1)
        committed[:, step] = torch.where(
            active,
            sampled.to(dtype),
            committed[:, step],
        )
        lengths = torch.where(active, step + 1, lengths)

        matches = (
            parent_indices.eq(current.unsqueeze(1))
            & tree_token_ids.eq(sampled.unsqueeze(1))
            & active.unsqueeze(1)
        )
        has_child = matches.any(dim=1)
        child = matches.to(torch.int64).argmax(dim=1)
        can_descend = active & has_child & (step <= max_depth)
        if step <= max_depth:
            accepted_nodes[:, step] = torch.where(
                can_descend,
                child,
                accepted_nodes[:, step],
            )
        current = torch.where(can_descend, child, current)
        active = can_descend

    return committed, accepted_nodes, lengths
