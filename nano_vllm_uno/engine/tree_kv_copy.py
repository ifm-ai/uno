"""Direct all-layer KV cache compaction for accepted tree paths."""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_tree_kv_slots_kernel(
    accepted_nodes, # [C, L] accepted node indices e.g. [[0,2,3,-1]], including root.
    cache_lengths, # [C] actual num accepted nodes e.g.[[3]].
    prefix_lengths, # [C] committed length before tree.
    block_tables, # [C, blocks]
    source, # flattened [C * PATH_LENGTH].
    destination, # flattened [C * PATH_LENGTH].
    ACCEPTED_BATCH_STRIDE: tl.constexpr,
    ACCEPTED_PATH_STRIDE: tl.constexpr,
    BLOCK_TABLE_BATCH_STRIDE: tl.constexpr,
    BLOCK_TABLE_BLOCK_STRIDE: tl.constexpr,
    PATH_LENGTH: tl.constexpr, # = diffusion_block_size.
    KV_BLOCK_SIZE: tl.constexpr,
    BLOCK_PATH: tl.constexpr, # next power of 2 of PATH_LENGTH.
):
    batch = tl.program_id(0)
    positions = tl.arange(0, BLOCK_PATH)
    in_path = positions < PATH_LENGTH
    cache_length = tl.load(cache_lengths + batch)
    accepted = tl.load(
        accepted_nodes
        + batch * ACCEPTED_BATCH_STRIDE
        + positions * ACCEPTED_PATH_STRIDE,
        mask=in_path,
        other=0,
    )
    nodes = tl.where(positions < cache_length, accepted, positions)
    prefix = tl.load(prefix_lengths + batch)
    source_positions = prefix + nodes
    destination_positions = prefix + positions

    source_blocks = tl.load(
        block_tables
        + batch * BLOCK_TABLE_BATCH_STRIDE
        + (source_positions // KV_BLOCK_SIZE) * BLOCK_TABLE_BLOCK_STRIDE,
        mask=in_path,
    ).to(tl.int64)
    destination_blocks = tl.load(
        block_tables
        + batch * BLOCK_TABLE_BATCH_STRIDE
        + (destination_positions // KV_BLOCK_SIZE) * BLOCK_TABLE_BLOCK_STRIDE,
        mask=in_path,
    ).to(tl.int64)
    output_offset = batch * PATH_LENGTH + positions
    tl.store(
        source + output_offset,
        source_blocks * KV_BLOCK_SIZE + source_positions % KV_BLOCK_SIZE,
        mask=in_path,
    )
    tl.store(
        destination + output_offset,
        destination_blocks * KV_BLOCK_SIZE
        + destination_positions % KV_BLOCK_SIZE,
        mask=in_path,
    )


def build_tree_kv_slots(
    accepted_nodes: torch.Tensor,
    cache_lengths: torch.Tensor,
    prefix_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    kv_block_size: int,
) -> None:
    """
    Build physical source/destination slots (in block tables) for accepted nodes.
    Followed immediately by copy_tree_kv which does the actual copy.
    """
    batch_size, path_length = accepted_nodes.shape
    _build_tree_kv_slots_kernel[(batch_size,)](
        accepted_nodes,
        cache_lengths,
        prefix_lengths,
        block_tables,
        source,
        destination,
        ACCEPTED_BATCH_STRIDE=accepted_nodes.stride(0),
        ACCEPTED_PATH_STRIDE=accepted_nodes.stride(1),
        BLOCK_TABLE_BATCH_STRIDE=block_tables.stride(0),
        BLOCK_TABLE_BLOCK_STRIDE=block_tables.stride(1),
        PATH_LENGTH=path_length,
        KV_BLOCK_SIZE=kv_block_size,
        BLOCK_PATH=triton.next_power_of_2(path_length),
        num_warps=1,
        num_stages=1,
    )


# Adapted from SGLang's tiled KV-cache move kernel.
@triton.jit
def _copy_tree_kv_kernel(
    cache,
    destination,
    source,
    num_locations,
    LAYER_STRIDE_BYTES: tl.constexpr,
    SLOT_STRIDE_BYTES: tl.constexpr,
    LOCATIONS_PER_PROGRAM: tl.constexpr,
    BLOCK_LOCATIONS: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    layer = tl.program_id(0)
    tile = tl.program_id(1)
    location_block = tl.program_id(2)

    location_offset = tl.arange(0, BLOCK_LOCATIONS)
    location = location_block * LOCATIONS_PER_PROGRAM + location_offset
    location_mask = (
        (location_offset < LOCATIONS_PER_PROGRAM)
        & (location < num_locations)
    )
    src = tl.load(source + location, mask=location_mask, other=0)
    dst = tl.load(destination + location, mask=location_mask, other=0)

    byte = tile * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    byte_mask = byte < SLOT_STRIDE_BYTES
    base = (
        tl.cast(cache, tl.pointer_type(tl.uint8))
        + layer.to(tl.int64) * LAYER_STRIDE_BYTES
    )
    # Tree compaction is an overlapping in-place gather. Skip identity lanes,
    # then finish every source load before any warp overwrites a destination.
    move_mask = location_mask & (src != dst)
    mask = move_mask[:, None] & byte_mask[None, :]
    values = tl.load(
        base + src[:, None] * SLOT_STRIDE_BYTES + byte[None, :],
        mask=mask,
    )
    tl.debug_barrier()
    tl.store(
        base + dst[:, None] * SLOT_STRIDE_BYTES + byte[None, :],
        values,
        mask=mask,
    )


def copy_tree_kv(
    cache: torch.Tensor,
    destination: torch.Tensor,
    source: torch.Tensor,
    path_length: int,
) -> None:
    """Copy physical cache slots directly, without an intermediate KV tensor."""
    if not cache.is_cuda:
        accepted = cache.index_select(2, source)
        cache.index_copy_(2, destination, accepted)
        return

    num_locations = source.numel()
    if num_locations == 0:
        return
    slot_stride_bytes = cache.stride(2) * cache.element_size()
    layer_stride_bytes = cache.stride(1) * cache.element_size()
    bytes_per_tile = 512 if slot_stride_bytes >= 8192 else (
        256 if slot_stride_bytes >= 4096 else 128
    )
    # Keep every request's overlapping path moves inside one synchronized CTA.
    requests_per_program = max(1, 128 // path_length)
    locations_per_program = requests_per_program * path_length
    block_locations = triton.next_power_of_2(locations_per_program)
    grid = (
        cache.size(0) * cache.size(1),
        triton.cdiv(slot_stride_bytes, bytes_per_tile),
        triton.cdiv(num_locations, locations_per_program),
    )
    _copy_tree_kv_kernel[grid](
        cache,
        destination,
        source,
        num_locations,
        LAYER_STRIDE_BYTES=layer_stride_bytes,
        SLOT_STRIDE_BYTES=slot_stride_bytes,
        LOCATIONS_PER_PROGRAM=locations_per_program,
        BLOCK_LOCATIONS=block_locations,
        BYTES_PER_TILE=bytes_per_tile,
        num_warps=8 if bytes_per_tile == 512 else 4,
        num_stages=2,
    )
