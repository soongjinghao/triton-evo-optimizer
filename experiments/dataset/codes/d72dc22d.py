import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def reshape_and_cache_flash_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    block_stride,
    key_stride,
    value_stride,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size,
    k_scale,
    v_scale,
    n: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return

    NUM_CHUNKS: tl.constexpr = (n + CHUNK_ELEMENTS - 1) // CHUNK_ELEMENTS

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n

    key_base = key + token_idx * key_stride
    value_base = value + token_idx * value_stride
    key_cache_base = key_cache + base_cache_offset
    value_cache_base = value_cache + base_cache_offset

    if (CHUNK_ELEMENTS // head_size) * head_size == CHUNK_ELEMENTS:
        BH: tl.constexpr = CHUNK_ELEMENTS // head_size
        head_idx = tl.arange(0, BH)
        dim_idx = tl.arange(0, head_size)
        head_offsets = head_idx[:, None] * head_size + dim_idx[None, :]
        for chunk_idx in range(NUM_CHUNKS):
            offset = chunk_idx * CH