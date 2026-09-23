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
    num_heads,
    head_size,
    block_size,
    k_scale,
    v_scale,
    n: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    BT: tl.constexpr,
    num_tokens,
):
    pid = tl.program_id(0)
    tt = pid * BT + tl.arange(0, BT)
    token_mask = tt < num_tokens

    slot_idx = tl.load(slot_mapping + tt, mask=token_mask)
    slot_valid = token_mask & (slot_idx >= 0)
    safe_slot = tl.where(slot_valid, slot_idx, 0)

    block_idx = safe_slot // block_size
    block_offset = safe_slot % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n

    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    for chunk_idx in range(0, num_chunks):
        offs = chunk_idx * CHUNK_ELEMENTS + tl.arange(0, CHUNK_ELEMENTS)
        valid_offs = offs < n
        mem_mask = token_mask[:, None] & slot_valid[:, None] & valid_offs[None, :]

        key_ptrs = key + tt[:, None] * key_stride + offs[None, :]
        value_ptrs = value + tt[:, None] * value_stride + offs[None, :]
        k = tl.load(key_ptrs, mask=mem_mask)
        v = tl.load(value_ptrs, mask=mem_mask)

        key_cache_ptrs = key_cache + base_cache_offset[:, None] + offs[None, :]
        value_cache_ptrs = value_cache + base_cache_offset[:, None] + offs[None, :]
        tl.store(key_cache_ptrs, k, mask=mem_mask)
        tl.store(value_cache_ptrs, v, mask=mem_mask)


def reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    logger.debug("GEMS RESHAPE_AND_CACHE_FLASH")
    num_tokens = slot_mapping.size(0)
    num_heads = key.size(1)
    head_size = key.size(2)
    block_size = key_cache.size(1)
    key_stride = key.stride(0)
    value_stride = value.stride(0)
    block_stride = key_cache.stride(0)
    assert key_cache.stride(0) == value_cache.stride(0)
    n = num_heads * head_size

    CHUNK_ELEMENTS = 4096
    BT = 2
    grid = ((num_tokens + BT - 1) // BT,)
    reshape_and_cache_flash_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        block_stride,
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        k_scale,
        v_scale,
        n,
        CHUNK_ELEMENTS,
        BT,
        num_tokens,
    )