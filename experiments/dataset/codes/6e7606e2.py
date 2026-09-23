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
    num_tokens,
    n: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    token_id = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_id < num_tokens

    slot_ids = tl.load(slot_mapping + token_id, mask=token_mask, other=-1)
    valid = token_mask & (slot_ids >= 0)

    block_idx = slot_ids // block_size
    block_offset = slot_ids % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n

    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)

    for chunk_idx in range(0, num_chunks):
        offset = chunk_idx * CHUNK_ELEMENTS
        chunk_mask = (offset + chunk_i) < n
        combined_mask = valid[:, None] & chunk_mask[None, :]

        key_ptrs = key + token_id[:, None] * key_stride + (offset + chunk_i)[None, :]
        value_ptrs = value + token_id[:, None] * value_stride + (offset + chunk_i)[None, :]

        k = tl.load(key_ptrs, mask=combined_mask, other=0.0)
        v = tl.load(value_ptrs, mask=combined_mask, other=0.0)

        cache_offset = base_cache_offset[:, None] + (offset + chunk_i)[None, :]

        tl.store(key_cache + cache_offset, k, mask=combined_mask)
        tl.store(value_cache + cache_offset, v, mask=combined_mask)


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
    BLOCK_M = 8

    grid = (triton.cdiv(num_tokens, BLOCK_M),)
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
        num_tokens,
        n,
        CHUNK_ELEMENTS,
        BLOCK_M,
    )