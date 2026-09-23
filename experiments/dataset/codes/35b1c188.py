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
    BT: tl.constexpr,
):
    token_ids = tl.program_id(0) * BT + tl.arange(0, BT)
    token_mask = token_ids < num_tokens

    slot_idx = tl.load(slot_mapping + token_ids, mask=token_mask, other=-1)
    valid = token_mask & (slot_idx >= 0)

    safe_slot_idx = tl.where(valid, slot_idx, 0)
    block_idx = safe_slot_idx // block_size
    block_offset = safe_slot_idx % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n

    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_off = tl.arange(0, CHUNK_ELEMENTS)

    token_ids_2d = tl.expand_dims(token_ids, 1)
    valid_2d = tl.expand_dims(valid, 1)
    base_cache_2d = tl.expand_dims(base_cache_offset, 1)

    for chunk_idx in range(0, num_chunks):
        offset = chunk_idx * CHUNK_ELEMENTS
        cols = offset + chunk_off
        col_mask = cols < n

        cols_2d = tl.expand_dims(cols, 0)
        col_mask_2d = tl.expand_dims(col_mask, 0)
        load_mask = valid_2d & col_mask_2d

        key_ptrs = key + token_ids_2d * key_stride + cols_2d
        value_ptrs = value + token_ids_2d * value_stride + cols_2d

        k = tl.load(key_ptrs, mask=load_mask, other=0.0)
        v = tl.load(value_ptrs, mask=load_mask, other=0.0)

        cache_offsets = base_cache_2d + cols_2d
        key_cache_ptrs = key_cache + cache_offsets
        value_cache_ptrs = value_cache + cache_offsets

        tl.store(key_cache_ptrs, k, mask=load_mask)
        tl.store(value_cache_ptrs, v, mask=load_mask)


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
    BT = 4
    grid = (triton.cdiv(num_tokens, BT),)

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
        BT,
    )