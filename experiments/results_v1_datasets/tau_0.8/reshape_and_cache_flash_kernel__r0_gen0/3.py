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
    num_tokens,
    CHUNK_ELEMENTS: tl.constexpr,
    BT2: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BT2)
    token_offs = pid * BT2 + lane
    token_mask = token_offs < num_tokens

    slots = tl.load(slot_mapping + token_offs, mask=token_mask, other=-1)
    valid_slots = token_mask & (slots >= 0)

    block_idx_vec = slots // block_size
    block_offset_vec = slots % block_size
    base_vec = block_idx_vec * block_stride + block_offset_vec * n

    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)

    for b in range(BT2):
        is_b = lane == b
        valid_b = tl.sum(tl.where(is_b & valid_slots, 1, 0)) > 0

        if valid_b:
            token_idx = pid * BT2 + b
            block_idx_b = tl.sum(tl.where(is_b, block_idx_vec, 0))
            block_offset_b = tl.sum(tl.where(is_b, block_offset_vec, 0))
            base_b = tl.sum(tl.where(is_b, base_vec, 0))

            key_base = key + token_idx * key_stride
            value_base = value + token_idx * value_stride
            key_cache_base = key_cache + base_b
            value_cache_base = value_cache + base_b

            for chunk_idx in range(0, num_chunks):
                offset = chunk_idx * CHUNK_ELEMENTS
                chunk_mask = (offset + chunk_i) < n

                key_ptr = key_base + offset
                value_ptr = value_base + offset

                k = tl.load(key_ptr + chunk_i, mask=chunk_mask)
                v = tl.load(value_ptr + chunk_i, mask=chunk_mask)

                tl.store(key_cache_base + offset + chunk_i, k, mask=chunk_mask)
                tl.store(value_cache_base + offset + chunk_i, v, mask=chunk_mask)


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
    BT2 = 2

    grid = (triton.cdiv(num_tokens, BT2),)
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
        num_tokens,
        CHUNK_ELEMENTS,
        BT2,
    )