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
    num_tokens,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    tok = pid * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)

    tok_valid = tok < num_tokens
    slot = tl.load(slot_mapping + tok, mask=tok_valid, other=-1)
    valid = tok_valid & (slot >= 0)

    safe_tok = tl.where(tok_valid, tok, 0)
    safe_slot = tl.where(valid, slot, 0)

    block_idx = safe_slot // block_size
    block_offset = safe_slot % block_size
    cache_offsets = block_idx * block_stride + block_offset * n

    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)

    for chunk_idx in range(0, num_chunks):
        off = chunk_idx * CHUNK_ELEMENTS + chunk_i
        chunk_mask = off < n
        load_store_mask = valid[:, None] & chunk_mask[None, :]

        key_ptr = key + safe_tok[:, None] * key_stride + off[None, :]
        value_ptr = value + safe_tok[:, None] * value_stride + off[None, :]
        key_cache_ptr = key_cache + cache_offsets[:, None] + off[None, :]
        value_cache_ptr = value_cache + cache_offsets[:, None] + off[None, :]

        k = tl.load(key_ptr, mask=load_store_mask)
        v = tl.load(value_ptr, mask=load_store_mask)

        tl.store(key_cache_ptr, k, mask=load_store_mask)
        tl.store(value_cache_ptr, v, mask=load_store_mask)


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
    BLOCK_TOKENS = 8
    grid = ((num_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS,)

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
        num_tokens,
        BLOCK_TOKENS,
    )