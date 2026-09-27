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
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n

    key_base = key + token_idx * key_stride
    value_base = value + token_idx * value_stride
    key_cache_base = key_cache + base_cache_offset
    value_cache_base = value_cache + base_cache_offset

    if n <= 0:
        return

    # Static fast path: n already fits in one chunk, no loop, no tail mask.
    if n <= CHUNK_ELEMENTS:
        idx = tl.arange(0, n)
        k = tl.load(key_base + idx)
        v = tl.load(value_base + idx)
        tl.store(key_cache_base + idx, k)
        tl.store(value_cache_base + idx, v)
        return

    num_chunks = (n + CHUNK_ELEMENTS - 1) // CHUNK_ELEMENTS
    indices = tl.arange(0, CHUNK_ELEMENTS)

    # Divisible path: all chunks are full tiles, no mask required.
    if n % CHUNK_ELEMENTS == 0:
        for chunk_idx in tl.static_range(0, num_chunks):
            offset = chunk_idx * CHUNK_ELEMENTS
            k = tl.load(key_base + offset + indices)
            v = tl.load(value_base + offset + indices)
            tl.store(key_cache_base + offset + indices, k)
            tl.store(value_cache_base + offset + indices, v)
    else:
        # Non-divisible path: all full chunks unmasked, only tail chunk masked.
        full_chunks = num_chunks - 1
        for chunk_idx in tl.static_range(0, full_chunks):
            offset = chunk_idx * CHUNK_ELEMENTS
            k = tl.load(key_base + offset + indices)
            v = tl.load(value_base + offset + indices)
            tl.store(key_cache_base + offset + indices, k)
            tl.store(value_cache_base + offset + indices, v)

        tail_offset = full_chunks * CHUNK_ELEMENTS
        tail_mask = (tail_offset + indices) < n
        k = tl.load(key_base + tail_offset + indices, mask=tail_mask)
        v = tl.load(value_base + tail_offset + indices, mask=tail_mask)
        tl.store(key_cache_base + tail_offset + indices, k, mask=tail_mask)
        tl.store(value_cache_base + tail_offset + indices, v, mask=tail_mask)


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
    CHUNK_ELEMENTS = min(max(n, 1), 4096)
    grid = (num_tokens,)

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
    )