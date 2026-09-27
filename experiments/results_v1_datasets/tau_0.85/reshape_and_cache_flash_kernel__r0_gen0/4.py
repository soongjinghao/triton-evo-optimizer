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

    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)

    for chunk_idx in range(0, num_chunks):
        offset = chunk_idx * CHUNK_ELEMENTS
        chunk_mask = (offset + chunk_i) < n

        key_ptr = key + token_idx * key_stride + offset
        value_ptr = value + token_idx * value_stride + offset

        k = tl.load(key_ptr + chunk_i, mask=chunk_mask)
        v = tl.load(value_ptr + chunk_i, mask=chunk_mask)

        cache_offset = base_cache_offset + offset
        tl.store(key_cache + cache_offset + chunk_i, k, mask=chunk_mask)
        tl.store(value_cache + cache_offset + chunk_i, v, mask=chunk_mask)


@triton.jit
def _precompute_cache_offsets_kernel(
    slot_mapping,
    base_cache_offsets,
    valid_flags,
    block_stride,
    block_size,
    n,
    num_tokens,
    CACHE_CONTIGUOUS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    token_mask = token_offsets < num_tokens

    slot_idx = tl.load(slot_mapping + token_offsets, mask=token_mask)
    valid = slot_idx >= 0
    slot_clamped = tl.where(valid, slot_idx, 0).to(tl.int64)

    if CACHE_CONTIGUOUS:
        base_offset = slot_clamped * n
    else:
        block_idx = slot_clamped // block_size
        block_offset = slot_clamped % block_size
        base_offset = block_idx * block_stride + block_offset * n

    tl.store(base_cache_offsets + token_offsets, base_offset, mask=token_mask)
    tl.store(valid_flags + token_offsets,
             tl.where(valid, 1, 0).to(tl.int32),
             mask=token_mask)


@triton.jit
def _reshape_and_cache_flash_precomputed_kernel(
    key,
    value,
    key_cache,
    value_cache,
    base_cache_offsets,
    valid_flags,
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

    valid = tl.load(valid_flags + token_idx)
    if valid == 0:
        return

    base_cache_offset = tl.load(base_cache_offsets + token_idx)

    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)

    for chunk_idx in range(0, num_chunks):
        offset = chunk_idx * CHUNK_ELEMENTS
        chunk_mask = (offset + chunk_i) < n

        key_ptr = key + token_idx * key_stride + offset
        value_ptr = value + token_idx * value_stride + offset

        k = tl.load(key_ptr + chunk_i, mask=chunk_mask)
        v = tl.load(value_ptr + chunk_i, mask=chunk_mask)

        cache_offset = base_cache_offset + offset
        tl.store(key_cache + cache_offset + chunk_i, k, mask=chunk_mask)
        tl.store(value_cache + cache_offset + chunk_i, v, mask=chunk_mask)


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
    grid = (num_tokens,)

    PRECOMPUTE_BLOCK = 1024
    PRECOMPUTE_MIN_TOKENS = 2048

    if num_tokens >= PRECOMPUTE_MIN_TOKENS:
        base_cache_offsets = torch.empty(
            num_tokens, device=slot_mapping.device, dtype=torch.int64
        )
        valid_flags = torch.empty(
            num_tokens, device=slot_mapping.device, dtype=torch.int32
        )

        cache_contiguous = (block_stride == block_size * n)
        precompute_grid = (triton.cdiv(num_tokens, PRECOMPUTE_BLOCK),)

        _precompute_cache_offsets_kernel[precompute_grid](
            slot_mapping,
            base_cache_offsets,
            valid_flags,
            block_stride,
            block_size,
            n,
            num_tokens,
            cache_contiguous,
            PRECOMPUTE_BLOCK,
        )

        _reshape_and_cache_flash_precomputed_kernel[grid](
            key,
            value,
            key_cache,
            value_cache,
            base_cache_offsets,
            valid_flags,
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
    else:
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