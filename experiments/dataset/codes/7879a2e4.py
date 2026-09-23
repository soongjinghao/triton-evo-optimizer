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
    chunk_idx = tl.program_id(1)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n
    offset = chunk_idx * CHUNK_ELEMENTS
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)
    chunk_mask = (offset + chunk_i) < n
    key_ptr = key + token_idx * key_stride + offset
    value_ptr = value + token_idx * value_stride + offset
    k = tl.load(key_ptr + chunk_i, mask=chunk_mask)
    v = tl.load(value_ptr + chunk_i, mask=chunk_mask)
    cache_offset = base_cache_offset + offset
    tl.store(key_cache + cache_offset + chunk_i, k, mask=chunk_mask)
    tl.store(value_cache + cache_offset + chunk_i, v, mask=chunk_mask)


@triton.jit
def reshape_and_cache_flash_kernel_int32(
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
    chunk_idx = tl.program_id(1)
    token_idx_i32 = token_idx.to(tl.int32)
    block_stride_i32 = block_stride.to(tl.int32)
    key_stride_i32 = key_stride.to(tl.int32)
    value_stride_i32 = value_stride.to(tl.int32)
    block_size_i32 = block_size.to(tl.int32)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size_i32
    block_offset = slot_idx % block_size_i32
    base_cache_offset = block_idx * block_stride_i32 + block_offset * n
    offset = chunk_idx * CHUNK_ELEMENTS
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)
    chunk_mask = (offset + chunk_i) < n
    key_ptr = key + token_idx_i32 * key_stride_i32 + offset
    value_ptr = value + token_idx_i32 * value_stride_i32 + offset
    k = tl.load(key_ptr + chunk_i, mask=chunk_mask)
    v = tl.load(value_ptr + chunk_i, mask=chunk_mask)
    cache_offset = base_cache_offset + offset
    tl.store(key_cache + cache_offset + chunk_i, k, mask=chunk_mask)
    tl.store(value_cache + cache_offset + chunk_i, v, mask=chunk_mask)


@triton.jit
def reshape_and_cache_flash_kernel_single_chunk(
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
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n
    chunk_i = tl.arange(0, n)
    chunk_mask = chunk_i < n
    key_base = key + token_idx * key_stride
    value_base = value + token_idx * value_stride
    k = tl.load(key_base + chunk_i, mask=chunk_mask)
    v = tl.load(value_base + chunk_i, mask=chunk_mask)
    tl.store(key_cache + base_cache_offset + chunk_i, k, mask=chunk_mask)
    tl.store(value_cache + base_cache_offset + chunk_i, v, mask=chunk_mask)


@triton.jit
def reshape_and_cache_flash_kernel_single_chunk_int32(
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
):
    token_idx = tl.program_id(0)
    token_idx_i32 = token_idx.to(tl.int32)
    block_stride_i32 = block_stride.to(tl.int32)
    key_stride_i32 = key_stride.to(tl.int32)
    value_stride_i32 = value_stride.to(tl.int32)
    block_size_i32 = block_size.to(tl.int32)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size_i32
    block_offset = slot_idx % block_size_i32
    base_cache_offset = block_idx * block_stride_i32 + block_offset * n
    chunk_i = tl.arange(0, n)
    chunk_mask = chunk_i < n
    key_base = key + token_idx_i32 * key_stride_i32
    value_base = value + token_idx_i32 * value_stride_i32
    k = tl.load(key_base + chunk_i, mask=chunk_mask)
    v = tl.load(value_base + chunk_i, mask=chunk_mask)
    tl.store(key_cache + base_cache_offset + chunk_i, k, mask=chunk_mask)
    tl.store(value_cache + base_cache_offset + chunk_i, v, mask=chunk_mask)


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
    num_chunks = triton.cdiv(n, CHUNK_ELEMENTS)
    max_int32 = 2 ** 31 - 1
    use_int32 = (
        num_tokens <= max_int32
        and block_size <= max_int32
        and block_stride <= max_int32
        and key_stride <= max_int32
        and value_stride <= max_int32
        and n <= max_int32
    )
    if use_int32 and num_tokens > 0:
        max_slot_val = int(slot_mapping.max().item())
    else:
        max_slot_val = -1
    if use_int32 and max_slot_val >= 0:
        max_cache_offset = (
            (max_slot_val // block_size) * block_stride
            + (max_slot_val % block_size) * n
            + (n - 1)
        )
        max_key_offset = (num_tokens - 1) * key_stride + (n - 1)
        max_value_offset = (num_tokens - 1) * value_stride + (n - 1)
        use_int32 = (
            max_cache_offset < max_int32
            and max_key_offset < max_int32
            and max_value_offset < max_int32
        )
    if num_chunks == 1:
        if use_int32:
            grid = (num_tokens,)
            reshape_and_cache_flash_kernel_single_chunk_int32[grid](
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
            )
        else:
            grid = (num_tokens,)
            reshape_and_cache_flash_kernel_single_chunk[grid](
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
            )
    else:
        grid = (num_tokens, num_chunks)
        if use_int32:
            reshape_and_cache_flash_kernel_int32[grid](
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