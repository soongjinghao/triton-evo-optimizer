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
    n,
    CHUNK_ELEMENTS: tl.constexpr,
    num_tokens,
):
    pid_token = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    if pid_token >= num_tokens:
        return
    slot = tl.load(slot_mapping + pid_token)
    if slot < 0:
        return
    block_idx = slot // block_size
    block_offset = slot % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n
    chunk_idx = pid_chunk
    offset = chunk_idx * CHUNK_ELEMENTS
    chunk_offs = tl.arange(0, CHUNK_ELEMENTS)
    chunk_mask = (offset + chunk_offs) < n
    key_ptr = key + pid_token * key_stride + offset
    value_ptr = value + pid_token * value_stride + offset
    k = tl.load(key_ptr + chunk_offs, mask=chunk_mask)
    v = tl.load(value_ptr + chunk_offs, mask=chunk_mask)
    cache_offset = base_cache_offset + offset
    tl.store(key_cache + cache_offset + chunk_offs, k, mask=chunk_mask)
    tl.store(value_cache + cache_offset + chunk_offs, v, mask=chunk_mask)

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
    grid = (num_tokens, num_chunks)
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
    )