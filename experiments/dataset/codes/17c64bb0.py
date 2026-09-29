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
    block_idx_ptr,
    block_offset_ptr,
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
    pid = tl.program_id(0)
    num_tokens_per_block = 4
    token_idx = pid * num_tokens_per_block + tl.arange(0, num_tokens_per_block)
    block_idx = tl.load(block_idx_ptr + token_idx, mask=token_idx < tl.num_programs(0) * num_tokens_per_block, other=-1)
    block_offset = tl.load(block_offset_ptr + token_idx, mask=token_idx < tl.num_programs(0) * num_tokens_per_block, other=0)
    valid_mask = block_idx >= 0
    base_cache_offset = block_idx * block_stride + block_offset * n
    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)
    for chunk_idx in range(0, num_chunks):
        offset = chunk_idx * CHUNK_ELEMENTS
        chunk_mask = (offset + chunk_i) < n
        key_ptr = key + token_idx[:, None] * key_stride + offset + chunk_i[None, :]
        value_ptr = value + token_idx[:, None] * value_stride + offset + chunk_i[None, :]
        k = tl.load(key_ptr, mask=valid_mask[:, None] & chunk_mask[None, :])
        v = tl.load(value_ptr, mask=valid_mask[:, None] & chunk_mask[None, :])
        cache_offset = base_cache_offset[:, None] + offset + chunk_i[None, :]
        tl.store(key_cache + cache_offset, k, mask=valid_mask[:, None] & chunk_mask[None, :])
        tl.store(value_cache + cache_offset, v, mask=valid_mask[:, None] & chunk_mask[None, :])
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
    slot_mapping_int32 = slot_mapping.to(torch.int32)
    block_idx = slot_mapping_int32 // block_size
    block_offset = slot_mapping_int32 % block_size
    num_tokens_per_block = 4
    grid = (triton.cdiv(num_tokens, num_tokens_per_block),)
    reshape_and_cache_flash_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        block_idx,
        block_offset,
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