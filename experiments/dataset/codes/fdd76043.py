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
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    program_id = tl.program_id(0)
    token_base = program_id * TOKENS_PER_PROGRAM
    
    # Load 4 consecutive slot_mapping entries
    slot_offsets = tl.arange(0, TOKENS_PER_PROGRAM)
    slot_mapping_ptrs = slot_mapping + token_base + slot_offsets
    slot_indices = tl.load(slot_mapping_ptrs)
    
    # Early exit if all slots are invalid
    all_invalid = slot_indices < 0
    if tl.sum(all_invalid.to(tl.int32)) == TOKENS_PER_PROGRAM:
        return
    
    num_chunks = tl.cdiv(n, CHUNK_ELEMENTS)
    chunk_i = tl.arange(0, CHUNK_ELEMENTS)
    
    for chunk_idx in range(0, num_chunks):
        offset = chunk_idx * CHUNK_ELEMENTS
        chunk_mask = (offset + chunk_i) < n
        
        # Process each token in the group
        for i in range(TOKENS_PER_PROGRAM):
            token_idx = token_base + i
            slot_idx = slot_indices[i]
            
            # Skip invalid slots
            if slot_idx < 0:
                continue
            
            block_idx = slot_idx // block_size
            block_offset = slot_idx % block_size
            base_cache_offset = block_idx * block_stride + block_offset * n
            
            # Continuous load for key and value (key_stride == n, value_stride == n)
            key_base = key + token_idx * key_stride
            value_base = value + token_idx * value_stride
            
            k = tl.load(key_base + offset + chunk_i, mask=chunk_mask)
            v = tl.load(value_base + offset + chunk_i, mask=chunk_mask)
            
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
    
    # Grid consolidation: process 4 tokens per program
    TOKENS_PER_PROGRAM = 4
    assert num_tokens % TOKENS_PER_PROGRAM == 0, "num_tokens must be divisible by TOKENS_PER_PROGRAM"
    assert key_stride == n, "key_stride must equal n for continuous access"
    assert value_stride == n, "value_stride must equal n for continuous access"
    
    grid = (num_tokens // TOKENS_PER_PROGRAM,)
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
        TOKENS_PER_PROGRAM,
    )