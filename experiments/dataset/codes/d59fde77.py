import torch
import triton
import triton.language as tl

@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    GROUP_SIZE: tl.constexpr = 4
    pid = tl.program_id(axis=0)
    
    # Pack multiple requests into one program
    base_req = pid * GROUP_SIZE
    
    # Compute address ranges for all requests in the group
    kv_starts = tl.load(start_offset + base_req + tl.arange(0, GROUP_SIZE), mask=tl.arange(0, GROUP_SIZE) + base_req < bs_upper, other=0)
    kv_ends = tl.load(end_offset + base_req + tl.arange(0, GROUP_SIZE), mask=tl.arange(0, GROUP_SIZE) + base_req < bs_upper, other=0)
    
    # Compute output offsets for each request
    out_offsets = tl.zeros((GROUP_SIZE,), dtype=tl.int64)
    for i in range(GROUP_SIZE):
        if base_req + i < bs_upper:
            prev_starts = tl.load(start_offset + tl.arange(0, base_req + i), mask=tl.arange(0, base_req + i) < bs_upper, other=0)
            prev_ends = tl.load(end_offset + tl.arange(0, base_req + i), mask=tl.arange(0, base_req + i) < bs_upper, other=0)
            out_offsets = tl.where(tl.arange(0, GROUP_SIZE) == i, tl.sum(prev_ends - prev_starts, axis=0), out_offsets)
    
    # Load token pool indices
    pool_indices = tl.load(req_pool_indices + base_req + tl.arange(0, GROUP_SIZE), mask=tl.arange(0, GROUP_SIZE) + base_req < bs_upper, other=0)
    
    # Compute max length needed across all requests in group
    max_length = tl.max(kv_ends - kv_starts, axis=0)
    num_loop = tl.cdiv(max_length, BLOCK_SIZE)
    
    # Shared memory buffer for prefetching token_pool data
    # Use local registers as shared memory is not directly available in Triton
    # Instead, we batch load and store in a tiled manner
    
    for loop_idx in range(num_loop):
        load_base = loop_idx * BLOCK_SIZE
        
        # Process each request in the group
        for req_idx in range(GROUP_SIZE):
            req_id = base_req + req_idx
            if req_id >= bs_upper:
                continue
            
            kv_start = kv_starts[req_idx]
            kv_end = kv_ends[req_idx]
            pool_idx = pool_indices[req_idx]
            out_offset = out_offsets[req_idx]
            
            load_offset = tl.arange(0, BLOCK_SIZE) + kv_start + load_base
            save_offset = tl.arange(0, BLOCK_SIZE) + load_base
            
            mask = load_offset < kv_end
            token_pool = req_to_token + pool_idx * pool_len
            
            data = tl.load(token_pool + load_offset, mask=mask)
            tl.store(out_cache_loc + out_offset + save_offset, data, mask=mask)