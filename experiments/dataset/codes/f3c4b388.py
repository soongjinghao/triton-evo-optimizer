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
    # Consolidate multiple requests per program to reduce launch overhead
    # Each program handles a chunk of consecutive pids
    CONSOLIDATE_FACTOR: tl.constexpr = 4
    base_pid = tl.program_id(axis=0) * CONSOLIDATE_FACTOR
    
    # Precompute cumulative offsets for all pids in this chunk
    # Use a single scan to avoid repeated prefix loops
    chunk_start = tl.load(start_offset + base_pid)
    chunk_end = tl.load(end_offset + base_pid)
    chunk_len = chunk_end - chunk_start
    
    # Accumulate offsets from previous pids
    prev_len = tl.zeros([1], dtype=tl.int32)
    for i in range(CONSOLIDATE_FACTOR):
        pid = base_pid + i
        if pid >= bs_upper:
            break
        
        kv_start = tl.load(start_offset + pid)
        kv_end = tl.load(end_offset + pid)
        token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
        
        # Compute output offset using precomputed cumulative length
        out_offset = tl.sum(prev_len)
        out_cache_ptr = out_cache_loc + out_offset
        
        load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
        save_offset = tl.arange(0, BLOCK_SIZE)
        num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
        
        for _ in range(num_loop):
            mask = load_offset < kv_end
            data = tl.load(token_pool + load_offset, mask=mask)
            tl.store(out_cache_ptr + save_offset, data, mask=mask)
            load_offset += BLOCK_SIZE
            save_offset += BLOCK_SIZE
        
        # Update cumulative length for next pid
        prev_len = prev_len + (kv_end - kv_start)