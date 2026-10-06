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
    pid = tl.program_id(axis=0)
    
    # Strategy_LoadMerge: merge start_offset and end_offset loads into one contiguous block
    # Since start_offset and end_offset are consecutive in memory (same base, stride=1),
    # we load both as a single 2-element block and extract via tl.where
    pid_offsets = tl.arange(0, 2) + pid * 2
    pid_mask = pid_offsets < bs_upper * 2
    pid_data = tl.load(start_offset + pid_offsets, mask=pid_mask, other=0)
    kv_start = tl.where(pid_offsets % 2 == 0, pid_data, 0)
    kv_end = tl.where(pid_offsets % 2 == 1, pid_data, 0)
    kv_start = tl.sum(kv_start, axis=0)
    kv_end = tl.sum(kv_end, axis=0)
    
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    
    # Strategy_LoadMerge: merge start_offset and end_offset for prefix sum computation
    length_offset = tl.arange(0, bs_upper)
    # Load both start and end offsets in one contiguous block (2 * bs_upper elements)
    all_offsets = tl.arange(0, bs_upper * 2)
    all_data = tl.load(start_offset + all_offsets, mask=all_offsets < bs_upper * 2, other=0)
    start = tl.where(all_offsets < bs_upper, all_data, 0)
    end = tl.where(all_offsets >= bs_upper, all_data, 0)
    start = tl.sum(start * (length_offset < pid), axis=0)
    end = tl.sum(end * (length_offset < pid), axis=0)
    out_offset = end - start
    
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

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    grid = (req_pool_indices_tensor.numel(),)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper
    )
    return out_cache_loc_tensor