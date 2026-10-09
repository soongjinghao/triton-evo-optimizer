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
    
    # Each program handles 2 consecutive requests (pid*2 and pid*2+1)
    base_idx = pid * 2
    
    # Process first request
    kv_start_0 = tl.load(start_offset + base_idx)
    kv_end_0 = tl.load(end_offset + base_idx)
    token_pool_0 = req_to_token + tl.load(req_pool_indices + base_idx) * pool_len
    
    length_offset = tl.arange(0, bs_upper)
    start_0 = tl.load(start_offset + length_offset, mask=length_offset < base_idx, other=0)
    end_0 = tl.load(end_offset + length_offset, mask=length_offset < base_idx, other=0)
    out_offset_0 = tl.sum(end_0 - start_0, axis=0)
    out_cache_ptr_0 = out_cache_loc + out_offset_0
    
    load_offset_0 = tl.arange(0, BLOCK_SIZE) + kv_start_0
    save_offset_0 = tl.arange(0, BLOCK_SIZE)
    num_loop_0 = tl.cdiv(kv_end_0 - kv_start_0, BLOCK_SIZE)
    for _ in range(num_loop_0):
        mask_0 = load_offset_0 < kv_end_0
        data_0 = tl.load(token_pool_0 + load_offset_0, mask=mask_0)
        tl.store(out_cache_ptr_0 + save_offset_0, data_0, mask=mask_0)
        load_offset_0 += BLOCK_SIZE
        save_offset_0 += BLOCK_SIZE
    
    # Process second request (with mask for odd total count)
    has_second = (base_idx + 1) < tl.num_programs() * 2
    if has_second:
        kv_start_1 = tl.load(start_offset + base_idx + 1)
        kv_end_1 = tl.load(end_offset + base_idx + 1)
        token_pool_1 = req_to_token + tl.load(req_pool_indices + base_idx + 1) * pool_len
        
        start_1 = tl.load(start_offset + length_offset, mask=length_offset < (base_idx + 1), other=0)
        end_1 = tl.load(end_offset + length_offset, mask=length_offset < (base_idx + 1), other=0)
        out_offset_1 = tl.sum(end_1 - start_1, axis=0)
        out_cache_ptr_1 = out_cache_loc + out_offset_1
        
        load_offset_1 = tl.arange(0, BLOCK_SIZE) + kv_start_1
        save_offset_1 = tl.arange(0, BLOCK_SIZE)
        num_loop_1 = tl.cdiv(kv_end_1 - kv_start_1, BLOCK_SIZE)
        for _ in range(num_loop_1):
            mask_1 = load_offset_1 < kv_end_1
            data_1 = tl.load(token_pool_1 + load_offset_1, mask=mask_1)
            tl.store(out_cache_ptr_1 + save_offset_1, data_1, mask=mask_1)
            load_offset_1 += BLOCK_SIZE
            save_offset_1 += BLOCK_SIZE

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    grid = (triton.cdiv(req_pool_indices_tensor.numel(), 2),)
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