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
    
    # Each program handles 2 consecutive sequences
    pid0 = pid * 2
    pid1 = pid * 2 + 1
    
    # Load kv_start and kv_end for both sequences
    kv_start0 = tl.load(start_offset + pid0)
    kv_end0 = tl.load(end_offset + pid0)
    kv_start1 = tl.load(start_offset + pid1)
    kv_end1 = tl.load(end_offset + pid1)
    
    # Compute token pool base for both sequences
    token_pool0 = req_to_token + tl.load(req_pool_indices + pid0) * pool_len
    token_pool1 = req_to_token + tl.load(req_pool_indices + pid1) * pool_len
    
    # Compute output offset for both sequences
    length_offset = tl.arange(0, bs_upper)
    start_all = tl.load(start_offset + length_offset, mask=length_offset < pid1, other=0)
    end_all = tl.load(end_offset + length_offset, mask=length_offset < pid1, other=0)
    out_offset0 = tl.sum(end_all - start_all, mask=length_offset < pid0, axis=0)
    out_offset1 = tl.sum(end_all - start_all, mask=length_offset < pid1, axis=0)
    
    out_cache_ptr0 = out_cache_loc + out_offset0
    out_cache_ptr1 = out_cache_loc + out_offset1
    
    # Process sequence 0
    load_offset0 = tl.arange(0, BLOCK_SIZE) + kv_start0
    save_offset0 = tl.arange(0, BLOCK_SIZE)
    num_loop0 = tl.cdiv(kv_end0 - kv_start0, BLOCK_SIZE)
    for _ in range(num_loop0):
        mask0 = load_offset0 < kv_end0
        data0 = tl.load(token_pool0 + load_offset0, mask=mask0)
        tl.store(out_cache_ptr0 + save_offset0, data0, mask=mask0)
        load_offset0 += BLOCK_SIZE
        save_offset0 += BLOCK_SIZE
    
    # Process sequence 1
    load_offset1 = tl.arange(0, BLOCK_SIZE) + kv_start1
    save_offset1 = tl.arange(0, BLOCK_SIZE)
    num_loop1 = tl.cdiv(kv_end1 - kv_start1, BLOCK_SIZE)
    for _ in range(num_loop1):
        mask1 = load_offset1 < kv_end1
        data1 = tl.load(token_pool1 + load_offset1, mask=mask1)
        tl.store(out_cache_ptr1 + save_offset1, data1, mask=mask1)
        load_offset1 += BLOCK_SIZE
        save_offset1 += BLOCK_SIZE

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    numel = req_pool_indices_tensor.numel()
    # Use consolidated grid if numel is even, otherwise fallback to original grid
    if numel % 2 == 0:
        grid = (numel // 2,)
    else:
        grid = (numel,)
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