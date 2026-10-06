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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    batch_idx = pid // tl.cdiv(num_blocks, tl.constexpr(1))
    block_idx = pid % tl.cdiv(num_blocks, tl.constexpr(1))
    
    kv_start = tl.load(start_offset + batch_idx)
    kv_end = tl.load(end_offset + batch_idx)
    token_pool = req_to_token + tl.load(req_pool_indices + batch_idx) * pool_len
    
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < batch_idx, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < batch_idx, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset
    
    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start + block_idx * BLOCK_SIZE
    save_offset = tl.arange(0, BLOCK_SIZE) + block_idx * BLOCK_SIZE
    mask = load_offset < kv_end
    data = tl.load(token_pool + load_offset, mask=mask)
    tl.store(out_cache_ptr + save_offset, data, mask=mask)

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    BLOCK_SIZE = 32
    num_batches = req_pool_indices_tensor.numel()
    total_blocks = 0
    for i in range(num_batches):
        kv_start = start_offset_tensor[i].item()
        kv_end = end_offset_tensor[i].item()
        total_blocks += triton.cdiv(kv_end - kv_start, BLOCK_SIZE)
    grid = (total_blocks,)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_cache_loc_tensor