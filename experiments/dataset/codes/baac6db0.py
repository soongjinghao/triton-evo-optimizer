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
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    kv_start = tl.load(start_offset + pid0)
    kv_end = tl.load(end_offset + pid0)
    token_pool = req_to_token + tl.load(req_pool_indices + pid0) * pool_len
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid0, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid0, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset
    load_offset = kv_start + pid1 * BLOCK_SIZE
    save_offset = pid1 * BLOCK_SIZE
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
    max_kv_len = 0
    for i in range(req_pool_indices_tensor.numel()):
        kv_start = start_offset_tensor[i].item()
        kv_end = end_offset_tensor[i].item()
        max_kv_len = max(max_kv_len, kv_end - kv_start)
    grid = (req_pool_indices_tensor.numel(), triton.cdiv(max_kv_len, BLOCK_SIZE))
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