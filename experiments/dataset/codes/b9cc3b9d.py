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
    K: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    base_req_idx = pid * K
    num_requests = tl.load(req_pool_indices + base_req_idx + K - 1, mask=base_req_idx + K - 1 < tl.num_programs(axis=0) * K, other=0)
    for k in range(K):
        req_idx = base_req_idx + k
        mask_req = req_idx < tl.num_programs(axis=0) * K
        kv_start = tl.load(start_offset + req_idx, mask=mask_req, other=0)
        kv_end = tl.load(end_offset + req_idx, mask=mask_req, other=0)
        token_pool = req_to_token + tl.load(req_pool_indices + req_idx, mask=mask_req, other=0) * pool_len
        length_offset = tl.arange(0, bs_upper)
        start = tl.load(start_offset + length_offset, mask=length_offset < req_idx, other=0)
        end = tl.load(end_offset + length_offset, mask=length_offset < req_idx, other=0)
        out_offset = tl.sum(end - start, axis=0)
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
    K = 4
    num_requests = req_pool_indices_tensor.numel()
    grid = ((num_requests + K - 1) // K,)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper,
        K=K
    )
    return out_cache_loc_tensor