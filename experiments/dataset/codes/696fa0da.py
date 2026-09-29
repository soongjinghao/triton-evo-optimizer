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
    K: tl.constexpr = 4
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    base_idx = pid * K
    kv_start_offsets = tl.load(start_offset + base_idx + tl.arange(0, K), mask=tl.arange(0, K) < K)
    kv_end_offsets = tl.load(end_offset + base_idx + tl.arange(0, K), mask=tl.arange(0, K) < K)
    req_indices = tl.load(req_pool_indices + base_idx + tl.arange(0, K), mask=tl.arange(0, K) < K)
    token_pool_bases = req_to_token + req_indices * pool_len
    length_offsets = tl.arange(0, bs_upper)
    start_vals = tl.load(start_offset + length_offsets, mask=length_offsets < base_idx, other=0)
    end_vals = tl.load(end_offset + length_offsets, mask=length_offsets < base_idx, other=0)
    out_offset = tl.sum(end_vals - start_vals, axis=0)
    out_cache_ptr = out_cache_loc + out_offset
    kv_starts = kv_start_offsets
    kv_ends = kv_end_offsets
    max_len = tl.max(kv_ends - kv_starts)
    num_loop = tl.cdiv(max_len, BLOCK_SIZE)
    for i in range(num_loop):
        load_offsets = tl.arange(0, BLOCK_SIZE) + kv_starts[:, None] + i * BLOCK_SIZE
        save_offsets = tl.arange(0, BLOCK_SIZE) + out_offset + i * BLOCK_SIZE
        mask = load_offsets < kv_ends[:, None]
        data = tl.load(token_pool_bases[:, None] + load_offsets, mask=mask)
        tl.store(out_cache_ptr + save_offsets, data, mask=mask)

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    K: tl.constexpr = 4
    grid = ((req_pool_indices_tensor.numel() + K - 1) // K,)
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