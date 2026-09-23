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
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)

    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset

    num_tokens = kv_end - kv_start
    num_loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    token_pool_start = token_pool + kv_start
    rel_offset = tl.arange(0, BLOCK_SIZE)

    for _ in range(num_loop):
        mask = rel_offset < num_tokens
        data = tl.load(token_pool_start + rel_offset, mask=mask)
        tl.store(out_cache_ptr + rel_offset, data, mask=mask)
        rel_offset += BLOCK_SIZE


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