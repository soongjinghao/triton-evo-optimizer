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
    kv_len = kv_end - kv_start

    token_base = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    load_base = token_base + kv_start

    length_offset = tl.arange(0, bs_upper)
    prefix_mask = length_offset < pid
    start = tl.load(start_offset + length_offset, mask=prefix_mask, other=0)
    end = tl.load(end_offset + length_offset, mask=prefix_mask, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_base = out_cache_loc + out_offset

    num_loop = tl.cdiv(kv_len, BLOCK_SIZE)
    for i in range(num_loop):
        off = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = kv_start + off < kv_end
        data = tl.load(load_base + off, mask=mask)
        tl.store(out_base + off, data, mask=mask)


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