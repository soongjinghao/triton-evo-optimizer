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
    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


@triton.jit
def _assign_extend_cache_locs_precomputed_prefix(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    prefix_offsets,
    out_cache_loc,
    pool_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    out_offset = tl.load(prefix_offsets + pid)
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
    n_requests = req_pool_indices_tensor.numel()
    grid = (n_requests,)

    if n_requests < 16:
        assign_extend_cache_locs[grid](
            req_pool_indices_tensor,
            req_to_token_tensor,
            start_offset_tensor,
            end_offset_tensor,
            out_cache_loc_tensor,
            pool_len=pool_len,
            bs_upper=bs_upper
        )
    else:
        lengths = end_offset_tensor - start_offset_tensor
        exclusive_prefix = torch.cat([
            torch.zeros_like(lengths[:1]),
            torch.cumsum(lengths, dim=0)[:-1]
        ])
        _assign_extend_cache_locs_precomputed_prefix[grid](
            req_pool_indices_tensor,
            req_to_token_tensor,
            start_offset_tensor,
            end_offset_tensor,
            exclusive_prefix,
            out_cache_loc_tensor,
            pool_len=pool_len
        )

    return out_cache_loc_tensor