import torch
import triton
import triton.language as tl

@triton.jit
def _prefix_cum_lengths_kernel(
    start_offset,
    end_offset,
    cum_lengths,
    n: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid == 0:
        tl.store(cum_lengths, 0)
    else:
        length = tl.load(end_offset + pid - 1) - tl.load(start_offset + pid - 1)
        prev_cum = tl.load(cum_lengths + pid - 1)
        tl.store(cum_lengths + pid, prev_cum + length)

@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    cum_lengths,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    out_offset = tl.load(cum_lengths + pid, mask=pid > 0, other=0)
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
    n = req_pool_indices_tensor.numel()
    cum_lengths = torch.zeros(n + 1, dtype=torch.int64, device=req_pool_indices_tensor.device)
    grid_prefix = (n + 1,)
    _prefix_cum_lengths_kernel[grid_prefix](
        start_offset_tensor,
        end_offset_tensor,
        cum_lengths,
        n=n,
    )
    grid = (n,)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        cum_lengths,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper
    )
    return out_cache_loc_tensor