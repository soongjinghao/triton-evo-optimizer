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
def _assign_extend_cache_locs_pre_grid(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    prefix_offsets,
    pool_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_req = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)

    kv_start = tl.load(start_offset + pid_req)
    kv_end = tl.load(end_offset + pid_req)
    out_offset = tl.load(prefix_offsets + pid_req)
    token_pool = req_to_token + tl.load(req_pool_indices + pid_req) * pool_len

    block_start = kv_start + pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < kv_end

    data = tl.load(token_pool + offsets, mask=mask)
    save_offsets = out_offset + pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_cache_loc + save_offsets, data, mask=mask)

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    num_reqs = req_pool_indices_tensor.numel()
    if num_reqs == 0:
        return out_cache_loc_tensor

    valid_len = min(num_reqs, bs_upper)
    device = start_offset_tensor.device
    dtype = start_offset_tensor.dtype

    if valid_len <= 0:
        prefix_offsets = torch.zeros(num_reqs, dtype=dtype, device=device)
    else:
        start = start_offset_tensor[:valid_len]
        end = end_offset_tensor[:valid_len]
        lengths = end - start
        cumsum = torch.cumsum(lengths, dim=0)
        prefix_offsets = torch.zeros(num_reqs, dtype=dtype, device=device)
        prefix_offsets[:valid_len] = cumsum - lengths
        if valid_len < num_reqs:
            prefix_offsets[valid_len:] = cumsum[-1]

    prefix_offsets = prefix_offsets.contiguous()

    lengths_all = end_offset_tensor - start_offset_tensor
    max_len = int(lengths_all.max().item())
    if max_len <= 0:
        return out_cache_loc_tensor

    BLOCK_SIZE = 128
    num_blocks = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_reqs, num_blocks)

    _assign_extend_cache_locs_pre_grid[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        prefix_offsets,
        pool_len=pool_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_cache_loc_tensor