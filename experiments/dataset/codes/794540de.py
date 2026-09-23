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
def _assign_extend_cache_locs_opt(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    prefix_offsets,
    pool_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid_req = tl.program_id(axis=0)
    pid_blk = tl.program_id(axis=1)
    kv_start = tl.load(start_offset + pid_req)
    kv_end = tl.load(end_offset + pid_req)
    out_base = tl.load(prefix_offsets + pid_req)
    token_pool = req_to_token + tl.load(req_pool_indices + pid_req) * pool_len
    total = kv_end - kv_start
    num_full = total // BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = kv_start + pid_blk * BLOCK_SIZE
    out_block_start = out_base + pid_blk * BLOCK_SIZE
    if pid_blk < num_full:
        load_offset = block_start + offsets
        data = tl.load(token_pool + load_offset)
        tl.store(out_cache_loc + out_block_start + offsets, data)
    else:
        load_offset = block_start + offsets
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_loc + out_block_start + offsets, data, mask=mask)

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
    start = start_offset_tensor[:num_reqs]
    end = end_offset_tensor[:num_reqs]
    lengths = end - start
    prefix_offsets = torch.cumsum(lengths, dim=0) - lengths
    prefix_offsets = prefix_offsets.contiguous()
    max_len = int(lengths.max().item())
    if max_len < 0:
        max_len = 0
    max_copy = max(1, (max_len + 32 - 1) // 32)
    grid = (num_reqs, max_copy)
    _assign_extend_cache_locs_opt[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        prefix_offsets,
        pool_len=pool_len,
    )
    return out_cache_loc_tensor