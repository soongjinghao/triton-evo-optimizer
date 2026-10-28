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
    base_offset = tl.load(req_pool_indices + pid) * pool_len
    token_pool = req_to_token + base_offset
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
def assign_extend_cache_locs_half(
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
    req_idx = pid // 2
    half_idx = pid % 2
    kv_start = tl.load(start_offset + req_idx)
    kv_end = tl.load(end_offset + req_idx)
    half_len = (kv_end - kv_start) // 2
    half_start = kv_start + half_idx * half_len
    half_end = half_start + half_len
    base_offset = tl.load(req_pool_indices + req_idx) * pool_len
    token_pool = req_to_token + base_offset
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < req_idx, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < req_idx, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset + half_idx * half_len
    load_offset = tl.arange(0, BLOCK_SIZE) + half_start
    save_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(half_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < half_end
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
    num_reqs = req_pool_indices_tensor.numel()
    use_half = True
    for i in range(num_reqs):
        kv_start = start_offset_tensor[i].item()
        kv_end = end_offset_tensor[i].item()
        if (kv_end - kv_start) % 2 != 0:
            use_half = False
            break
    if use_half:
        grid = (num_reqs * 2,)
        assign_extend_cache_locs_half[grid](
            req_pool_indices_tensor,
            req_to_token_tensor,
            start_offset_tensor,
            end_offset_tensor,
            out_cache_loc_tensor,
            pool_len=pool_len,
            bs_upper=bs_upper
        )
    else:
        grid = (num_reqs,)
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