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

    total = kv_end - kv_start
    num_full = total // BLOCK_SIZE
    rem = total - num_full * BLOCK_SIZE

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)
    for _ in range(num_full):
        data = tl.load(token_pool + load_offset)
        tl.store(out_cache_ptr + save_offset, data)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE

    tail_load_offset = kv_start + num_full * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tail_mask = tl.arange(0, BLOCK_SIZE) < rem
    data = tl.load(token_pool + tail_load_offset, mask=tail_mask)
    tl.store(out_cache_ptr + num_full * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), data, mask=tail_mask)


@triton.jit
def _assign_extend_cache_locs_precomputed(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    out_offsets,
    pool_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    out_cache_ptr = out_cache_loc + tl.load(out_offsets + pid)

    total = kv_end - kv_start
    num_full = total // BLOCK_SIZE
    rem = total - num_full * BLOCK_SIZE

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)
    for _ in range(num_full):
        data = tl.load(token_pool + load_offset)
        tl.store(out_cache_ptr + save_offset, data)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE

    tail_load_offset = kv_start + num_full * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tail_mask = tl.arange(0, BLOCK_SIZE) < rem
    data = tl.load(token_pool + tail_load_offset, mask=tail_mask)
    tl.store(out_cache_ptr + num_full * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), data, mask=tail_mask)


def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    lengths = end_offset_tensor - start_offset_tensor
    out_offsets = torch.cumsum(lengths, dim=0) - lengths
    out_offsets = out_offsets.to(start_offset_tensor.dtype)

    grid = (req_pool_indices_tensor.numel(),)
    _assign_extend_cache_locs_precomputed[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        out_offsets,
        pool_len=pool_len
    )
    return out_cache_loc_tensor