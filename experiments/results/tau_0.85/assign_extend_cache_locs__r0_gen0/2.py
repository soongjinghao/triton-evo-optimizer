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

    kv_len = kv_end - kv_start
    num_full = kv_len // BLOCK_SIZE

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    for _ in range(num_full):
        data = tl.load(token_pool + load_offset)
        tl.store(out_cache_ptr + save_offset, data)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE

    tail_base = num_full * BLOCK_SIZE
    tail_off = tl.arange(0, BLOCK_SIZE)
    tail_mask = tail_off < (kv_len - tail_base)

    tail_load_offset = kv_start + tail_base + tail_off
    tail_save_offset = tail_base + tail_off

    data = tl.load(token_pool + tail_load_offset, mask=tail_mask)
    tl.store(out_cache_ptr + tail_save_offset, data, mask=tail_mask)


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