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
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)

    req_idx = tl.load(req_pool_indices + pid)
    token_pool = req_to_token + req_idx * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_base = out_cache_loc + out_offset

    copy_len = kv_end - kv_start
    src_base = token_pool + kv_start

    off = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = off < copy_len
        data = tl.load(src_base + off, mask=mask)
        tl.store(out_base + off, data, mask=mask)
        off += BLOCK_SIZE