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
    BLOCK_SIZE: tl.constexpr = 64
    pid = tl.program_id(axis=0)
    req_id = pid // tl.cdiv(bs_upper, 1)
    tile_id = pid % tl.cdiv(bs_upper, 1)
    kv_start = tl.load(start_offset + req_id)
    kv_end = tl.load(end_offset + req_id)
    token_pool = req_to_token + tl.load(req_pool_indices + req_id) * pool_len
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < req_id, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < req_id, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset
    load_offset = kv_start + tile_id * BLOCK_SIZE
    save_offset = tile_id * BLOCK_SIZE
    num_loop = (kv_end - kv_start + BLOCK_SIZE - 1) >> 6
    for _ in range(num_loop):
        mask = load_offset + tl.arange(0, BLOCK_SIZE) < kv_end
        data = tl.load(token_pool + load_offset + tl.arange(0, BLOCK_SIZE), mask=mask)
        tl.store(out_cache_ptr + save_offset + tl.arange(0, BLOCK_SIZE), data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE