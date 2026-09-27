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
    BLOCK_REQUESTS: tl.constexpr = 4
    pid = tl.program_id(axis=0)
    base_req = pid * BLOCK_REQUESTS
    kv_start = tl.load(start_offset + base_req + tl.arange(0, BLOCK_REQUESTS))
    kv_end = tl.load(end_offset + base_req + tl.arange(0, BLOCK_REQUESTS))
    token_pool_base = req_to_token + tl.load(req_pool_indices + base_req + tl.arange(0, BLOCK_REQUESTS)) * pool_len
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < base_req, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < base_req, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset
    load_offset = tl.arange(0, BLOCK_SIZE * BLOCK_REQUESTS)
    save_offset = tl.arange(0, BLOCK_SIZE * BLOCK_REQUESTS)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool_base + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE * BLOCK_REQUESTS
        save_offset += BLOCK_SIZE * BLOCK_REQUESTS