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
    BLOCK_PID: tl.constexpr = 4
    pid_base = tl.program_id(axis=0) * BLOCK_PID
    offs = tl.arange(0, BLOCK_PID)
    req_indices = tl.load(req_pool_indices + pid_base + offs, mask=pid_base + offs < bs_upper, other=0)
    starts = tl.load(start_offset + pid_base + offs, mask=pid_base + offs < bs_upper, other=0)
    ends = tl.load(end_offset + pid_base + offs, mask=pid_base + offs < bs_upper, other=0)
    out_offsets = tl.cumsum(ends - starts, axis=0) - (ends - starts)
    for i in tl.static_range(BLOCK_PID):
        kv_start = starts[i]
        kv_end = ends[i]
        token_pool = req_to_token + req_indices[i] * pool_len
        out_cache_ptr = out_cache_loc + out_offsets[i]
        load_off = kv_start
        save_off = 0
        num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
        for _ in range(num_loop):
            mask = load_off + tl.arange(0, BLOCK_SIZE) < kv_end
            data = tl.load(token_pool + load_off + tl.arange(0, BLOCK_SIZE), mask=mask)
            tl.store(out_cache_ptr + save_off + tl.arange(0, BLOCK_SIZE), data, mask=mask)
            load_off += BLOCK_SIZE
            save_off += BLOCK_SIZE