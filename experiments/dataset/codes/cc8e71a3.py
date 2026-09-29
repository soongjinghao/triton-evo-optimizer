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
    BLOCK_BS: tl.constexpr = 4
    pid = tl.program_id(axis=0)
    pid_base = pid * BLOCK_BS
    offs = tl.arange(0, BLOCK_BS)
    req_indices = tl.load(req_pool_indices + pid_base + offs, mask=offs < bs_upper)
    starts = tl.load(start_offset + pid_base + offs, mask=offs < bs_upper)
    ends = tl.load(end_offset + pid_base + offs, mask=offs < bs_upper)
    lengths = ends - starts
    out_offsets = tl.cumsum(lengths, axis=0) - lengths
    out_cache_ptr_base = out_cache_loc + out_offsets
    for i in tl.static_range(BLOCK_BS):
        mask_i = offs < bs_upper
        kv_start = tl.load(start_offset + pid_base + i, mask=mask_i, other=0)
        kv_end = tl.load(end_offset + pid_base + i, mask=mask_i, other=0)
        token_pool = req_to_token + tl.load(req_pool_indices + pid_base + i, mask=mask_i, other=0) * pool_len
        out_cache_ptr = out_cache_ptr_base + i
        load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
        save_offset = tl.arange(0, BLOCK_SIZE)
        num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
        for _ in range(num_loop):
            mask = load_offset < kv_end
            data = tl.load(token_pool + load_offset, mask=mask)
            tl.store(out_cache_ptr + save_offset, data, mask=mask)
            load_offset += BLOCK_SIZE
            save_offset += BLOCK_SIZE