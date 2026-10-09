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
    # Prefetch first block
    mask_curr = load_offset < kv_end
    data_curr = tl.load(token_pool + load_offset, mask=mask_curr)
    load_offset += BLOCK_SIZE
    for _ in range(1, num_loop):
        mask_next = load_offset < kv_end
        data_next = tl.load(token_pool + load_offset, mask=mask_next)
        tl.store(out_cache_ptr + save_offset, data_curr, mask=mask_curr)
        data_curr = data_next
        mask_curr = mask_next
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE
    # Store last block
    tl.store(out_cache_ptr + save_offset, data_curr, mask=mask_curr)