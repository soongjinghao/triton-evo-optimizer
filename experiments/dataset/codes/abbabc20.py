import torch
import triton
import triton.language as tl

@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = 8
    BLOCK_SIZE: tl.constexpr = 256
    pid = tl.program_id(axis=0)
    base_pid = pid * GROUP_SIZE
    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + base_pid * copy_len
    pid_offsets = tl.arange(0, GROUP_SIZE)
    seq_lens_vals = tl.load(seq_lens + base_pid + pid_offsets)
    pool_indices = tl.load(req_pool_indices + base_pid + pid_offsets)
    num_full_blocks = copy_len // BLOCK_SIZE
    remainder = copy_len % BLOCK_SIZE
    for i in range(num_full_blocks):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        for g in range(GROUP_SIZE):
            pid_val = base_pid + g
            kv_start = tl.load(seq_lens_vals + g)
            pool_index = tl.load(pool_indices + g)
            token_offset = pool_index * pool_len + kv_start
            token_pool_base = req_to_token + token_offset
            data_int64 = tl.load(token_pool_base + copy_offset)
            data_int32 = data_int64.to(tl.int32)
            tl.store(out_cache_ptr + g * copy_len + copy_offset, data_int32)
    if remainder > 0:
        remainder_offset = tl.arange(0, BLOCK_SIZE) + num_full_blocks * BLOCK_SIZE
        remainder_mask = remainder_offset < copy_len
        for g in range(GROUP_SIZE):
            pid_val = base_pid + g
            kv_start = tl.load(seq_lens_vals + g)
            pool_index = tl.load(pool_indices + g)
            token_offset = pool_index * pool_len + kv_start
            token_pool_base = req_to_token + token_offset
            data_int64 = tl.load(token_pool_base + remainder_offset, mask=remainder_mask)
            data_int32 = data_int64.to(tl.int32)
            tl.store(out_cache_ptr + g * copy_len + remainder_offset, data_int32, mask=remainder_mask)