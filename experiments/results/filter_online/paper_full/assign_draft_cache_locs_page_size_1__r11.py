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
    BLOCK_SIZE: tl.constexpr = 128
    copy_len = topk * speculative_num_steps
    pid = tl.program_id(axis=0)
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    kv_start = tl.load(seq_lens + pid)
    token_pool_base = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    base_ptr = token_pool_base + kv_start
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(base_ptr + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)