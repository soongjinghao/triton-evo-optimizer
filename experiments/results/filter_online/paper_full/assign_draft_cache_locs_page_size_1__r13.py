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
    copy_len: tl.constexpr = topk * speculative_num_steps
    BLOCK_SIZE: tl.constexpr = copy_len
    pid = tl.program_id(axis=0)
    out_cache_ptr = out_cache_loc + pid * copy_len
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    copy_offset = tl.arange(0, BLOCK_SIZE)
    data = tl.load(token_pool + kv_start + copy_offset)
    tl.store(out_cache_ptr + copy_offset, data)