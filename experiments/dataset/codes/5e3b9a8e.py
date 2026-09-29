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
    BLOCK_SIZE: tl.constexpr = 512
    copy_len = topk * speculative_num_steps
    num_pids = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    out_cache_ptr = out_cache_loc + pid * copy_len
    kv_start = tl.load(seq_lens + pid)
    pool_index = tl.load(req_pool_indices + pid)
    token_offset = pool_index * pool_len + kv_start
    token_pool_base = req_to_token + token_offset
    num_full_blocks = copy_len // BLOCK_SIZE
    remainder = copy_len % BLOCK_SIZE
    for i in range(num_full_blocks):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        data_int64 = tl.load(token_pool_base + copy_offset)
        data_int32 = data_int64.to(tl.int32)
        tl.store(out_cache_ptr + copy_offset, data_int32)
    if remainder > 0:
        remainder_offset = tl.arange(0, BLOCK_SIZE) + num_full_blocks * BLOCK_SIZE
        remainder_mask = remainder_offset < copy_len
        data_int64 = tl.load(token_pool_base + remainder_offset, mask=remainder_mask)
        data_int32 = data_int64.to(tl.int32)
        tl.store(out_cache_ptr + remainder_offset, data_int32, mask=remainder_mask)

def assign_draft_cache_locs_page_size_1_wrapper(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len,
    topk,
    speculative_num_steps,
):
    copy_len = topk * speculative_num_steps
    batch_size = req_pool_indices.shape[0]
    grid = (batch_size,)
    num_warps = 4
    num_stages = 2
    assign_draft_cache_locs_page_size_1[grid](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        pool_len,
        topk,
        speculative_num_steps,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_cache_loc