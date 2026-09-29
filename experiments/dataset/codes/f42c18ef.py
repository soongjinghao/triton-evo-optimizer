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
    BLOCK_SIZE: tl.constexpr = 256
    copy_len: tl.constexpr = topk * speculative_num_steps
    num_blocks_per_request: tl.constexpr = (copy_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(axis=0)
    request_id = pid // num_blocks_per_request
    block_id = pid % num_blocks_per_request
    out_cache_ptr = out_cache_loc + request_id * copy_len
    kv_start = tl.load(seq_lens + request_id)
    pool_index = tl.load(req_pool_indices + request_id)
    token_offset = pool_index * pool_len + kv_start
    token_pool_base = req_to_token + token_offset
    copy_offset = tl.arange(0, BLOCK_SIZE) + block_id * BLOCK_SIZE
    mask = copy_offset < copy_len
    data_int64 = tl.load(token_pool_base + copy_offset, mask=mask)
    data_int32 = data_int64.to(tl.int32)
    tl.store(out_cache_ptr + copy_offset, data_int32, mask=mask)