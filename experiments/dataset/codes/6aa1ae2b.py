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
    total_elements = tl.num_programs(axis=0) * copy_len
    pid = tl.program_id(axis=0)
    elem_start = pid * BLOCK_SIZE
    elem_end = elem_start + BLOCK_SIZE
    if elem_end <= total_elements:
        batch_id = elem_start // copy_len
        copy_offset_in_batch = elem_start % copy_len
        kv_start = tl.load(seq_lens + batch_id)
        pool_index = tl.load(req_pool_indices + batch_id)
        token_offset = pool_index * pool_len + kv_start
        token_pool_base = req_to_token + token_offset
        out_cache_ptr = out_cache_loc + batch_id * copy_len
        copy_offset = tl.arange(0, BLOCK_SIZE) + copy_offset_in_batch
        data_int64 = tl.load(token_pool_base + copy_offset)
        data_int32 = data_int64.to(tl.int32)
        tl.store(out_cache_ptr + copy_offset, data_int32)
    else:
        batch_id = elem_start // copy_len
        copy_offset_in_batch = elem_start % copy_len
        kv_start = tl.load(seq_lens + batch_id)
        pool_index = tl.load(req_pool_indices + batch_id)
        token_offset = pool_index * pool_len + kv_start
        token_pool_base = req_to_token + token_offset
        out_cache_ptr = out_cache_loc + batch_id * copy_len
        copy_offset = tl.arange(0, BLOCK_SIZE) + copy_offset_in_batch
        mask = copy_offset < batch_id * copy_len + copy_len
        data_int64 = tl.load(token_pool_base + copy_offset, mask=mask)
        data_int32 = data_int64.to(tl.int32)
        tl.store(out_cache_ptr + copy_offset, data_int32, mask=mask)