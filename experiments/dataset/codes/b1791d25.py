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
    BLOCK_REQ: tl.constexpr = 4
    copy_len: tl.constexpr = topk * speculative_num_steps
    log2_pool_len: tl.constexpr = 31 - tl.clz(pool_len) if (pool_len & (pool_len - 1)) == 0 else 0
    is_pow2_pool_len: tl.constexpr = (pool_len & (pool_len - 1)) == 0
    pid = tl.program_id(axis=0)
    for r in range(BLOCK_REQ):
        req_id = pid * BLOCK_REQ + r
        out_cache_ptr = out_cache_loc + req_id * copy_len
        kv_start = tl.load(seq_lens + req_id)
        pool_index = tl.load(req_pool_indices + req_id)
        if is_pow2_pool_len:
            token_offset = (pool_index << log2_pool_len) + kv_start
        else:
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