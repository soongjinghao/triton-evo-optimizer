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
    max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 32,
):
    num_blocks_per_seq = tl.cdiv(max_seq_len, BLOCK_SIZE)
    pid = tl.program_id(axis=0)
    seq_idx = pid // num_blocks_per_seq
    block_idx = pid % num_blocks_per_seq

    kv_start = tl.load(start_offset + seq_idx)
    kv_end = tl.load(end_offset + seq_idx)
    token_pool = req_to_token + tl.load(req_pool_indices + seq_idx) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < seq_idx, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < seq_idx, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = kv_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    save_offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = load_offset < kv_end
    data = tl.load(token_pool + load_offset, mask=mask)
    tl.store(out_cache_ptr + save_offset, data, mask=mask)