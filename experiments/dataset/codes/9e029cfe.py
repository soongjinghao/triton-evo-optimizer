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
    pid2 = pid * 2
    pid2_plus_1 = pid2 + 1

    kv_start_0 = tl.load(start_offset + pid2)
    kv_end_0 = tl.load(end_offset + pid2)
    token_pool_0 = req_to_token + tl.load(req_pool_indices + pid2) * pool_len

    kv_start_1 = tl.load(start_offset + pid2_plus_1)
    kv_end_1 = tl.load(end_offset + pid2_plus_1)
    token_pool_1 = req_to_token + tl.load(req_pool_indices + pid2_plus_1) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start_0 = tl.load(start_offset + length_offset, mask=length_offset < pid2, other=0)
    end_0 = tl.load(end_offset + length_offset, mask=length_offset < pid2, other=0)
    out_offset_0 = tl.sum(end_0 - start_0, axis=0)
    out_cache_ptr_0 = out_cache_loc + out_offset_0

    start_1 = tl.load(start_offset + length_offset, mask=length_offset < pid2_plus_1, other=0)
    end_1 = tl.load(end_offset + length_offset, mask=length_offset < pid2_plus_1, other=0)
    out_offset_1 = tl.sum(end_1 - start_1, axis=0)
    out_cache_ptr_1 = out_cache_loc + out_offset_1

    load_offset_0 = tl.arange(0, BLOCK_SIZE) + kv_start_0
    save_offset_0 = tl.arange(0, BLOCK_SIZE)
    num_loop_0 = tl.cdiv(kv_end_0 - kv_start_0, BLOCK_SIZE)
    for _ in range(num_loop_0):
        mask = load_offset_0 < kv_end_0
        data = tl.load(token_pool_0 + load_offset_0, mask=mask)
        tl.store(out_cache_ptr_0 + save_offset_0, data, mask=mask)
        load_offset_0 += BLOCK_SIZE
        save_offset_0 += BLOCK_SIZE

    load_offset_1 = tl.arange(0, BLOCK_SIZE) + kv_start_1
    save_offset_1 = tl.arange(0, BLOCK_SIZE)
    num_loop_1 = tl.cdiv(kv_end_1 - kv_start_1, BLOCK_SIZE)
    for _ in range(num_loop_1):
        mask = load_offset_1 < kv_end_1
        data = tl.load(token_pool_1 + load_offset_1, mask=mask)
        tl.store(out_cache_ptr_1 + save_offset_1, data, mask=mask)
        load_offset_1 += BLOCK_SIZE
        save_offset_1 += BLOCK_SIZE

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    N = req_pool_indices_tensor.numel()
    assert N % 2 == 0, "N must be even for grid consolidation"
    grid = (N // 2,)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper
    )
    return out_cache_loc_tensor