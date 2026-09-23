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
    exclusive_prefix,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
    N: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    MAX_LOOP: tl.constexpr = tl.cdiv(pool_len, BLOCK_SIZE)
    pid = tl.program_id(axis=0)
    req_start = pid * REQ_BLOCK

    for i in tl.static_range(REQ_BLOCK):
        req_idx = req_start + i
        valid = req_idx < N

        kv_start = tl.load(start_offset + req_idx, mask=valid, other=0)
        kv_end = tl.load(end_offset + req_idx, mask=valid, other=0)
        out_offset = tl.load(exclusive_prefix + req_idx, mask=valid, other=0)
        req_pool_idx = tl.load(req_pool_indices + req_idx, mask=valid, other=0)
        token_pool = req_to_token + req_pool_idx * pool_len

        for j in tl.static_range(MAX_LOOP):
            load_offset = tl.arange(0, BLOCK_SIZE) + kv_start + j * BLOCK_SIZE
            mask = (load_offset < kv_end) & valid
            data = tl.load(token_pool + load_offset, mask=mask, other=0)
            save_offset = tl.arange(0, BLOCK_SIZE) + out_offset + j * BLOCK_SIZE
            tl.store(out_cache_loc + save_offset, data, mask=mask)


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
    lengths = end_offset_tensor - start_offset_tensor
    exclusive_prefix = torch.zeros_like(lengths)
    if N > 0:
        exclusive_prefix[1:] = torch.cumsum(lengths, dim=0)[:-1]

    # 选择请求块大小,避免静态展开过大造成寄存器压力
    if pool_len <= 1024:
        REQ_BLOCK = 4
    else:
        REQ_BLOCK = 1

    grid = (triton.cdiv(N, REQ_BLOCK),)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        exclusive_prefix,
        pool_len=pool_len,
        bs_upper=bs_upper,
        N=N,
        REQ_BLOCK=REQ_BLOCK
    )
    return out_cache_loc_tensor