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
    out_offsets,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 64
    pid = tl.program_id(0)
    req_start = pid * GROUP_SIZE
    req_end = tl.minimum(req_start + GROUP_SIZE, NUM_REQUESTS)
    for i in range(GROUP_SIZE):
        req_idx = req_start + i
        if req_idx < req_end:
            kv_start = tl.load(start_offset + req_idx)
            kv_end = tl.load(end_offset + req_idx)
            out_offset = tl.load(out_offsets + req_idx)
            token_pool = req_to_token + tl.load(req_pool_indices + req_idx) * pool_len
            load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
            save_offset = tl.arange(0, BLOCK_SIZE)
            num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
            for _ in range(num_loop):
                mask = load_offset < kv_end
                data = tl.load(token_pool + load_offset, mask=mask)
                tl.store(out_cache_loc + out_offset + save_offset, data, mask=mask)
                load_offset += BLOCK_SIZE
                save_offset += BLOCK_SIZE

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    num_requests = req_pool_indices_tensor.numel()
    lens = end_offset_tensor - start_offset_tensor
    out_offsets = torch.cumsum(lens, dim=0) - lens
    GROUP_SIZE = 4
    grid = (triton.cdiv(num_requests, GROUP_SIZE),)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        out_offsets,
        pool_len=pool_len,
        bs_upper=bs_upper,
        GROUP_SIZE=GROUP_SIZE,
        NUM_REQUESTS=num_requests,
    )
    return out_cache_loc_tensor