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
    precomputed_offset,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    kv_start = tl.load(start_offset + bid)
    kv_end = tl.load(end_offset + bid)
    token_pool = req_to_token + tl.load(req_pool_indices + bid) * pool_len
    out_offset = tl.load(precomputed_offset + bid)
    out_cache_ptr = out_cache_loc + out_offset
    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start + pid * BLOCK_SIZE
    save_offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = load_offset < kv_end
    data = tl.load(token_pool + load_offset, mask=mask)
    tl.store(out_cache_ptr + save_offset, data, mask=mask)

def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    num_tokens = req_pool_indices_tensor.numel()
    max_kv_len = 0
    for i in range(num_tokens):
        s = start_offset_tensor[i].item()
        e = end_offset_tensor[i].item()
        if e - s > max_kv_len:
            max_kv_len = e - s
    precomputed_offset = torch.zeros(num_tokens, dtype=torch.int64, device=req_pool_indices_tensor.device)
    cum = 0
    for i in range(num_tokens):
        precomputed_offset[i] = cum
        cum += (end_offset_tensor[i].item() - start_offset_tensor[i].item())
    BLOCK_SIZE = 32
    num_blocks = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks, num_tokens)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        precomputed_offset,
        pool_len=pool_len,
        bs_upper=bs_upper
    )
    return out_cache_loc_tensor