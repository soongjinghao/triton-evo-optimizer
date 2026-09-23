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
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset
    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


@triton.jit
def _assign_extend_cache_locs_meta(
    metadata_tensor,
    req_to_token,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    meta_ptr = metadata_tensor + pid * 4

    req_pool_idx = tl.load(meta_ptr + 0)
    kv_start = tl.load(meta_ptr + 1)
    kv_end = tl.load(meta_ptr + 2)
    out_offset = tl.load(meta_ptr + 3)

    token_pool = req_to_token + req_pool_idx * pool_len
    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)
    out_cache_ptr = out_cache_loc + out_offset
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
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
    req_pool_indices = req_pool_indices_tensor.contiguous()
    start_offset = start_offset_tensor.contiguous()
    end_offset = end_offset_tensor.contiguous()

    lengths = end_offset - start_offset
    exclusive_prefix = torch.cumsum(lengths, dim=0) - lengths

    metadata_tensor = torch.stack([
        req_pool_indices.to(torch.int64),
        start_offset.to(torch.int64),
        end_offset.to(torch.int64),
        exclusive_prefix.to(torch.int64),
    ], dim=-1).contiguous()

    grid = (req_pool_indices.numel(),)
    _assign_extend_cache_locs_meta[grid](
        metadata_tensor,
        req_to_token_tensor,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper,
    )
    return out_cache_loc_tensor