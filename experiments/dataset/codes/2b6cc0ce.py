import torch
import triton
import triton.language as tl

@triton.jit
def _precompute_out_offsets_kernel(
    start_offset,
    end_offset,
    precomputed_out_offsets,
    num_pids: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= num_pids:
        return
    # Compute prefix sum of (end - start) for all pids < current pid
    # Use a simple serial scan since num_pids is typically small
    total = 0
    for i in range(pid):
        s = tl.load(start_offset + i)
        e = tl.load(end_offset + i)
        total += e - s
    tl.store(precomputed_out_offsets + pid, total)

@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    precomputed_out_offsets,
    pool_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    out_offset = tl.load(precomputed_out_offsets + pid)
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

def assign_extend_cache_locs_wrapper(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len,
):
    num_pids = start_offset.shape[0]
    precomputed_out_offsets = torch.empty(num_pids, dtype=torch.int64, device=start_offset.device)
    grid = (num_pids,)
    _precompute_out_offsets_kernel[grid](
        start_offset,
        end_offset,
        precomputed_out_offsets,
        num_pids,
    )
    assign_extend_cache_locs[grid](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        precomputed_out_offsets,
        pool_len,
    )
    return out_cache_loc