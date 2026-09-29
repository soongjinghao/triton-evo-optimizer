import torch
import triton
import triton.language as tl
import torch_npu
device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream
@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    masks = offsets < size_upper
    accept_vals = tl.load(accept_index + offsets, mask=masks, other=-1)
    valid_mask = accept_vals != -1
    valid_count = tl.sum(valid_mask.to(tl.int64))
    prefix_sum = tl.cumsum(valid_mask.to(tl.int64), axis=0) - 1
    valid_prefix = tl.where(valid_mask, prefix_sum, -1)
    base_dst = tl.sum(tl.where(offsets < block_start, (tl.load(accept_index + offsets, mask=offsets < size_upper, other=-1) != -1).to(tl.int64), 0))
    for i in range(BLOCK_SIZE):
        src = tl.load(accept_index + block_start + i)
        if src > -1:
            dst = base_dst + tl.sum(tl.where(tl.arange(0, i) < i, (tl.load(accept_index + block_start + tl.arange(0, i), mask=(block_start + tl.arange(0, i)) < size_upper, other=-1) != -1).to(tl.int64), 0))
            value = tl.load(out_cache_loc + src)
            tl.store(accepted_out_cache_loc + dst, value)
def fill_accepted_out_cache_loc_wrapper(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
):
    size_upper = accept_index.shape[0]
    BLOCK_SIZE = 128
    grid = ((size_upper + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fill_accepted_out_cache_loc[grid](
        accept_index,
        out_cache_loc,
        accepted_out_cache_loc,
        size_upper,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return accepted_out_cache_loc