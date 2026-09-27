import torch
import triton
import triton.language as tl

@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK: tl.constexpr = 128
    offset = tl.arange(0, BLOCK) + pid * BLOCK
    masks = (tl.load(accept_index + offset, offset < size_upper, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks, axis=0)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)