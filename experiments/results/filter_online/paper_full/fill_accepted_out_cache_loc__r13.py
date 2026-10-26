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
    BLOCK: tl.constexpr = 128
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK
    offs = tl.arange(0, BLOCK)
    masks = (tl.load(accept_index + base_offset + offs, mask=base_offset + offs < size_upper, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)