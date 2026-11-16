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
    BLOCK = 256
    dst = 0
    for start in range(0, size_upper, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < pid
        vals = tl.load(accept_index + offsets, mask=mask, other=-1)
        dst += tl.sum((vals != -1).to(tl.int64))
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)