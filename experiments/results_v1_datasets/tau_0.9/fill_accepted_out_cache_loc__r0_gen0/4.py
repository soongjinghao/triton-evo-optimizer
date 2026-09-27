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
    dst = 0

    for i in range(0, size_upper, BLOCK):
        off = i + tl.arange(0, BLOCK)
        mask = off < pid
        loaded = tl.load(accept_index + off, mask=mask, other=-1)
        dst += tl.sum((loaded != -1).to(tl.int32))

    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)