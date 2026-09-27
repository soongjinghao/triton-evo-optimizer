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
    offset = tl.arange(0, BLOCK)
    cumsum = 0
    for start in range(0, size_upper, BLOCK):
        off = start + offset
        mask = off < size_upper
        idx = tl.load(accept_index + off, mask=mask, other=-1)
        valid = (idx != -1).to(tl.int64)
        valid_masked = tl.where(off < pid, valid, 0)
        cumsum += tl.sum(valid_masked)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + cumsum, value)