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

    offset = tl.arange(0, size_upper + 1)
    loaded = tl.load(accept_index + offset, mask=offset < pid + 1, other=-1)

    dst = tl.sum(((loaded != -1) & (offset < pid)).to(tl.int64))
    src = tl.sum(tl.where(offset == pid, loaded, 0))

    value = tl.load(out_cache_loc + src, mask=src > -1, other=0)
    tl.store(accepted_out_cache_loc + dst, value, mask=src > -1)