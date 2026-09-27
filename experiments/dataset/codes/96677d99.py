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
    offset = tl.arange(0, size_upper)
    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src_vec = tl.load(accept_index + offset, offset == pid, other=-1)
    value_vec = tl.load(out_cache_loc + src_vec, src_vec > -1, other=0)
    tl.store(accepted_out_cache_loc + dst, value_vec)