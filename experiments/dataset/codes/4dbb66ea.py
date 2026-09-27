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
    base_pid = pid * 4
    offset = tl.arange(0, size_upper)
    for i in range(4):
        cur_pid = base_pid + i
        if cur_pid < size_upper:
            masks = (tl.load(accept_index + offset, offset < cur_pid, other=-1) != -1).to(tl.int64)
            dst = tl.sum(masks)
            src = tl.load(accept_index + cur_pid)
            if src > -1:
                value = tl.load(out_cache_loc + src)
                tl.store(accepted_out_cache_loc + dst, value)