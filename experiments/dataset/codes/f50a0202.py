import triton
import triton.language as tl

@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    GROUP = 16
    pid = tl.program_id(0)
    group = pid
    if group * GROUP >= size_upper:
        return

    offset = tl.arange(0, size_upper)
    acc_vec = tl.load(accept_index + offset)
    cond = (acc_vec != -1).to(tl.int32)
    group_start = group * GROUP
    base = tl.sum(tl.where(offset < group_start, cond, 0))

    group_cnt = tl.zeros((1,), tl.int32)
    for k in range(GROUP):
        idx = group_start + k
        src = tl.load(accept_index + idx, mask=(idx < size_upper), other=-1)
        is_valid = src > -1
        src_safe = tl.maximum(src, 0)
        value = tl.load(out_cache_loc + src_safe, mask=is_valid, other=0)
        tl.store(accepted_out_cache_loc + base + group_cnt, value, mask=is_valid)
        group_cnt += is_valid.to(tl.int32)