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
):
    GROUP = 16
    pid = tl.program_id(axis=0)
    group = pid // GROUP
    local = pid % GROUP
    if local != 0:
        return
    offset = tl.arange(0, size_upper)
    group_start = group * GROUP
    acc_vec = tl.load(accept_index + offset, mask=(offset < group_start), other=-1)
    cond = (acc_vec != -1).to(tl.int32)
    base = tl.sum(cond)
    group_cnt = tl.zeros((1,), tl.int32)
    for k in range(GROUP):
        idx = group_start + k
        src = tl.load(accept_index + idx, mask=(idx < size_upper), other=-1)
        is_valid = src > -1
        value = tl.load(out_cache_loc + src, mask=is_valid, other=0)
        tl.store(accepted_out_cache_loc + base + group_cnt, value, mask=is_valid)
        group_cnt += is_valid.to(tl.int32)