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
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)
    acc_vec = tl.load(accept_index + offset, mask=(offset < pid), other=-1)
    cond = (acc_vec != -1).to(tl.int32)
    dst = tl.sum(cond)
    src = tl.load(accept_index + pid)
    src_safe = tl.maximum(src, 0)
    value = tl.load(out_cache_loc + src_safe)
    tl.store(accepted_out_cache_loc + dst, value, mask=(src > -1))