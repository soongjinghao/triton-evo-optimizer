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
    if pid == 0:
        offs = tl.arange(0, size_upper)
        acc = tl.load(accept_index + offs)
        valid = (acc != -1).to(tl.int32)
        inc = tl.cumsum(valid, axis=0)
        dst = inc - valid
        src_safe = tl.maximum(acc, 0)
        value = tl.load(out_cache_loc + src_safe)
        tl.store(accepted_out_cache_loc + dst, value, mask=(acc > -1))