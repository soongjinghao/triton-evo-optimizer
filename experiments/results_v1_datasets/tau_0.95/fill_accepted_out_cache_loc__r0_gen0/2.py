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
    if size_upper <= 2048:
        if pid == 0:
            cnt = tl.zeros([], dtype=tl.int32)
            for i in range(size_upper):
                src = tl.load(accept_index + i)
                if src > -1:
                    value = tl.load(out_cache_loc + src)
                    tl.store(accepted_out_cache_loc + cnt, value)
                    cnt = cnt + 1
        else:
            return
    else:
        offset = tl.arange(0, size_upper)
        acc_vec = tl.load(accept_index + offset)
        cond = (acc_vec != -1).to(tl.int32)
        dst = tl.sum(tl.where(offset < pid, cond, 0))
        src = tl.load(accept_index + pid)
        if src > -1:
            value = tl.load(out_cache_loc + src)
            tl.store(accepted_out_cache_loc + dst, value)