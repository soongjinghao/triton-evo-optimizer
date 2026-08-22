import torch
import triton
import triton.language as tl
import torch_npu

device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1, 'num_stages': 1}),
        triton.Config({'num_warps': 2, 'num_stages': 1}),
        triton.Config({'num_warps': 4, 'num_stages': 1}),
        triton.Config({'num_warps': 8, 'num_stages': 1}),
        triton.Config({'num_warps': 1, 'num_stages': 2}),
        triton.Config({'num_warps': 2, 'num_stages': 2}),
        triton.Config({'num_warps': 4, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 1, 'num_stages': 4}),
        triton.Config({'num_warps': 2, 'num_stages': 4}),
        triton.Config({'num_warps': 4, 'num_stages': 4}),
        triton.Config({'num_warps': 8, 'num_stages': 4}),
    ],
    key=['size_upper'],
)
@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        local_acc = 0
        for i in range(size_upper):
            src = tl.load(accept_index + i)
            if src > -1:
                value = tl.load(out_cache_loc + src)
                tl.store(accepted_out_cache_loc + local_acc, value)
                local_acc += 1