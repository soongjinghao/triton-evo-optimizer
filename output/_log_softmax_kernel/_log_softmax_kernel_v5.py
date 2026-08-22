import torch
import triton
import triton.language as tl
import math

@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    col_idx = tl.max_contiguous(
        tl.multiple_of(col_idx, BLOCK_SIZE),
        BLOCK_SIZE,
    )
    log2e = tl.constexpr(1.0 / math.log(2))
    ln2 = tl.constexpr(math.log(2))
    for row_idx in range(pid, n_rows, num_programs):
        row_offset = row_idx * n_cols
        in_ptrs = input_ptr + row_offset + col_idx
        out_ptrs = output_ptr + row_offset + col_idx
        values = tl.load(
            in_ptrs,
            mask=mask,
            other=-float("inf"),
        )
        row_max = tl.max(values, axis=0)
        centered = values - row_max
        sum_exp2 = tl.sum(tl.exp2(centered * log2e), axis=0)
        result = centered - tl.log2(sum_exp2) * ln2
        tl.store(
            out_ptrs,
            result,
            mask=mask,
        )

def log_softmax(
    input: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    normalized_dim = dim if dim >= 0 else input.ndim + dim
    if normalized_dim != input.ndim - 1:
        raise ValueError(
            "Only supports log_softmax along the last dimension"
        )
    if input.device.type != "npu":
        input = input.to("npu")
    input_c = input if input.is_contiguous() else input.contiguous()
    if input_c.ndim == 0:
        raise ValueError("log_softmax requires at least one dimension")
    n_cols = input_c.shape[-1]
    if n_cols == 0:
        raise ValueError("Input tensor cannot have an empty last dimension")
    if input_c.numel() == 0:
        return torch.empty_like(input_c)
    n_rows = input_c.numel() // n_cols
    output = torch.empty_like(input_c)
    BLOCK_SIZE = max(
        triton.next_power_of_2(n_cols),
        16,
    )
    if BLOCK_SIZE >= 4096:
        num_warps = 8
    elif BLOCK_SIZE >= 2048:
        num_warps = 4
    elif BLOCK_SIZE >= 512:
        num_warps = 2
    else:
        num_warps = 1
    max_grid = 512
    grid = (min(n_rows, max_grid),)
    _log_softmax_kernel[grid](
        input_c,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )
    return output