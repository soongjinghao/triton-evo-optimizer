import torch
import triton
import triton.language as tl

@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    IS_SMALL: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    if IS_SMALL:
        col_idx = tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float('inf'))
        max_val = tl.max(vals)
        exp_vals = tl.exp(vals - max_val)
        sum_exp = tl.sum(exp_vals)
        log_sum_exp = tl.log(sum_exp)
        output = vals - max_val - log_sum_exp
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)
    else:
        max_val = -float('inf')
        sum_exp = 0.0
        offs = tl.arange(0, BLOCK_SIZE)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + offs
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float('inf'))
            block_max = tl.max(vals)
            if block_max > max_val:
                sum_exp = sum_exp * tl.exp(max_val - block_max) + tl.sum(tl.exp(vals - block_max))
                max_val = block_max
            else:
                sum_exp += tl.sum(tl.exp(vals - max_val))
        log_sum_exp = tl.log(sum_exp)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + offs
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            output = vals - max_val - log_sum_exp
            tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    if input.device.type != 'npu':
        input = input.to('npu')

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    if n_cols == 0:
        raise ValueError("Input tensor cannot have empty last dimension")

    output = torch.empty_like(input_2d, device='npu')

    if n_cols <= 32768:
        IS_SMALL = True
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
    else:
        IS_SMALL = False
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))

    if BLOCK_SIZE <= 512:
        num_warps = 2
    elif BLOCK_SIZE <= 1024:
        num_warps = 4
    elif BLOCK_SIZE <= 2048:
        num_warps = 8
    else:
        num_warps = 16

    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_SMALL=IS_SMALL,
        num_warps=num_warps,
    )

    return output.reshape(original_shape)