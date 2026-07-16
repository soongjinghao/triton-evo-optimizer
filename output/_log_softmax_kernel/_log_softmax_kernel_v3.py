import torch
import triton
import triton.language as tl

@triton.jit
def _log_softmax_kernel_small(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols

    vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float('inf'))

    max_val = tl.max(vals)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.sum(exp_vals)
    log_sum_exp = tl.log(sum_exp)

    output = vals - max_val - log_sum_exp
    tl.store(output_row_start_ptr + col_idx, output, mask=mask)


@triton.jit
def _log_softmax_kernel_large(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    max_val = -float('inf')
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        remaining = n_cols - col_offset
        col_idx = tl.arange(0, BLOCK_SIZE)
        mask = col_idx < remaining
        vals = tl.load(row_start_ptr + col_offset + col_idx, mask=mask, other=-float('inf'))

        block_max = tl.max(vals)

        if block_max > max_val:
            sum_exp = sum_exp * tl.exp(max_val - block_max) + tl.sum(tl.exp(vals - block_max))
            max_val = block_max
        else:
            sum_exp += tl.sum(tl.exp(vals - max_val))

    log_sum_exp = tl.log(sum_exp)

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        remaining = n_cols - col_offset
        col_idx = tl.arange(0, BLOCK_SIZE)
        mask = col_idx < remaining
        vals = tl.load(row_start_ptr + col_offset + col_idx, mask=mask, other=0.0)
        output = vals - max_val - log_sum_exp
        tl.store(output_row_start_ptr + col_offset + col_idx, output, mask=mask)


def _get_num_warps(block_size):
    if block_size <= 1024:
        return 2
    elif block_size <= 2048:
        return 4
    elif block_size <= 4096:
        return 8
    else:
        return 16


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
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        kernel = _log_softmax_kernel_small
    else:
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
        kernel = _log_softmax_kernel_large

    num_warps = _get_num_warps(BLOCK_SIZE)
    grid = (n_rows,)
    kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return output.reshape(original_shape)