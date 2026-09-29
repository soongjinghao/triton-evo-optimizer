import os
from collections.abc import Callable
from functools import cache
from typing import Any
import torch
import torch_npu
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
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    m_i = -float("inf")
    l_i = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        curr_vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        m_next = tl.max(curr_vals, axis=0)
        m_new = tl.maximum(m_i, m_next)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(curr_vals - m_new), axis=0)
        m_i = m_new
    log_sum_exp = tl.log(l_i) + m_i
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask)
        output = vals - log_sum_exp
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
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)