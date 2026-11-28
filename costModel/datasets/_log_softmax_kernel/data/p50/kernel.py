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
    
    # 使用更高效的在线softmax计算,减少重复加载
    m_i = -float("inf")
    l_i = 0.0
    
    # 第一遍:计算最大值和log-sum-exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        curr_vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        
        # 在线softmax稳定计算
        m_next = tl.max(curr_vals, axis=0)
        m_new = tl.maximum(m_i, m_next)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(curr_vals - m_new), axis=0)
        m_i = m_new
    
    # 计算最终的log-sum-exp
    log_sum_exp = tl.log(l_i) + m_i
    
    # 第二遍:计算输出,使用更高效的存储
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
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
    
    # 优化BLOCK_SIZE选择:使用更大的块减少循环次数,同时考虑NPU缓存行对齐
    BLOCK_SIZE = min(2048, triton.next_power_of_2(n_cols))
    
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