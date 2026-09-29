import functools
import json
import os
from collections.abc import Callable
from typing import Any
import torch
import torch.nn.functional as F
import logging
import triton
import triton.language as tl
PAD_SLOT_ID = -1
logger = logging.getLogger(__name__)
def is_torch_equal_or_newer(version: str) -> bool:
    major, minor, *patch = version.split('.')
    torch_major, torch_minor, torch_patch = torch.__version__.split('.')[:3]
    return (int(torch_major), int(torch_minor)) >= (int(major), int(minor))
def direct_register_custom_op(op_name, op_func, **kwargs):
    pass
def activation_without_mul(activation: str) -> str:
    return activation

@triton.jit
def compute_scales_kernel(
    top_k: int,
    expert_scales_ptr: tl.tensor,
    scales_accum_ptr: tl.tensor,
    num_tokens: int,
    scales_stride: int,
    BLOCK_B: tl.constexpr,
    TOP_K_BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    token_start = pid * BLOCK_B
    token_offsets = tl.arange(0, BLOCK_B)
    token_mask = token_offsets < (num_tokens - token_start)
    scales_accum = tl.zeros([BLOCK_B], tl.float32)
    for k_start in range(0, top_k, TOP_K_BLOCK):
        k_offsets = tl.arange(0, TOP_K_BLOCK)
        k_mask = k_offsets < (top_k - k_start)
        scale_ptr = expert_scales_ptr + token_start * scales_stride + k_start
        scales_block = tl.load(
            scale_ptr + token_offsets[:, None] * scales_stride + k_offsets[None, :],
            mask=token_mask[:, None] & k_mask[None, :]
        )
        scales_sum = tl.sum(scales_block, axis=1)
        scales_accum += tl.where(token_mask, scales_sum, 0.0)
    tl.store(
        scales_accum_ptr + token_start,
        scales_accum,
        mask=token_mask
    )

@triton.jit
def compute_hidden_kernel(
    hidden_states_ptr: tl.tensor,
    scales_accum_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_tokens: int,
    hidden_dim: int,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    token_idx = pid
    d_start = tl.program_id(1) * BLOCK_D
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < (hidden_dim - d_start)
    scales_accum = tl.load(scales_accum_ptr + token_idx)
    h_ptr = hidden_states_ptr + token_idx * hidden_dim + d_start
    h = tl.load(
        h_ptr + d_offsets,
        mask=d_mask
    )
    result = h * scales_accum
    out_ptr = output_ptr + token_idx * hidden_dim + d_start
    tl.store(
        out_ptr + d_offsets,
        result,
        mask=d_mask
    )

def zero_experts_compute_triton(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    assert hidden_states.device.type == 'npu', "Hidden states must be on NPU"
    N = expert_indices.numel()
    top_k = expert_indices.size(-1)
    zero_expert_scales = expert_scales.clone()
    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales[zero_expert_mask] = 0.0
    normal_expert_mask = expert_indices >= num_experts
    expert_indices_masked = expert_indices.clone()
    expert_scales_masked = expert_scales.clone()
    expert_indices_masked[normal_expert_mask] = 0
    expert_scales_masked[normal_expert_mask] = 0.0
    output = torch.zeros_like(hidden_states, device='npu')
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)
    BLOCK_B = 32
    BLOCK_D = 256
    TOP_K_BLOCK = 8
    grid_scales = lambda meta: (triton.cdiv(num_tokens, BLOCK_B),)
    scales_accum = torch.empty(num_tokens, dtype=torch.float32, device='npu')
    compute_scales_kernel[grid_scales](
        top_k,
        zero_expert_scales,
        scales_accum,
        num_tokens,
        zero_expert_scales.stride(0),
        BLOCK_B=BLOCK_B,
        TOP_K_BLOCK=TOP_K_BLOCK,
    )
    grid_hidden = lambda meta: (num_tokens, triton.cdiv(hidden_dim, BLOCK_D))
    compute_hidden_kernel[grid_hidden](
        hidden_states,
        scales_accum,
        output,
        num_tokens,
        hidden_dim,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
    )
    return output