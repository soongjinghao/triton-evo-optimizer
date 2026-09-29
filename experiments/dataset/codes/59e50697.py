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
def compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_tokens: int,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOP_K_BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    token_start = pid * BLOCK_B
    token_offsets = tl.arange(0, BLOCK_B)
    token_mask = token_offsets < (num_tokens - token_start)
    scales_accum = tl.zeros([BLOCK_B], tl.float32)
    for k_start in range(0, top_k, TOP_K_BLOCK * 2):
        k_offsets = tl.arange(0, TOP_K_BLOCK)
        k_mask = k_offsets < (top_k - k_start)
        scale_ptr = expert_scales_ptr + token_start * scales_stride + k_start
        scales_block_0 = tl.load(
            scale_ptr + token_offsets[:, None] * scales_stride + k_offsets[None, :],
            mask=token_mask[:, None] & k_mask[None, :]
        )
        scales_sum_0 = tl.sum(scales_block_0, axis=1)
        scales_accum += tl.where(token_mask, scales_sum_0, 0.0)
        k_start_1 = k_start + TOP_K_BLOCK
        k_offsets_1 = tl.arange(0, TOP_K_BLOCK)
        k_mask_1 = k_offsets_1 < (top_k - k_start_1)
        scale_ptr_1 = expert_scales_ptr + token_start * scales_stride + k_start_1
        scales_block_1 = tl.load(
            scale_ptr_1 + token_offsets[:, None] * scales_stride + k_offsets_1[None, :],
            mask=token_mask[:, None] & k_mask_1[None, :]
        )
        scales_sum_1 = tl.sum(scales_block_1, axis=1)
        scales_accum += tl.where(token_mask, scales_sum_1, 0.0)
    for d_start in range(0, hidden_dim, BLOCK_D * 4):
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < (hidden_dim - d_start)
        h_ptr = hidden_states_ptr + token_start * hidden_dim + d_start
        h_0 = tl.load(
            h_ptr + token_offsets[:, None] * hidden_dim + d_offsets[None, :],
            mask=token_mask[:, None] & d_mask[None, :]
        )
        result_0 = h_0 * scales_accum[:, None]
        out_ptr = output_ptr + token_start * hidden_dim + d_start
        tl.store(
            out_ptr + token_offsets[:, None] * hidden_dim + d_offsets[None, :],
            result_0,
            mask=token_mask[:, None] & d_mask[None, :]
        )
        d_start_1 = d_start + BLOCK_D
        d_offsets_1 = tl.arange(0, BLOCK_D)
        d_mask_1 = d_offsets_1 < (hidden_dim - d_start_1)
        h_ptr_1 = hidden_states_ptr + token_start * hidden_dim + d_start_1
        h_1 = tl.load(
            h_ptr_1 + token_offsets[:, None] * hidden_dim + d_offsets_1[None, :],
            mask=token_mask[:, None] & d_mask_1[None, :]
        )
        result_1 = h_1 * scales_accum[:, None]
        out_ptr_1 = output_ptr + token_start * hidden_dim + d_start_1
        tl.store(
            out_ptr_1 + token_offsets[:, None] * hidden_dim + d_offsets_1[None, :],
            result_1,
            mask=token_mask[:, None] & d_mask_1[None, :]
        )
        d_start_2 = d_start + BLOCK_D * 2
        d_offsets_2 = tl.arange(0, BLOCK_D)
        d_mask_2 = d_offsets_2 < (hidden_dim - d_start_2)
        h_ptr_2 = hidden_states_ptr + token_start * hidden_dim + d_start_2
        h_2 = tl.load(
            h_ptr_2 + token_offsets[:, None] * hidden_dim + d_offsets_2[None, :],
            mask=token_mask[:, None] & d_mask_2[None, :]
        )
        result_2 = h_2 * scales_accum[:, None]
        out_ptr_2 = output_ptr + token_start * hidden_dim + d_start_2
        tl.store(
            out_ptr_2 + token_offsets[:, None] * hidden_dim + d_offsets_2[None, :],
            result_2,
            mask=token_mask[:, None] & d_mask_2[None, :]
        )
        d_start_3 = d_start + BLOCK_D * 3
        d_offsets_3 = tl.arange(0, BLOCK_D)
        d_mask_3 = d_offsets_3 < (hidden_dim - d_start_3)
        h_ptr_3 = hidden_states_ptr + token_start * hidden_dim + d_start_3
        h_3 = tl.load(
            h_ptr_3 + token_offsets[:, None] * hidden_dim + d_offsets_3[None, :],
            mask=token_mask[:, None] & d_mask_3[None, :]
        )
        result_3 = h_3 * scales_accum[:, None]
        out_ptr_3 = output_ptr + token_start * hidden_dim + d_start_3
        tl.store(
            out_ptr_3 + token_offsets[:, None] * hidden_dim + d_offsets_3[None, :],
            result_3,
            mask=token_mask[:, None] & d_mask_3[None, :]
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
    BLOCK_B = 128
    BLOCK_D = 256
    TOP_K_BLOCK = 8
    grid = lambda meta: (triton.cdiv(num_tokens, BLOCK_B),)
    compute_identity_kernel[grid](
        top_k,
        hidden_states,
        zero_expert_scales,
        output,
        num_tokens,
        hidden_dim,
        zero_expert_scales.stride(0),
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
        TOP_K_BLOCK=TOP_K_BLOCK,
    )
    return output