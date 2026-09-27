

"""Fused MoE Triton kernels."""

import functools
import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

# Remove vllm-specific imports and replace with standard equivalents
import logging
import triton
import triton.language as tl

# Define constants for missing imports
PAD_SLOT_ID = -1

# Remove vllm-specific utility functions and replace with local implementations
logger = logging.getLogger(__name__)

def is_torch_equal_or_newer(version: str) -> bool:
    """Check if current torch version is equal or newer than specified."""
    major, minor, *patch = version.split('.')
    torch_major, torch_minor, torch_patch = torch.__version__.split('.')[:3]
    return (int(torch_major), int(torch_minor)) >= (int(major), int(minor))

def direct_register_custom_op(op_name, op_func, **kwargs):
    """Placeholder for direct_register_custom_op function."""
    pass

def activation_without_mul(activation: str) -> str:
    """Return activation string without multiplication."""
    return activation

@triton.jit
def compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
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
    
    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales[zero_expert_mask] = 0.0

    normal_expert_mask = expert_indices >= num_experts
    expert_indices_masked = expert_indices.clone()
    expert_scales_masked = expert_scales.clone()
    expert_indices_masked[normal_expert_mask] = 0
    expert_scales_masked[normal_expert_mask] = 0.0

    output = torch.zeros_like(hidden_states, device='npu')
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)

    grid = lambda meta: (num_tokens * (hidden_dim // meta["BLOCK_SIZE"]),)
    compute_identity_kernel[grid](
        top_k,
        hidden_states,
        zero_expert_scales,
        num_tokens,
        output,
        hidden_dim,
        zero_expert_scales.stride(0),
        BLOCK_SIZE=256,
    )

    return output
