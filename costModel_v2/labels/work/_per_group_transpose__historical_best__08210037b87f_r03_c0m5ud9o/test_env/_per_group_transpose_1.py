import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

@triton.jit
def _per_group_transpose(
    data_ptr: torch.Tensor,
    trans_data_ptr: torch.Tensor,
    expert_start_offsets: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    k: int,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_id = tl.program_id(1)
    k_id = tl.program_id(2)

    curr_expert_offset = tl.load(expert_start_offsets + expert_id)
    num_tokens_of_expert = tl.load(expert_num_tokens + expert_id)

    data_start_ptr = data_ptr + curr_expert_offset * k
    trans_data_start_ptr = trans_data_ptr + curr_expert_offset * k
    data_start_ptr = tl.multiple_of(data_start_ptr, 16)
    trans_data_start_ptr = tl.multiple_of(trans_data_start_ptr, 16)

    m_coord = m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_coord = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    m_mask = m_coord < num_tokens_of_expert
    k_mask = k_coord < k
    off = m_coord[:, None] * k + k_coord[None, :]
    trans_off = m_coord[:, None] + k_coord[None, :] * num_tokens_of_expert
    mask = m_mask[:, None] & k_mask[None, :]

    data = tl.load(data_start_ptr + off, mask=mask)
    tl.store(trans_data_start_ptr + trans_off, data, mask=mask)

def per_group_transpose(
    a: torch.Tensor,
    expert_offsets: torch.Tensor,
    M_ALIGNMENT: int = 1,
) -> torch.Tensor:
    assert a.dim() == 2
    assert a.is_contiguous(), "`a` is not contiguous"

    m, k = a.size()
    trans_a = torch.empty_like(a)
    num_experts = expert_offsets.size(0) - 1

    expert_start = expert_offsets[:-1]
    expert_num_tokens = expert_offsets[1:] - expert_offsets[:-1]
    max_tokens = expert_num_tokens.max().item()

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 32

    grid = (
        num_experts,
        triton.cdiv(max_tokens, BLOCK_SIZE_M),
        triton.cdiv(k, BLOCK_SIZE_K),
    )

    _per_group_transpose[grid](
        a,
        trans_a,
        expert_start,
        expert_num_tokens,
        k,
        M_ALIGNMENT,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return trans_a