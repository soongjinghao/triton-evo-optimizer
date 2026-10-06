import logging
from typing import Optional
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

def ceil_div(a, b):
    return (a + b - 1) // b

def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y

@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tokens_per_program = tl.cdiv(numel, num_programs)
    start_idx = pid * tokens_per_program
    end_idx = tl.minimum(start_idx + tokens_per_program, numel)
    off_c = (pid + 1) * num_experts
    for i in range(start_idx, end_idx):
        idx = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
        tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)

@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)

@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)

@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tokens_per_program = tl.cdiv(numel, num_programs)
    start_idx = pid * tokens_per_program
    end_idx = tl.minimum(start_idx + tokens_per_program, numel)
    off_t = pid * num_experts
    for i in range(start_idx, end_idx):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)

def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    
    # Stage1: expanded grid for finer granularity
    grid_stage1 = (num_experts * ceil_div(numel, num_experts),)
    moe_align_block_size_stage1[grid_stage1](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
    )
    
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    
    # Stage4: expanded grid for finer granularity
    grid_stage4 = (num_experts * ceil_div(numel, num_experts),)
    moe_align_block_size_stage4[grid_stage4](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
    )