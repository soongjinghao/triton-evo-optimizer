import logging
from typing import Optional
import torch
import torch_npu
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
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid = pid_0 * num_experts + pid_1
    if pid < numel:
        idx = tl.load(topk_ids_ptr + pid)
        off_c = (pid_0 + 1) * num_experts
        token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
        tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)
@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    base_off = num_experts + pid
    vals = tl.zeros([num_experts], dtype=tl.int32)
    for i in range(num_experts):
        vals = tl.where(tl.arange(0, num_experts) == i, 
                       tl.load(tokens_cnts_ptr + (i + 1) * num_experts + pid),
                       vals)
    step = 1
    while step < num_experts:
        shifted = tl.where(tl.arange(0, num_experts) >= step,
                          tl.view(vals, (num_experts,))[tl.arange(0, num_experts) - step],
                          tl.zeros([num_experts], dtype=tl.int32))
        vals = vals + shifted
        step *= 2
    for i in range(num_experts):
        tl.store(tokens_cnts_ptr + (i + 1) * num_experts + pid, vals[i])
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
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid = pid_0 * num_experts + pid_1
    if pid < numel:
        start_idx = tl.load(cumsum_ptr + pid_0)
        end_idx = tl.load(cumsum_ptr + pid_0 + 1)
        for i in range(start_idx, end_idx, block_size):
            tl.store(expert_ids_ptr + i // block_size, pid_0)
        expert_id = tl.load(topk_ids_ptr + pid)
        off_t = pid_0 * num_experts
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, pid)
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
    if numel > num_experts:
        tokens_per_expert = ceil_div(numel, num_experts)
        grid = (tokens_per_expert, num_experts)
        tokens_cnts = torch.zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        )
        cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
        moe_align_block_size_stage1[grid](
            topk_ids,
            tokens_cnts,
            num_experts,
            numel,
        )
        moe_align_block_size_stage2[(num_experts,)](
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
        moe_align_block_size_stage4[grid](
            topk_ids,
            sorted_token_ids,
            expert_ids,
            tokens_cnts,
            cumsum,
            num_experts,
            block_size,
            numel,
        )
    else:
        grid = (num_experts,)
        tokens_cnts = torch.zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        )
        cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
        tokens_per_thread = ceil_div(numel, num_experts)
        moe_align_block_size_stage1[grid](
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
        moe_align_block_size_stage4[grid](
            topk_ids,
            sorted_token_ids,
            expert_ids,
            tokens_cnts,
            cumsum,
            num_experts,
            block_size,
            numel,
        )