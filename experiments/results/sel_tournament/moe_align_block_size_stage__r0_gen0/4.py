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
def moe_align_block_size_stage1_fused(
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
    expert_id = pid // tl.cdiv(numel, num_experts)
    token_idx = pid % tl.cdiv(numel, num_experts)
    
    if token_idx < numel:
        topk_id = tl.load(topk_ids_ptr + token_idx)
        tl.atomic_add(tokens_cnts_ptr + (expert_id + 1) * num_experts + topk_id, 1)
    
    tl.debug_barrier()
    
    if token_idx == 0:
        last_cnt = 0
        for i in range(1, num_experts + 1):
            token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + expert_id)
            last_cnt = last_cnt + token_cnt
            tl.store(tokens_cnts_ptr + i * num_experts + expert_id, last_cnt)
        
        if expert_id == 0:
            last_cumsum = 0
            off_cnt = num_experts * num_experts
            for i in range(1, num_experts + 1):
                token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
                last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
                tl.store(cumsum_ptr + i, last_cumsum)
            tl.store(cumsum_ptr, 0)
    
    tl.debug_barrier()
    
    if token_idx < numel:
        topk_id = tl.load(topk_ids_ptr + token_idx)
        token_cnt = tl.load(tokens_cnts_ptr + (expert_id + 1) * num_experts + topk_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + topk_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, token_idx)
        tl.store(tokens_cnts_ptr + (expert_id + 1) * num_experts + topk_id, token_cnt + 1)
        
        if token_idx == 0:
            start_idx = tl.load(cumsum_ptr + expert_id)
            end_idx = tl.load(cumsum_ptr + expert_id + 1)
            for i in range(start_idx, end_idx, block_size):
                tl.store(expert_ids_ptr + i // block_size, expert_id)

def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    
    if numel <= num_experts:
        grid = (num_experts,)
        tokens_cnts = torch.zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        )
        cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
        tokens_per_thread = ceil_div(numel, num_experts)
        
        @triton.jit
        def moe_align_block_size_stage1(
            topk_ids_ptr,
            tokens_cnts_ptr,
            num_experts: tl.constexpr,
            numel,
            tokens_per_thread,
        ):
            pid = tl.program_id(0)
            start_idx = pid * tokens_per_thread
            off_c = (pid + 1) * num_experts
            for i in range(tokens_per_thread):
                if start_idx + i < numel:
                    idx = tl.load(topk_ids_ptr + start_idx + i)
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
            tokens_per_thread,
        ):
            pid = tl.program_id(0)
            start_idx = tl.load(cumsum_ptr + pid)
            end_idx = tl.load(cumsum_ptr + pid + 1)
            for i in range(start_idx, end_idx, block_size):
                tl.store(expert_ids_ptr + i // block_size, pid)
            start_idx = pid * tokens_per_thread
            off_t = pid * num_experts
            for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
                expert_id = tl.load(topk_ids_ptr + i)
                token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
                rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
                tl.store(sorted_token_ids_ptr + rank_post_pad, i)
                tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)
        
        moe_align_block_size_stage1[grid](
            topk_ids,
            tokens_cnts,
            num_experts,
            numel,
            tokens_per_thread,
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
            tokens_per_thread,
        )
    else:
        tokens_per_expert = ceil_div(numel, num_experts)
        grid = (num_experts * tokens_per_expert,)
        tokens_cnts = torch.zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        )
        cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
        
        moe_align_block_size_stage1_fused[grid](
            topk_ids,
            sorted_token_ids,
            expert_ids,
            tokens_cnts,
            cumsum,
            num_experts,
            block_size,
            numel,
        )
        
        last_cumsum = cumsum[num_experts].item()
        num_tokens_post_pad[0] = last_cumsum