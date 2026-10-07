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
    # Stage2: 使用单program向量化处理所有expert的累加
    # 将grid从(num_experts,)改为(1,),利用tl.arange向量化加载和前缀和
    # 前提: num_experts较小(如≤256)且UB可容纳
    # 使用向量化方式加载所有expert的token计数
    off_base = num_experts  # 从第1行开始(跳过第0行)
    # 向量化加载所有expert的token计数
    # 对于每个expert i,加载tokens_cnts_ptr[off_base + i * num_experts + expert_id]
    # 这里expert_id是当前处理的expert,由于grid=(1,),我们向量化处理所有expert
    # 加载所有expert的token计数到寄存器
    # 使用tl.arange(0, num_experts)作为expert索引
    expert_ids = tl.arange(0, num_experts)
    # 对于每个expert i,需要加载tokens_cnts_ptr[off_base + i * num_experts + expert_ids]
    # 但这里我们实际上需要按列处理:对于每个expert_id,累加所有行的token计数
    # 原始逻辑:for i in range(1, num_experts + 1): token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
    # 其中pid是expert_id,i是行索引
    # 向量化版本:对于每个expert_id,加载所有行的token计数并累加
    # 使用向量化前缀和计算
    # 初始化累加结果
    cumsum = tl.zeros([num_experts], dtype=tl.int32)
    # 逐行处理,但使用向量化加载
    for i in range(1, num_experts + 1):
        # 向量化加载当前行所有expert的token计数
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + expert_ids, mask=expert_ids < num_experts)
        # 向量化累加
        cumsum = cumsum + token_cnt
        # 向量化存储累加结果
        tl.store(tokens_cnts_ptr + i * num_experts + expert_ids, cumsum, mask=expert_ids < num_experts)
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
    tokens_per_thread = ceil_div(numel, num_experts)
    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    # Stage2: 使用单program向量化处理所有expert的累加
    # 将grid从(num_experts,)改为(1,),利用向量化减少标量循环次数
    moe_align_block_size_stage2[(1,)](
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