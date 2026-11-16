import torch
import triton
import triton.language as tl

@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,
    topk_numel: tl.constexpr,
    expert_map_ptr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tasks = num_experts
    num_blocks_per_expert = tl.cdiv(topk_numel, BLOCK_SIZE)
    total_blocks = num_tasks * num_blocks_per_expert
    if pid >= total_blocks:
        return
    curr_expert = pid // num_blocks_per_expert
    block_idx = pid % num_blocks_per_expert
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = block_idx * BLOCK_SIZE
    mask = offsets < (topk_numel - block_start)
    expert_ids = tl.load(topk_ids_ptr + block_start + offsets, mask=mask, other=-1)
    if HAS_EXPERT_MAP:
        map_mask = expert_ids >= 0
        expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)
    has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
    block_sum = tl.sum(has_curr_expert)
    tl.atomic_add(expert_num_tokens_ptr + curr_expert, block_sum)

def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.zeros(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
    num_blocks_per_expert = triton.cdiv(topk_ids.numel(), BLOCK_SIZE)
    total_blocks = num_local_experts * num_blocks_per_expert
    grid = (total_blocks,)
    _count_expert_num_tokens[grid](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return expert_num_tokens