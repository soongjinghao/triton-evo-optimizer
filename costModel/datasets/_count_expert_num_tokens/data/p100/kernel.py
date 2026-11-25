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
    num_experts_per_block: tl.constexpr = 4
    expert_start = pid * num_experts_per_block
    offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(topk_numel, BLOCK_SIZE)
    base_ptr = topk_ids_ptr + offsets
    acc0 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    acc1 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    acc3 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(num_blocks):
        block_start = x * BLOCK_SIZE
        mask = offsets < (topk_numel - block_start)
        expert_ids = tl.load(base_ptr + block_start, mask=mask, other=-1)
        if HAS_EXPERT_MAP:
            map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)
        has_curr0 = tl.where(expert_ids == expert_start, 1, 0)
        has_curr1 = tl.where(expert_ids == (expert_start + 1), 1, 0)
        has_curr2 = tl.where(expert_ids == (expert_start + 2), 1, 0)
        has_curr3 = tl.where(expert_ids == (expert_start + 3), 1, 0)
        acc0 = acc0 + has_curr0
        acc1 = acc1 + has_curr1
        acc2 = acc2 + has_curr2
        acc3 = acc3 + has_curr3
    if expert_start < num_experts:
        tl.store(expert_num_tokens_ptr + expert_start, tl.sum(acc0))
    if expert_start + 1 < num_experts:
        tl.store(expert_num_tokens_ptr + expert_start + 1, tl.sum(acc1))
    if expert_start + 2 < num_experts:
        tl.store(expert_num_tokens_ptr + expert_start + 2, tl.sum(acc2))
    if expert_start + 3 < num_experts:
        tl.store(expert_num_tokens_ptr + expert_start + 3, tl.sum(acc3))

def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )
    num_experts_per_block: tl.constexpr = 4
    grid = (num_local_experts + num_experts_per_block - 1) // num_experts_per_block
    total_elements = topk_ids.numel()
    BLOCK_SIZE = min(total_elements, 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        total_elements,
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return expert_num_tokens