import torch
import triton
import triton.language as tl

@triton.jit
def _count_expert_num_tokens_kernel(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,
    topk_numel: tl.constexpr,
    expert_map_ptr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_id = tl.program_id(0)
    block_id = tl.program_id(1)
    if expert_id >= num_experts:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = block_id * BLOCK_SIZE
    mask = offsets < (topk_numel - block_start)
    ptr = topk_ids_ptr + block_start + offsets
    aligned_ptr = tl.multiple_of(ptr, 16)
    expert_ids = tl.load(aligned_ptr, mask=mask, other=-1)
    if HAS_EXPERT_MAP:
        map_mask = expert_ids >= 0
        mapped_ids = tl.load(
            tl.multiple_of(expert_map_ptr + expert_ids, 16),
            mask=map_mask,
            other=-1,
        )
        has_curr_expert = tl.where(mapped_ids == expert_id, 1, 0)
    else:
        has_curr_expert = tl.where(expert_ids == expert_id, 1, 0)
    count = tl.sum(has_curr_expert).to(tl.int32)
    tl.atomic_add(expert_num_tokens_ptr + expert_id, count)

def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.zeros(
        (num_local_experts,), device=topk_ids.device, dtype=torch.int32
    )
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
    num_blocks = triton.cdiv(topk_ids.numel(), BLOCK_SIZE)
    grid = (num_local_experts, num_blocks)
    _count_expert_num_tokens_kernel[grid](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return expert_num_tokens