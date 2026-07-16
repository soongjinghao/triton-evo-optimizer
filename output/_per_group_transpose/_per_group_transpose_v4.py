import torch
import triton
import triton.language as tl

@triton.jit
def _per_group_transpose(
    data_ptr: torch.Tensor,
    trans_data_ptr: torch.Tensor,
    expert_offsets: torch.Tensor,
    k: int,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_id = tl.program_id(1)
    k_id = tl.program_id(2)

    curr_expert_offset = tl.load(expert_offsets + expert_id)
    next_expert_offset = tl.load(expert_offsets + expert_id + 1)
    num_tokens_of_expert = next_expert_offset - curr_expert_offset
    tl.multiple_of(curr_expert_offset, M_ALIGNMENT)
    tl.multiple_of(next_expert_offset, M_ALIGNMENT)

    data_start_ptr = data_ptr + curr_expert_offset * k
    trans_data_start_ptr = trans_data_ptr + curr_expert_offset * k

    k_coord = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_coord < k

    for start_m in tl.range(0, num_tokens_of_expert, BLOCK_SIZE_M * tl.num_programs(1)):
        m_coord = start_m + m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_coord < num_tokens_of_expert
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

    # Dynamic block sizing: use next_power_of_2, cap at 64, ensure multiple of 16
    avg_tokens = (m + num_experts - 1) // num_experts
    BLOCK_SIZE_M = triton.next_power_of_2(min(64, avg_tokens))
    BLOCK_SIZE_M = max(16, BLOCK_SIZE_M)  # ensure at least 16
    BLOCK_SIZE_K = triton.next_power_of_2(min(64, k))
    BLOCK_SIZE_K = max(16, BLOCK_SIZE_K)

    grid = lambda META: (
        num_experts,
        triton.cdiv(avg_tokens, META["BLOCK_SIZE_M"]),
        triton.cdiv(k, META["BLOCK_SIZE_K"]),
    )
    _per_group_transpose[grid](
        a, trans_a, expert_offsets, k, M_ALIGNMENT,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return trans_a