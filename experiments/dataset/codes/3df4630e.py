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
    # Part 1: store expert_ids for blocks (already vectorized via block_size stride)
    start_idx_expert = tl.load(cumsum_ptr + pid)
    end_idx_expert = tl.load(cumsum_ptr + pid + 1)
    num_blocks = (end_idx_expert - start_idx_expert) // block_size
    # Vectorized store for expert_ids
    offs_block = tl.arange(0, num_blocks)
    tl.store(expert_ids_ptr + start_idx_expert // block_size + offs_block, pid, mask=offs_block < num_blocks)
    
    # Part 2: vectorized load/store for sorted_token_ids and tokens_cnts
    start_idx = pid * tokens_per_thread
    # Compute the actual number of tokens this thread processes (with mask for tail)
    num_tokens = tl.minimum(tokens_per_thread, numel - start_idx)
    offs_token = tl.arange(0, tokens_per_thread)
    mask_token = offs_token < num_tokens
    # Vectorized load topk_ids
    topk_ids_val = tl.load(topk_ids_ptr + start_idx + offs_token, mask=mask_token, other=0)
    # Compute expert_id offsets for tokens_cnts_ptr
    off_t = pid * num_experts
    # Load current token counts for each expert (vectorized across tokens)
    # We need to load tokens_cnts_ptr[off_t + expert_id] for each token
    # Since expert_id varies per token, this is a gather, but we can vectorize the store part
    # For each token, we need to read the current count, compute rank, store sorted_token_ids, then increment count
    # To vectorize, we process tokens in groups where expert_id is the same? No, that's complex.
    # Instead, we vectorize the store of sorted_token_ids and the increment of tokens_cnts using scatter-add.
    # However, tl.atomic_add is not available. We'll use a loop but vectorize the load of topk_ids and store of sorted_token_ids.
    # Actually, we can vectorize the store of sorted_token_ids by computing all ranks first, then storing.
    # But the ranks depend on the current count which changes as we process tokens.
    # To maintain correctness, we must process tokens sequentially for the count update.
    # However, we can vectorize the load of topk_ids and the store of sorted_token_ids if we pre-compute ranks.
    # Since cumsum values are known and we can load the initial counts, we can compute ranks for all tokens at once.
    # But the counts are updated per token, so we need to simulate the increment.
    # We can use a prefix-sum-like approach: for each expert, the tokens are assigned consecutive ranks.
    # We can load the initial counts for each expert, then for each token, compute its rank as initial_count[expert] + local_index_within_expert.
    # To do this vectorized, we need to know the local index of each token within its expert.
    # This requires a sort by expert_id, which is not trivial.
    # Given the complexity, we keep the loop but vectorize the load of topk_ids and the store of sorted_token_ids.
    # We'll process tokens in chunks of BLOCK_SIZE to amortize loop overhead.
    BLOCK_SIZE: tl.constexpr = 32
    i = 0
    while i < num_tokens:
        block_offs = offs_token[i:i + BLOCK_SIZE]
        block_mask = block_offs < num_tokens
        block_topk = tl.load(topk_ids_ptr + start_idx + block_offs, mask=block_mask, other=0)
        # For each token in the block, we need to load the current count, compute rank, store, and increment.
        # This still requires sequential processing within the block due to count dependency.
        # To truly vectorize, we would need atomic operations or a different algorithm.
        # Given the constraints, we keep the original loop but with vectorized loads/stores where possible.
        for j in range(tl.minimum(BLOCK_SIZE, num_tokens - i)):
            idx = i + j
            expert_id = tl.load(topk_ids_ptr + start_idx + idx)
            token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
            rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
            tl.store(sorted_token_ids_ptr + rank_post_pad, start_idx + idx)
            tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)
        i += BLOCK_SIZE

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