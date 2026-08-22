import torch
import triton
import triton.language as tl
def ceil_div(a, b):
    return (a + b - 1) // b
@triton.jit
def moe_align_block_size_token(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread,
    stage: tl.constexpr,
    EXPERT_ID_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if stage == 0:
        off_c = (
            (pid + 1)
            * num_experts
        )
        start_idx = (
            pid
            * tokens_per_thread
        )
        for i in range(
            tokens_per_thread
        ):
            idx = (
                start_idx + i
            )
            if idx < numel:
                expert_id = tl.load(
                    topk_ids_ptr + idx
                )
                counter_ptr = (
                    tokens_cnts_ptr
                    + off_c
                    + expert_id
                )
                count = tl.load(
                    counter_ptr
                )
                tl.store(
                    counter_ptr,
                    count + 1,
                )
    else:
        cum_start = tl.load(
            cumsum_ptr + pid
        )
        cum_end = tl.load(
            cumsum_ptr + pid + 1
        )
        start_block = (
            cum_start
            // block_size
        )
        end_block = (
            cum_end
            // block_size
        )
        num_blocks = (
            end_block
            - start_block
        )
        for base in range(
            0,
            num_blocks,
            EXPERT_ID_BLOCK,
        ):
            lane = tl.arange(
                0,
                EXPERT_ID_BLOCK,
            )
            block_off = (
                base + lane
            )
            mask = (
                block_off
                < num_blocks
            )
            tl.store(
                expert_ids_ptr
                + start_block
                + block_off,
                pid,
                mask=mask,
            )
        token_start = (
            pid
            * tokens_per_thread
        )
        off_t = (
            pid
            * num_experts
        )
        for i in range(
            token_start,
            tl.minimum(
                token_start
                + tokens_per_thread,
                numel,
            ),
        ):
            expert_id = tl.load(
                topk_ids_ptr + i
            )
            counter_ptr = (
                tokens_cnts_ptr
                + off_t
                + expert_id
            )
            rank = tl.load(
                counter_ptr
            )
            tl.store(
                counter_ptr,
                rank + 1,
            )
            rank_post_pad = (
                rank
                + tl.load(
                    cumsum_ptr
                    + expert_id
                )
            )
            tl.store(
                sorted_token_ids_ptr
                + rank_post_pad,
                i,
            )
@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(
        0,
        BLOCK_E,
    )
    mask = (
        rows < num_experts
    )
    offsets = (
        (rows + 1)
        * num_experts
        + pid
    )
    values = tl.load(
        tokens_cnts_ptr
        + offsets,
        mask=mask,
        other=0,
    )
    prefix = tl.cumsum(
        values
    )
    tl.store(
        tokens_cnts_ptr
        + offsets,
        prefix,
        mask=mask,
    )
@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    lane = tl.arange(
        0,
        BLOCK_E,
    )
    mask = (
        lane < num_experts
    )
    off_cnt = (
        num_experts
        * num_experts
    )
    cnts = tl.load(
        tokens_cnts_ptr
        + off_cnt
        + lane,
        mask=mask,
        other=0,
    )
    padded = (
        tl.cdiv(
            cnts,
            block_size,
        )
        * block_size
    )
    padded = tl.where(
        mask,
        padded,
        0,
    )
    cumsum_vec = tl.cumsum(
        padded
    )
    tl.store(
        cumsum_ptr
        + 1
        + lane,
        cumsum_vec,
        mask=mask,
    )
    total_tokens = tl.sum(
        padded
    )
    tl.store(
        total_tokens_post_pad_ptr,
        total_tokens,
    )
def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = (
        topk_ids.numel()
    )
    grid = (
        num_experts,
    )
    tokens_per_thread = ceil_div(
        numel,
        num_experts,
    )
    tokens_cnts = torch.zeros(
        (
            num_experts + 1,
            num_experts,
        ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum = torch.zeros(
        (
            num_experts + 1,
        ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    BLOCK_E = triton.next_power_of_2(
        num_experts
    )
    EXPERT_ID_BLOCK = 64
    moe_align_block_size_token[
        grid
    ](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
        stage=0,
        EXPERT_ID_BLOCK=EXPERT_ID_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    moe_align_block_size_stage2[
        grid
    ](
        tokens_cnts,
        num_experts,
        BLOCK_E,
        num_warps=1,
        num_stages=1,
    )
    moe_align_block_size_stage3[
        (1,)
    ](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        BLOCK_E,
        num_warps=1,
        num_stages=1,
    )
    moe_align_block_size_token[
        grid
    ](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
        stage=1,
        EXPERT_ID_BLOCK=EXPERT_ID_BLOCK,
        num_warps=4,
        num_stages=2,
    )