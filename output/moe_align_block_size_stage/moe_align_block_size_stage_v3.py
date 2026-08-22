import torch
import triton
import triton.language as tl


def ceil_div(a, b):
    return (a + b - 1) // b


# ============================================================
# Stage 0 / Stage 1
#
# stage = 0:
#   统计每个 token 对应 expert 的数量
#
# stage = 1:
#   写 expert_ids + sorted_token_ids
# ============================================================

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

    # ========================================================
    # Stage 0
    # ========================================================
    if stage == 0:

        # 每个 pid 独占 tokens_cnts 中的一整行：
        #
        # pid=0 -> row 1
        # pid=1 -> row 2
        # ...
        #
        # 所以不存在不同 program 写同一个 counter 的竞争，
        # 不需要 atomic_add。
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

    # ========================================================
    # Stage 1
    # ========================================================
    else:

        # ----------------------------------------------------
        # 1. 写 expert_ids
        # ----------------------------------------------------

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

        # 原 best:
        #
        # tl.arange(0, end_idx - start_idx)
        #
        # end_idx-start_idx 是 runtime value，
        # 对 Ascend Triton 存在潜在编译风险。
        #
        # 这里改成固定 BLOCK + mask。
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

        # ----------------------------------------------------
        # 2. 写 sorted_token_ids
        # ----------------------------------------------------

        token_start = (
            pid
            * tokens_per_thread
        )

        # row pid 保存的是当前 pid 之前的累计数量
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

            # 每个 pid 同样独占一行，
            # 不需要 atomic_add。
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


# ============================================================
# Stage 2
#
# 对 tokens_cnts 的每个 expert column 做 prefix sum。
#
# 保留当前有效的 tl.cumsum 思路。
#
# 相比原 best:
#
#   tl.arange(1, num_experts + 1)
#
# 改成 power-of-two BLOCK_E + mask，
# 对隐藏 num_experts 更稳健。
# ============================================================

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


# ============================================================
# Stage 3
#
# 当前平台 runtime error 的核心修复点。
# ============================================================

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

    # stage2 完成以后，
    # 最后一行就是每个 expert 的总 token 数。
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

    # padded expert token count
    padded = (
        tl.cdiv(
            cnts,
            block_size,
        )
        * block_size
    )

    # padding lane 必须严格为 0，
    # 避免影响 cumsum / total。
    padded = tl.where(
        mask,
        padded,
        0,
    )

    # --------------------------------------------------------
    # prefix
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 关键修正
    #
    # ❌ 原版：
    #
    # cumsum_vec[num_experts - 1]
    #
    # Ascend:
    #
    # unsupported tensor index: constexpr
    #
    # ✅ 新版：
    #
    # 最后的 prefix 本质就是 sum(padded)
    # --------------------------------------------------------

    total_tokens = tl.sum(
        padded
    )

    tl.store(
        total_tokens_post_pad_ptr,
        total_tokens,
    )


# ============================================================
# Wrapper
# ============================================================

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

    # 一个 program 对应一个 expert/token partition
    grid = (
        num_experts,
    )

    tokens_per_thread = ceil_div(
        numel,
        num_experts,
    )

    # --------------------------------------------------------
    # tokens_cnts layout
    #
    # row 0:
    #   prefix base
    #
    # row 1..E:
    #   per-program counts
    # --------------------------------------------------------

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

    # power-of-two vector width
    BLOCK_E = triton.next_power_of_2(
        num_experts
    )

    # 固定 expert_ids 写入块。
    #
    # 不使用 runtime tl.arange。
    EXPERT_ID_BLOCK = 64

    # ========================================================
    # Stage 0
    # ========================================================

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

        # token loop 保持比较稳健的配置
        num_warps=4,
        num_stages=2,
    )

    # ========================================================
    # Stage 2
    #
    # tiny vector prefix scan:
    # 这里用 1 warp 更适合小 expert 数。
    # ========================================================

    moe_align_block_size_stage2[
        grid
    ](
        tokens_cnts,

        num_experts,
        BLOCK_E,

        num_warps=1,
        num_stages=1,
    )

    # ========================================================
    # Stage 3
    # ========================================================

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

    # ========================================================
    # Stage 1
    # ========================================================

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