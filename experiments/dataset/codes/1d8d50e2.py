import torch
import triton
import triton.language as tl

@triton.jit
def fn_triton_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    num_tokens,
    QK_NOPE_HEAD_DIM: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    NUM_LOCAL_HEADS: tl.constexpr,
    K_NOPE_STRIDE_0: tl.constexpr,
    K_NOPE_STRIDE_1: tl.constexpr,
    K_STRIDE_0: tl.constexpr,
    K_STRIDE_1: tl.constexpr,
    K_ROPE_STRIDE_0: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)

    token_id = pid_token * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    head_id = pid_head * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)

    token_mask = token_id < num_tokens
    head_mask = head_id < NUM_LOCAL_HEADS
    mask = token_mask[:, None, None] & head_mask[None, :, None]

    head_offs_nope = head_id * K_NOPE_STRIDE_1
    nope_sub_id = tl.arange(0, QK_NOPE_HEAD_DIM)
    offs_nope = (
        token_id[:, None, None] * K_NOPE_STRIDE_0
        + head_offs_nope[None, :, None]
        + nope_sub_id[None, None, :]
    )

    head_offs_k = head_id * K_STRIDE_1
    offs_k = (
        token_id[:, None, None] * K_STRIDE_0
        + head_offs_k[None, :, None]
        + nope_sub_id[None, None, :]
    )

    vals_nope = tl.load(k_nope_ptr + offs_nope, mask=mask)
    tl.store(k_ptr + offs_k, vals_nope, mask=mask)

    rope_sub_id = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_rope = token_id[:, None, None] * K_ROPE_STRIDE_0 + rope_sub_id[None, None, :]
    vals_rope = tl.load(k_rope_ptr + offs_rope, mask=token_mask[:, None, None])

    if BLOCK_HEADS <= 32:
        for h in tl.static_range(BLOCK_HEADS):
            head_id_h = pid_head * BLOCK_HEADS + h
            head_mask_h = head_id_h < NUM_LOCAL_HEADS
            mask_h = token_mask[:, None, None] & head_mask_h

            offs_k_h = (
                token_id[:, None, None] * K_STRIDE_0
                + head_id_h * K_STRIDE_1
                + rope_sub_id[None, None, :]
                + QK_NOPE_HEAD_DIM
            )
            tl.store(k_ptr + offs_k_h, vals_rope, mask=mask_h)
    else:
        offs_k_rope = (
            token_id[:, None, None] * K_STRIDE_0
            + head_offs_k[None, :, None]
            + rope_sub_id[None, None, :]
            + QK_NOPE_HEAD_DIM
        )
        tl.store(k_ptr + offs_k_rope, vals_rope, mask=mask)


def fn_triton(k, k_nope, k_rope, qk_nope_head_dim, qk_rope_head_dim, num_local_heads):
    num_tokens, _, _ = k.shape
    grid = lambda meta: (
        triton.cdiv(num_tokens, meta["BLOCK_ROWS"]),
        triton.cdiv(num_local_heads, meta["BLOCK_HEADS"]),
    )
    fn_triton_kernel[grid](
        k,
        k_nope,
        k_rope,
        num_tokens,
        QK_NOPE_HEAD_DIM=qk_nope_head_dim,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        NUM_LOCAL_HEADS=num_local_heads,
        K_NOPE_STRIDE_0=k_nope.stride(0),
        K_NOPE_STRIDE_1=k_nope.stride(1),
        K_STRIDE_0=k.stride(0),
        K_STRIDE_1=k.stride(1),
        K_ROPE_STRIDE_0=k_rope.stride(0),
        BLOCK_ROWS=16,
        BLOCK_HEADS=16,
    )