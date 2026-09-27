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
    offs_k = (
        token_id[:, None, None] * K_STRIDE_0
        + head_offs_k[None, :, None]
        + rope_sub_id[None, None, :]
        + QK_NOPE_HEAD_DIM
    )
    vals_rope = tl.load(k_rope_ptr + offs_rope, mask=token_mask[:, None, None])
    tl.store(k_ptr + offs_k, vals_rope, mask=mask)


def fn_triton(k, k_nope, k_rope, qk_nope_head_dim, qk_rope_head_dim, num_local_heads):
    num_tokens, _, _ = k.shape

    total_head_dim = qk_nope_head_dim + qk_rope_head_dim
    block_rows, block_heads = 16, 16

    many_tokens = num_tokens >= 32
    many_heads = num_local_heads >= 32

    if total_head_dim <= 192:
        if many_tokens and num_tokens >= 2 * num_local_heads:
            block_rows, block_heads = 32, 16
        elif many_heads and num_local_heads >= 2 * num_tokens:
            if num_tokens <= 8:
                block_rows, block_heads = 8, 32
            else:
                block_rows, block_heads = 16, 32
    else:
        if many_heads and num_tokens <= 16:
            block_rows, block_heads = 8, 32

    if block_rows * block_heads > 512:
        block_rows, block_heads = 16, 16

    grid = (
        triton.cdiv(num_tokens, block_rows),
        triton.cdiv(num_local_heads, block_heads),
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
        BLOCK_ROWS=block_rows,
        BLOCK_HEADS=block_heads,
    )