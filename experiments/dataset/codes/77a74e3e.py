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

    token_start = pid_token * BLOCK_ROWS
    head_start = pid_head * BLOCK_HEADS

    k_nope_block = tl.make_block_ptr(
        base=k_nope_ptr,
        shape=(num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM),
        strides=(K_NOPE_STRIDE_0, K_NOPE_STRIDE_1, 1),
        offsets=(token_start, head_start, 0),
        block_shape=(BLOCK_ROWS, BLOCK_HEADS, QK_NOPE_HEAD_DIM),
        order=(2, 1, 0),
    )
    vals_nope = tl.load(k_nope_block)

    k_nope_out_block = tl.make_block_ptr(
        base=k_ptr,
        shape=(num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM),
        strides=(K_STRIDE_0, K_STRIDE_1, 1),
        offsets=(token_start, head_start, 0),
        block_shape=(BLOCK_ROWS, BLOCK_HEADS, QK_NOPE_HEAD_DIM),
        order=(2, 1, 0),
    )
    tl.store(k_nope_out_block, vals_nope)

    k_rope_block = tl.make_block_ptr(
        base=k_rope_ptr,
        shape=(num_tokens, 1, QK_ROPE_HEAD_DIM),
        strides=(K_ROPE_STRIDE_0, 1, 1),
        offsets=(token_start, 0, 0),
        block_shape=(BLOCK_ROWS, 1, QK_ROPE_HEAD_DIM),
        order=(2, 1, 0),
    )
    vals_rope = tl.load(k_rope_block)

    k_rope_out_block = tl.make_block_ptr(
        base=k_ptr,
        shape=(num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM),
        strides=(K_STRIDE_0, K_STRIDE_1, 1),
        offsets=(token_start, head_start, QK_NOPE_HEAD_DIM),
        block_shape=(BLOCK_ROWS, BLOCK_HEADS, QK_ROPE_HEAD_DIM),
        order=(2, 1, 0),
    )
    vals_rope = tl.broadcast_to(
        vals_rope, (BLOCK_ROWS, BLOCK_HEADS, QK_ROPE_HEAD_DIM)
    )
    tl.store(k_rope_out_block, vals_rope)


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