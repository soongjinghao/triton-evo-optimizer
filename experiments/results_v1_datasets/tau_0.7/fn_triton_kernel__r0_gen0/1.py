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
):
    pid = tl.program_id(axis=0)
    token_id = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    token_mask = token_id < num_tokens
    head_id = tl.arange(0, NUM_LOCAL_HEADS)

    # 公共 k 行基址:token/head 部分只计算一次,供 nope 与 rope store 复用。
    k_row_base = (
        token_id[:, None, None] * K_STRIDE_0
        + head_id[None, :, None] * K_STRIDE_1
    )

    nope_sub_id = tl.arange(0, QK_NOPE_HEAD_DIM)
    offs_nope = (
        token_id[:, None, None] * K_NOPE_STRIDE_0
        + head_id[None, :, None] * K_NOPE_STRIDE_1
        + nope_sub_id[None, None, :]
    )
    vals_nope = tl.load(k_nope_ptr + offs_nope, mask=token_mask[:, None, None])

    offs_k_nope = k_row_base + nope_sub_id[None, None, :]
    tl.store(k_ptr + offs_k_nope, vals_nope, mask=token_mask[:, None, None])

    rope_sub_id = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_rope = (
        token_id[:, None, None] * K_ROPE_STRIDE_0
        + rope_sub_id[None, None, :]
    )
    vals_rope = tl.load(k_rope_ptr + offs_rope, mask=token_mask[:, None, None])

    offs_k_rope = k_row_base + rope_sub_id[None, None, :] + QK_NOPE_HEAD_DIM
    tl.store(k_ptr + offs_k_rope, vals_rope, mask=token_mask[:, None, None])


def fn_triton(k, k_nope, k_rope, qk_nope_head_dim, qk_rope_head_dim, num_local_heads):
    num_tokens, _, _ = k.shape
    grid = lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_ROWS"]),)
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
    )