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
    K_STRIDE_0: tl.constexpr,
    K_ROPE_STRIDE_0: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)
    items_per_program = BLOCK_ROWS * BLOCK_HEADS
    start_idx = pid * items_per_program
    offsets = tl.arange(0, items_per_program)
    flat_idx = start_idx + offsets
    token_id = flat_idx // BLOCK_HEADS
    head_id = flat_idx % BLOCK_HEADS
    token_mask = token_id < num_tokens
    head_mask = head_id < NUM_LOCAL_HEADS
    mask = token_mask[:, None] & head_mask[:, None]
    nope_sub_id = tl.arange(0, QK_NOPE_HEAD_DIM)
    offs_nope = (
        token_id[:, None] * K_NOPE_STRIDE_0
        + head_id[:, None] * K_NOPE_STRIDE_0
        + nope_sub_id[None, :]
    )
    offs_k = (
        token_id[:, None] * K_STRIDE_0
        + head_id[:, None] * K_STRIDE_0
        + nope_sub_id[None, :]
    )
    vals_nope = tl.load(k_nope_ptr + offs_nope, mask=mask)
    tl.store(k_ptr + offs_k, vals_nope, mask=mask)
    rope_sub_id = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_rope = token_id[:, None] * K_ROPE_STRIDE_0 + rope_sub_id[None, :]
    offs_k = (
        token_id[:, None] * K_STRIDE_0
        + head_id[:, None] * K_STRIDE_0
        + rope_sub_id[None, :]
        + QK_NOPE_HEAD_DIM
    )
    vals_rope = tl.load(k_rope_ptr + offs_rope, mask=token_mask[:, None])
    tl.store(k_ptr + offs_k, vals_rope, mask=mask)

def fn_triton(k, k_nope, k_rope, qk_nope_head_dim, qk_rope_head_dim, num_local_heads):
    num_tokens, _, _ = k.shape
    BLOCK_ROWS = 32
    BLOCK_HEADS = 32
    items_per_program = BLOCK_ROWS * BLOCK_HEADS
    grid = lambda meta: (triton.cdiv(num_tokens * num_local_heads, items_per_program),)
    fn_triton_kernel[grid](
        k,
        k_nope,
        k_rope,
        num_tokens,
        QK_NOPE_HEAD_DIM=qk_nope_head_dim,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        NUM_LOCAL_HEADS=num_local_heads,
        K_NOPE_STRIDE_0=k_nope.stride(0),
        K_STRIDE_0=k.stride(0),
        K_ROPE_STRIDE_0=k_rope.stride(0),
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HEADS=BLOCK_HEADS,
    )