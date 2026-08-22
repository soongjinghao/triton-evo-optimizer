import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_k_cache_fast_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NOPE_GRID_BLOCKS: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    # 预计算倒数,将除法替换为乘法
    inv_fp8_max = 1.0 / FP8_MAX

    if raw_block_id < NOPE_GRID_BLOCKS:
        # GROUPS_PER_BLOCK = BLOCK_N // GROUP_SIZE = 8
        effective_block_id = raw_block_id * 8
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, BLOCK_N)
        mask = offs < DIM_NOPE

        ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs
        ptr = tl.multiple_of(ptr, 16)
        y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

        # 将数据重塑为 (8, GROUP_SIZE)
        y_2d = tl.view(y, (BLOCK_N // GROUP_SIZE, GROUP_SIZE))

        # 利用向量空闲单元,用乘法代替标量除法
        y_s = tl.max(tl.abs(y_2d), axis=1) * inv_fp8_max
        y_s_inv = 1.0 / y_s  # 向量倒数,利用向量单元

        y_q = tl.clamp(y_2d * y_s_inv[:, None], FP8_MIN, FP8_MAX).to(
            output_nope_q_ptr.dtype.element_ty
        )
        y_q_flat = tl.view(y_q, (BLOCK_N,))

        dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
        dst_q_ptr = tl.multiple_of(dst_q_ptr, 16)
        tl.store(dst_q_ptr, y_q_flat, mask=mask)

        offs_s = effective_block_id + tl.arange(0, 8)
        mask_s = offs_s < NUM_NOPE_BLOCKS
        dst_s_ptr = output_nope_s_ptr + token_id * output_nope_s_stride_0 + offs_s
        tl.store(dst_s_ptr, y_s, mask=mask_s)
    else:
        # rope 部分,直接拷贝
        effective_block_id = (raw_block_id - NOPE_GRID_BLOCKS) * 8
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, BLOCK_N)
        mask = offs < DIM_ROPE

        src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs
        src_ptr = tl.multiple_of(src_ptr, 16)
        dst_ptr = tl.multiple_of(dst_ptr, 16)
        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)


def _quantize_k_cache_fast(k_nope, k_rope, group_size: int = 128):
    assert k_nope.dtype == torch.bfloat16
    assert k_rope.dtype == torch.bfloat16
    num_tokens, dim_nope = k_nope.shape
    num_tokens_, dim_rope = k_rope.shape
    assert num_tokens == num_tokens_
    assert k_nope.dtype == k_rope.dtype
    assert k_nope.stride(1) == 1
    assert k_rope.stride(1) == 1

    num_tiles = triton.cdiv(dim_nope, group_size)
    NUM_NOPE_BLOCKS = num_tiles
    NUM_ROPE_BLOCKS = triton.cdiv(dim_rope, group_size)

    # 网格收拢:GROUPS_PER_BLOCK 从 4 提升至 8
    GROUPS_PER_BLOCK = 8
    BLOCK_N = group_size * GROUPS_PER_BLOCK

    NOPE_GRID_BLOCKS = triton.cdiv(NUM_NOPE_BLOCKS, GROUPS_PER_BLOCK)
    ROPE_GRID_BLOCKS = triton.cdiv(NUM_ROPE_BLOCKS, GROUPS_PER_BLOCK)
    total_blocks = NOPE_GRID_BLOCKS + ROPE_GRID_BLOCKS

    output = torch.empty(
        (num_tokens, dim_nope + num_tiles * 4 + dim_rope),
        dtype=torch.bfloat16,
        device=k_nope.device,
    )
    output_nope_q = output[..., :dim_nope]
    output_nope_s = output[..., dim_nope : dim_nope + num_tiles * 4].view(torch.bfloat16)
    output_rope = output[..., dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    grid = lambda meta: (min(num_tokens, 65535), min(total_blocks, 65535))
    _quantize_k_cache_fast_kernel[grid](
        output_nope_q,
        output_nope_s,
        output_rope,
        k_nope,
        k_rope,
        output_nope_q.stride(0),
        output_nope_s.stride(0),
        output_rope.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        FP8_MIN=-448.0,
        FP8_MAX=448.0,
        BLOCK_N=BLOCK_N,
        NOPE_GRID_BLOCKS=NOPE_GRID_BLOCKS,
    )
    return output