import torch
import triton
import triton.language as tl
import math

def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128, TOKENS_PER_BLOCK: int = 4):
    num_tokens, dim_quant = quant_k_cache.shape
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size
    total_dim = dim_nope + dim_rope

    BLOCK_SIZE = total_dim
    assert BLOCK_SIZE % 16 == 0

    if group_size & (group_size - 1) == 0:
        SHIFT = int(math.log2(group_size))
    else:
        SHIFT = -1

    output = torch.empty(
        (num_tokens, total_dim),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    input_nope_q = quant_k_cache[:, :dim_nope].contiguous()
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].contiguous().view(torch.float32)
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].contiguous().view(torch.bfloat16)

    grid = (triton.cdiv(num_tokens, TOKENS_PER_BLOCK),)
    _dequantize_k_cache_fast_kernel[grid](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        BLOCK_SIZE=BLOCK_SIZE,
        SHIFT=SHIFT,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        num_tokens=num_tokens,
    )

    return output


@triton.jit
def _dequantize_k_cache_fast_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SHIFT: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    num_tokens: int,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask_nope = offs < DIM_NOPE
    mask_rope = (offs >= DIM_NOPE) & (offs < DIM_NOPE + DIM_ROPE)
    mask_out = offs < (DIM_NOPE + DIM_ROPE)

    for t in range(TOKENS_PER_BLOCK):
        token_id = pid * TOKENS_PER_BLOCK + t
        if token_id < num_tokens:
            base_nope_q = tl.multiple_of(input_nope_q_ptr + token_id * input_nope_q_stride_0, 16)
            base_nope_s = tl.multiple_of(input_nope_s_ptr + token_id * input_nope_s_stride_0, 16)
            base_rope = tl.multiple_of(input_rope_ptr + token_id * input_rope_stride_0, 16)
            base_out = tl.multiple_of(output_ptr + token_id * output_stride_0, 16)

            q = tl.load(base_nope_q + offs, mask=mask_nope, other=0.0).to(tl.float32)

            s0 = tl.load(base_nope_s + 0)
            s1 = tl.load(base_nope_s + 1)
            s2 = tl.load(base_nope_s + 2)
            s3 = tl.load(base_nope_s + 3)

            if SHIFT >= 0:
                group_id = offs >> SHIFT
            else:
                group_id = offs // GROUP_SIZE

            s = (
                tl.where(group_id == 0, s0, 0.0)
                + tl.where(group_id == 1, s1, 0.0)
                + tl.where(group_id == 2, s2, 0.0)
                + tl.where(group_id == 3, s3, 0.0)
            )

            y_nope = (q * s).to(tl.bfloat16)

            offs_rope = tl.maximum(offs - DIM_NOPE, 0)
            rope_data = tl.load(base_rope + offs_rope, mask=mask_rope, other=0.0).to(tl.bfloat16)

            out = tl.where(mask_nope, y_nope, rope_data)

            tl.store(base_out + offs, out, mask=mask_out)