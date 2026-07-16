import torch
import triton
import triton.language as tl
import math

def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128, ROWS_PER_BLOCK: int = 4):
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

    grid = (triton.cdiv(num_tokens, ROWS_PER_BLOCK),)

    _dequantize_k_cache_fast_kernel[grid](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        num_tokens,
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        BLOCK_SIZE=BLOCK_SIZE,
        SHIFT=SHIFT,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
    )

    return output


@triton.jit
def _dequantize_k_cache_fast_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    num_tokens: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SHIFT: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask_nope = offs < DIM_NOPE
    mask_rope = (offs >= DIM_NOPE) & (offs < DIM_NOPE + DIM_ROPE)
    mask_out = offs < (DIM_NOPE + DIM_ROPE)

    if SHIFT >= 0:
        group_id = offs >> SHIFT
    else:
        group_id = offs // GROUP_SIZE

    for i in tl.static_range(ROWS_PER_BLOCK):
        token_id = pid * ROWS_PER_BLOCK + i
        if token_id < num_tokens:
            base_nope_q = tl.multiple_of(input_nope_q_ptr + token_id * DIM_NOPE, 16)
            base_nope_s = tl.multiple_of(input_nope_s_ptr + token_id * NUM_NOPE_BLOCKS, 16)
            base_rope = tl.multiple_of(input_rope_ptr + token_id * DIM_ROPE, 16)
            base_out = tl.multiple_of(output_ptr + token_id * (DIM_NOPE + DIM_ROPE), 16)

            q = tl.load(base_nope_q + offs, mask=mask_nope, other=0.0).to(tl.float32)
            s = tl.load(base_nope_s + group_id, mask=mask_nope, other=0.0).to(tl.float32)
            y_nope = (q * s).to(tl.bfloat16)

            offs_rope = tl.maximum(offs - DIM_NOPE, 0)
            rope_data = tl.load(base_rope + offs_rope, mask=mask_rope, other=0.0).to(tl.bfloat16)

            out = tl.where(mask_nope, y_nope, rope_data)
            tl.store(base_out + offs, out, mask=mask_out)