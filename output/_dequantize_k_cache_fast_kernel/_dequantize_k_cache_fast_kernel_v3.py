import torch
import triton
import triton.language as tl


def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128):
    num_tokens, dim_quant = quant_k_cache.shape
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size
    assert dim_quant == 656
    output = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    grid = (num_tokens,)
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
):
    token_id = tl.program_id(0)

    row_off = token_id * input_nope_q_stride_0

    offs_nope = tl.arange(0, DIM_NOPE)
    ptr_q = tl.multiple_of(input_nope_q_ptr + row_off, 16) + offs_nope
    q = tl.load(ptr_q).to(tl.float32)

    offs_s = tl.arange(0, NUM_NOPE_BLOCKS)
    ptr_s = tl.multiple_of(input_nope_s_ptr + row_off, 16) + offs_s
    scales = tl.load(ptr_s)

    q_2d = tl.view(q, (NUM_NOPE_BLOCKS, GROUP_SIZE))
    scales_2d = tl.view(scales, (NUM_NOPE_BLOCKS, 1))
    y_nope = tl.view((q_2d * scales_2d).to(tl.bfloat16), (DIM_NOPE,))

    out_row_off = token_id * output_stride_0
    ptr_out_nope = tl.multiple_of(output_ptr + out_row_off, 16) + offs_nope
    tl.store(ptr_out_nope, y_nope)

    offs_rope = tl.arange(0, DIM_ROPE)
    ptr_rope = tl.multiple_of(input_rope_ptr + row_off, 16) + offs_rope
    rope = tl.load(ptr_rope).to(tl.bfloat16)
    offs_out_rope = DIM_NOPE + offs_rope
    ptr_out_rope = tl.multiple_of(output_ptr + out_row_off, 16) + offs_out_rope
    tl.store(ptr_out_rope, rope)