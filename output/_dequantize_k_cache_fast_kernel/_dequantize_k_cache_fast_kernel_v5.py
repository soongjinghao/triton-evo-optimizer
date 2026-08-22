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

    output_stride_0 = output.stride(0)
    input_nope_q_stride_0 = input_nope_q.stride(0)
    input_nope_s_stride_0 = input_nope_s.stride(0)
    input_rope_stride_0 = input_rope.stride(0)

    assert input_nope_q_stride_0 == dim_quant, (
        "input_nope_q must be contiguous in the innermost dimension"
    )

    grid = (num_tokens, num_tiles)

    _dequantize_k_cache_fast_kernel[grid](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        output_stride_0,
        input_nope_q_stride_0,
        input_nope_s_stride_0,
        input_rope_stride_0,
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
    output_stride_0: tl.constexpr,
    input_nope_q_stride_0: tl.constexpr,
    input_nope_s_stride_0: tl.constexpr,
    input_rope_stride_0: tl.constexpr,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_block = tl.program_id(1)

    row_off_q = pid_token * input_nope_q_stride_0
    row_off_s = pid_token * input_nope_s_stride_0
    row_off_out = pid_token * output_stride_0
    row_off_rope = pid_token * input_rope_stride_0

    offs_q = pid_block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    ptr_q = tl.multiple_of(input_nope_q_ptr + row_off_q, 16) + offs_q
    q = tl.load(ptr_q).to(tl.float32)

    scale = tl.load(input_nope_s_ptr + row_off_s + pid_block).to(tl.float32)

    y = (q * scale).to(output_ptr.dtype.element_ty)

    offs_out = pid_block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    ptr_out = tl.multiple_of(output_ptr + row_off_out, 16) + offs_out
    tl.store(ptr_out, y)

    if pid_block == 0:
        offs_rope = tl.arange(0, DIM_ROPE)
        ptr_rope = tl.multiple_of(input_rope_ptr + row_off_rope, 16) + offs_rope
        rope = tl.load(ptr_rope).to(tl.bfloat16)
        offs_out_rope = DIM_NOPE + offs_rope
        ptr_out_rope = tl.multiple_of(output_ptr + row_off_out, 16) + offs_out_rope
        tl.store(ptr_out_rope, rope)