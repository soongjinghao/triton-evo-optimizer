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
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5
    assert dim_nope % group_size == 0
    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    BLOCK_SIZE = num_blocks_per_token * group_size
    _dequantize_k_cache_fast_kernel[(num_tokens,)](
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
):
    token_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    offs_q = tl.arange(0, NUM_NOPE_BLOCKS * GROUP_SIZE)
    offs_rope = tl.arange(0, DIM_ROPE) + NUM_NOPE_BLOCKS * GROUP_SIZE

    # Load nope quantized data
    mask_q = offs_q < DIM_NOPE
    ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_q
    y_q = tl.load(ptr_q, mask=mask_q, other=0.0).to(tl.float32)

    # Load nope scales
    ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + tl.arange(0, NUM_NOPE_BLOCKS)
    y_s = tl.load(ptr_s)

    # Reshape and multiply
    y_q_2d = tl.view(y_q, (NUM_NOPE_BLOCKS, GROUP_SIZE))
    y_s_2d = tl.view(y_s, (NUM_NOPE_BLOCKS, 1))
    y_nope = (y_q_2d * y_s_2d).to(output_ptr.dtype.element_ty)
    y_nope_flat = tl.view(y_nope, (NUM_NOPE_BLOCKS * GROUP_SIZE,))

    # Store nope part
    dst_ptr_nope = output_ptr + token_id * output_stride_0 + offs_q
    tl.store(dst_ptr_nope, y_nope_flat, mask=mask_q)

    # Load and store rope part
    mask_rope = offs_rope < DIM_NOPE + DIM_ROPE
    src_ptr_rope = input_rope_ptr + token_id * input_rope_stride_0 + offs_rope - NUM_NOPE_BLOCKS * GROUP_SIZE
    dst_ptr_rope = output_ptr + token_id * output_stride_0 + offs_rope
    data_rope = tl.load(src_ptr_rope, mask=mask_rope).to(tl.bfloat16)
    tl.store(dst_ptr_rope, data_rope, mask=mask_rope)