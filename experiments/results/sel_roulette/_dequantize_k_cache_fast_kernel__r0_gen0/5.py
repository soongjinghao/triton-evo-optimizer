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
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 : dim_nope + num_tiles * 4 + dim_rope].view(torch.bfloat16)
    BLOCK_M = 4
    _dequantize_k_cache_fast_kernel[(triton.cdiv(num_tokens, BLOCK_M), 1)](
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
        BLOCK_M=BLOCK_M,
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
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    token_start = pid * BLOCK_M
    offs_m = token_start + tl.arange(0, BLOCK_M)
    token_mask = offs_m < tl.num_programs(0) * BLOCK_M

    # Process all nope blocks: total nope elements = NUM_NOPE_BLOCKS * GROUP_SIZE
    total_nope = NUM_NOPE_BLOCKS * GROUP_SIZE
    offs_nope = tl.arange(0, total_nope)
    nope_mask = offs_nope < DIM_NOPE

    # Load nope quantized data: shape (BLOCK_M, total_nope)
    ptr_q = input_nope_q_ptr + offs_m[:, None] * input_nope_q_stride_0 + offs_nope[None, :]
    y_q = tl.load(ptr_q, mask=token_mask[:, None] & nope_mask[None, :], other=0.0).to(tl.float32)

    # Load nope scales: shape (BLOCK_M, NUM_NOPE_BLOCKS)
    offs_s = tl.arange(0, NUM_NOPE_BLOCKS)
    ptr_s = input_nope_s_ptr + offs_m[:, None] * input_nope_s_stride_0 + offs_s[None, :]
    y_s = tl.load(ptr_s, mask=token_mask[:, None], other=0.0)

    # Dequantize: y_q * y_s, need to expand y_s to match y_q shape
    # y_s shape: (BLOCK_M, NUM_NOPE_BLOCKS) -> expand to (BLOCK_M, total_nope)
    y_s_expanded = tl.view(tl.broadcast_to(y_s[:, :, None], (BLOCK_M, NUM_NOPE_BLOCKS, GROUP_SIZE)), (BLOCK_M, total_nope))
    y = (y_q * y_s_expanded).to(output_ptr.dtype.element_ty)

    # Store nope output
    dst_ptr = output_ptr + offs_m[:, None] * output_stride_0 + offs_nope[None, :]
    tl.store(dst_ptr, y, mask=token_mask[:, None] & nope_mask[None, :])

    # Process rope blocks: total rope elements = DIM_ROPE
    offs_rope = tl.arange(0, DIM_ROPE)
    rope_mask = offs_rope < DIM_ROPE

    # Load rope data: shape (BLOCK_M, DIM_ROPE)
    ptr_rope = input_rope_ptr + offs_m[:, None] * input_rope_stride_0 + offs_rope[None, :]
    data = tl.load(ptr_rope, mask=token_mask[:, None] & rope_mask[None, :], other=0.0).to(tl.bfloat16)

    # Store rope output
    dst_rope_ptr = output_ptr + offs_m[:, None] * output_stride_0 + DIM_NOPE + offs_rope[None, :]
    tl.store(dst_rope_ptr, data, mask=token_mask[:, None] & rope_mask[None, :])