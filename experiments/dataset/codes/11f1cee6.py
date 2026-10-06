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
    
    # Pre-pack input_nope_q and input_nope_s into interleaved layout
    # Layout: [q0, s0, q1, s1, ...] for each group, contiguous in memory
    # Each group: group_size q values + 1 s value (4 bytes as float32)
    # Total per group: group_size * 2 + 4 bytes (q as bf16, s as f32)
    # We pack as bf16 interleaved: q0_bf16, s0_bf16, q1_bf16, s1_bf16, ...
    # s is float32, we reinterpret as two bf16 values for packing
    packed_list = []
    for g in range(num_tiles):
        q_start = g * group_size
        q_end = q_start + group_size
        q_block = input_nope_q[:, q_start:q_end].contiguous()  # (num_tokens, group_size) bf16
        s_block = input_nope_s[:, g:g+1].contiguous()  # (num_tokens, 1) float32
        # Reinterpret s as bf16 (2 bf16 values per float32)
        s_bf16 = s_block.view(torch.bfloat16)  # (num_tokens, 2)
        # Interleave: q0, s0_low, s0_high, q1, s1_low, s1_high, ...
        # For simplicity, pack as [q_block, s_bf16] concatenated along dim=1
        packed_block = torch.cat([q_block, s_bf16], dim=1)  # (num_tokens, group_size + 2) bf16
        packed_list.append(packed_block)
    input_packed = torch.cat(packed_list, dim=1)  # (num_tokens, num_tiles * (group_size + 2)) bf16
    
    _dequantize_k_cache_fast_kernel[(triton.cdiv(num_tokens, 1), num_blocks_per_token)](
        output,
        input_packed,
        input_rope,
        output.stride(0),
        input_packed.stride(0),
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
    input_packed_ptr,
    input_rope_ptr,
    output_stride_0: int,
    input_packed_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)
    if raw_block_id < NUM_NOPE_BLOCKS:
        effective_block_id = raw_block_id
        # Load packed data: group_size q values + 2 bf16 values for s (reinterpreted from float32)
        offs_packed = effective_block_id * (GROUP_SIZE + 2) + tl.arange(0, GROUP_SIZE + 2)
        mask_packed = offs_packed < (NUM_NOPE_BLOCKS * (GROUP_SIZE + 2))
        ptr_packed = input_packed_ptr + token_id * input_packed_stride_0 + offs_packed
        packed_data = tl.load(ptr_packed, mask=mask_packed, other=0.0)  # (GROUP_SIZE + 2,) bf16
        
        # Split: first GROUP_SIZE are q values, last 2 are s (reinterpreted as float32)
        offs_q = tl.arange(0, GROUP_SIZE)
        mask_q = offs_q < GROUP_SIZE
        y_q = packed_data[offs_q].to(tl.float32)
        
        # Reconstruct s from two bf16 values
        s_low = packed_data[GROUP_SIZE].to(tl.float32)
        s_high = packed_data[GROUP_SIZE + 1].to(tl.float32)
        # Reinterpret as float32: s_low is lower 16 bits, s_high is upper 16 bits
        # Use bit manipulation: s = (s_high << 16) | s_low (as uint32)
        s_uint32 = (s_high.to(tl.uint32) << 16) | s_low.to(tl.uint32)
        y_s = tl.inline_asm_elementwise(
            "mov {0}, {1};",
            constraints="=r,r",
            args=[s_uint32],
            dtype=tl.float32,
            is_pure=True,
        )
        
        y = (y_q * y_s).to(output_ptr.dtype.element_ty)
        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask_q)
    else:
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        src_ptr = input_rope_ptr + token_id * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs
        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)