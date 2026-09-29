import torch
import triton
import triton.language as tl

@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0).to(tl.int32)
    head_idx = tl.program_id(axis=1).to(tl.int32)
    
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)
    
    # Load all LSE values for this (B, H)
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )
    lse = tl.load(lses_ptr + lse_offsets)
    
    # Handle NaN and inf
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    
    # Compute logsumexp
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_shifted = lse - lse_max
    lse_exp = tl.exp(lse_shifted)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse_log = tl.log(lse_acc)
    lse_final = lse_log + lse_max
    
    # Store final LSE
    lse_offsets_out = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets_out, lse_final)
    
    # Load specific LSE for correction
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    
    # Compute correction factor
    lse_diff = lse_tmp - lse_final
    lse_diff = tl.where(
        (lse_diff != lse_diff) | (lse_diff == float("inf")),
        -float("inf"),
        lse_diff,
    )
    factor = tl.exp(lse_diff)
    
    # Apply correction to output
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + output_offsets, output)

def correct_attn_cp_out(outputs, lses, lse_idx):
    B, H, D = outputs.shape
    N = lses.shape[0]
    N_ROUNDED = triton.next_power_of_2(N)
    
    new_output = torch.empty_like(outputs, device='npu')
    vlse = torch.empty((B, H), device='npu', dtype=torch.float32)
    
    grid = (B, H)
    _correct_attn_cp_out_kernel[grid](
        outputs, 
        new_output, 
        lses, 
        vlse,
        outputs.stride(0),
        outputs.stride(1),
        outputs.stride(2),
        lses.stride(0),
        lses.stride(1),
        lses.stride(2),
        lse_idx,
        HEAD_DIM=D,
        N_ROUNDED=N_ROUNDED
    )
    
    return new_output, vlse