import torch
import triton
import triton.language as tl

@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_permuted_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_permuted_stride_B,
    lses_permuted_stride_H,
    lses_permuted_stride_N,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    N: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)
    
    # Continuous load from permuted lses (B, H, N) layout
    lse_offsets = (
        batch_idx * lses_permuted_stride_B
        + head_idx * lses_permuted_stride_H
        + num_n_offsets * lses_permuted_stride_N
    )
    lse = tl.load(lses_permuted_ptr + lse_offsets, mask=num_n_offsets < N, other=-float("inf"))
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    lse_exp = tl.exp(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse = tl.log(lse_acc)
    lse += lse_max
    
    # Store vlse
    lse_offsets = batch_idx * lses_permuted_stride_B + head_idx * lses_permuted_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)
    
    # Load lse at lse_idx
    lse_offset = (
        batch_idx * lses_permuted_stride_B
        + head_idx * lses_permuted_stride_H
        + lse_idx * lses_permuted_stride_N
    )
    lse_tmp = tl.load(lses_permuted_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally)
    
    # Load and scale output
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
    
    # Permute lses from (N, B, H) to (B, H, N) for continuous access
    lses_permuted = lses.permute(1, 2, 0).contiguous()
    
    new_output = torch.empty_like(outputs, device='npu')
    vlse = torch.empty((B, H), device='npu', dtype=torch.float32)
    
    grid = (B, H)
    _correct_attn_cp_out_kernel[grid](
        outputs, 
        new_output, 
        lses_permuted, 
        vlse,
        outputs.stride(0),
        outputs.stride(1),
        outputs.stride(2),
        lses_permuted.stride(0),
        lses_permuted.stride(1),
        lses_permuted.stride(2),
        lse_idx,
        HEAD_DIM=D,
        N_ROUNDED=N_ROUNDED,
        N=N,
    )
    return new_output, vlse