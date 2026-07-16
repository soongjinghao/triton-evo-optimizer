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

    # Load all LSEs (padded with -inf, so no mask needed)
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    lse_exp = tl.exp(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse = tl.log(lse_acc)
    lse += lse_max

    # Store global LSE for this (B,H)
    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # Load specific LSE for correction (lse_idx is valid)
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally)

    # Load and correct output (use mask for D dimension to avoid out-of-bounds)
    mask_d = d_offsets < HEAD_DIM
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    output = tl.load(outputs_ptr + output_offsets, mask=mask_d, other=0.0)
    output = output * factor
    tl.store(new_output_ptr + output_offsets, output, mask=mask_d)

def correct_attn_cp_out(outputs, lses, lse_idx):
    B, H, D = outputs.shape
    N = lses.shape[0]
    N_ROUNDED = triton.next_power_of_2(N)

    # Pad lses to N_ROUNDED with -inf for unmasked vectorized loads
    lses_padded = torch.full((N_ROUNDED, B, H), -float('inf'), device='npu', dtype=lses.dtype)
    lses_padded[:N] = lses

    new_output = torch.empty_like(outputs, device='npu')
    vlse = torch.empty((B, H), device='npu', dtype=torch.float32)

    grid = (B, H)

    _correct_attn_cp_out_kernel[grid](
        outputs,
        new_output,
        lses_padded,
        vlse,
        outputs.stride(0),
        outputs.stride(1),
        outputs.stride(2),
        lses_padded.stride(0),
        lses_padded.stride(1),
        lses_padded.stride(2),
        lse_idx,
        HEAD_DIM=D,
        N_ROUNDED=N_ROUNDED,
        num_warps=8,
    )

    return new_output, vlse