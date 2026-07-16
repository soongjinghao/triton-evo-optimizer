import torch
import triton
import triton.language as tl

@triton.jit
def _lse_correct_output_fused_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0).to(tl.int32)
    head_idx = tl.program_id(axis=1).to(tl.int32)
    block_idx = tl.program_id(axis=2).to(tl.int32)

    # Compute LSE over N dimension (padded to power of 2)
    num_n_offsets = tl.arange(0, N_ROUNDED)
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )
    mask_n = num_n_offsets < N

    lse = tl.load(lses_ptr + lse_offsets, mask=mask_n, other=-float("inf"))

    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse -= lse_max
    lse_exp = tl.exp(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse = tl.log(lse_acc)
    lse += lse_max

    # Load lse_tmp for the specific lse_idx
    lse_offset_tmp = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset_tmp)

    # Compute factor = exp(lse_tmp - lse)
    factor = tl.exp(lse_tmp - lse)

    # Correct output for this block of HEAD_DIM
    d_start = block_idx * BLOCK_SIZE_D
    d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
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

    new_output = torch.empty_like(outputs, device='npu')

    BLOCK_SIZE_D = min(triton.next_power_of_2(D), 1024)
    BLOCK_SIZE_D = max(16, (BLOCK_SIZE_D // 16) * 16)

    grid = (B, H, triton.cdiv(D, BLOCK_SIZE_D))

    _lse_correct_output_fused_kernel[grid](
        outputs,
        new_output,
        lses,
        outputs.stride(0),
        outputs.stride(1),
        outputs.stride(2),
        lses.stride(0),
        lses.stride(1),
        lses.stride(2),
        lse_idx,
        HEAD_DIM=D,
        N=N,
        N_ROUNDED=N_ROUNDED,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return new_output