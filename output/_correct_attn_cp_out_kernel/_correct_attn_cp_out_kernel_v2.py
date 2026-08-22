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
    H,
    BLOCK_H: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    num_n_offsets = tl.arange(0, N_ROUNDED)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_offsets = (
        num_n_offsets[:, None] * lses_stride_N
        + batch_idx * lses_stride_B
        + h_offs[None, :] * lses_stride_H
    )
    lse = tl.load(lses_ptr + lse_offsets, mask=h_mask[None, :], other=-float("inf"))
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)

    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_stable = lse - lse_max[None, :]
    lse_exp = tl.exp(lse_stable)
    sum_exp = tl.sum(lse_exp, axis=0)

    inv_sum = tl.where(sum_exp == 0.0, 0.0, 1.0 / sum_exp)

    lse_final = tl.log(sum_exp) + lse_max
    vlse_offsets = batch_idx * lses_stride_B + h_offs * lses_stride_H
    tl.store(tl.multiple_of(vlse_ptr + vlse_offsets, 16), lse_final, mask=h_mask)

    lse_tmp_offsets = (
        lse_idx * lses_stride_N
        + batch_idx * lses_stride_B
        + h_offs * lses_stride_H
    )
    lse_tmp = tl.load(tl.multiple_of(lses_ptr + lse_tmp_offsets, 16), mask=h_mask, other=-float("inf"))
    lse_tmp = tl.where((lse_tmp != lse_tmp) | (lse_tmp == float("inf")), -float("inf"), lse_tmp)

    factor = tl.exp(lse_tmp - lse_max) * inv_sum

    output_offsets = (
        batch_idx * outputs_stride_B
        + h_offs[:, None] * outputs_stride_H
        + d_offsets[None, :] * outputs_stride_D
    )
    output = tl.load(outputs_ptr + output_offsets, mask=h_mask[:, None])
    output = output * factor[:, None]
    tl.store(new_output_ptr + output_offsets, output, mask=h_mask[:, None])


def correct_attn_cp_out(outputs, lses, lse_idx):
    B, H, D = outputs.shape
    N = lses.shape[0]
    N_ROUNDED = triton.next_power_of_2(N)
    BLOCK_H = 16

    new_output = torch.empty_like(outputs, device='npu')
    vlse = torch.empty((B, H), device='npu', dtype=torch.float32)

    grid = (B, triton.cdiv(H, BLOCK_H))

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
        N_ROUNDED=N_ROUNDED,
        H=H,
        BLOCK_H=BLOCK_H,
    )
    return new_output, vlse