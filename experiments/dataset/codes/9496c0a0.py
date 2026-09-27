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
    IS_SMALL_N: tl.constexpr,
    IS_MEDIUM_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    
    if IS_SMALL_N:
        num_n_offsets = tl.arange(0, N_ROUNDED)
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
    elif IS_MEDIUM_N:
        lse_max = tl.full((), -float("inf"), dtype=tl.float32)
        lse_acc = tl.full((), 0.0, dtype=tl.float32)
        for start_n in range(0, N_ROUNDED, BLOCK_N):
            num_n_offsets = start_n + tl.arange(0, BLOCK_N)
            mask = num_n_offsets < N_ROUNDED
            lse_offsets = (
                num_n_offsets * lses_stride_N
                + batch_idx * lses_stride_B
                + head_idx * lses_stride_H
            )
            lse_block = tl.load(lses_ptr + lse_offsets, mask=mask, other=-float("inf"))
            lse_block = tl.where((lse_block != lse_block) | (lse_block == float("inf")), -float("inf"), lse_block)
            block_max = tl.max(lse_block, axis=0)
            block_max = tl.where(block_max == -float("inf"), -float("inf"), block_max)
            new_lse_max = tl.maximum(lse_max, block_max)
            lse_acc = lse_acc * tl.exp(lse_max - new_lse_max) + tl.sum(tl.exp(lse_block - new_lse_max), axis=0)
            lse_max = new_lse_max
        lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
        lse = tl.log(lse_acc) + lse_max
    else:
        lse_max = tl.full((), -float("inf"), dtype=tl.float32)
        lse_acc = tl.full((), 0.0, dtype=tl.float32)
        for start_n in range(0, N_ROUNDED, 64):
            num_n_offsets = start_n + tl.arange(0, 64)
            mask = num_n_offsets < N_ROUNDED
            lse_offsets = (
                num_n_offsets * lses_stride_N
                + batch_idx * lses_stride_B
                + head_idx * lses_stride_H
            )
            lse_block = tl.load(lses_ptr + lse_offsets, mask=mask, other=-float("inf"))
            lse_block = tl.where((lse_block != lse_block) | (lse_block == float("inf")), -float("inf"), lse_block)
            block_max = tl.max(lse_block, axis=0)
            block_max = tl.where(block_max == -float("inf"), -float("inf"), block_max)
            new_lse_max = tl.maximum(lse_max, block_max)
            lse_acc = lse_acc * tl.exp(lse_max - new_lse_max) + tl.sum(tl.exp(lse_block - new_lse_max), axis=0)
            lse_max = new_lse_max
        lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
        lse = tl.log(lse_acc) + lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)
    
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
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
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + output_offsets, output)

def correct_attn_cp_out(outputs, lses, lse_idx):
    B, H, D = outputs.shape
    N = lses.shape[0]
    N_ROUNDED = triton.next_power_of_2(N)
    
    if N_ROUNDED <= 32:
        IS_SMALL_N = True
        IS_MEDIUM_N = False
        BLOCK_N = 0
    elif N_ROUNDED <= 128:
        IS_SMALL_N = False
        IS_MEDIUM_N = True
        BLOCK_N = N_ROUNDED
    else:
        IS_SMALL_N = False
        IS_MEDIUM_N = False
        BLOCK_N = 64
    
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
        N_ROUNDED=N_ROUNDED,
        IS_SMALL_N=IS_SMALL_N,
        IS_MEDIUM_N=IS_MEDIUM_N,
        BLOCK_N=BLOCK_N
    )
    return new_output, vlse