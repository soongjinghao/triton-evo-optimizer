from typing import Tuple

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BLOCK_M, "num_warps": num_warps})
        for BLOCK_M in [1, 2, 4, 8, 16]
        for num_warps in [2, 4, 8]
    ],
    key=["M", "N", "group_size"],
)
@triton.jit
def _per_token_group_quant_8bit_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    M,
    N,
    y_row_stride,
    y_q_row_stride,
    y_s_col_stride,
    eps,
    bit8_min,
    bit8_max,
    BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    One-pass fused quantize kernel: load entire group once, compute absmax,
    quantize in-register, and store.  Processes BLOCK_M rows per program.
    Eliminates double HBM loads and keeps all intermediates in registers.
    """
    pid = tl.program_id(0)
    num_groups_per_row = tl.cdiv(N, group_size)

    # Row tile: BLOCK_M consecutive rows
    row_start = pid * BLOCK_M
    row_end = min(row_start + BLOCK_M, M)

    # Precompute column indices for the full group (padded to BLOCK)
    cols = tl.arange(0, BLOCK)

    for m in range(row_start, row_end):
        y_row_ptr = y_ptr + m * y_row_stride
        y_q_row_ptr = y_q_ptr + m * y_q_row_stride

        for group_idx in range(num_groups_per_row):
            col_start = group_idx * group_size
            actual_group_size = min(group_size, N - col_start)
            if actual_group_size <= 0:
                continue

            # Pointer to the scale output: (group_idx, m)
            y_s_ptr_local = y_s_ptr + group_idx * y_s_col_stride + m

            # Load the entire group (padded to BLOCK) once
            mask = cols < actual_group_size
            y_chunk = tl.load(y_row_ptr + col_start + cols, mask=mask, other=0.0).to(tl.float32)

            # Compute absmax in-register
            _absmax = tl.max(tl.abs(y_chunk))
            y_s = _absmax / bit8_max

            if SCALE_UE8M0:
                y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))

            # Quantize in-register
            y_q_chunk = tl.clamp(y_chunk / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

            # Store quantized data and scale
            tl.store(y_q_row_ptr + col_start + cols, y_q_chunk, mask=mask)
            tl.store(y_s_ptr_local, y_s)


def per_token_group_quant_8bit_colmajor(
    y: torch.Tensor, group_size: int, eps: float = 1e-5, scale_ue8m0: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform per-token-group quantization on a tensor.
    Uses one-pass fused kernel to minimize memory traffic.
    """
    assert y.is_contiguous(), "Input tensor must be contiguous"
    y_shape = y.shape
    y = y.view(-1, y_shape[-1])
    M, N = y.shape

    num_groups_per_row = (N + group_size - 1) // group_size
    y_q = torch.empty_like(y, dtype=torch.int8)
    y_s = torch.empty((num_groups_per_row, M), dtype=torch.float32, device=y.device)

    bit8_min = -128.0
    bit8_max = 127.0
    BLOCK = triton.next_power_of_2(group_size)

    # Dynamic grid: launch enough programs to cover all rows
    # BLOCK_M will be chosen by autotuner; default to 4 for grid calculation
    default_block_m = 4
    grid = (triton.cdiv(M, default_block_m),)

    _per_token_group_quant_8bit_colmajor[grid](
        y,
        y_q,
        y_s,
        group_size,
        M,
        N,
        y.stride(0),
        y_q.stride(0),
        y_s.stride(0),
        eps,
        bit8_min,
        bit8_max,
        BLOCK=BLOCK,
        SCALE_UE8M0=scale_ue8m0,
        # BLOCK_M is not passed explicitly; autotuner injects it via config
    )

    y_q = y_q.view(y_shape)
    return y_q, y_s