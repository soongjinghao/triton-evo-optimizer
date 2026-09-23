

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

PAD_SLOT_ID = -1

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)"""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()

@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):

    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    cols_float = cols.to(tl.float32)
    row_mask_float = rows[:, None].to(tl.float32) < M
    col_mask_float = cols_float[None, :] < N
    mask_float = row_mask_float & col_mask_float

    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        x *= z * tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N

        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)

        xbar = tl.where(mask_float, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    else:

        xbar = tl.where(mask_float, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        mean = 0.0

    rstd = tl.rsqrt(var + eps)

    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        y *= z * tl.sigmoid(z)

    tl.store(Y_base, y, mask=mask)

def calc_rows_per_block(M: int) -> int:
    sm_count = 1
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, 4)
    return rows_per_block

def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)

    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    num_warps = min(max(BLOCK_N // 256, 1), 8)

    rows_per_block = calc_rows_per_block(M)

    grid = (cdiv(M, rows_per_block), ngroups)
    layer_norm_fwd_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    return out, mean, rstd
