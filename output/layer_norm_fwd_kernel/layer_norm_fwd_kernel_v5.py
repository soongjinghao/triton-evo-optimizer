from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
PAD_SLOT_ID = -1
def cdiv(a: int, b: int) -> int:
    return -(a // -b)
def next_power_of_2(n: int) -> int:
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
    group = tl.program_id(0)
    const_off = group * N
    w_base = W + const_off
    if HAS_BIAS:
        b_base = B + const_off
    for row_start in range(0, M, ROWS_PER_BLOCK):
        rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
        row_mask = rows < M
        x_base_ptr = X + rows * stride_x_row + const_off
        y_base_ptr = Y + rows * stride_y_row + const_off
        if HAS_Z:
            z_base_ptr = Z + rows * stride_z_row + const_off
        if N > BLOCK_N:
            num_full_blocks = N // BLOCK_N
            remainder = N % BLOCK_N
            sum_x = tl.zeros([ROWS_PER_BLOCK], dtype=tl.float32)
            sum_x2 = tl.zeros([ROWS_PER_BLOCK], dtype=tl.float32)
            for block_idx in range(num_full_blocks):
                col_start = block_idx * BLOCK_N
                cols = col_start + tl.arange(0, BLOCK_N)
                mask = row_mask[:, None]
                X_base = x_base_ptr[:, None] + cols[None, :]
                x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)
                if HAS_Z and not NORM_BEFORE_GATE:
                    Z_base = z_base_ptr[:, None] + cols[None, :]
                    z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                    x = x * z * tl.sigmoid(z)
                sum_x += tl.sum(x, axis=1)
                sum_x2 += tl.sum(x * x, axis=1)
            if remainder > 0:
                col_start = num_full_blocks * BLOCK_N
                cols = col_start + tl.arange(0, BLOCK_N)
                col_mask = cols < N
                mask = row_mask[:, None] & col_mask[None, :]
                X_base = x_base_ptr[:, None] + cols[None, :]
                x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)
                if HAS_Z and not NORM_BEFORE_GATE:
                    Z_base = z_base_ptr[:, None] + cols[None, :]
                    z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                    x = x * z * tl.sigmoid(z)
                sum_x += tl.sum(x, axis=1)
                sum_x2 += tl.sum(x * x, axis=1)
            inv_N = 1.0 / N
            mean = sum_x * inv_N if not IS_RMS_NORM else 0.0
            var = sum_x2 * inv_N - mean * mean
            if not IS_RMS_NORM:
                mean_offsets = group * M + rows
                tl.store(Mean + mean_offsets, mean, mask=row_mask)
            rstd = tl.rsqrt(var + eps)
            rstd_offsets = group * M + rows
            tl.store(Rstd + rstd_offsets, rstd, mask=row_mask)
            for block_idx in range(num_full_blocks):
                col_start = block_idx * BLOCK_N
                cols = col_start + tl.arange(0, BLOCK_N)
                mask = row_mask[:, None]
                X_base = x_base_ptr[:, None] + cols[None, :]
                x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)
                if HAS_Z and not NORM_BEFORE_GATE:
                    Z_base = z_base_ptr[:, None] + cols[None, :]
                    z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                    x = x * z * tl.sigmoid(z)
                w = tl.load(w_base + cols, mask=None, other=0.0).to(tl.float32)
                if HAS_BIAS:
                    b = tl.load(b_base + cols, mask=None, other=0.0).to(tl.float32)
                if not IS_RMS_NORM:
                    x_hat = (x - mean[:, None]) * rstd[:, None]
                else:
                    x_hat = x * rstd[:, None]
                y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]
                if HAS_Z and NORM_BEFORE_GATE:
                    Z_base = z_base_ptr[:, None] + cols[None, :]
                    z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                    y = y * z * tl.sigmoid(z)
                Y_base = y_base_ptr[:, None] + cols[None, :]
                tl.store(Y_base, y, mask=mask)
            if remainder > 0:
                col_start = num_full_blocks * BLOCK_N
                cols = col_start + tl.arange(0, BLOCK_N)
                col_mask = cols < N
                mask = row_mask[:, None] & col_mask[None, :]
                X_base = x_base_ptr[:, None] + cols[None, :]
                x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)
                if HAS_Z and not NORM_BEFORE_GATE:
                    Z_base = z_base_ptr[:, None] + cols[None, :]
                    z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                    x = x * z * tl.sigmoid(z)
                w = tl.load(w_base + cols, mask=col_mask, other=0.0).to(tl.float32)
                if HAS_BIAS:
                    b = tl.load(b_base + cols, mask=col_mask, other=0.0).to(tl.float32)
                if not IS_RMS_NORM:
                    x_hat = (x - mean[:, None]) * rstd[:, None]
                else:
                    x_hat = x * rstd[:, None]
                y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]
                if HAS_Z and NORM_BEFORE_GATE:
                    Z_base = z_base_ptr[:, None] + cols[None, :]
                    z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                    y = y * z * tl.sigmoid(z)
                Y_base = y_base_ptr[:, None] + cols[None, :]
                tl.store(Y_base, y, mask=mask)
        else:
            cols = tl.arange(0, BLOCK_N)
            col_mask = cols[None, :] < N
            mask = row_mask[:, None] & col_mask
            X_base = x_base_ptr[:, None] + cols[None, :]
            Y_base = y_base_ptr[:, None] + cols[None, :]
            x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)
            if HAS_Z and not NORM_BEFORE_GATE:
                Z_base = z_base_ptr[:, None] + cols[None, :]
                z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                x = x * z * tl.sigmoid(z)
            sum_x = tl.sum(x, axis=1)
            sum_x2 = tl.sum(x * x, axis=1)
            inv_N = 1.0 / N
            mean = sum_x * inv_N if not IS_RMS_NORM else 0.0
            var = sum_x2 * inv_N - mean * mean
            if not IS_RMS_NORM:
                mean_offsets = group * M + rows
                tl.store(Mean + mean_offsets, mean, mask=row_mask)
            rstd = tl.rsqrt(var + eps)
            rstd_offsets = group * M + rows
            tl.store(Rstd + rstd_offsets, rstd, mask=row_mask)
            w = tl.load(w_base + cols, mask=col_mask, other=0.0).to(tl.float32)
            if HAS_BIAS:
                b = tl.load(b_base + cols, mask=col_mask, other=0.0).to(tl.float32)
            if not IS_RMS_NORM:
                x_hat = (x - mean[:, None]) * rstd[:, None]
            else:
                x_hat = x * rstd[:, None]
            y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]
            if HAS_Z and NORM_BEFORE_GATE:
                Z_base = z_base_ptr[:, None] + cols[None, :]
                z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
                y = y * z * tl.sigmoid(z)
            tl.store(Y_base, y, mask=mask)
def calc_rows_per_block(
    M: int,
    BLOCK_N: int,
    LOGICAL_N: int,
    MAX_FUSED_SIZE: int,
) -> int:
    rows_per_block = next_power_of_2(cdiv(M, 2))
    if BLOCK_N <= 256:
        row_cap = 32
    elif BLOCK_N <= 512:
        row_cap = 16
    else:
        row_cap = 8
    rows_per_block = min(rows_per_block, row_cap)
    max_rows_by_tile = max(MAX_FUSED_SIZE // BLOCK_N, 1)
    rows_per_block = min(rows_per_block, max_rows_by_tile)
    max_rows_by_logical_n = max(MAX_FUSED_SIZE // LOGICAL_N, 1)
    rows_per_block = min(rows_per_block, max_rows_by_logical_n)
    return max(rows_per_block, 1)
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
    if group_size > 8192:
        BLOCK_N = 1024
    else:
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    rows_per_block = calc_rows_per_block(M, BLOCK_N, group_size, MAX_FUSED_SIZE)
    if BLOCK_N <= 256:
        num_warps = 4
    elif BLOCK_N <= 512:
        num_warps = 4
    else:
        num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (ngroups,)
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