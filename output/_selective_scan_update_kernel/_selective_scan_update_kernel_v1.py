import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from packaging import version
from typing import Optional

PAD_SLOT_ID = -1

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.jit
def _selective_scan_update_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Adjust base pointers for batch and head
    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < dim

    # Load x (shared across all dstate tiles)
    x = tl.load(x_ptr + offs_m * stride_x_dim, mask=mask_m, other=0.0).to(tl.float32)

    # Load dt (per dim or scalar)
    if not TIE_HDIM:
        dt = tl.load(dt_ptr + offs_m * stride_dt_dim, mask=mask_m, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr + offs_m * stride_dt_bias_dim, mask=mask_m, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)

    # Load D and z once
    if HAS_D:
        D = tl.load(D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim, mask=mask_m, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptr + offs_m * stride_z_dim, mask=mask_m, other=0.0).to(tl.float32)

    # Precompute dA scalar for TIE_HDIM
    if TIE_HDIM:
        A_scalar = tl.load(A_ptr).to(tl.float32)
        dA_scalar = tl.exp(A_scalar * dt)  # dt is scalar

    # Output accumulator
    out_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Loop over dstate tiles
    for n_start in range(0, dstate, BLOCK_SIZE_DSTATE):
        offs_n = n_start + tl.arange(0, BLOCK_SIZE_DSTATE)
        mask_n = offs_n < dstate
        # Combined mask for state tile
        mask_state = mask_m[:, None] & mask_n[None, :]
        if HAS_STATE_BATCH_INDICES:
            mask_state &= state_batch_idx != pad_slot_id

        # Load state tile
        state_ptrs = state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        state_tile = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)

        # Load B and C chunks
        B_chunk = tl.load(B_ptr + offs_n * stride_B_dstate, mask=mask_n, other=0.0).to(tl.float32)
        C_chunk = tl.load(C_ptr + offs_n * stride_C_dstate, mask=mask_n, other=0.0).to(tl.float32)

        if not TIE_HDIM:
            # Load A tile
            A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
            A_tile = tl.load(A_ptrs, mask=mask_state, other=0.0).to(tl.float32)
            dA_tile = tl.exp(A_tile * dt[:, None])
            dB_chunk = B_chunk[None, :] * dt[:, None]
        else:
            dA_tile = dA_scalar  # broadcast
            dB_chunk = B_chunk * dt  # dt scalar, B_chunk vector -> broadcast to (1, BLOCK_SIZE_DSTATE) then broadcast with x

        # Update state
        new_state_tile = state_tile * dA_tile + dB_chunk * x[:, None]

        # Store updated state tile
        tl.store(state_ptrs, new_state_tile, mask=mask_state)

        # Accumulate output for this tile: sum over dstate
        out_acc += tl.sum(new_state_tile * C_chunk[None, :], axis=1)

    # Apply D and z
    if HAS_D:
        out_acc += x * D
    if HAS_Z:
        out_acc *= z * tl.sigmoid(z)

    # Store output
    tl.store(out_ptr + offs_m * stride_out_dim, out_acc, mask=mask_m)


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    out=None,
):
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]

    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    assert out.shape == x.shape

    # Dynamic tile size selection: prefer large BLOCK_SIZE_M and moderate BLOCK_SIZE_DSTATE
    # to maximize parallelism and reduce launch overhead.
    # Ascend NPU requires block sizes to be multiples of 16.
    if dstate <= 32:
        # Small dstate: load whole row at once, no tiling over dstate
        BLOCK_SIZE_DSTATE = triton.next_power_of_2(dstate)
        BLOCK_SIZE_M = 64
    else:
        # Larger dstate: tile over dstate with fixed chunk size 32
        BLOCK_SIZE_DSTATE = 32
        BLOCK_SIZE_M = 64

    num_warps = 4
    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and dt_bias.stride(-1) == 0
    )

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)

    _selective_scan_update_kernel[grid](
        state,
        x,
        dt,
        dt_bias,
        A,
        B,
        C,
        D,
        z,
        out,
        state_batch_indices,
        pad_slot_id,
        batch,
        nheads,
        dim,
        dstate,
        nheads // ngroups,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else (0, 0),
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
        z_strides[0],
        z_strides[1],
        z_strides[2],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dt_softplus,
        tie_hdim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_DSTATE,
        num_warps=num_warps,
    )