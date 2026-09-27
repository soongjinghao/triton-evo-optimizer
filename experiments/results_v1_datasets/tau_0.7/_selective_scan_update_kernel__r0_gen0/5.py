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
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
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
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

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
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < dim

    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    x = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    if not TIE_HDIM:
        dt = tl.load(dt_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)

    if HAS_D:
        D = tl.load(D_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    if TIE_HDIM:
        A_scalar = tl.load(A_ptr).to(tl.float32)
        dA_scalar = tl.exp(A_scalar * dt)

    if BLOCK_SIZE_DSTATE <= 128:
        offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
        state_ptrs = state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )
        state_mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
        if HAS_STATE_BATCH_INDICES:
            state_mask &= state_batch_idx != pad_slot_id

        state = tl.load(state_ptrs, mask=state_mask, other=0.0)

        if not TIE_HDIM:
            A_ptrs = A_ptr + (
                offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
            )
            A = tl.load(
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dA = tl.exp(A * dt[:, None])
        else:
            dA = dA_scalar

        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        state = state * dA + dB * x[:, None]

        tl.store(state_ptrs, state, mask=state_mask)
        out = tl.sum(state * C[None, :], axis=1)
    else:
        out = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        offs_chunk = tl.arange(0, 64)

        for chunk in range(0, BLOCK_SIZE_DSTATE, 64):
            offs_n = chunk + offs_chunk
            state_ptrs = state_ptr + (
                offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
            )
            state_mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
            if HAS_STATE_BATCH_INDICES:
                state_mask &= state_batch_idx != pad_slot_id

            state_chunk = tl.load(state_ptrs, mask=state_mask, other=0.0)

            if not TIE_HDIM:
                A_ptrs = A_ptr + (
                    offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
                )
                A = tl.load(
                    A_ptrs,
                    mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                    other=0.0,
                ).to(tl.float32)
                dA_chunk = tl.exp(A * dt[:, None])
            else:
                dA_chunk = dA_scalar

            B_ptrs = B_ptr + offs_n * stride_B_dstate
            C_ptrs = C_ptr + offs_n * stride_C_dstate
            B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
            C = tl