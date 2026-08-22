import math
import torch
import triton
import triton.language as tl
@triton.jit
def softplus(dt):
    dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
    return dt
@triton.jit
def _chunk_state_fwd_kernel(
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr = tl.multiple_of(x_ptr, 16) + (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    b_ptr = tl.multiple_of(b_ptr, 16) + (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    dt_ptr = tl.multiple_of(dt_ptr, 16) + (
        pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    )
    dA_cumsum_ptr = tl.multiple_of(dA_cumsum_ptr, 16) + (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )
    if HAS_SEQ_IDX:
        seq_idx_ptr = tl.multiple_of(seq_idx_ptr, 16) + (
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        )
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(
            seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen
        )
    x_off = x_ptr
    b_off = b_ptr
    dt_off = dt_ptr
    dA_cs_off = dA_cumsum_ptr
    if HAS_SEQ_IDX:
        seq_idx_off = seq_idx_ptr
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(chunk_size_limit):
        x_off = tl.multiple_of(x_off, 16)
        x_col = tl.load(
            x_off + offs_m,
            mask=offs_m < hdim,
            other=0.0,
        )
        b_off = tl.multiple_of(b_off, 16)
        b_row = tl.load(
            b_off + offs_n,
            mask=offs_n < dstate,
            other=0.0,
        ).to(tl.float32)
        dA_cs_off = tl.multiple_of(dA_cs_off, 16)
        dA_cs_k = tl.load(dA_cs_off).to(tl.float32)
        dt_off = tl.multiple_of(dt_off, 16)
        dt_k = tl.load(dt_off).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        else:
            seq_idx_off = tl.multiple_of(seq_idx_off, 16)
            seq_idx_k = tl.load(seq_idx_off)
            scale = tl.where(
                seq_idx_k == seq_idx_last,
                tl.exp(dA_cs_last - dA_cs_k) * dt_k,
                0,
            )
        b_row *= scale
        acc += x_col[:, None] * b_row[None, :]
        x_off += stride_x_seqlen
        b_off += stride_b_seqlen
        dt_off += stride_dt_csize
        dA_cs_off += stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_off += stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += (
        pid_b * stride_states_batch
        + pid_c * stride_states_chunk
        + pid_h * stride_states_head
    )
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)
def _chunk_state_fwd(
    B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (batch, nchunks, nheads, headdim, dstate),
            device=B.device,
            dtype=states_dtype,
        )
    assert x.stride(3) == 1 and B.stride(3) == 1, (
        "x and B must have contiguous innermost dimension (hdim/dstate), "
        f"got strides: x.stride(3)={x.stride(3)}, B.stride(3)={B.stride(3)}"
    )
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    _chunk_state_fwd_kernel[grid](
        x,
        B,
        states,
        dt,
        dA_cumsum,
        seq_idx,
        headdim,
        dstate,
        chunk_size,
        batch,
        seqlen,
        nheads // ngroups,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        dt.stride(0),
        dt.stride(2),
        dt.stride(1),
        dt.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        *(seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0),
        HAS_SEQ_IDX=seq_idx is not None,
    )
    return states