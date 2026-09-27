import math
import torch
import triton
import triton.language as tl
from packaging import version
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
@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    batch,
    seqlen,
    nheads,
    chunk_size,
    dt_min,
    dt_max,
    stride_dt_batch,
    stride_dt_seqlen,
    stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch,
    stride_dt_out_chunk,
    stride_dt_out_head,
    stride_dt_out_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 16,
    NHEADS_DIVISIBLE: tl.constexpr = False,
    CHUNK_SIZE_POW2: tl.constexpr = False,
    SEQLEN_MULTIPLE_CHUNK: tl.constexpr = False,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if NHEADS_DIVISIBLE and CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        dt = tl.load(dt_ptrs, mask=offs_c[None, :] < chunk_size_limit, other=0.0).to(tl.float32)
    elif NHEADS_DIVISIBLE:
        dt = tl.load(
            dt_ptrs,
            mask=offs_c[None, :] < chunk_size_limit,
            other=0.0,
        ).to(tl.float32)
    elif CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        dt = tl.load(
            dt_ptrs,
            mask=offs_h[:, None] < nheads,
            other=0.0,
        ).to(tl.float32)
    else:
        dt = tl.load(
            dt_ptrs,
            mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
            other=0.0,
        ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    if NHEADS_DIVISIBLE and CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        dt = tl.where(offs_c[None, :] < chunk_size_limit, dt, 0.0)
    elif NHEADS_DIVISIBLE:
        dt = tl.where(offs_c[None, :] < chunk_size_limit, dt, 0.0)
    elif CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        dt = tl.where(offs_h[:, None] < nheads, dt, 0.0)
    else:
        dt = tl.where(
            (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
        )
    if NHEADS_DIVISIBLE and CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        tl.store(dt_out_ptrs, dt, mask=offs_c[None, :] < chunk_size)
    elif NHEADS_DIVISIBLE:
        tl.store(dt_out_ptrs, dt, mask=offs_c[None, :] < chunk_size)
    elif CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        tl.store(dt_out_ptrs, dt, mask=offs_h[:, None] < nheads)
    else:
        tl.store(
            dt_out_ptrs,
            dt,
            mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
        )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    if NHEADS_DIVISIBLE and CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        tl.store(dA_cs_ptrs, dA_cs, mask=offs_c[None, :] < chunk_size)
    elif NHEADS_DIVISIBLE:
        tl.store(dA_cs_ptrs, dA_cs, mask=offs_c[None, :] < chunk_size)
    elif CHUNK_SIZE_POW2 and SEQLEN_MULTIPLE_CHUNK:
        tl.store(dA_cs_ptrs, dA_cs, mask=offs_h[:, None] < nheads)
    else:
        tl.store(
            dA_cs_ptrs,
            dA_cs,
            mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
        )
def _chunk_cumsum_fwd(
    dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))
):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    grid_chunk_cs = lambda META: (
        batch,
        nchunks,
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
    )
    nheads_divisible = (nheads % 16 == 0)
    chunk_size_pow2 = (chunk_size & (chunk_size - 1)) == 0
    seqlen_multiple_chunk = (seqlen % chunk_size == 0)
    _chunk_cumsum_fwd_kernel[grid_chunk_cs](
        dt,
        A,
        dt_bias,
        dt_out,
        dA_cumsum,
        batch,
        seqlen,
        nheads,
        chunk_size,
        dt_limit[0],
        dt_limit[1],
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_out.stride(0),
        dt_out.stride(2),
        dt_out.stride(1),
        dt_out.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        NHEADS_DIVISIBLE=nheads_divisible,
        CHUNK_SIZE_POW2=chunk_size_pow2,
        SEQLEN_MULTIPLE_CHUNK=seqlen_multiple_chunk,
    )
    return dA_cumsum, dt_out