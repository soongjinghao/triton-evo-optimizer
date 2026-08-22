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
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1).to(tl.int32)
    pid_h = tl.program_id(axis=2)

    dt_base = dt_ptr + pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_base = dt_out_ptr + pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_base = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H).to(tl.int32)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK).to(tl.int32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    h_mask = offs_h < nheads
    c_mask = offs_c < chunk_size_limit
    full_mask = h_mask[:, None] & c_mask[None, :]

    # dt load with multiple_of hint
    dt = tl.load(
        tl.multiple_of(dt_base, 16) + offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen,
        mask=full_mask,
        other=0.0
    ).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias = tl.load(
            tl.multiple_of(dt_bias_ptr, 16) + offs_h * stride_dt_bias_head,
            mask=h_mask,
            other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]

    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(full_mask, dt, 0.0)

    # store dt_out
    tl.store(
        tl.multiple_of(dt_out_base, 16) + offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize,
        dt,
        mask=h_mask[:, None] & (offs_c[None, :] < chunk_size)
    )

    # A load
    A = tl.load(
        tl.multiple_of(A_ptr, 16) + offs_h * stride_A_head,
        mask=h_mask,
        other=0.0
    ).to(tl.float32)

    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)

    # store dA_cumsum
    tl.store(
        tl.multiple_of(dA_cumsum_base, 16) + offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize,
        dA_cs,
        mask=h_mask[:, None] & (offs_c[None, :] < chunk_size)
    )

@triton.jit
def _chunk_cumsum_fwd_kernel_aligned(
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
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1).to(tl.int32)
    pid_h = tl.program_id(axis=2)

    dt_base = dt_ptr + pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_base = dt_out_ptr + pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_base = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H).to(tl.int32)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK).to(tl.int32)

    dt = tl.load(
        tl.multiple_of(dt_base, 16) + offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    ).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias = tl.load(
            tl.multiple_of(dt_bias_ptr, 16) + offs_h * stride_dt_bias_head
        ).to(tl.float32)
        dt += dt_bias[:, None]

    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)

    tl.store(
        tl.multiple_of(dt_out_base, 16) + offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize,
        dt
    )

    A = tl.load(
        tl.multiple_of(A_ptr, 16) + offs_h * stride_A_head
    ).to(tl.float32)

    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)

    tl.store(
        tl.multiple_of(dA_cumsum_base, 16) + offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize,
        dA_cs
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
    MAX_ELEMS = 8192
    ROWS_PER_BLOCK = 8
    max_h_by_ub = MAX_ELEMS // max(chunk_size, 1)
    BLOCK_SIZE_H = min(nheads, ROWS_PER_BLOCK, max_h_by_ub)
    BLOCK_SIZE_H = max(1, triton.next_power_of_2(BLOCK_SIZE_H))
    grid_h = triton.cdiv(nheads, BLOCK_SIZE_H)
    is_pow2 = (chunk_size > 0) and (chunk_size & (chunk_size - 1) == 0)
    full_chunks = (seqlen % chunk_size == 0)
    heads_aligned = (nheads % BLOCK_SIZE_H == 0)
    if is_pow2 and heads_aligned and full_chunks:
        kernel = _chunk_cumsum_fwd_kernel_aligned
        BLOCK_SIZE_CHUNK = chunk_size
    else:
        kernel = _chunk_cumsum_fwd_kernel
        BLOCK_SIZE_CHUNK = triton.next_power_of_2(chunk_size)
    grid = (batch, nchunks, grid_h)
    stride_dt_batch, stride_dt_seqlen, stride_dt_head = dt.stride()
    stride_A_head = A.stride(0)
    stride_dt_bias_head = dt_bias.stride(0) if dt_bias is not None else 0
    stride_dt_out_batch, stride_dt_out_head, stride_dt_out_chunk, stride_dt_out_csize = dt_out.stride()
    stride_dA_cs_batch, stride_dA_cs_head, stride_dA_cs_chunk, stride_dA_cs_csize = dA_cumsum.stride()
    kernel[grid](
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
        DT_SOFTPLUS=dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        BLOCK_SIZE_CHUNK=BLOCK_SIZE_CHUNK,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=8,
        num_stages=2,
    )
    return dA_cumsum, dt_out