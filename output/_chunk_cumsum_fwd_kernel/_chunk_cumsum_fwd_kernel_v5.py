import math
import torch
import triton
import triton.language as tl

@triton.jit
def softplus(dt):
    # Always use stable softplus (no log1p to avoid unsupported op)
    dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1.0), dt)
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

    # Base pointers for this chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    # Indices as int32 for efficiency
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H).to(tl.int32)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK).to(tl.int32)

    # Effective chunk size (may be smaller at boundary)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # Load dt with masks and bias/softplus/clamp
    dt_ptrs = dt_ptr + offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0)

    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0)
        dt += dt_bias[:, None]

    if DT_SOFTPLUS:
        dt = softplus(dt)

    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)

    # Store dt_out (full chunk size, padded with zeros)
    tl.store(dt_out_ptr + offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize,
             dt,
             mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))

    # Load A and compute dA = dt * A broadcast
    A = tl.load(A_ptr + offs_h * stride_A_head, mask=offs_h < nheads, other=0.0)
    dA = dt * A[:, None]

    # Cumulative sum over chunk dimension
    dA_cs = tl.cumsum(dA, axis=1)

    # Store cumulative sum (full chunk size)
    tl.store(dA_cumsum_ptr + offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize,
             dA_cs,
             mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)

    nchunks = math.ceil(seqlen / chunk_size)

    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)

    # Dynamic block sizes: must be multiples of 16, cap for occupancy/resource
    BLOCK_SIZE_H = max(16, triton.next_power_of_2(nheads))
    BLOCK_SIZE_H = min(32, BLOCK_SIZE_H)  # cap for occupancy
    BLOCK_SIZE_CHUNK = max(16, triton.next_power_of_2(chunk_size))
    BLOCK_SIZE_CHUNK = min(1024, BLOCK_SIZE_CHUNK)  # cap to avoid resource exhaustion

    # Choose num_warps based on block product to balance occupancy
    num_warps = 4
    if BLOCK_SIZE_H * BLOCK_SIZE_CHUNK > 1024:
        num_warps = 8

    grid = lambda META: (
        batch,
        nchunks,
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
    )

    _chunk_cumsum_fwd_kernel[grid](
        dt, A, dt_bias, dt_out, dA_cumsum,
        batch, seqlen, nheads, chunk_size,
        dt_limit[0], dt_limit[1],
        dt.stride(0), dt.stride(1), dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        BLOCK_SIZE_CHUNK=BLOCK_SIZE_CHUNK,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=num_warps,
    )
    return dA_cumsum, dt_out