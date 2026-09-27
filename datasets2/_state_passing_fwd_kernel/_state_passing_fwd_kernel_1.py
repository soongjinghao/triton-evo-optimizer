import torch
import triton
import triton.language as tl

@triton.jit
def _state_passing_fwd_kernel(
    states_ptr,
    out_ptr,
    final_states_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    chunk_offsets_ptr,
    DIM: tl.constexpr,
    NCHUNKS: tl.constexpr,
    SEQLEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CHUNK_META_NUM: tl.constexpr,
    STRIDE_STATES_BATCH: tl.constexpr,
    STRIDE_STATES_CHUNK: tl.constexpr,
    STRIDE_STATES_HEAD: tl.constexpr,
    STRIDE_STATES_DIM: tl.constexpr,
    STRIDE_OUT_BATCH: tl.constexpr,
    STRIDE_OUT_CHUNK: tl.constexpr,
    STRIDE_OUT_HEAD: tl.constexpr,
    STRIDE_OUT_DIM: tl.constexpr,
    STRIDE_FINAL_BATCH: tl.constexpr,
    STRIDE_FINAL_HEAD: tl.constexpr,
    STRIDE_FINAL_DIM: tl.constexpr,
    STRIDE_DA_BATCH: tl.constexpr,
    STRIDE_DA_CHUNK: tl.constexpr,
    STRIDE_DA_HEAD: tl.constexpr,
    STRIDE_DA_CSIZE: tl.constexpr,
    STRIDE_INIT_BATCH: tl.constexpr,
    STRIDE_INIT_HEAD: tl.constexpr,
    STRIDE_INIT_DIM: tl.constexpr,
    STRIDE_SEQ_BATCH: tl.constexpr,
    STRIDE_SEQ_SEQLEN: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    IS_CONT_BATCHED: tl.constexpr,
    NEEDS_CONT_SEQ: tl.constexpr,
    NEEDS_SEQ_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FULL_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(offs, BLOCK_SIZE)

    if IS_FULL_TILE:
        mask = None
    else:
        mask = offs < DIM

    # 基地址计算并对齐宣告
    states_base = states_ptr + pid_b * STRIDE_STATES_BATCH + pid_h * STRIDE_STATES_HEAD
    states_base = tl.multiple_of(states_base, 16)

    out_base = out_ptr + pid_b * STRIDE_OUT_BATCH + pid_h * STRIDE_OUT_HEAD
    out_base = tl.multiple_of(out_base, 16)

    final_base = final_states_ptr + pid_b * STRIDE_FINAL_BATCH + pid_h * STRIDE_FINAL_HEAD
    final_base = tl.multiple_of(final_base, 16)

    dA_base = (
        dA_cs_ptr
        + pid_b * STRIDE_DA_BATCH
        + pid_h * STRIDE_DA_HEAD
        + (CHUNK_SIZE - 1) * STRIDE_DA_CSIZE
    )
    dA_base = tl.multiple_of(dA_base, 16)

    if HAS_INITSTATES:
        init_base = initstates_ptr + pid_h * STRIDE_INIT_HEAD
        if not IS_CONT_BATCHED:
            init_base += pid_b * STRIDE_INIT_BATCH
        init_base = tl.multiple_of(init_base, 16)
        if IS_FULL_TILE:
            state = tl.load(init_base + offs * STRIDE_INIT_DIM).to(tl.float32)
        else:
            state = tl.load(init_base + offs * STRIDE_INIT_DIM, mask=mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 存储首个 chunk 的初始状态
    if IS_FULL_TILE:
        tl.store(out_base + offs * STRIDE_OUT_DIM, state)
    else:
        tl.store(out_base + offs * STRIDE_OUT_DIM, state, mask=mask)

    states_ptrs = states_base + offs * STRIDE_STATES_DIM
    states_ptrs = tl.multiple_of(states_ptrs, 16)

    out_ptrs = out_base + STRIDE_OUT_CHUNK + offs * STRIDE_OUT_DIM
    out_ptrs = tl.multiple_of(out_ptrs, 16)

    final_ptrs = final_base + offs * STRIDE_FINAL_DIM
    final_ptrs = tl.multiple_of(final_ptrs, 16)

    dA_ptr = dA_base

    if NEEDS_CONT_SEQ:
        seq_base = seq_idx_ptr + pid_b * STRIDE_SEQ_BATCH
        prev_seq_idx_chunk_end = tl.zeros((), dtype=tl.int32)
        logical_chunk_idx = tl.zeros((), dtype=tl.int32)
    elif NEEDS_SEQ_MASK:
        seq_base = seq_idx_ptr + pid_b * STRIDE_SEQ_BATCH
        prev_seq_idx_chunk_end = tl.zeros((), dtype=tl.int32)

    for c in range(NCHUNKS):
        if IS_FULL_TILE:
            new_state = tl.load(states_ptrs).to(tl.float32)
        else:
            new_state = tl.load(states_ptrs, mask=mask, other=0.0).to(tl.float32)

        dA = tl.load(tl.multiple_of(dA_ptr, 16), cache_modifier=".ca").to(tl.float32)

        scale_mask = tl.full((), True, dtype=tl.int1)

        if NEEDS_CONT_SEQ:
            seq_end_idx = min((c + 1) * CHUNK_SIZE, SEQLEN) - 1
            seq_start_idx = min(c * CHUNK_SIZE, SEQLEN)
            seq_idx_chunk_end = tl.load(
                seq_base + seq_end_idx * STRIDE_SEQ_SEQLEN
            ).to(tl.int32)
            seq_changed = prev_seq_idx_chunk_end != seq_idx_chunk_end
            if seq_changed:
                if IS_FULL_TILE:
                    state = tl.load(
                        initstates_ptr
                        + seq_idx_chunk_end * STRIDE_INIT_BATCH
                        + pid_h * STRIDE_INIT_HEAD
                        + offs * STRIDE_INIT_DIM
                    ).to(tl.float32)
                else:
                    state = tl.load(
                        initstates_ptr
                        + seq_idx_chunk_end * STRIDE_INIT_BATCH
                        + pid_h * STRIDE_INIT_HEAD
                        + offs * STRIDE_INIT_DIM,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                seq_idx_chunk_start = tl.load(
                    seq_base + seq_start_idx * STRIDE_SEQ_SEQLEN
                ).to(tl.int32)
                logical_chunk_idx += seq_idx_chunk_end - seq_idx_chunk_start
                if logical_chunk_idx < CHUNK_META_NUM:
                    c_off = tl.load(chunk_offsets_ptr + logical_chunk_idx)
                    if c_off > 0:
                        boundary_idx = c_off - 1
                        dA_boundary = tl.load(
                            dA_ptr
                            - (CHUNK_SIZE - 1) * STRIDE_DA_CSIZE
                            + boundary_idx * STRIDE_DA_CSIZE,
                            cache_modifier=".ca",
                        ).to(tl.float32)
                        dA -= dA_boundary
            logical_chunk_idx += 1
            prev_seq_idx_chunk_end = seq_idx_chunk_end
        elif NEEDS_SEQ_MASK:
            seq_end_idx = min((c + 1) * CHUNK_SIZE, SEQLEN) - 1
            seq_idx_chunk_end = tl.load(
                seq_base + seq_end_idx * STRIDE_SEQ_SEQLEN
            ).to(tl.int32)
            scale_mask = seq_idx_chunk_end == prev_seq_idx_chunk_end
            prev_seq_idx_chunk_end = seq_idx_chunk_end

        scale = tl.where(scale_mask, tl.exp(dA), 0.0)
        state = scale * state + new_state

        if c < NCHUNKS - 1:
            if IS_FULL_TILE:
                tl.store(out_ptrs, state)
            else:
                tl.store(out_ptrs, state, mask=mask)
        else:
            if IS_FULL_TILE:
                tl.store(final_ptrs, state)
            else:
                tl.store(final_ptrs, state, mask=mask)

        states_ptrs += STRIDE_STATES_CHUNK
        out_ptrs += STRIDE_OUT_CHUNK
        dA_ptr += STRIDE_DA_CHUNK


def _state_passing_fwd(
    states,
    dA_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
    is_cont_batched=False,
    chunk_offsets=None,
):
    batch, nchunks, nheads, dim = states.shape
    if chunk_size is None:
        chunk_size = dA_cumsum.shape[-1]
    else:
        assert chunk_size == dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)

    if initial_states is not None:
        if is_cont_batched:
            assert seq_idx is not None
            assert chunk_offsets is not None
        else:
            assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    else:
        seqlen = 0

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)

    block_size = min(256, triton.next_power_of_2(max(1, dim)))
    if block_size <= 32:
        num_warps = 1
    elif block_size <= 128:
        num_warps = 2
    else:
        num_warps = 4

    grid = (triton.cdiv(dim, block_size), batch, nheads)
    IS_FULL_TILE = (dim % block_size == 0)

    with torch.npu.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            final_states,
            dA_cumsum,
            initial_states,
            seq_idx,
            chunk_offsets,
            DIM=dim,
            NCHUNKS=nchunks,
            SEQLEN=seqlen,
            CHUNK_SIZE=chunk_size,
            CHUNK_META_NUM=len(chunk_offsets) if chunk_offsets is not None else 0,
            STRIDE_STATES_BATCH=states.stride(0),
            STRIDE_STATES_CHUNK=states.stride(1),
            STRIDE_STATES_HEAD=states.stride(2),
            STRIDE_STATES_DIM=states.stride(3),
            STRIDE_OUT_BATCH=out.stride(0),
            STRIDE_OUT_CHUNK=out.stride(1),
            STRIDE_OUT_HEAD=out.stride(2),
            STRIDE_OUT_DIM=out.stride(3),
            STRIDE_FINAL_BATCH=final_states.stride(0),
            STRIDE_FINAL_HEAD=final_states.stride(1),
            STRIDE_FINAL_DIM=final_states.stride(2),
            STRIDE_DA_BATCH=dA_cumsum.stride(0),
            STRIDE_DA_CHUNK=dA_cumsum.stride(2),
            STRIDE_DA_HEAD=dA_cumsum.stride(1),
            STRIDE_DA_CSIZE=dA_cumsum.stride(3),
            STRIDE_INIT_BATCH=initial_states.stride(0) if initial_states is not None else 0,
            STRIDE_INIT_HEAD=initial_states.stride(1) if initial_states is not None else 0,
            STRIDE_INIT_DIM=initial_states.stride(2) if initial_states is not None else 0,
            STRIDE_SEQ_BATCH=seq_idx.stride(0) if seq_idx is not None else 0,
            STRIDE_SEQ_SEQLEN=seq_idx.stride(1) if seq_idx is not None else 0,
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_CONT_BATCHED=is_cont_batched,
            NEEDS_CONT_SEQ=seq_idx is not None and initial_states is not None and is_cont_batched,
            NEEDS_SEQ_MASK=seq_idx is not None and initial_states is None,
            BLOCK_SIZE=block_size,
            IS_FULL_TILE=IS_FULL_TILE,
            num_stages=1,
            num_warps=num_warps,
        )
    return out, final_states