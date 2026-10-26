from typing import Tuple
import torch
import triton
import triton.language as tl
@triton.jit
def experts_combine_kernel(
    out_hidden_states,
    moe_hidden_states,
    mlp_hidden_states,
    combine_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_CHUNKED_REDUCTION: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index_mlp = pid * hidden_dim
    start_index_rmoe = pid * hidden_dim * combine_k
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    combine_k_offsets = tl.arange(0, combine_k)
    if USE_CHUNKED_REDUCTION:
        chunk_size = 1024
        num_chunks = tl.cdiv(hidden_dim, chunk_size)
        acc = tl.zeros([chunk_size], dtype=tl.float32)
        for chunk_idx in range(0, num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_offsets = tl.arange(0, chunk_size)
            chunk_mask = chunk_offsets < tl.minimum(chunk_size, hidden_dim - chunk_start)
            moe_x_chunk = tl.load(
                moe_hidden_states
                + start_index_rmoe
                + combine_k_offsets[:, None] * hidden_dim
                + (chunk_start + chunk_offsets[None, :]),
                mask=chunk_mask[None, :],
                other=0.0,
            )
            moe_x_chunk = tl.sum(moe_x_chunk, axis=0)
            acc = acc + moe_x_chunk
        moe_x = acc[:hidden_dim]
    else:
        moe_x = tl.load(
            moe_hidden_states
            + start_index_rmoe
            + combine_k_offsets[:, None] * hidden_dim
            + offsets[None, :],
            mask=mask[None, :],
            other=0.0,
        )
        moe_x = tl.sum(moe_x, axis=0)
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)
    combined_x = (moe_x + mlp_x) * 0.7071067811865475
    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)
def experts_combine_triton(moe_hidden_states, mlp_hidden_states, output_buffer=None):
    assert moe_hidden_states.is_contiguous()
    assert mlp_hidden_states.is_contiguous()
    if len(moe_hidden_states.shape) == 2:
        combine_k = 1
    else:
        combine_k = moe_hidden_states.shape[1]
    if output_buffer is None:
        out_hidden_states = torch.empty_like(mlp_hidden_states)
    else:
        flat_output_buffer = output_buffer.view(mlp_hidden_states.dtype).reshape(-1)
        assert flat_output_buffer.numel() >= mlp_hidden_states.numel()
        out_hidden_states = flat_output_buffer[: mlp_hidden_states.numel()].reshape(
            mlp_hidden_states.shape
        )
    bs, hidden_dim = mlp_hidden_states.shape
    use_chunked = (hidden_dim > 2048) and (hidden_dim <= 8192) and (combine_k <= 4)
    if use_chunked:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    num_warps = max(
        min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
    )
    experts_combine_kernel[(bs,)](
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        BLOCK_SIZE,
        use_chunked,
        num_warps=num_warps,
    )
    return out_hidden_states