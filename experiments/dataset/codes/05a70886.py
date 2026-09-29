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
    BLOCK_M: tl.constexpr,
    IS_POWER_OF_TWO: tl.constexpr,
):
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    offsets = tl.arange(0, BLOCK_SIZE)
    combine_k_offsets = tl.arange(0, combine_k)
    if IS_POWER_OF_TWO:
        for i in range(BLOCK_M):
            row_idx = row_base + i
            start_index_mlp = row_idx * hidden_dim
            start_index_rmoe = row_idx * hidden_dim * combine_k
            moe_x = tl.load(
                moe_hidden_states
                + start_index_rmoe
                + combine_k_offsets[:, None] * hidden_dim
                + offsets[None, :],
            )
            moe_x = tl.sum(moe_x, axis=0)
            mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets)
            combined_x = (moe_x + mlp_x) / 1.4142135623730951
            tl.store(out_hidden_states + start_index_mlp + offsets, combined_x)
    else:
        mask = offsets < hidden_dim
        for i in range(BLOCK_M):
            row_idx = row_base + i
            start_index_mlp = row_idx * hidden_dim
            start_index_rmoe = row_idx * hidden_dim * combine_k
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
            combined_x = (moe_x + mlp_x) / 1.4142135623730951
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
    BLOCK_M = 4
    grid_bs = (bs + BLOCK_M - 1) // BLOCK_M
    is_power_of_two = (hidden_dim & (hidden_dim - 1)) == 0
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "BLOCK_M": BLOCK_M,
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
        ),
    }
    experts_combine_kernel[(grid_bs,)](
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        IS_POWER_OF_TWO=is_power_of_two,
        **config,
    )
    return out_hidden_states