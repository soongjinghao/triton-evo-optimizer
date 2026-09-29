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
):
    pid = tl.program_id(0)
    row0_start = pid * 2 * hidden_dim
    row1_start = (pid * 2 + 1) * hidden_dim
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    combine_k_offsets = tl.arange(0, combine_k)
    
    # Process row 0
    start_index_mlp_0 = row0_start
    start_index_rmoe_0 = pid * 2 * hidden_dim * combine_k
    
    moe_x_0 = tl.load(
        moe_hidden_states
        + start_index_rmoe_0
        + combine_k_offsets[:, None] * hidden_dim
        + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x_0 = tl.sum(moe_x_0, axis=0)
    mlp_x_0 = tl.load(mlp_hidden_states + start_index_mlp_0 + offsets, mask=mask, other=0.0)
    combined_x_0 = (moe_x_0 + mlp_x_0) / 1.4142135623730951
    tl.store(out_hidden_states + start_index_mlp_0 + offsets, combined_x_0, mask=mask)
    
    # Process row 1
    start_index_mlp_1 = row1_start
    start_index_rmoe_1 = (pid * 2 + 1) * hidden_dim * combine_k
    
    moe_x_1 = tl.load(
        moe_hidden_states
        + start_index_rmoe_1
        + combine_k_offsets[:, None] * hidden_dim
        + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x_1 = tl.sum(moe_x_1, axis=0)
    mlp_x_1 = tl.load(mlp_hidden_states + start_index_mlp_1 + offsets, mask=mask, other=0.0)
    combined_x_1 = (moe_x_1 + mlp_x_1) / 1.4142135623730951
    tl.store(out_hidden_states + start_index_mlp_1 + offsets, combined_x_1, mask=mask)

def experts_combine_triton(moe_hidden_states, mlp_hidden_states, output_buffer=None):
    assert moe_hidden_states.is_contiguous()
    assert mlp_hidden_states.is_contiguous()
    if len(moe_hidden_states.shape) == 2:
        combine_k = 1  # pre-combined
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
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim * 2),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
        ),
    }
    experts_combine_kernel[(bs // 2 + (1 if bs % 2 != 0 else 0),)](
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        **config,
    )
    return out_hidden_states