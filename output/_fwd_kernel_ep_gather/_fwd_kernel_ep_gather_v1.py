import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def apply_expert_map(expert_id, expert_map):
    """Apply expert mapping if expert_map is provided"""
    return tl.load(expert_map + expert_id)

@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
    hidden_size: tl.constexpr,
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_idx = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_idx < total_token_num

    for d_block in tl.static_range(NUM_D_BLOCKS):
        off_d = tl.arange(0, BLOCK_D)
        d_mask = (d_block * BLOCK_D + off_d) < hidden_size

        for b_offset in tl.static_range(BLOCK_B):
            b_token = b_start + b_offset
            if b_token < total_token_num:
                accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)

                for topk_index in tl.static_range(topk_num):
                    expert_id = tl.load(
                        recv_topk_ids + b_token * recv_topk_ids_stride0 + topk_index
                    )
                    if HAS_EXPERT_MAP:
                        expert_id = apply_expert_map(expert_id, expert_map)

                    if expert_id >= 0:
                        source_token_index = tl.load(
                            input_index + b_token * input_index_stride0 + topk_index
                        )
                        acc_weight = tl.load(
                            recv_topk_weight + b_token * recv_topk_weight_stride0 + topk_index
                        )
                        tmp = tl.load(
                            input_tensor
                            + source_token_index * input_tensor_stride0
                            + d_block * BLOCK_D
                            + off_d,
                            mask=d_mask,
                            other=0.0
                        )
                        accumulator += tmp.to(tl.float32) * acc_weight

                tl.store(
                    output_tensor
                    + b_token * output_tensor_stride0
                    + d_block * BLOCK_D
                    + off_d,
                    accumulator.to(output_tensor.dtype.element_ty),
                    mask=d_mask
                )

@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: torch.Tensor | None,
    output_tensor: torch.Tensor,
):
    assert input_tensor.device.type == 'npu', "input_tensor must be on NPU"
    assert recv_topk_ids.device.type == 'npu', "recv_topk_ids must be on NPU"
    assert recv_topk_weight.device.type == 'npu', "recv_topk_weight must be on NPU"
    assert input_index.device.type == 'npu', "input_index must be on NPU"
    assert output_tensor.device.type == 'npu', "output_tensor must be on NPU"
    if expert_map is not None:
        assert expert_map.device.type == 'npu', "expert_map must be on NPU"

    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]

    # BLOCK_D: at least 16 for Ascend, power of 2, capped at 1024
    BLOCK_D = min(hidden_size, 1024)
    BLOCK_D = max(triton.next_power_of_2(BLOCK_D), 16)

    # Grid size limitation ~40 programs
    BLOCK_B = 4
    grid = (triton.cdiv(num_tokens, BLOCK_B),)
    if grid[0] > 40:
        BLOCK_B = triton.cdiv(num_tokens, 40)
        grid = (40,)

    # Ensure we have at least one program
    if grid[0] == 0:
        grid = (1,)
        BLOCK_B = num_tokens

    NUM_D_BLOCKS = triton.cdiv(hidden_size, BLOCK_D)

    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
        hidden_size=hidden_size,
    )
    return output_tensor