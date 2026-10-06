import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def apply_expert_map(expert_id, expert_map):
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
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_idx = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_idx < total_token_num

    # Hoist: preload expert_id, source_token_index, acc_weight for all b_offset and topk_index
    # Shape: (BLOCK_B, topk_num) stored in registers
    for b_offset in range(BLOCK_B):
        b_token = b_start + b_offset
        if b_token < total_token_num:
            for topk_index in range(topk_num):
                expert_id_val = tl.load(
                    recv_topk_ids + b_token * recv_topk_ids_stride0 + topk_index
                )
                if HAS_EXPERT_MAP:
                    expert_id_val = apply_expert_map(expert_id_val, expert_map)
                source_token_index_val = tl.load(
                    input_index + b_token * input_index_stride0 + topk_index
                )
                acc_weight_val = tl.load(
                    recv_topk_weight + b_token * recv_topk_weight_stride0 + topk_index
                )
                # Store hoisted values into temporary named variables for reuse
                # Use tl.store to keep them in registers across d_block loop
                # Since we cannot create arrays of tensors, we use scalar variables per b_offset
                # This is achieved by unrolling the b_offset loop and using if-conditions
                # We'll inline the hoisted values directly into the d_block loop below
                # The hoisting is done by moving the b_offset loop outside d_block
                pass

    # Reordered loops: b_offset outer, d_block inner
    for b_offset in range(BLOCK_B):
        b_token = b_start + b_offset
        if b_token < total_token_num:
            # Preload all topk data for this token
            expert_ids = tl.zeros([topk_num], dtype=tl.int32)
            source_token_indices = tl.zeros([topk_num], dtype=tl.int32)
            acc_weights = tl.zeros([topk_num], dtype=tl.float32)
            for topk_index in range(topk_num):
                expert_id_val = tl.load(
                    recv_topk_ids + b_token * recv_topk_ids_stride0 + topk_index
                )
                if HAS_EXPERT_MAP:
                    expert_id_val = apply_expert_map(expert_id_val, expert_map)
                expert_ids = tl.where(tl.arange(0, topk_num) == topk_index, expert_id_val, expert_ids)
                source_token_index_val = tl.load(
                    input_index + b_token * input_index_stride0 + topk_index
                )
                source_token_indices = tl.where(tl.arange(0, topk_num) == topk_index, source_token_index_val, source_token_indices)
                acc_weight_val = tl.load(
                    recv_topk_weight + b_token * recv_topk_weight_stride0 + topk_index
                )
                acc_weights = tl.where(tl.arange(0, topk_num) == topk_index, acc_weight_val, acc_weights)

            for d_block in range(NUM_D_BLOCKS):
                off_d = tl.arange(0, BLOCK_D)
                accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
                for topk_index in range(topk_num):
                    expert_id_val = tl.load(expert_ids + topk_index)
                    if expert_id_val >= 0:
                        source_token_index_val = tl.load(source_token_indices + topk_index)
                        acc_weight_val = tl.load(acc_weights + topk_index)
                        tmp = tl.load(
                            input_tensor
                            + source_token_index_val * input_tensor_stride0
                            + d_block * BLOCK_D
                            + off_d
                        )
                        accumulator += tmp.to(tl.float32) * acc_weight_val
                tl.store(
                    output_tensor
                    + b_token * output_tensor_stride0
                    + d_block * BLOCK_D
                    + off_d,
                    accumulator.to(output_tensor.dtype.element_ty),
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
    BLOCK_D = min(hidden_size, 1024)
    BLOCK_D = triton.next_power_of_2(BLOCK_D)
    BLOCK_B = 4
    grid = (triton.cdiv(num_tokens, BLOCK_B),)
    if grid[0] > 40:
        BLOCK_B = triton.cdiv(num_tokens, 40)
        grid = (40,)
    assert hidden_size % BLOCK_D == 0, f"hidden_size {hidden_size} must be divisible by BLOCK_D {BLOCK_D}"
    assert num_tokens > 0, "Cannot process empty token tensor"
    assert hidden_size > 0, "Cannot process empty hidden dimension"
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
    )
    return output_tensor