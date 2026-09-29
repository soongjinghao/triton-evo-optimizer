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

    for d_block in range(NUM_D_BLOCKS):
        off_d = tl.arange(0, BLOCK_D)

        # Vectorized load recv_topk_ids for all tokens in block
        # recv_topk_ids shape: (total_token_num, topk_num)
        # For each token, we need topk_num expert ids
        # We load them as a 2D block: (BLOCK_B, topk_num)
        topk_arange = tl.arange(0, topk_num)
        # Compute linear indices: b_idx[:, None] * recv_topk_ids_stride0 + topk_arange[None, :]
        recv_topk_ids_offsets = b_idx[:, None] * recv_topk_ids_stride0 + topk_arange[None, :]
        recv_topk_ids_vals = tl.load(recv_topk_ids + recv_topk_ids_offsets, mask=b_mask[:, None])

        if HAS_EXPERT_MAP:
            recv_topk_ids_vals = apply_expert_map(recv_topk_ids_vals, expert_map)

        # Vectorized load input_index: (total_token_num, topk_num)
        input_index_offsets = b_idx[:, None] * input_index_stride0 + topk_arange[None, :]
        input_index_vals = tl.load(input_index + input_index_offsets, mask=b_mask[:, None])

        # Vectorized load recv_topk_weight: (total_token_num, topk_num)
        recv_topk_weight_offsets = b_idx[:, None] * recv_topk_weight_stride0 + topk_arange[None, :]
        recv_topk_weight_vals = tl.load(recv_topk_weight + recv_topk_weight_offsets, mask=b_mask[:, None])

        # Initialize accumulator: (BLOCK_B, BLOCK_D)
        accumulator = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.float32)

        # Loop over topk_index
        for topk_index in range(0, topk_num):
            # Extract per-token expert_id for this topk_index: (BLOCK_B,)
            expert_id = tl.load(recv_topk_ids + b_idx * recv_topk_ids_stride0 + topk_index, mask=b_mask)
            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            # Extract source_token_index: (BLOCK_B,)
            source_token_index = tl.load(input_index + b_idx * input_index_stride0 + topk_index, mask=b_mask)

            # Extract weight: (BLOCK_B,)
            acc_weight = tl.load(recv_topk_weight + b_idx * recv_topk_weight_stride0 + topk_index, mask=b_mask)

            # Compute valid expert mask: expert_id >= 0
            valid_expert = expert_id >= 0

            # Load input_tensor for valid tokens
            # source_token_index: (BLOCK_B,), off_d: (BLOCK_D,)
            # input_tensor shape: (total_token_num, hidden_size)
            # We need to load for each valid token: input_tensor[source_token_index, d_block*BLOCK_D + off_d]
            # Use tl.where to handle invalid tokens
            safe_source_idx = tl.where(valid_expert, source_token_index, 0)
            input_offsets = safe_source_idx[:, None] * input_tensor_stride0 + d_block * BLOCK_D + off_d[None, :]
            input_mask = valid_expert[:, None] & (b_mask[:, None])
            tmp = tl.load(input_tensor + input_offsets, mask=input_mask, other=0.0)

            # Accumulate: tmp * acc_weight
            accumulator += tmp.to(tl.float32) * acc_weight[:, None]

        # Store output: (BLOCK_B, BLOCK_D)
        output_offsets = b_idx[:, None] * output_tensor_stride0 + d_block * BLOCK_D + off_d[None, :]
        tl.store(output_tensor + output_offsets, accumulator.to(output_tensor.dtype.element_ty), mask=b_mask[:, None])

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
    BLOCK_B = 8  # Increased from 4 to 8 for consolidation
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