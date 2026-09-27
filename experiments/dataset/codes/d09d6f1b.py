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
        d_start = d_block * BLOCK_D

        # Preload topk_ids, topk_weight, input_index for all tokens in BLOCK_B
        # Shape: (BLOCK_B, topk_num)
        topk_ids_block = tl.zeros([BLOCK_B, topk_num], dtype=tl.int64)
        topk_weight_block = tl.zeros([BLOCK_B, topk_num], dtype=tl.float32)
        input_index_block = tl.zeros([BLOCK_B, topk_num], dtype=tl.int64)

        for k in range(topk_num):
            k_offset = k
            topk_ids_val = tl.load(
                recv_topk_ids + b_idx * recv_topk_ids_stride0 + k_offset,
                mask=b_mask,
                other=-1
            )
            if HAS_EXPERT_MAP:
                topk_ids_val = apply_expert_map(topk_ids_val, expert_map)
            topk_ids_block = tl.where(
                tl.arange(0, BLOCK_B)[:, None] < BLOCK_B,
                tl.where(tl.arange(0, topk_num)[None, :] == k, topk_ids_val, topk_ids_block),
                topk_ids_block
            )
            topk_weight_val = tl.load(
                recv_topk_weight + b_idx * recv_topk_weight_stride0 + k_offset,
                mask=b_mask,
                other=0.0
            )
            topk_weight_block = tl.where(
                tl.arange(0, BLOCK_B)[:, None] < BLOCK_B,
                tl.where(tl.arange(0, topk_num)[None, :] == k, topk_weight_val, topk_weight_block),
                topk_weight_block
            )
            input_index_val = tl.load(
                input_index + b_idx * input_index_stride0 + k_offset,
                mask=b_mask,
                other=0
            )
            input_index_block = tl.where(
                tl.arange(0, BLOCK_B)[:, None] < BLOCK_B,
                tl.where(tl.arange(0, topk_num)[None, :] == k, input_index_val, input_index_block),
                input_index_block
            )

        # Flatten to (BLOCK_B * topk_num) for vectorized gather
        flat_size = BLOCK_B * topk_num
        flat_topk_ids = tl.view(topk_ids_block, [flat_size])
        flat_topk_weight = tl.view(topk_weight_block, [flat_size])
        flat_input_index = tl.view(input_index_block, [flat_size])

        # Vectorized gather: load input_tensor for all (token, topk) pairs
        # source_token_index = flat_input_index
        # input_tensor address: input_tensor + source_token_index * stride0 + d_start + off_d
        # We need to load BLOCK_D elements for each of flat_size entries
        # Use a loop over flat_size to avoid 3D view
        accumulator = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.float32)
        for i in range(flat_size):
            src_token = flat_input_index[i]
            weight = flat_topk_weight[i]
            # Compute which b_token this belongs to
            b_token_idx = i // topk_num
            # Load input_tensor[src_token, d_start:d_start+BLOCK_D]
            tmp = tl.load(
                input_tensor + src_token * input_tensor_stride0 + d_start + off_d,
                mask=off_d < BLOCK_D,
                other=0.0
            )
            # Accumulate into the correct b_token row
            row_mask = tl.arange(0, BLOCK_B) == b_token_idx
            accumulator = tl.where(
                row_mask[:, None],
                accumulator + tmp[None, :].to(tl.float32) * weight,
                accumulator
            )

        # Store result
        for b_offset in range(BLOCK_B):
            b_token = b_start + b_offset
            if b_token < total_token_num:
                tl.store(
                    output_tensor + b_token * output_tensor_stride0 + d_start + off_d,
                    accumulator[b_offset, :].to(output_tensor.dtype.element_ty),
                    mask=off_d < BLOCK_D
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
    topk_num = recv_topk_ids.shape[1]

    # Guard: topk_num must be compile-time constant and ≤ 16 for packing
    assert topk_num <= 16, f"topk_num {topk_num} > 16, packing disabled"
    assert isinstance(topk_num, int) and topk_num > 0, "topk_num must be positive integer"

    BLOCK_D = min(hidden_size, 1024)
    BLOCK_D = triton.next_power_of_2(BLOCK_D)
    BLOCK_B = 4  # Process 4 tokens per program
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
        topk_num=topk_num,
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
    )
    return output_tensor