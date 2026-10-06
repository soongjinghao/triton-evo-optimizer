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
    SORTED: tl.constexpr,
    sorted_input_tensor,
    sorted_input_tensor_stride0,
    sorted_input_tensor_stride1,
    sorted_source_token_start,
    sorted_source_token_start_stride0,
    sorted_source_token_start_stride1,
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_idx = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_idx < total_token_num

    for d_block in range(NUM_D_BLOCKS):
        off_d = tl.arange(0, BLOCK_D)
        for b_offset in range(BLOCK_B):
            b_token = b_start + b_offset
            if b_token < total_token_num:
                accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
                for topk_index in range(0, topk_num):
                    expert_id = tl.load(
                        recv_topk_ids + b_token * recv_topk_ids_stride0 + topk_index
                    )
                    if HAS_EXPERT_MAP:
                        expert_id = apply_expert_map(expert_id, expert_map)
                    if expert_id >= 0:
                        if SORTED:
                            source_token_start = tl.load(
                                sorted_source_token_start + b_token * sorted_source_token_start_stride0 + topk_index
                            )
                            tmp = tl.load(
                                sorted_input_tensor
                                + source_token_start * sorted_input_tensor_stride0
                                + d_block * BLOCK_D
                                + off_d,
                                mask=None,
                            )
                        else:
                            source_token_index = tl.load(
                                input_index + b_token * input_index_stride0 + topk_index
                            )
                            tmp = tl.load(
                                input_tensor
                                + source_token_index * input_tensor_stride0
                                + d_block * BLOCK_D
                                + off_d
                            )
                        acc_weight = tl.load(
                            recv_topk_weight + b_token * recv_topk_weight_stride0 + topk_index
                        )
                        accumulator += tmp.to(tl.float32) * acc_weight
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

    # Attempt to sort input_index for contiguous reads
    SORTED = False
    sorted_input_tensor = input_tensor
    sorted_input_tensor_stride0 = input_tensor.stride(0)
    sorted_input_tensor_stride1 = input_tensor.stride(1)
    sorted_source_token_start = input_index
    sorted_source_token_start_stride0 = input_index.stride(0)
    sorted_source_token_start_stride1 = input_index.stride(1)

    try:
        # Check if input_tensor is contiguous in the hidden dimension
        if input_tensor.stride(1) == 1:
            # Flatten input_index and sort per token
            num_tokens_local = input_index.shape[0]
            topk_num_local = input_index.shape[1]
            flat_indices = input_index.view(-1)
            # Sort indices globally to get contiguous blocks
            sorted_flat_indices, sort_order = torch.sort(flat_indices)
            # Build sorted_input_tensor by gathering rows according to sorted indices
            sorted_input_tensor = input_tensor[sorted_flat_indices, :].contiguous()
            sorted_input_tensor_stride0 = sorted_input_tensor.stride(0)
            sorted_input_tensor_stride1 = sorted_input_tensor.stride(1)
            # Build sorted_source_token_start: for each token, map original topk positions to sorted positions
            # We need to know for each token, the start index in the sorted array for its topk entries
            # Since we sorted globally, we can compute per-token offsets
            # For simplicity, we create a mapping from original index to sorted position
            # This is O(N*K) but done on CPU/host side; for NPU we do it on device
            # We'll use a scatter approach: create an empty tensor and fill
            sorted_source_token_start = torch.zeros_like(input_index)
            # For each token, find its topk indices in the sorted array
            # We can do this by scanning sorted_flat_indices
            # Since this is a pre-processing step, we do it on CPU for simplicity
            # But to keep it on NPU, we use a simple loop (small K)
            for k in range(topk_num_local):
                token_indices = input_index[:, k]  # [num_tokens]
                # Find positions in sorted_flat_indices
                # Use searchsorted
                positions = torch.searchsorted(sorted_flat_indices, token_indices)
                sorted_source_token_start[:, k] = positions
            sorted_source_token_start_stride0 = sorted_source_token_start.stride(0)
            sorted_source_token_start_stride1 = sorted_source_token_start.stride(1)
            SORTED = True
    except Exception:
        # Fallback: keep original path
        pass

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
        SORTED=SORTED,
        sorted_input_tensor=sorted_input_tensor,
        sorted_input_tensor_stride0=sorted_input_tensor_stride0,
        sorted_input_tensor_stride1=sorted_input_tensor_stride1,
        sorted_source_token_start=sorted_source_token_start,
        sorted_source_token_start_stride0=sorted_source_token_start_stride0,
        sorted_source_token_start_stride1=sorted_source_token_start_stride1,
    )
    return output_tensor