import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    precomputed_indices,
    precomputed_indices_stride0,
    recv_topk_weight_precomputed,
    recv_topk_weight_precomputed_stride0,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
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
        for b_offset in range(BLOCK_B):
            b_token = b_start + b_offset
            if b_token < total_token_num:
                accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
                base_offset = b_token * precomputed_indices_stride0
                weight_base_offset = b_token * recv_topk_weight_precomputed_stride0
                for topk_index in range(0, topk_num):
                    source_token_index = tl.load(precomputed_indices + base_offset + topk_index)
                    acc_weight = tl.load(recv_topk_weight_precomputed + weight_base_offset + topk_index)
                    tmp = tl.load(
                        input_tensor
                        + source_token_index * input_tensor_stride0
                        + d_block * BLOCK_D
                        + off_d
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

    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    topk_num = recv_topk_ids.shape[1]

    # Precompute source_token_index and expert_id into a single contiguous tensor
    # Also pre-arrange recv_topk_weight in the same layout
    if expert_map is not None:
        mapped_expert_ids = expert_map[recv_topk_ids.long()]
    else:
        mapped_expert_ids = recv_topk_ids.long()
    valid_mask = mapped_expert_ids >= 0
    precomputed_indices = torch.where(valid_mask, input_index, torch.zeros_like(input_index))
    recv_topk_weight_precomputed = torch.where(valid_mask, recv_topk_weight, torch.zeros_like(recv_topk_weight))

    num_warps = 2
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
        precomputed_indices,
        precomputed_indices.stride(0),
        recv_topk_weight_precomputed,
        recv_topk_weight_precomputed.stride(0),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=topk_num,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
    )
    return output_tensor