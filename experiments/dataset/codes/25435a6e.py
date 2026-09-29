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
        accumulator = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.float32)

        # Load all topk_ids and input_index for BLOCK_B tokens at once
        recv_topk_ids_2d = tl.load(
            recv_topk_ids + b_idx[:, None] * recv_topk_ids_stride0 + tl.arange(0, topk_num)[None, :] * recv_topk_ids_stride1,
            mask=b_mask[:, None] & (tl.arange(0, topk_num)[None, :] < topk_num),
            other=0
        )
        input_index_2d = tl.load(
            input_index + b_idx[:, None] * input_index_stride0 + tl.arange(0, topk_num)[None, :] * input_index_stride1,
            mask=b_mask[:, None] & (tl.arange(0, topk_num)[None, :] < topk_num),
            other=0
        )
        recv_topk_weight_2d = tl.load(
            recv_topk_weight + b_idx[:, None] * recv_topk_weight_stride0 + tl.arange(0, topk_num)[None, :] * recv_topk_weight_stride1,
            mask=b_mask[:, None] & (tl.arange(0, topk_num)[None, :] < topk_num),
            other=0.0
        )

        if HAS_EXPERT_MAP:
            expert_id_2d = recv_topk_ids_2d
            expert_id_2d = apply_expert_map(expert_id_2d, expert_map)
            valid_mask = expert_id_2d >= 0
        else:
            valid_mask = recv_topk_ids_2d >= 0

        # Vectorized accumulation over topk dimension
        for topk_index in range(topk_num):
            source_token_index = tl.where(valid_mask[:, topk_index], input_index_2d[:, topk_index], 0)
            acc_weight = tl.where(valid_mask[:, topk_index], recv_topk_weight_2d[:, topk_index], 0.0)
            tmp = tl.load(
                input_tensor
                + source_token_index[:, None] * input_tensor_stride0
                + d_block * BLOCK_D
                + off_d[None, :],
                mask=b_mask[:, None],
                other=0.0
            )
            accumulator += tmp.to(tl.float32) * acc_weight[:, None]

        tl.store(
            output_tensor
            + b_idx[:, None] * output_tensor_stride0
            + d_block * BLOCK_D
            + off_d[None, :],
            accumulator.to(output_tensor.dtype.element_ty),
            mask=b_mask[:, None]
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
        topk_num=topk_num,
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
    )
    return output_tensor