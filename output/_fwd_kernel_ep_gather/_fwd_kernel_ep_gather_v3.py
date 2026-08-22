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
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    off_d = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    for d_block in range(NUM_D_BLOCKS):
        d_offset = d_block * BLOCK_D
        in_base_d = input_tensor + d_offset
        out_base_d = output_tensor + d_offset
        for b_offset in range(BLOCK_B):
            b_token = b_start + b_offset
            if b_token < total_token_num:
                recv_base = recv_topk_ids + b_token * recv_topk_ids_stride0
                weight_base = recv_topk_weight + b_token * recv_topk_weight_stride0
                index_base = input_index + b_token * input_index_stride0
                out_base_b = out_base_d + b_token * output_tensor_stride0
                accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
                for topk_index in range(0, topk_num):
                    expert_id = tl.load(recv_base + topk_index)
                    if HAS_EXPERT_MAP:
                        expert_id = apply_expert_map(expert_id, expert_map)
                    if expert_id >= 0:
                        source_token_index = tl.load(index_base + topk_index)
                        acc_weight = tl.load(weight_base + topk_index)
                        in_ptr = in_base_d + source_token_index * input_tensor_stride0
                        in_ptr = tl.multiple_of(in_ptr, 16)
                        tmp = tl.load(in_ptr + off_d)
                        accumulator += tmp.to(tl.float32) * acc_weight
                out_ptr = tl.multiple_of(out_base_b, 16)
                tl.store(out_ptr + off_d, accumulator.to(output_tensor.dtype.element_ty))

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
    assert input_tensor.stride(1) == 1, "input_tensor must be contiguous in the last dimension"
    assert output_tensor.stride(1) == 1, "output_tensor must be contiguous in the last dimension"
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
        topk_num=recv_topk_ids.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
        num_warps=num_warps,
    )
    return output_tensor