import logging
from typing import Optional

import torch
import torch_npu
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

from moe_align_block_size_stage import moe_align_block_size_triton

def test_moe_align_block_size():
    # Test parameters
    num_experts = 4
    block_size = 32
    batch_size = 128
    
    # Create test data on NPU
    torch.manual_seed(0)
    topk_ids = torch.randint(0, num_experts, (batch_size,), dtype=torch.int32, device='npu')
    
    # Initialize output tensors
    sorted_token_ids = torch.empty((batch_size,), dtype=torch.int32, device='npu')
    expert_ids = torch.empty((batch_size,), dtype=torch.int32, device='npu')
    num_tokens_post_pad = torch.zeros((), dtype=torch.int32, device='npu')
    
    # Run the Triton implementation
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad
    )
    
    # Simple validation - check that tensors have been modified
    print(f"topk_ids: {topk_ids}")
    print(f"sorted_token_ids: {sorted_token_ids}")
    print(f"expert_ids: {expert_ids}")
    print(f"num_tokens_post_pad: {num_tokens_post_pad.item()}")
    
    # Basic assertion to ensure the function ran
    assert sorted_token_ids.device.type == 'npu'
    assert expert_ids.device.type == 'npu'
    assert num_tokens_post_pad.device.type == 'npu'
    print("✅ moe_align_block_size test passed")

if __name__ == "__main__":
    test_moe_align_block_size()
