import torch

from reshape_and_cache_flash_kernel import reshape_and_cache_flash

def test_reshape_and_cache_flash():

    device = torch.device('npu')

    num_tokens = 16
    num_heads = 8
    head_size = 64
    num_blocks = 4
    block_size = 8

    key = torch.randn(num_tokens, num_heads, head_size, device=device)
    value = torch.randn(num_tokens, num_heads, head_size, device=device)

    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device=device)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device=device)

    slot_mapping = torch.tensor([i % (num_blocks * block_size) for i in range(num_tokens)], 
                               device=device, dtype=torch.int32)

    kv_cache_dtype = torch.float32
    k_scale = 1.0
    v_scale = 1.0

    assert key.device.type == 'npu'
    assert value.device.type == 'npu'
    assert key_cache.device.type == 'npu'
    assert value_cache.device.type == 'npu'
    assert slot_mapping.device.type == 'npu'

    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, 
                           kv_cache_dtype, k_scale, v_scale)

    for token_idx in range(num_tokens):
        slot_idx = slot_mapping[token_idx].item()
        if slot_idx >= 0:
            block_idx = slot_idx // block_size
            block_offset = slot_idx % block_size

            cached_key = key_cache[block_idx, block_offset]
            cached_value = value_cache[block_idx, block_offset]
            original_key = key[token_idx]
            original_value = value[token_idx]

            torch.testing.assert_close(cached_key, original_key, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(cached_value, original_value, rtol=1e-5, atol=1e-5)
    
    print("reshape_and_cache_flash test passed!")

if __name__ == "__main__":
    test_reshape_and_cache_flash()
    print("All tests passed!")

