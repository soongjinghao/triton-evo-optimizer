import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    valid_start_ptr,
    valid_end_ptr,
    k_scale,
    v_scale,
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    valid_start = tl.load(valid_start_ptr).to(tl.int64)
    valid_end = tl.load(valid_end_ptr).to(tl.int64)
    token_idx = token_idx + valid_start
    if token_idx >= valid_end:
        return
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride
    tgt_idx = block_idx * block_stride + block_offset * page_stride
    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load
    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load
    tl.store(
        key_cache_ptr + tgt_idx + tile_pos,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx + tile_pos,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )

def triton_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
):
    assert key.device.type == 'npu', f"Key tensor must be on NPU, got {key.device}"
    assert value.device.type == 'npu', f"Value tensor must be on NPU, got {value.device}"
    assert key_cache.device.type == 'npu', f"Key cache must be on NPU, got {key_cache.device}"
    assert value_cache.device.type == 'npu', f"Value cache must be on NPU, got {value_cache.device}"
    assert slot_mapping.device.type == 'npu', f"Slot mapping must be on NPU, got {slot_mapping.device}"
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]
    head_stride = key_cache.stride()[2]
    assert head_stride == head_size, "only continous heads are supported"
    assert kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8"), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        torch.float16
        if kv_cache_dtype.startswith("fp8")
        else key_cache.dtype
    )
    if key_cache.dtype != kv_cache_torch_dtype and kv_cache_dtype.startswith("fp8"):
        key_cache = key_cache.to(kv_cache_torch_dtype)
        value_cache = value_cache.to(kv_cache_torch_dtype)
    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")
    TILE_SIZE = min(512, triton.next_power_of_2(n))
    num_warps = 4
    num_stages = 2

    # Precompute valid_start and valid_end from slot_mapping
    # Assumption: invalid tokens (-1) are contiguous at the tail
    slot_mapping_cpu = slot_mapping.cpu()
    valid_mask = slot_mapping_cpu >= 0
    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() > 0:
        valid_start = int(valid_indices[0].item())
        valid_end = int(valid_indices[-1].item()) + 1
    else:
        valid_start = 0
        valid_end = 0
    valid_start_tensor = torch.tensor([valid_start], device=slot_mapping.device, dtype=torch.int64)
    valid_end_tensor = torch.tensor([valid_end], device=slot_mapping.device, dtype=torch.int64)

    num_valid_tokens = valid_end - valid_start
    if num_valid_tokens <= 0:
        return

    grid = lambda meta: (
        num_valid_tokens,
        triton.cdiv(n, meta["TILE_SIZE"]),
    )
    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        valid_start_ptr=valid_start_tensor,
        valid_end_ptr=valid_end_tensor,
        k_scale=k_scale,
        v_scale=v_scale,
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        FP8_KV_CACHE=FP8_KV_CACHE,
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

def test_reshape_and_cache_npu():
    device = torch.npu.current_device()
    num_tokens = 128
    num_heads = 32
    head_size = 128
    block_size = 16
    num_blocks = 10
    key = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    value = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device='npu', dtype=torch.float16)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device='npu', dtype=torch.float16)
    slot_mapping = torch.cat([
        torch.randint(0, num_blocks * block_size - 10, (num_tokens - 20,), device='npu'),
        torch.full((20,), -1, device='npu')
    ])
    k_scale = torch.tensor([1.0], device='npu')
    v_scale = torch.tensor([1.0], device='npu')
    triton_reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, 
        "auto", k_scale, v_scale
    )
    print("NPU reshape_and_cache test completed successfully!")

if __name__ == "__main__":
    test_reshape_and_cache_npu()