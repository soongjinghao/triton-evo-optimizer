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
    BLOCK_TOKENS: tl.constexpr,
):
    block_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    
    tile_i = tl.program_id(axis=2)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    
    base_slot_offset = block_id * BLOCK_TOKENS
    slot_offsets = tl.arange(0, BLOCK_TOKENS)
    slot_indices = base_slot_offset + slot_offsets
    
    slot_mapping = tl.load(slot_mapping_ptr + slot_indices, mask=slot_indices < slot_mapping_ptr.shape[0], other=-1)
    valid_mask = slot_mapping >= 0
    
    block_idx = slot_mapping // block_size
    block_offset = slot_mapping % block_size
    
    src_key_base = slot_indices * key_stride
    src_value_base = slot_indices * value_stride
    
    tgt_base = block_idx * block_stride + block_offset * page_stride
    
    head_offset = head_id * head_size
    tile_head_offs = tile_pos + head_offset
    
    src_key_idx = src_key_base[:, None] + tile_head_offs[None, :]
    src_value_idx = src_value_base[:, None] + tile_head_offs[None, :]
    tgt_idx = tgt_base[:, None] + tile_head_offs[None, :]
    
    key_load = tl.load(key_ptr + src_key_idx, mask=valid_mask[:, None] & (tile_head_offs[None, :] < ((head_id + 1) * head_size)))
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load
    
    value_load = tl.load(value_ptr + src_value_idx, mask=valid_mask[:, None] & (tile_head_offs[None, :] < ((head_id + 1) * head_size)))
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load
    
    tl.store(key_cache_ptr + tgt_idx, key_tile, mask=valid_mask[:, None] & (tile_head_offs[None, :] < ((head_id + 1) * head_size)))
    tl.store(value_cache_ptr + tgt_idx, value_tile, mask=valid_mask[:, None] & (tile_head_offs[None, :] < ((head_id + 1) * head_size)))

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
    
    num_tokens = key.shape[0]
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
    TILE_SIZE = min(512, triton.next_power_of_2(head_size))
    BLOCK_TOKENS = min(32, triton.next_power_of_2(num_tokens))
    num_blocks = triton.cdiv(num_tokens, BLOCK_TOKENS)
    num_warps = 4
    num_stages = 2
    
    grid = lambda meta: (
        num_blocks,
        num_heads,
        triton.cdiv(meta["head_size"], meta["TILE_SIZE"]),
    )
    
    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
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
        BLOCK_TOKENS=BLOCK_TOKENS,
        num_warps=num_warps,
        num_stages=num_stages,
    )