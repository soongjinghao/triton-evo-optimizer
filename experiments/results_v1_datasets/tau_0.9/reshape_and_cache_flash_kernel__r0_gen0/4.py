import logging
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def reshape_and_cache_flash_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    k_scale,
    v_scale,
    n: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    num_tokens,
    BLOCK_M: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    token_offs = pid_token * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_token = token_offs < num_tokens

    slots = tl.load(slot_mapping + token_offs, mask=valid_token, other=-1)
    valid_slot = slots >= 0
    row_valid = valid_token & valid_slot

    safe_token_offs = tl.where(valid_token, token_offs, 0)
    safe_slots = tl.where(row_valid, slots, 0)

    block_idx = safe_slots // block_size
    block_offset = safe_slots % block_size
    base_cache_offset = block_idx * block_stride + block_offset * n

    col_offs = pid_chunk * CHUNK_ELEMENTS + tl.arange(0, CHUNK_ELEMENTS)
    chunk_mask = col_offs < n
    tile_mask = row_valid[:, None] & chunk_mask[None, :]

    key_ptrs = key + safe_token_offs[:, None] * key_stride + col_offs[None, :]
    value_ptrs = value + safe_token_offs[:, None] * value_stride + col_offs[None, :]

    k = tl.load(key_ptrs, mask=tile_mask, other=0.0)
    v = tl.load(value_ptrs, mask=tile_mask, other=0.0)

    key_cache_ptrs = key_cache + base_cache_offset[:, None] + col_offs[None, :]
    value_cache_ptrs = value_cache + base_cache_offset[:, None] + col_offs[None, :]

    tl.store(key_cache_ptrs, k, mask=tile_mask)
    tl.store(value_cache_ptrs, v, mask=tile_mask)


def reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    logger.debug("GEMS RESHAPE_AND_CACHE_FLASH")
    num_tokens = slot_mapping.size(0)
    num_heads = key.size(1)
    head_size = key.size(2)
    block_size = key_cache.size(1)
    key_stride = key.stride(0)
    value_stride = value.stride(0)
    block_stride = key_cache.stride(0)
    assert key_cache.stride(0) == value_cache.stride(0)

    n = num_heads * head_size
    if num_tokens == 0 or n == 0:
        return None

    # Shape-aware chunk width: avoid unnecessary grid fragmentation when n is small.
    CHUNK_ELEMENTS = min(n, 4096)

    # Shape-aware token packing: use larger token blocks for small n, bounded UB tile area.
    if n <= 128:
        target_block_m = 16
    elif n <= 512:
        target_block_m = 8
    elif n <= 1024:
        target_block_m = 4
    else:
        target_block_m = 4

    token_block_cap = 1
    while token_block_cap < num_tokens:
        token_block_cap <<= 1
    BLOCK_M = max(1, min(target_block_m, token_block_cap))

    grid = (triton.cdiv(num_tokens, BLOCK_M), triton.cdiv(n, CHUNK_ELEMENTS))

    reshape_and_cache_flash_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        block_stride,
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        k_scale,
        v_scale,
        n,
        CHUNK_ELEMENTS,
        num_tokens,
        BLOCK_M,
    )
    return None