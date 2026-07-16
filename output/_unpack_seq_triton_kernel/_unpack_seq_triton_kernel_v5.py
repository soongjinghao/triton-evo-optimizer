import torch
import triton
import triton.language as tl

@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,
    out_ptr,
    cum_lengths_ptr,
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # Simple boundary check (parent 1 style)
    if pid >= B:
        return

    # Load precomputed cumulative lengths
    cum_len_start = tl.load(cum_lengths_ptr + pid)
    cum_len_end = tl.load(cum_lengths_ptr + pid + 1)
    seq_len = cum_len_end - cum_len_start

    # Early exit for empty sequences
    if seq_len == 0:
        return

    # Tile over time dimension
    for t_start in range(0, seq_len, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_idx < seq_len

        # Tile over feature dimension
        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D

            # Combined 2D mask
            mask = t_mask[:, None] & d_mask[None, :]

            # Source offsets
            packed_offset = (pid * Lmax * D +
                             t_idx[:, None] * D +
                             d_idx[None, :])

            # Destination offsets
            out_row = cum_len_start + t_idx
            out_offset = (out_row[:, None] * D + d_idx[None, :])

            # Coalesced loads and stores with predicated mask
            packed_vals = tl.load(packed_ptr + packed_offset, mask=mask)
            tl.store(out_ptr + out_offset, packed_vals, mask=mask)


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = None,
    block_d: int = None,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Hybrid offspring: combines the low-latency small-tile approach of Parent 1
    with the dynamic shape adaptation of Parent 2, plus resource-aware warps tuning.
    """
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    # Dynamic tile selection – bias toward smaller tiles for better occupancy,
    # while scaling up for large dimensions to keep loop count reasonable.
    if block_t is None:
        # Time tile: prefer 32–64 for typical lengths, up to 128 for very long sequences
        block_t = min(128, max(16, (Lmax + 15) // 16 * 16))
        # Further refinement: if length is moderate, clamp to 64 to mimic Parent 1's efficiency
        if Lmax <= 256:
            block_t = min(block_t, 64)
        # Ensure at least 16
        block_t = max(16, block_t)
    if block_d is None:
        # Feature tile: prefer 64 for typical dims, up to 128 for large dims
        block_d = min(128, max(16, (D + 15) // 16 * 16))
        if D <= 512:
            block_d = min(block_d, 64)
        block_d = max(16, block_d)

    N = int(lengths.sum().item())
    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    # Precompute cumulative sequence lengths
    cum_lengths = torch.zeros(B + 1, dtype=torch.int32, device=packed_tensor.device)
    torch.cumsum(lengths.int(), dim=0, out=cum_lengths[1:])

    # One program per batch (1D grid)
    grid = (B,)

    # Resource-aware warp count: larger tiles need more warps to hide latency
    total_tile_size = block_t * block_d
    if total_tile_size <= 2048:
        num_warps = 4
    elif total_tile_size <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    _unpack_seq_triton_kernel[grid](
        packed_reshaped,
        out,
        cum_lengths,
        B,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out