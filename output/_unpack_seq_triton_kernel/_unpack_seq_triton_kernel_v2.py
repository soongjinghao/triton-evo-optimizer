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
    # Mask-based boundary check to avoid branch divergence
    valid_batch = pid < B
    if not valid_batch:
        return

    # Load precomputed cumulative lengths (two global reads per batch)
    cum_len_start = tl.load(cum_lengths_ptr + pid)
    cum_len_end = tl.load(cum_lengths_ptr + pid + 1)
    seq_len = cum_len_end - cum_len_start

    # Early exit for empty sequences using mask (predicated)
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

            # Source offsets: batch offset + time * D + feature
            packed_offset = (pid * Lmax * D +
                             t_idx[:, None] * D +
                             d_idx[None, :])

            # Destination offsets: (cumulative_start + time) * D + feature
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
    Optimized Triton implementation for Ascend NPU.
    Uses precomputed cumulative lengths to eliminate per-batch loops inside the kernel,
    reducing global memory reads from O(B*L) to O(B).

    Dynamic tile sizes are chosen based on tensor shapes to maximize utilization
    while respecting Ascend's 16-byte alignment requirement.
    """
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    # Dynamic tile selection (multiples of 16 for Ascend Cube)
    if block_t is None:
        # Time tile: balance between parallelism and register pressure
        block_t = min(128, max(16, (Lmax + 15) // 16 * 16))
    if block_d is None:
        # Feature tile: adapt to D, keep within 256 for large dims
        block_d = min(256, max(16, (D + 15) // 16 * 16))

    N = int(lengths.sum().item())
    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    # Precompute cumulative sequence lengths (passed as kernel argument)
    cum_lengths = torch.zeros(B + 1, dtype=torch.int32, device=packed_tensor.device)
    torch.cumsum(lengths.int(), dim=0, out=cum_lengths[1:])

    # One program per batch, each loads only its own cum lengths (O(1) reads per batch)
    grid = (B,)

    _unpack_seq_triton_kernel[grid](
        packed_reshaped,
        out,
        cum_lengths,
        B,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out