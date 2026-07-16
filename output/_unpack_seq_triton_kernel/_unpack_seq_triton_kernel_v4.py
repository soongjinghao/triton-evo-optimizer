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
    valid_batch = pid < B
    if not valid_batch:
        return

    cum_len_start = tl.load(cum_lengths_ptr + pid)
    cum_len_end = tl.load(cum_lengths_ptr + pid + 1)
    seq_len = cum_len_end - cum_len_start

    if seq_len == 0:
        return

    packed_base = packed_ptr + pid * Lmax * D
    out_base = out_ptr + cum_len_start * D

    for t_start in range(0, seq_len, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_idx < seq_len

        packed_row_off = t_idx[:, None] * D
        out_row_off = t_idx[:, None] * D

        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D
            mask = t_mask[:, None] & d_mask[None, :]

            packed_offs = packed_base + packed_row_off + d_idx[None, :]
            out_offs = out_base + out_row_off + d_idx[None, :]

            vals = tl.load(packed_offs, mask=mask)
            tl.store(out_offs, vals, mask=mask)


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = None,
    block_d: int = None,
) -> torch.Tensor:
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    if block_t is None:
        block_t = min(128, max(32, (Lmax + 15) // 16 * 16))
    if block_d is None:
        block_d = min(256, max(64, (D + 15) // 16 * 16))

    N = int(lengths.sum().item())
    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    cum_lengths = torch.zeros(B + 1, dtype=torch.int32, device=packed_tensor.device)
    torch.cumsum(lengths.int(), dim=0, out=cum_lengths[1:])

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
        num_warps=8,
        num_stages=2,
    )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out