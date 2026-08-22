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
    batch_id = tl.program_id(0)
    t_block = tl.program_id(1)
    if batch_id >= B:
        return
    cum_len_start = tl.load(cum_lengths_ptr + batch_id)
    cum_len_end = tl.load(cum_lengths_ptr + batch_id + 1)
    seq_len = cum_len_end - cum_len_start
    if seq_len == 0:
        return
    t_start = t_block * BLOCK_T
    t_idx = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_idx < seq_len
    base_packed = batch_id * Lmax * D
    base_out = cum_len_start * D
    packed_base = packed_ptr + base_packed
    packed_base = tl.multiple_of(packed_base, 16)
    out_base = out_ptr + base_out
    out_base = tl.multiple_of(out_base, 16)
    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D
        mask = t_mask[:, None] & d_mask[None, :]
        r_offset = t_idx[:, None] * D + d_idx[None, :]
        packed_vals = tl.load(packed_base + r_offset, mask=mask)
        tl.store(out_base + r_offset, packed_vals, mask=mask)

def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 32,
    block_d: int = 64,
) -> torch.Tensor:
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor
    N = int(lengths.sum().item())
    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)
    cum_lengths = torch.zeros(B + 1, dtype=torch.int32, device=packed_tensor.device)
    torch.cumsum(lengths.int(), dim=0, out=cum_lengths[1:])
    grid = (B, triton.cdiv(Lmax, block_t))
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