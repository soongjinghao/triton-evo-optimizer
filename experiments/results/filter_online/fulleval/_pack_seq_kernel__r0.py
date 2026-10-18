import torch
import triton
import triton.language as tl
@triton.jit
def _pack_seq_kernel(
    x_ptr,
    out_ptr,
    lengths_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // tl.cdiv(Lmax, BLOCK_T)
    pid_t = pid % tl.cdiv(Lmax, BLOCK_T)
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = off_t < Lmax
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)
    valid_row = (off_t < seq_len) & t_mask
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d[None, :] < D
        in_row = in_start + off_t
        x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]
        out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
        tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)
        x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
        tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)
def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x
    B = lengths.numel()
    Lmax = int(lengths.max().item())
    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)
    grid = (B * triton.cdiv(Lmax, block_t), 1, 1)
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=float(pad_value),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    if len(original_shape) > 2:
        output_shape = (B, Lmax) + original_shape[1:]
        out = out.reshape(output_shape)
    return out