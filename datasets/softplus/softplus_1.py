import logging
import torch
import torch_npu
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def softplus_kernel(x_ptr, out_ptr, N, beta, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp = x.to(tl.float32)
    z = x_fp * beta
    soft_z = tl.where(z > threshold, z, tl.log(1 + tl.exp(z)))
    out = (soft_z / beta).to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def softplus_kernel_default(x_ptr, out_ptr, N, beta, threshold, BLOCK_SIZE: tl.constexpr):
    # When beta=1.0 and threshold=20.0, we can skip the multiplication and division.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp = x.to(tl.float32)
    # z = x_fp * 1.0  => z = x_fp, and threshold is exactly 20.0
    soft_z = tl.where(x_fp > 20.0, x_fp, tl.log(1 + tl.exp(x_fp)))
    out = soft_z.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


def softplus_forward(x, beta, threshold):
    logger.debug("GEMS SOFTPLUS FORWARD")
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    if beta == 1.0 and threshold == 20.0:
        softplus_kernel_default[grid](x, out, N, beta, threshold, BLOCK_SIZE=1024)
    else:
        softplus_kernel[grid](x, out, N, beta, threshold, BLOCK_SIZE=1024)
    return out


def softplus(self, beta=1.0, threshold=20.0):
    output = softplus_forward(self, beta, threshold)
    return output