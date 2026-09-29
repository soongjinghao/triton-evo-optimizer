import logging
import torch
import torch_npu
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N, beta, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 2
    offsets1 = block_start + tl.arange(0, BLOCK_SIZE)
    mask1 = offsets1 < N
    x1 = tl.load(x_ptr + offsets1, mask=mask1)
    x_fp1 = x1 if x1.dtype == tl.float32 else x1.to(tl.float32)
    z1 = x_fp1 * beta
    soft_z1 = tl.where(z1 > threshold, z1, tl.log(1 + tl.exp(z1)))
    out1 = (soft_z1 / beta).to(x1.dtype)
    tl.store(out_ptr + offsets1, out1, mask=mask1)
    offsets2 = block_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask2 = offsets2 < N
    x2 = tl.load(x_ptr + offsets2, mask=mask2)
    x_fp2 = x2 if x2.dtype == tl.float32 else x2.to(tl.float32)
    z2 = x_fp2 * beta
    soft_z2 = tl.where(z2 > threshold, z2, tl.log(1 + tl.exp(z2)))
    out2 = (soft_z2 / beta).to(x2.dtype)
    tl.store(out_ptr + offsets2, out2, mask=mask2)


@triton.jit
def softplus_kernel_default(x_ptr, out_ptr, N, beta, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 2
    offsets1 = block_start + tl.arange(0, BLOCK_SIZE)
    mask1 = offsets1 < N
    x1 = tl.load(x_ptr + offsets1, mask=mask1)
    x_fp1 = x1 if x1.dtype == tl.float32 else x1.to(tl.float32)
    soft_z1 = tl.where(x_fp1 > 20.0, x_fp1, tl.log(1 + tl.exp(x_fp1)))
    out1 = soft_z1.to(x1.dtype)
    tl.store(out_ptr + offsets1, out1, mask=mask1)
    offsets2 = block_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask2 = offsets2 < N
    x2 = tl.load(x_ptr + offsets2, mask=mask2)
    x_fp2 = x2 if x2.dtype == tl.float32 else x2.to(tl.float32)
    soft_z2 = tl.where(x_fp2 > 20.0, x_fp2, tl.log(1 + tl.exp(x_fp2)))
    out2 = soft_z2.to(x2.dtype)
    tl.store(out_ptr + offsets2, out2, mask=mask2)


def softplus_forward(x, beta, threshold):
    logger.debug("GEMS SOFTPLUS FORWARD")
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE'] * 2),)
    if beta == 1.0 and threshold == 20.0:
        softplus_kernel_default[grid](x, out, N, beta, threshold, BLOCK_SIZE=1024)
    else:
        softplus_kernel[grid](x, out, N, beta, threshold, BLOCK_SIZE=1024)
    return out


def softplus(self, beta=1.0, threshold=20.0):
    output = softplus_forward(self, beta, threshold)
    return output