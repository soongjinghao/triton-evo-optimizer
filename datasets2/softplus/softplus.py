import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def softplus_forward(x, beta, threshold):
    x_fp = x.to(tl.float32)
    z = x_fp * beta
    soft_z = tl.where(z > threshold, z, tl.log(1 + tl.exp(z)))
    out = (soft_z / beta).to(x.dtype)
    return out


def softplus(self, beta=1.0, threshold=20.0):
    logger.debug("GEMS SOFTPLUS FORWARD")
    output = softplus_forward(self, beta, threshold)
    return output
