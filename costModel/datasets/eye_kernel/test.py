import logging

import torch

from eye_kernel import eye_m

logger = logging.getLogger(__name__)


def test_eye_m_64x64():
    device = torch.device("npu")
    n, m = 64, 64

    result_triton = eye_m(n, m, device=device)
    result_torch = torch.eye(n, m, device=device)

    assert torch.allclose(
        result_triton, result_torch, rtol=0.0, atol=0.0
    ), f"Mismatch for shape {(n, m)}"


if __name__ == "__main__":
    test_eye_m_64x64()
    print("eye_kernel test_1 passed")

