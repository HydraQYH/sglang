import sys

import pytest
import torch

from sglang.jit_kernel.nvfp4 import nvfp4_per_token_gemm, nvfp4_per_token_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def _nvfp4_per_token_gemm_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and torch.cuda.get_device_capability() < (12, 0)
    )


DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [
    (128, 80, 64),
    (128, 128, 64),
    (256, 80, 64),
]

BLOCK_SIZE = 16

E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def cast_from_fp4(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    v_2nd = (x & 0xF).to(torch.long)
    v_1st = ((x >> 4) & 0xF).to(torch.long)
    c = torch.stack((v_2nd, v_1st), dim=-1).flatten()
    lut = torch.tensor(E2M1_TO_FLOAT32, device=x.device, dtype=torch.float32)
    return lut[c].reshape(m, n)


def recover_swizzled_scales(scale: torch.Tensor, m: int, k: int) -> torch.Tensor:
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_k = k // BLOCK_SIZE
    rounded_k = ((scale_k + 4 - 1) // 4) * 4
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_k // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_k)).to(torch.float32)
    return result[:m, :scale_k]


def dequantize_per_token_nvfp4(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    per_token_sf: torch.Tensor,
) -> torch.Tensor:
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = cast_from_fp4(tensor_fp4, m, k)
    tensor_f32 = tensor_f32.reshape(m, k // BLOCK_SIZE, BLOCK_SIZE)
    tensor_sf = recover_swizzled_scales(tensor_sf, m, k)
    return (tensor_f32 * tensor_sf.unsqueeze(-1) * per_token_sf[:, None, None]).reshape(
        m, k
    )


@pytest.mark.skipif(
    not _nvfp4_per_token_gemm_supported(),
    reason="NVFP4 per-token GEMM currently requires Sm100.",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_nvfp4_per_token_gemm(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(42)
    m, n, packed_k = shape
    k = packed_k * 2

    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = torch.randn((n, k), dtype=dtype, device="cuda")

    a_fp4, a_sf, a_per_token_sf = nvfp4_per_token_quant(a)
    b_fp4, b_sf, b_per_token_sf = nvfp4_per_token_quant(b)

    a_dequant = dequantize_per_token_nvfp4(a_fp4, a_sf, a_per_token_sf)
    b_dequant = dequantize_per_token_nvfp4(b_fp4, b_sf, b_per_token_sf)
    expected = torch.matmul(a_dequant, b_dequant.t())

    out = nvfp4_per_token_gemm(
        a_fp4,
        b_fp4,
        a_sf,
        b_sf,
        a_per_token_sf,
        b_per_token_sf,
        dtype,
    )

    torch.testing.assert_close(out, expected.to(dtype=dtype), atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
