from typing import Tuple

import pytest
import torch
from sgl_kernel import fp8_blockwise_scaled_mm


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    m, n = x.shape[-2], x.shape[-1]
    aligned_m = get_tma_aligned_size(m, x.element_size())
    if x.dim() == 2:
        if x.stride(0) == 1 and x.stride(1) == aligned_m:
            return x
        x, remove_dim = x.unsqueeze(0), True

    b = x.shape[0]

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(
        torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2
    )
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def is_sm100_supported(device=None) -> bool:
    return (torch.cuda.get_device_capability(device)[0] == 10) and (
        torch.version.cuda >= "12.8"
    )


def is_sm90_supported(device=None) -> bool:
    return (torch.cuda.get_device_capability(device)[0] == 9) and (
        torch.version.cuda >= "12.3"
    )


def _test_accuracy_once(M, N, K, out_dtype, device):
    cc = torch.cuda.get_device_capability(None)[0]
    a = torch.randn((M, K), device=device, dtype=out_dtype)  # (M, K):(K, 1)
    b = torch.randn((N, K), device=device, dtype=out_dtype).t()  # (K, N):(1, K)

    a_g, a_scale = per_token_cast_to_fp8(
        a
    )  # a_g -- (M, K):(K, 1), a_scale() -- (M, k):(k, 1)
    b_g, b_scale = per_block_cast_to_fp8(
        b
    )  # b_g -- (K, N):(N, 1), b_scale() -- (k, n):(n, 1)
    baseline = torch.mm(a, b)
    # Transpose Matrix B to Column-Major
    b_g = b_g.t().contiguous().t()  # b_g -- (K, N):(1, N)
    # Transpose SFA to MN-Major (for both SM90 and SM100)
    a_scale = a_scale.t().contiguous().t()  # (M, k):(1, M)
    if cc == 10:
        # For SM100, we need K-Major SFB
        b_scale = b_scale.t().contiguous().t()  # (k, n):(1, n)

    o1 = fp8_blockwise_scaled_mm(a_g, b_g, a_scale, b_scale, out_dtype)
    diff = calc_diff(o1, baseline)
    assert diff < 0.001
    print(f"cc={cc} M={M}, N={N}, K={K}, out_dtype={out_dtype}, diff={diff:.5f}: OK")
    # torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)


@pytest.mark.parametrize("M", [1, 3, 5, 127, 128, 512, 1024, 4096])
@pytest.mark.parametrize("N", [128, 512, 1024, 4096, 8192, 14080])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 14080, 16384])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy(M, N, K, out_dtype):
    _test_accuracy_once(M, N, K, out_dtype, "cuda")


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
        ]
    )
