import itertools

import pytest
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
    restore_value=['X', 'RES']
)
@triton.jit
def rmsnorm_forward_kernel(
    X,            # pointer to input
    RES,          # pointer to residual
    W,            # weight (gamma)
    stride_x_row,
    N,            # num cols
    eps,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RES + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    x += res
    tl.store(RES + row * stride_x_row + cols, x, mask=mask)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    x_hat = x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = x_hat * w

    tl.store(X + row * stride_x_row + cols, y, mask=mask)


def triton_jit_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    num_tokens, dim = input.shape
    stride_x_row = dim
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(dim))
    if dim > BLOCK_N:
        raise RuntimeError("RMSNorm feature dim >= 64KB not supported.")

    grid = (num_tokens,)
    rmsnorm_forward_kernel[grid](
        input, residual, weight,
        stride_x_row, N=dim, eps=eps,
        BLOCK_N=BLOCK_N,
    )


def sglang_jit_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    from sglang.jit_kernel.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps)


def flashinfer_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    from flashinfer.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps=eps)


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
HIDDEN_SIZE_LIST = [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
DEVICE = "cuda"
DTYPE = torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,hidden_size", list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST))
)
def test_fused_add_rmsnorm(batch_size: int, hidden_size: int) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)

    input_sglang = input.clone()
    residual_sglang = residual.clone()
    input_flashinfer = input.clone()
    residual_flashinfer = residual.clone()
    input_triton = input.clone()
    residual_triton = residual.clone()
    sglang_jit_fused_add_rmsnorm(
        input_sglang, residual_sglang, weight, torch.finfo(torch.bfloat16).eps
    )
    flashinfer_fused_add_rmsnorm(
        input_flashinfer, residual_flashinfer, weight, torch.finfo(torch.bfloat16).eps
    )
    triton_jit_fused_add_rmsnorm(
        input_triton, residual_triton, weight, torch.finfo(torch.bfloat16).eps
    )
    torch.testing.assert_close(input_sglang, input_flashinfer, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        residual_sglang, residual_flashinfer, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(input_triton, input_flashinfer, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        residual_triton, residual_flashinfer, atol=1e-2, rtol=1e-2
    )

if __name__ == "__main__":
    pytest.main([__file__])
