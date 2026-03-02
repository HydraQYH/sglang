import itertools

import torch
import triton
import triton.testing
from flashinfer import fused_add_rmsnorm as fi_fused_add_rmsnorm
import triton
import triton.language as tl

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm

IS_CI = is_in_ci()

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
    jit_fused_add_rmsnorm(input, residual, weight, eps)


def flashinfer_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    fi_fused_add_rmsnorm(input, residual, weight, eps=eps)


DTYPE = torch.bfloat16
DEVICE = "cuda"

if IS_CI:
    BS_LIST = [16]
    HIDDEN_SIZE_LIST = [512, 2048]
else:
    BS_LIST = [2**n for n in range(0, 14)]
    HIDDEN_SIZE_LIST = [1536, 3072, 4096, 5120, 8192]

LINE_VALS = ["jit", "fi", "triton"]
LINE_NAMES = ["SGL JIT Kernel", "FlashInfer", "Triton"]
STYLES = [("orange", "-"), ("blue", "--"), ("green", "-."), ("red", ":")]

configs = list(itertools.product(HIDDEN_SIZE_LIST, BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark(hidden_size: int, batch_size: int, provider: str):
    input = torch.randn((batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    residual = torch.randn((batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    weight = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    FN_MAP = {
        "jit": sglang_jit_fused_add_rmsnorm,
        "fi": flashinfer_fused_add_rmsnorm,
        "triton": triton_jit_fused_add_rmsnorm,
    }
    fn = lambda: FN_MAP[provider](
        input.clone(), residual.clone(), weight, torch.finfo(torch.bfloat16).eps
    )
    fn()    # Autotune
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
