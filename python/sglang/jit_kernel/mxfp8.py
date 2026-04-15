from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_FLOAT4_E2M1_MAX = 6.0
_FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _mxfp8_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-extended-lambda",
    ]


@cache_once
def _jit_es_sm100_mxfp8_blocksclaed_group_quant(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "es_sm100_mxfp8_blocksclaed_group_quant",
        *args,
        cuda_files=[
            "moe/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh"
        ],
        cuda_wrappers=[
            (
                "es_sm100_mxfp8_blocksclaed_group_quant",
                f"EsSm100MXFP8BlockscaledGroupQuant<{args}>::run",
            )
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_mxfp8_cuda_flags(),
    )


def es_sm100_mxfp8_blockscaled_grouped_quant(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
):
    module = _jit_es_sm100_mxfp8_blocksclaed_group_quant(input.dtype)
    module.es_sm100_mxfp8_blocksclaed_group_quant(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )
