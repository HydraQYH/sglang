from typing import Tuple
import numpy as np
import random
import torch
import triton
from triton import Config
import triton.language as tl

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

configs = [
  Config(
    {"BLOCK_M": block_m}, num_warps=num_warps
  )
  for block_m in [16, 32, 64, 128]
  for num_warps in [2, 4, 8]
]

@triton.autotune(configs=configs, key=["K", "BLOCK_K", "M_ALIGNMENT"])
@triton.jit
def hopper_per_token_quant_fp8(
  a,  # (M, K):(K, 1)
  expert_offset,  # (num_experts + 1,)
  problem_sizes,  # (num_experts, 3)
  a_fp8,  # (M, K):(K, 1)
  sfa,  # (M, k)
  K: tl.constexpr,
  BLOCK_K: tl.constexpr,
  M_ALIGNMENT: tl.constexpr,
  BLOCK_M: tl.constexpr # tune
):
  k_offset = tl.program_id(0)
  expert_id = tl.program_id(1)

  m = tl.load(problem_sizes + expert_id * 3)
  current_expert_offset = tl.load(expert_offset + expert_id).to(tl.int64)
  tl.multiple_of(m, M_ALIGNMENT)
  tl.multiple_of(current_expert_offset, M_ALIGNMENT)

  coord_k = k_offset * BLOCK_K + tl.arange(0, BLOCK_K)
  for i in tl.range(tl.cdiv(m, BLOCK_M)):
    coord_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
    a_ptrs = a + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]
    a_mask = (coord_m < m)[:, None] & (coord_k < K)[None, :]

    inp = tl.load(a_ptrs, mask=a_mask).to(tl.float32)  # [BLOCK_M, BLOCK_K]
    inp_amax = tl.max(tl.abs(inp), axis=1)  # [BLOCK_M,]
    inp_amax = tl.clamp(inp_amax, min=1e-4, max=float("inf"))
    inp_fp8 = inp * (448.0 / inp_amax[:, None]).to(a_fp8.dtype.element_ty)

    # Store fp8
    a_fp8_ptrs = a_fp8 + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]
    tl.store(a_fp8_ptrs, inp_fp8, mask=a_mask)

    # Store sfa
    k = tl.cdiv(K, BLOCK_K)
    sfa_ptrs = sfa + current_expert_offset * k + k_offset * m + coord_m  # MN-Major with sfa
    tl.store(sfa_ptrs, inp_amax / 448.0, mask=coord_m < m)

if __name__ == '__main__':
  num_experts = 256
  expected_m_per_expert = 128
  n_g = 512
  k_g = 7168
  group_size = 128
  alignment = 1
  expert_ms = [ceil_div(int(expected_m_per_expert * random.uniform(0.7, 1.3)), alignment) * alignment for _ in range(num_experts)]
  print(expert_ms)
  m = sum(expert_ms)
  print(m)

  _expert_offset = [0]
  offset = 0
  for i in range(num_experts):
    offset += expert_ms[i]
    _expert_offset.append(offset)
  expert_offset =torch.from_numpy(np.asarray(_expert_offset)).to(dtype=torch.int32, device='cuda')

  problem_sizes = torch.stack([torch.from_numpy(np.asarray(expert_ms)).to(dtype=torch.int32, device='cuda'),
    torch.full((num_experts,), n_g, dtype=torch.int32, device='cuda'),
    torch.full((num_experts,), k_g, dtype=torch.int32, device='cuda')], dim=1)
  print(problem_sizes)
  print(problem_sizes.shape, problem_sizes.stride())

  # Input
  a_original = []
  a_tensors = []
  a_scales_tensors = []
  for g in range(num_experts):
    a = torch.randn((expert_ms[g], k_g), dtype=torch.bfloat16, device='cuda')
    a_g, a_scale = per_token_cast_to_fp8(a)
    a_original.append(a)
    a_tensors.append(a_g)
    a_scales_tensors.append(a_scale)
  a = torch.empty((m, k_g), device='cuda', dtype=torch.bfloat16)
  a_stack = torch.empty((m, k_g), device='cuda', dtype=torch.float8_e4m3fn)
  a_scale_stack = torch.empty((m * ceil_div(k_g, group_size)), device='cuda', dtype=torch.float32)

  for g in range(num_experts):
    # Matrix A is Row-Major.
    a[_expert_offset[g] : _expert_offset[g + 1]] = a_original[g]
    a_stack[_expert_offset[g] : _expert_offset[g + 1]] = a_tensors[g]
    a_scale_stack[_expert_offset[g] * ceil_div(k_g, group_size) : _expert_offset[g + 1] * ceil_div(k_g, group_size)] = (a_scales_tensors[g].t().contiguous().view(-1))
  a_scale_stack = a_scale_stack.view(m, ceil_div(k_g, group_size))

  # Output
  a_fp8 = torch.empty_like(a_stack)
  sfa = torch.empty((m, ceil_div(k_g, group_size)), dtype=torch.float32, device='cuda')

  grid = (ceil_div(k_g, group_size), num_experts)
  hopper_per_token_quant_fp8[grid](
    a, expert_offset, problem_sizes, a_fp8, sfa, k_g, group_size, alignment
  )
  print(a_stack)
  print(a_fp8)
  print(a_scale_stack)
  print(sfa)
  print(torch.mean(a_scale_stack - sfa))
  print(torch.max(a_scale_stack - sfa))
  torch.testing.assert_close(sfa, a_scale_stack)

