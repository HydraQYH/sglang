#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups/reduce.h>

#include "nvfp4_quant.cuh"
#include <cooperative_groups.h>

template <typename T>
struct BlockTypeTrait;

template <>
struct BlockTypeTrait<bf16_t> {
  using packed_t = packed_t<bf16_t>;
  using block_t = device::AlignedVector<packed_t, 8>;
};

template <>
struct BlockTypeTrait<fp16_t> {
  using packed_t = packed_t<fp16_t>;
  using block_t = device::AlignedVector<packed_t, 8>;
};

SGL_DEVICE uint64_t fp32_vec_to_e2m1(float2 (&array)[8]) {
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED || CUTLASS_ARCH_MMA_SM120A_ENABLED || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000))

  uint64_t val;

  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      ".reg .b8 byte4;\n"
      ".reg .b8 byte5;\n"
      ".reg .b8 byte6;\n"
      ".reg .b8 byte7;\n"

      // Temporary 32-bit registers
      ".reg .b32 lo;\n"
      ".reg .b32 hi;\n"

      // Convert two FP32 -> one packed e2m1x2 byte
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2,  %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4,  %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6,  %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8,  %7;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte4, %10, %9;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte5, %12, %11;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte6, %14, %13;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte7, %16, %15;\n"

      // Pack 4 bytes -> one uint32
      "mov.b32 lo, {byte0, byte1, byte2, byte3};\n"
      "mov.b32 hi, {byte4, byte5, byte6, byte7};\n"

      // Pack two uint32 -> one uint64
      "mov.b64 %0, {lo, hi};\n"

      "}\n"
      : "=l"(val)
      : "f"(array[0].x),
        "f"(array[0].y),
        "f"(array[1].x),
        "f"(array[1].y),
        "f"(array[2].x),
        "f"(array[2].y),
        "f"(array[3].x),
        "f"(array[3].y),
        "f"(array[4].x),
        "f"(array[4].y),
        "f"(array[5].x),
        "f"(array[5].y),
        "f"(array[6].x),
        "f"(array[6].y),
        "f"(array[7].x),
        "f"(array[7].y));

  return val;

#else
  printf("fp32_vec_to_e2m1 is not supported on this architecture\n");
  __trap();
  return 0;
#endif
}

template <typename T>
__device__ T block_reduce_max(T val) {
  constexpr int kWarpSize = 32;
  __shared__ T warp_max[32];

  int tid = threadIdx.x;
  int lane = tid & (kWarpSize - 1);
  int warp = tid / kWarpSize;

  // -------------------------
  // Step1. Warp Reduce
  // -------------------------
  unsigned mask = __activemask();

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    T other = __shfl_down_sync(mask, val, offset);
    val = max(val, other);
  }

  if (lane == 0) {
    warp_max[warp] = val;
  }

  __syncthreads();

  // -------------------------
  // Step2. First Warp Reduce
  // -------------------------
  int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  if (warp == 0) {
    val = (lane < num_warps) ? warp_max[lane] : std::numeric_limits<T>::lowest();

    unsigned warp_mask = __ballot_sync(mask, lane < num_warps);

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      T other = __shfl_down_sync(warp_mask, val, offset);
      val = max(val, other);
    }

    if (lane == 0) {
      warp_max[0] = val;
    }
  }
  __syncthreads();
  return warp_max[0];
}

// Quantizes the provided PackedVec into the uint32_t output
template <typename DType, typename BlockType>
SGL_DEVICE uint64_t nvfp4_per_token_quant_core(BlockType& block, float* SFScaleOut, uint8_t* SFout) {
  constexpr float E4M3MAX = 448.0f;
  constexpr float E2M1MAX = 6.0f;
  constexpr float E4M3MAX_MUL_E2M1MAX_RCP = 1.0f / (448.0f * 6.0f);
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(block[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < 8; i++) {
    localMax = __hmax2(localMax, __habs2(block[i]));
  }

  // Get the final absolute maximum values.
  float blockMax = float(__hmax(localMax.x, localMax.y));

  float tokenMax = block_reduce_max(blockMax);

  // Outer quant factor
  float SFScaleVal = tokenMax * E4M3MAX_MUL_E2M1MAX_RCP;
  if (threadIdx.x == 0) {
    // STG.32
    *SFScaleOut = SFScaleVal;
  }

  // Get the SF (max value of the vector / max value of e2m1).
  float SFValue = E4M3MAX * blockMax * reciprocal_approximate_ftz(tokenMax);
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  // Here SFValue is always positive, so E4M3 is the same as UE4M3.
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  fp8SFVal = tmp.__x;
  SFValue = static_cast<float>(tmp);
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) *
  //                       reciprocal(SFScaleVal))
  // float outputScale = SFValue != 0 ? (E2M1MAX * reciprocal_approximate_ftz(blockMax)) : 0.0f;
  float outputScale = SFValue != 0 ? reciprocal_approximate_ftz(SFValue * SFScaleVal) : 0.0f;

  // Write the SF to global memory (STG.8).
  *SFout = fp8SFVal;

  // Convert the input to float.
  float2 fp2Vals[8];

#pragma unroll
  for (int i = 0; i < 8; i++) {
    if constexpr (std::is_same_v<DType, half>) {
      fp2Vals[i] = __half22float2(block[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(block[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint64_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
}

template <class DType>
__global__ void nvfp4_per_token_quant(
    DType const* input, uint8_t* output, fp8_e4m3_t* output_sf, float* per_token_sf, int32_t hidden_size) {
  using namespace device;
  constexpr int BLOCK_SIZE = 16;
  auto token_id = blockIdx.x;
  auto block_id = threadIdx.x;
  DType const* block_ptr = input + token_id * hidden_size + block_id * BLOCK_SIZE;

  using block_t = typename BlockTypeTrait<DType>::block_t;
  block_t block;
  block.load(reinterpret_cast<const void*>(block_ptr));
  float* token_sf = per_token_sf + token_id;

  auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<fp8_e4m3_t, 1>(
      static_cast<int>(token_id), static_cast<int>(block_id), hidden_size, output_sf);

  uint64_t nvfp4_out = nvfp4_per_token_quant_core<DType, block_t>(block, token_sf, sf_out);
  uint64_t* output_ptr = reinterpret_cast<uint64_t*>(output + token_id * (hidden_size / 2) + block_id * 8);
  *output_ptr = nvfp4_out;
}

template <typename DType>
struct Nvfp4PerTokenQuant {
  static void
  run(tvm::ffi::TensorView output,
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView output_sf,
      tvm::ffi::TensorView per_token_sf) {
    using namespace host;
    auto num_tokens = SymbolicSize{"num_tokens"};
    auto padded_num_tokens = SymbolicSize{"padded_num_tokens"};
    auto hidden_size = SymbolicSize{"hidden_size"};
    auto half_hidden_size = SymbolicSize{"half_hidden_size"};
    auto scale_factors_per_token = SymbolicSize{"scale_factors_per_token"};

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_tokens, hidden_size}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({num_tokens, half_hidden_size}).with_dtype<uint8_t>().with_device(device).verify(output);
    TensorMatcher({padded_num_tokens, scale_factors_per_token})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output_sf);
    TensorMatcher({num_tokens}).with_dtype<float>().with_device(device).verify(per_token_sf);

    // Scale factors per token should align to 4 => hidden_size should align to 64
    auto _hidden_size = hidden_size.unwrap();
    RuntimeCheck(_hidden_size % 64 == 0, "hidden_size should align to 64");
    RuntimeCheck(_hidden_size == half_hidden_size.unwrap() * 2, "half_hidden_size mismatch hidden_size");
    RuntimeCheck(_hidden_size == scale_factors_per_token.unwrap() * 16, "scale_factors_per_token mismatch hidden_size");
    RuntimeCheck(padded_num_tokens.unwrap() % 128 == 0, "padded_num_tokens should align to 128");
    RuntimeCheck(
        padded_num_tokens.unwrap() == (num_tokens.unwrap() + 127) / 128 * 128, "padded_num_tokens mismatch num_tokens");

    cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
    int device_id = device.unwrap().device_id;

    if constexpr (std::is_same_v<DType, bf16_t> || std::is_same_v<DType, fp16_t>) {
      dim3 block(_hidden_size / 16, 1, 1);
      dim3 grid(num_tokens.unwrap(), 1, 1);
      nvfp4_per_token_quant<DType><<<grid, block, 0, stream>>>(
          reinterpret_cast<DType*>(input.data_ptr()),
          reinterpret_cast<uint8_t*>(output.data_ptr()),
          reinterpret_cast<fp8_e4m3_t*>(output_sf.data_ptr()),
          reinterpret_cast<float*>(per_token_sf.data_ptr()),
          _hidden_size);
    } else {
      Panic("Unsupported dtype");
    }
  }
};
