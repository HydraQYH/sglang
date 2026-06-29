/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "nvfp4_scaled_mm_common.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Config(half_t/bfloat16_t) for M <= 128
template <typename T>
struct PerTokenKernelConfigM128 {
  using OutputType = T;
  using MmaTileShape = Shape<_128, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 PerTokenKernelConfigM128<T>::preferred_cluster(1, 4, 1);
template <typename T>
const dim3 PerTokenKernelConfigM128<T>::fallback_cluster(1, 2, 1);

// Config(half_t/bfloat16_t) for M > 128
template <typename T>
struct PerTokenKernelConfigDefault {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 PerTokenKernelConfigDefault<T>::preferred_cluster(2, 4, 1);
template <typename T>
const dim3 PerTokenKernelConfigDefault<T>::fallback_cluster(2, 1, 1);

template <typename KernelConfig>
struct PerTokenFp4GemmSm100 {
  using Config = KernelConfig;
  using OutputType = typename KernelConfig::OutputType;

  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutputType;
  using ElementC = OutputType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ElementScale = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  using MmaTileShape = typename KernelConfig::MmaTileShape;
  using ClusterShape = typename KernelConfig::ClusterShape;
  using EpilogueTile = typename KernelConfig::EpilogueTile;
  using EpilogueSchedule = typename KernelConfig::EpilogueSchedule;
  using MainloopSchedule = typename KernelConfig::MainloopSchedule;

  // Custom epilogue:
  //   D[m, n] = Acc[m, n] * row_scale[m] * col_scale[n]
  // row_scale is a logical (M, 1) vector and is broadcast across columns.
  // col_scale is a logical (1, N) vector and is broadcast across rows.
  using RowScaleBroadcast = cutlass::epilogue::fusion::
      Sm90ColBroadcast<0, MmaTileShape, ElementScale, ElementAccumulator, Stride<_1, _0, int64_t>, 1>;
  using ColScaleBroadcast = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, MmaTileShape, ElementScale, ElementAccumulator, Stride<_0, _1, int64_t>, 1>;
  using PerTokenScaleEpilogue = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementAccumulator, RoundStyle>,
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::
              Sm90Compute<cutlass::multiplies, ElementAccumulator, ElementAccumulator, RoundStyle>,
          cutlass::epilogue::fusion::Sm90AccFetch,
          RowScaleBroadcast>,
      ColScaleBroadcast>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShape,
      ClusterShape,
      EpilogueTile,
      ElementAccumulator,
      ElementAccumulator,
      void,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      EpilogueSchedule,
      PerTokenScaleEpilogue>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <typename T>
typename T::Gemm::Arguments per_token_args_from_options(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView col_scale,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementScale = typename T::ElementScale;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       nullptr,
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args = {
      {// (acc * row_scale)
       {},
       {static_cast<ElementScale const*>(row_scale.data_ptr()), ElementScale(0), {_1{}, _0{}, 0}},
       {}},
      {static_cast<ElementScale const*>(col_scale.data_ptr()), ElementScale(0), {_0{}, _1{}, 0}},
      {}};

  using KernelConfig = typename T::Config;
  arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
  arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
  return arguments;
}

template <typename T>
void runPerTokenGemm(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView col_scale,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename T::Gemm gemm;
  auto arguments = per_token_args_from_options<T>(D, A, B, A_sf, B_sf, row_scale, col_scale, m, n, k);

  size_t workspace_size = T::Gemm::get_workspace_size(arguments);
  auto workspace_tensor = alloc_workspace_tensor(workspace_size, A.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

template <typename OutType>
void cutlassFp4PerTokenGemmDispatchSm100(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView col_scale,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  if (m <= 128) {
    runPerTokenGemm<PerTokenFp4GemmSm100<PerTokenKernelConfigM128<OutType>>>(
        D, A, B, A_sf, B_sf, row_scale, col_scale, m, n, k, stream);
  } else {
    runPerTokenGemm<PerTokenFp4GemmSm100<PerTokenKernelConfigDefault<OutType>>>(
        D, A, B, A_sf, B_sf, row_scale, col_scale, m, n, k, stream);
  }
}

#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

void nvfp4_per_token_gemm_sm100(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView col_scale) {
  RuntimeCheck(A.device().device_type == kDLCUDA, "a must be a CUDA tensor");
  RuntimeCheck(B.device().device_type == kDLCUDA, "b must be a CUDA tensor");
  RuntimeCheck(A_sf.device().device_type == kDLCUDA, "scale_a must be a CUDA tensor");
  RuntimeCheck(B_sf.device().device_type == kDLCUDA, "scale_b must be a CUDA tensor");
  RuntimeCheck(row_scale.device().device_type == kDLCUDA, "row_scale must be a CUDA tensor");
  RuntimeCheck(col_scale.device().device_type == kDLCUDA, "col_scale must be a CUDA tensor");
  RuntimeCheck(D.device().device_type == kDLCUDA, "out must be a CUDA tensor");

  RuntimeCheck(A.device() == B.device(), "a and b must be on same device");
  RuntimeCheck(A.device() == A_sf.device(), "a and scale_a must be on same device");
  RuntimeCheck(A.device() == B_sf.device(), "a and scale_b must be on same device");
  RuntimeCheck(A.device() == row_scale.device(), "a and row_scale must be on same device");
  RuntimeCheck(A.device() == col_scale.device(), "a and col_scale must be on same device");
  RuntimeCheck(A.device() == D.device(), "a and out must be on same device");

  RuntimeCheck(A.is_contiguous(), "a must be contiguous");
  RuntimeCheck(B.is_contiguous(), "b must be contiguous");
  RuntimeCheck(A_sf.is_contiguous(), "scale_a must be contiguous");
  RuntimeCheck(B_sf.is_contiguous(), "scale_b must be contiguous");
  RuntimeCheck(row_scale.is_contiguous(), "row_scale must be contiguous");
  RuntimeCheck(col_scale.is_contiguous(), "col_scale must be contiguous");
  RuntimeCheck(D.is_contiguous(), "out must be contiguous");

  RuntimeCheck(host::is_type<uint8_t>(A.dtype()), "a must be uint8");
  RuntimeCheck(host::is_type<uint8_t>(B.dtype()), "b must be uint8");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(A_sf.dtype()), "scale_a must be float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(B_sf.dtype()), "scale_b must be float8_e4m3fn");
  RuntimeCheck(host::is_type<float>(row_scale.dtype()), "row_scale must be float32");
  RuntimeCheck(host::is_type<float>(col_scale.dtype()), "col_scale must be float32");

  RuntimeCheck(A.dim() == 2, "a must be a matrix");
  RuntimeCheck(B.dim() == 2, "b must be a matrix");
  RuntimeCheck(A_sf.dim() == 2, "scale_a must be a matrix");
  RuntimeCheck(B_sf.dim() == 2, "scale_b must be a matrix");
  RuntimeCheck(row_scale.dim() == 1, "row_scale must be a vector");
  RuntimeCheck(col_scale.dim() == 1, "col_scale must be a vector");

  RuntimeCheck(
      A.size(1) == B.size(1),
      "a and b shapes cannot be multiplied (",
      A.size(0),
      "x",
      A.size(1),
      " and ",
      B.size(0),
      "x",
      B.size(1),
      ")");

  const auto m = static_cast<int64_t>(A.size(0));
  const auto n = static_cast<int64_t>(B.size(0));
  const auto k = static_cast<int64_t>(A.size(1) * 2);

  RuntimeCheck(D.dim() == 2, "out must be 2D");
  RuntimeCheck(D.size(0) == m, "out first dim must equal m");
  RuntimeCheck(D.size(1) == n, "out second dim must equal n");
  RuntimeCheck(row_scale.size(0) == m, "row_scale size must equal m");
  RuntimeCheck(col_scale.size(0) == n, "col_scale size must equal n");

  constexpr int alignment = 32;
  RuntimeCheck(k % alignment == 0, "Expected k to be divisible by ", alignment, ", but got k: ", k);

  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  const int64_t rounded_m = round_up(m, 128);
  const int64_t rounded_n = round_up(n, 128);
  const int64_t rounded_k = round_up(k / 16, 4);

  RuntimeCheck(
      A_sf.size(1) == B_sf.size(1),
      "scale_a and scale_b shapes cannot be multiplied (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      " and ",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");
  RuntimeCheck(
      A_sf.size(0) == rounded_m && A_sf.size(1) == rounded_k,
      "scale_a must be padded/swizzled to shape (",
      rounded_m,
      "x",
      rounded_k,
      "), got (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      ")");
  RuntimeCheck(
      B_sf.size(0) == rounded_n && B_sf.size(1) == rounded_k,
      "scale_b must be padded/swizzled to shape (",
      rounded_n,
      "x",
      rounded_k,
      "), got (",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");

  const cudaStream_t stream = LaunchKernel::resolve_device(A.device());
  const int sm_version = getSMVersion(A.device().device_id);
  RuntimeCheck(sm_version >= 100 && sm_version < 120, "nvfp4_per_token_gemm currently supports Sm100 only");

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  if (host::is_type<fp16_t>(D.dtype())) {
    cutlassFp4PerTokenGemmDispatchSm100<cutlass::half_t>(D, A, B, A_sf, B_sf, row_scale, col_scale, m, n, k, stream);
  } else if (host::is_type<bf16_t>(D.dtype())) {
    cutlassFp4PerTokenGemmDispatchSm100<cutlass::bfloat16_t>(
        D, A, B, A_sf, B_sf, row_scale, col_scale, m, n, k, stream);
  } else {
    Panic("Unsupported output data type of nvfp4_per_token_gemm");
  }
#else
  (void)stream;
  Panic("nvfp4_per_token_gemm was not compiled with Sm100 support");
#endif
}

void nvfp4_per_token_gemm(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView col_scale) {
  nvfp4_per_token_gemm_sm100(D, A, B, A_sf, B_sf, row_scale, col_scale);
}
