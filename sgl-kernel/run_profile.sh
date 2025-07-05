#!/bin/bash
set -ex
DEVICE=${1:-'0'}

nsys profile -f true -o dsr1_ep_timeline \
  -e CUDA_VISIBLE_DEVICES=${DEVICE} \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency=50000 \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  python3 ./benchmark/bench_fp8_blockwise_group_gemm.py

CUDA_VISIBLE_DEVICES=${DEVICE} ncu -f \
  -o dsr1_ep_sf \
  --set full \
  --disable-profiler-start-stop \
  --pm-sampling-interval 1000 \
  --clock-control none \
  --nvtx --nvtx-include "[cutlass" \
  python3 ./benchmark/bench_fp8_blockwise_group_gemm.py

CUDA_VISIBLE_DEVICES=${DEVICE} ncu -f \
  -o dsr1_ep_sr \
  --set roofline \
  --disable-profiler-start-stop \
  --clock-control none \
  --nvtx --nvtx-include "[cutlass" \
  python3 ./benchmark/bench_fp8_blockwise_group_gemm.py
