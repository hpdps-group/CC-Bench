#ifndef DIETGPU_COMMON_H
#define DIETGPU_COMMON_H

#include <cuda_runtime.h>
#include <nccl.h>

#include <dietgpu/float/GpuFloatCodec.h>

/* Convert ncclDataType_t to DietGPU FloatType */
static inline dietgpu::FloatType ncclTypeToFloat(ncclDataType_t t) {
  switch (t) {
    case ncclFloat32:  return dietgpu::FloatType::kFloat32;
    case ncclFloat16:  return dietgpu::FloatType::kFloat16;
    case ncclBFloat16: return dietgpu::FloatType::kBFloat16;
    default:           return dietgpu::FloatType::kUndefined;
  }
}

/* Size of ncclDataType_t in bytes */
static inline size_t ncclTypeSize(ncclDataType_t t) {
  switch (t) {
    case ncclInt8:    return 1;
    case ncclUint8:   return 1;
    case ncclFloat16: return 2;
    case ncclBFloat16:return 2;
    case ncclFloat32: return 4;
    case ncclInt32:   return 4;
    case ncclFloat64: return 8;
    case ncclInt64:   return 8;
    default:          return 4;
  }
}

/* Element-wise add kernel (for decompressReduce) */
template <typename T>
__global__ void addKernel(const T* a, const T* b, T* out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

#endif /* DIETGPU_COMMON_H */
