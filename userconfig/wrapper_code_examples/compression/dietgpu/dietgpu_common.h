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

#endif /* DIETGPU_COMMON_H */
