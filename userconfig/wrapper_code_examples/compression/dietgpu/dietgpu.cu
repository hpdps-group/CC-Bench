/**
 * dietgpu wrapper — implements ncclCompress / ncclDecompress / ncclDecompressReduce
 * backed by Meta's DietGPU library.
 *
 * Uses coccl_wrapper_template.cu — only 3 callbacks needed.
 *
 * Build:
 *   DIETGPU_DIR=~/compression_kernels/dietgpu
 *   nvcc -shared -O3 -lnccl -ldl -ldietgpu                     \
 *        -I${DIETGPU_DIR} -I${DIETGPU_DIR}/dietgpu              \
 *        -o libdietgpu_wrapper.so dietgpu.cu
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <dietgpu/float/GpuFloatCodec.h>
#include <dietgpu/ans/GpuANSCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>

#include "dietgpu_common.h"

/*-------------------------------------------------------------------------*
 *  Pull in coccl wrapper template — provides 4 entry-point functions        *
 *  Implement these below:  maxCompSize / compressOne / decompressOne       *
 *-------------------------------------------------------------------------*/
#include "../coccl_wrapper_template.cu"

/*===========================================================================*
 *  Callback 1 — max compressed size                                       *
 *===========================================================================*/
static size_t maxCompSize(size_t numElems, ncclDataType_t type) {
  dietgpu::FloatType ft = ncclTypeToFloat(type);
  if (ft == dietgpu::FloatType::kUndefined) return 0;
  return dietgpu::getMaxFloatCompressedSize(ft,
         static_cast<uint32_t>(numElems));
}

/*===========================================================================*
 *  Callback 2 — compress one chunk                                        *
 *===========================================================================*/
static ncclResult_t compressOne(const void* src, void* dst,
    size_t numElems, ncclDataType_t type, cudaStream_t stream) {

  dietgpu::FloatType ft = ncclTypeToFloat(type);
  if (ft == dietgpu::FloatType::kUndefined) return ncclInternalError;

  int dev;
  cudaError_t ce = cudaGetDevice(&dev);
  if (ce != cudaSuccess) return ncclInternalError;

  const void* inPtrs[1]  = { src };
  uint32_t    inSizes[1] = { static_cast<uint32_t>(numElems) };
  void*       outPtrs[1] = { dst };

  uint32_t* outSizeDev = nullptr;
  ce = cudaMallocAsync(&outSizeDev, sizeof(uint32_t), stream);
  if (ce != cudaSuccess) return ncclInternalError;

  {
    dietgpu::StackDeviceMemory mem(dev, dietgpu::kDefaultStackSize);
    dietgpu::FloatCompressConfig cfg(ft,
        dietgpu::ANSCodecConfig(dietgpu::kANSDefaultProbBits, false),
        false, false);

    dietgpu::floatCompress(mem, cfg, 1,
        inPtrs, inSizes, outPtrs, outSizeDev, stream);
  }

  cudaFreeAsync(outSizeDev, stream);
  return ncclSuccess;
}

/*===========================================================================*
 *  Callback 3 — decompress one chunk                                      *
 *===========================================================================*/
static ncclResult_t decompressOne(const void* src, void* dst,
    size_t numElems, ncclDataType_t type, cudaStream_t stream) {

  dietgpu::FloatType ft = ncclTypeToFloat(type);
  if (ft == dietgpu::FloatType::kUndefined) return ncclInternalError;

  int dev;
  cudaError_t ce = cudaGetDevice(&dev);
  if (ce != cudaSuccess) return ncclInternalError;

  const void* inPtrs[1]        = { src };
  void*       outPtrs[1]       = { dst };
  uint32_t    outCapacities[1] = { static_cast<uint32_t>(numElems) };

  {
    dietgpu::StackDeviceMemory mem(dev, dietgpu::kDefaultStackSize);
    dietgpu::FloatDecompressConfig cfg(ft,
        dietgpu::ANSCodecConfig(dietgpu::kANSDefaultProbBits, false),
        false, false);

    dietgpu::floatDecompress(mem, cfg, 1,
        inPtrs, outPtrs, outCapacities,
        nullptr, nullptr, stream);
  }

  return ncclSuccess;
}
