/**
 * dietgpu wrapper — implements ncclCompress / ncclDecompress / ncclDecompressReduce
 * backed by Meta's DietGPU library.
 *
 * Provides drop-in replacement for coccl's native compress.cc, using
 * DietGPU's float compression (floatCompress / floatDecompress) instead of the
 * custom minmax-uint8 kernel.
 *
 * Compile (example):
 *   DIETGPU_DIR=~/compression_kernels/dietgpu
 *   nvcc -shared -O3 -lnccl -ldl -ldietgpu                     \
 *        -I${DIETGPU_DIR} -I${DIETGPU_DIR}/dietgpu              \
 *        -o libdietgpu_wrapper.so dietgpu.cu
 *
 * LD_PRELOAD usage (overrides coccl's built-in compression):
 *   LD_PRELOAD=./libdietgpu_wrapper.so ./your_app
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <new>
#include <vector>

#include <dietgpu/float/GpuFloatCodec.h>
#include <dietgpu/ans/GpuANSCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>

#include "dietgpu_common.h"

/*===========================================================================*
 *  ncclCompress — compress using DietGPU float compression                 *
 *                                                                           *
 *  Maps each of `numChunks` independent segments (each `orgChunkCount`      *
 *  elements of type `orgType`) through DietGPU's floatCompress batch API.   *
 *===========================================================================*/

extern "C" ncclResult_t ncclCompress(const void* orgbuff,
    size_t orgChunkCount, ncclDataType_t orgType,
    void** compbuff, size_t* compChunkCount, ncclDataType_t* compType,
    size_t numChunks, cudaStream_t stream) {

  dietgpu::FloatType ft = ncclTypeToFloat(orgType);
  if (ft == dietgpu::FloatType::kUndefined)
    return ncclInternalError;

  int dev;
  CUDACHECK(cudaGetDevice(&dev));

  /* output is a raw byte stream */
  *compType = ncclUint8;

  /* max bytes needed per chunk */
  uint32_t maxCompSize = dietgpu::getMaxFloatCompressedSize(ft,
                               static_cast<uint32_t>(orgChunkCount));
  if (maxCompSize == 0)
    return ncclInternalError;

  *compChunkCount = maxCompSize;

  /* allocate output on first call */
  if (*compbuff == nullptr) {
    size_t total = maxCompSize * numChunks;
    CUDACHECK(cudaMallocAsync(compbuff, total, stream));
  }

  /* build batch descriptor arrays (host side) */
  size_t elemSize = ncclTypeSize(orgType);

  std::vector<const void*> inPtrs(numChunks);
  std::vector<uint32_t>   inSizes(numChunks);
  std::vector<void*>      outPtrs(numChunks);

  for (size_t i = 0; i < numChunks; ++i) {
    inPtrs[i]  = static_cast<const char*>(orgbuff) + i * orgChunkCount * elemSize;
    inSizes[i] = static_cast<uint32_t>(orgChunkCount);
    outPtrs[i] = static_cast<char*>(*compbuff) + i * maxCompSize;
  }

  /* device-side output-size array */
  uint32_t* outSizeDev = nullptr;
  CUDACHECK(cudaMallocAsync(&outSizeDev, numChunks * sizeof(uint32_t), stream));

  /* run compression */
  {
    dietgpu::StackDeviceMemory mem(dev, dietgpu::kDefaultStackSize);
    dietgpu::FloatCompressConfig cfg(ft,
        dietgpu::ANSCodecConfig(dietgpu::kANSDefaultProbBits, false),
        false,   /* is16ByteAligned */
        false);  /* useChecksum */

    dietgpu::floatCompress(mem, cfg,
        static_cast<uint32_t>(numChunks),
        inPtrs.data(), inSizes.data(),
        outPtrs.data(), outSizeDev,
        stream);
  }

  CUDACHECK(cudaFreeAsync(outSizeDev, stream));
  return ncclSuccess;
}

/*===========================================================================*
 *  ncclDecompress — decompress using DietGPU float decompression           *
 *===========================================================================*/

extern "C" ncclResult_t ncclDecompress(const void* compbuff,
    size_t compChunkCount, ncclDataType_t compType,
    void* decompbuff, size_t decompChunkCount, ncclDataType_t decompType,
    size_t numChunks, cudaStream_t stream) {

  (void)compType; /* always ncclUint8 from our compressor */

  dietgpu::FloatType ft = ncclTypeToFloat(decompType);
  if (ft == dietgpu::FloatType::kUndefined)
    return ncclInternalError;

  int dev;
  CUDACHECK(cudaGetDevice(&dev));

  size_t elemSize = ncclTypeSize(decompType);

  std::vector<const void*> inPtrs(numChunks);
  std::vector<void*>      outPtrs(numChunks);
  std::vector<uint32_t>   outCapacities(numChunks);

  for (size_t i = 0; i < numChunks; ++i) {
    inPtrs[i]        = static_cast<const char*>(compbuff) + i * compChunkCount;
    outPtrs[i]       = static_cast<char*>(decompbuff) + i * decompChunkCount * elemSize;
    outCapacities[i] = static_cast<uint32_t>(decompChunkCount);
  }

  {
    dietgpu::StackDeviceMemory mem(dev, dietgpu::kDefaultStackSize);
    dietgpu::FloatDecompressConfig cfg(ft,
        dietgpu::ANSCodecConfig(dietgpu::kANSDefaultProbBits, false),
        false,   /* is16ByteAligned */
        false);  /* useChecksum */

    dietgpu::floatDecompress(mem, cfg,
        static_cast<uint32_t>(numChunks),
        inPtrs.data(), outPtrs.data(), outCapacities.data(),
        nullptr,  /* outSuccess_dev */
        nullptr,  /* outSize_dev */
        stream);
  }

  return ncclSuccess;
}

/*===========================================================================*
 *  ncclDecompressReduce — decompress, then element-wise add with reduceInput
 *                                                                           *
 *  DietGPU has no fused decompress+reduce kernel, so we decompress into     *
 *  a temporary buffer first, then launch a simple add kernel.               *
 *===========================================================================*/

extern "C" ncclResult_t ncclDecompressReduce(const void* compbuff,
    size_t compChunkCount, ncclDataType_t compType,
    const void* reduceInput, void* output,
    size_t outputChunkCount, ncclDataType_t outputType,
    size_t numChunks, cudaStream_t stream) {

  (void)compType;

  int dev;
  CUDACHECK(cudaGetDevice(&dev));

  size_t elemSize = ncclTypeSize(outputType);
  size_t chunkBytes = outputChunkCount * elemSize;
  size_t totalBytes = chunkBytes * numChunks;

  /* temporary buffer for decompressed output */
  void* temp = nullptr;
  CUDACHECK(cudaMallocAsync(&temp, totalBytes, stream));

  /* step 1 — decompress into temp */
  ncclDecompress(compbuff, compChunkCount, ncclUint8,
                 temp, outputChunkCount, outputType,
                 numChunks, stream);

  /* step 2 — element-wise add (decompressed + reduceInput -> output) */
  {
    constexpr int blockSize = 256;
    int gridSize;

    if (outputType == ncclFloat32) {
      size_t totalElems = outputChunkCount * numChunks;
      gridSize = static_cast<int>((totalElems + blockSize - 1) / blockSize);
      gridSize = min(gridSize, 65535);
      addKernel<float><<<gridSize, blockSize, 0, stream>>>(
          static_cast<const float*>(temp),
          static_cast<const float*>(reduceInput),
          static_cast<float*>(output),
          totalElems);
    } else if (outputType == ncclFloat16) {
      size_t totalElems = outputChunkCount * numChunks;
      gridSize = static_cast<int>((totalElems + blockSize - 1) / blockSize);
      gridSize = min(gridSize, 65535);
      addKernel<__half><<<gridSize, blockSize, 0, stream>>>(
          static_cast<const __half*>(temp),
          static_cast<const __half*>(reduceInput),
          static_cast<__half*>(output),
          totalElems);
    } else {
      cudaFreeAsync(temp, stream);
      return ncclInternalError;
    }
  }

  CUDACHECK(cudaFreeAsync(temp, stream));
  return ncclSuccess;
}
