/**
 * coccl_wrapper_template.cu — Single-file template for writing GPU compression
 *                              wrappers for COCCL-family libraries.
 *
 * === Usage ===
 * 1. #include this file at the top of your backend .cu file
 * 2. Implement the 3 callback functions below (the dispatch will call them):
 *      size_t         maxCompSize(size_t numElems, ncclDataType_t type);
 *      ncclResult_t   compressOne(const void* src, void* dst, ...);
 *      ncclResult_t   decompressOne(const void* src, void* dst, ...);
 * 3. Compile: nvcc -shared -O3 -lnccl -std=c++17 -o libxxx.so xxx.cu
 *    (do NOT compile this file separately; #include brings it in)
 *
 * === Changing exported function names ===
 * If you want to export names other than ncclCompress (e.g., mylib_compress),
 * simply edit the 4 extern "C" function names below.
 * LD_PRELOAD requires symbol names matching whatever the intercepted library
 * uses, so name them accordingly.
 *
 * The callback names (maxCompSize / compressOne / decompressOne) don't need
 * to change; implement them in your own .cu file.
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstddef>
#include <cstdint>

/*-------------------------------------------------------------------------*
 *  COCCL comm operation enum                                              *
 *-------------------------------------------------------------------------*/
#ifdef __cplusplus
enum ncclCommOp {
  AlltoAll = 0, AlltoAll_Inter = 1, AllReduce = 2, AllReduce_Inter = 3,
  AllGather = 4, AllGather_Inter = 5, ReduceScatter = 6, ReduceScatter_Inter = 7,
  SendRecv = 8, SendRecv_BWD = 9
};
typedef ncclCommOp ncclCommOp_t;
#endif

/*-------------------------------------------------------------------------*
 *  Callback forward declarations — implement these in your .cu            *
 *-------------------------------------------------------------------------*/
static size_t      maxCompSize(size_t numElems, ncclDataType_t type);
static ncclResult_t compressOne(const void* src, void* dst,
                                size_t numElems, ncclDataType_t type,
                                cudaStream_t stream);
static ncclResult_t decompressOne(const void* src, void* dst,
                                  size_t numElems, ncclDataType_t type,
                                  cudaStream_t stream);

/*=========================================================================*
 *  4 entry-point functions — rename them here                             *
 *                                                                         *
 *  Default names: ncclCompress / ncclDecompress / ncclDecompressReduce /   *
 *                 ncclDecompReduceComp                                     *
 *  Suitable for LD_PRELOAD interception of coccl.                          *
 *  Change to other names (e.g., my_compress) for direct linking.          *
 *=========================================================================*/

/*─────────────────────────────────────────────────────────────────────────*
 *  1. Compress                                                          *
 *─────────────────────────────────────────────────────────────────────────*/
extern "C" ncclResult_t ncclCompress(
    const void* orgbuff, void** compbuff,
    const size_t orgChunkCount, ncclDataType_t orgDatatype,
    size_t* compChunkCount, ncclDataType_t* compDatatype,
    const size_t numChunks, const int rank, ncclCommOp_t commOp,
    cudaStream_t stream)
{
  (void)rank;
  (void)commOp;

  size_t perChunk = maxCompSize(orgChunkCount, orgDatatype);
  if (perChunk == 0) return ncclInternalError;

  *compDatatype  = ncclUint8;
  *compChunkCount = perChunk;

  if (*compbuff == nullptr) {
    cudaError_t e = cudaMallocAsync(compbuff, perChunk * numChunks, stream);
    if (e != cudaSuccess) return ncclInternalError;
  }

  size_t elemSize = 0;
  switch (orgDatatype) {
    case ncclInt8: case ncclUint8:    elemSize = 1; break;
    case ncclFloat16: case ncclBFloat16: elemSize = 2; break;
    case ncclFloat32: case ncclInt32: elemSize = 4; break;
    case ncclFloat64: case ncclInt64: elemSize = 8; break;
    default: return ncclInternalError;
  }

  size_t srcStride = orgChunkCount * elemSize;
  for (size_t c = 0; c < numChunks; c++) {
    ncclResult_t r = compressOne(
        static_cast<const char*>(orgbuff) + c * srcStride,
        static_cast<char*>(*compbuff) + c * perChunk,
        orgChunkCount, orgDatatype, stream);
    if (r != ncclSuccess) return r;
  }
  return ncclSuccess;
}

/*─────────────────────────────────────────────────────────────────────────*
 *  2. Decompress                                                        *
 *─────────────────────────────────────────────────────────────────────────*/
extern "C" ncclResult_t ncclDecompress(
    void* decompbuff, const void* compbuff,
    const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype,
    const size_t numChunks, ncclCommOp_t commOp,
    cudaStream_t stream)
{
  (void)compDatatype;
  (void)commOp;

  size_t elemSize = 0;
  switch (decompDatatype) {
    case ncclInt8: case ncclUint8:    elemSize = 1; break;
    case ncclFloat16: case ncclBFloat16: elemSize = 2; break;
    case ncclFloat32: case ncclInt32: elemSize = 4; break;
    case ncclFloat64: case ncclInt64: elemSize = 8; break;
    default: return ncclInternalError;
  }

  size_t dstStride = decompChunkCount * elemSize;
  for (size_t c = 0; c < numChunks; c++) {
    ncclResult_t r = decompressOne(
        static_cast<const char*>(compbuff) + c * compChunkCount,
        static_cast<char*>(decompbuff) + c * dstStride,
        decompChunkCount, decompDatatype, stream);
    if (r != ncclSuccess) return r;
  }
  return ncclSuccess;
}

/*─────────────────────────────────────────────────────────────────────────*
 *  3. Decompress + Reduce (ring reduce receiver side)                   *
 *─────────────────────────────────────────────────────────────────────────*/
extern "C" ncclResult_t ncclDecompressReduce(
    void* reducebuff, const void* compbuff,
    const size_t compChunkCount, ncclDataType_t compDatatype,
    const size_t reduceChunkCount, ncclDataType_t reduceDataType,
    const size_t numChunks, ncclCommOp_t commOp,
    cudaStream_t stream)
{
  (void)compDatatype;
  (void)commOp;

  size_t elemSize = 0;
  switch (reduceDataType) {
    case ncclFloat32: elemSize = 4; break;
    case ncclFloat64: elemSize = 8; break;
    case ncclFloat16: elemSize = 2; break;
    default: return ncclInternalError;
  }

  size_t chunkBytes = reduceChunkCount * elemSize;

  void* temp = nullptr;
  if (cudaMallocAsync(&temp, chunkBytes, stream) != cudaSuccess)
    return ncclInternalError;

  for (size_t c = 0; c < numChunks; c++) {
    ncclResult_t r = decompressOne(
        static_cast<const char*>(compbuff) + c * compChunkCount,
        temp, reduceChunkCount, reduceDataType, stream);
    if (r != ncclSuccess) { cudaFreeAsync(temp, stream); return r; }

    add_kernel(reduceDataType,
               temp,
               static_cast<const char*>(reducebuff) + c * chunkBytes,
               static_cast<char*>(reducebuff) + c * chunkBytes,
               reduceChunkCount, stream);
  }

  cudaFreeAsync(temp, stream);
  return ncclSuccess;
}

/*─────────────────────────────────────────────────────────────────────────*
 *  4. Decompress → Reduce → Recompress (inter-node for hierarchical allreduce) *
 *─────────────────────────────────────────────────────────────────────────*/
extern "C" ncclResult_t ncclDecompReduceComp(
    const void* compbuff, void** recompbuff,
    const size_t orgChunkCount, ncclDataType_t orgDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype,
    size_t* reCompChunkCount, ncclDataType_t* reCompDatatype,
    const size_t numChunks, ncclCommOp_t commOp,
    cudaStream_t stream)
{
  (void)commOp;
  (void)compDatatype;

  size_t elemSize = 0;
  switch (orgDatatype) {
    case ncclFloat32: elemSize = 4; break;
    case ncclFloat64: elemSize = 8; break;
    case ncclFloat16: elemSize = 2; break;
    default: return ncclInternalError;
  }

  size_t chunkBytes = orgChunkCount * elemSize;

  size_t perChunk = maxCompSize(orgChunkCount, orgDatatype);
  if (perChunk == 0) return ncclInternalError;

  *reCompDatatype  = ncclUint8;
  *reCompChunkCount = perChunk;

  /* temp holds all decompressed chunks */
  void* temp = nullptr;
  if (cudaMallocAsync(&temp, chunkBytes * numChunks, stream) != cudaSuccess)
    return ncclInternalError;

  if (*recompbuff == nullptr) {
    if (cudaMallocAsync(recompbuff, perChunk, stream) != cudaSuccess) {
      cudaFreeAsync(temp, stream);
      return ncclInternalError;
    }
  }

  /* Step 1 — Decompress all chunks */
  for (size_t c = 0; c < numChunks; c++) {
    ncclResult_t r = decompressOne(
        static_cast<const char*>(compbuff) + c * compChunkCount,
        static_cast<char*>(temp) + c * chunkBytes,
        orgChunkCount, orgDatatype, stream);
    if (r != ncclSuccess) { cudaFreeAsync(temp, stream); return r; }
  }

  /* Step 2 — Pairwise reduce into the first chunk */
  for (size_t c = 1; c < numChunks; c++) {
    add_kernel(orgDatatype,
               static_cast<const char*>(temp) + c * chunkBytes,
               static_cast<const char*>(temp),
               temp,
               orgChunkCount, stream);
  }

  /* Step 3 — Compress the reduced result */
  {
    ncclResult_t r = compressOne(temp, *recompbuff,
                                 orgChunkCount, orgDatatype, stream);
    if (r != ncclSuccess) { cudaFreeAsync(temp, stream); return r; }
  }

  cudaFreeAsync(temp, stream);
  return ncclSuccess;
}

/*=========================================================================*
 *  Utility functions — no need to modify                                  *
 *=========================================================================*/

/*── add_kernel ─────────────────────────────────────────────────────────*/
__global__ void add_f32_kernel(const float* a, const float* b,
                                float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void add_f64_kernel(const double* a, const double* b,
                                double* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void add_f16_kernel(const __half* a, const __half* b,
                                __half* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __hadd(a[i], b[i]);
}

static void add_kernel(ncclDataType_t dt,
                       const void* a, const void* b, void* out,
                       size_t n, cudaStream_t stream) {
  constexpr int BS = 256;
  int grid = static_cast<int>((n + BS - 1) / BS);
  if (grid > 65535) grid = 65535;

  switch (dt) {
    case ncclFloat32:
      add_f32_kernel<<<grid, BS, 0, stream>>>(
          static_cast<const float*>(a), static_cast<const float*>(b),
          static_cast<float*>(out), static_cast<int>(n));
      break;
    case ncclFloat64:
      add_f64_kernel<<<grid, BS, 0, stream>>>(
          static_cast<const double*>(a), static_cast<const double*>(b),
          static_cast<double*>(out), static_cast<int>(n));
      break;
    case ncclFloat16:
      add_f16_kernel<<<grid, BS, 0, stream>>>(
          static_cast<const __half*>(a), static_cast<const __half*>(b),
          static_cast<__half*>(out), static_cast<int>(n));
      break;
    default:
      break;
  }
}
