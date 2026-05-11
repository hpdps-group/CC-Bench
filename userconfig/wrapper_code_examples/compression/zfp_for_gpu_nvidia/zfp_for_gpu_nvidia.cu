/**
 * zfp_for_gpu_nvidia.cu — LD_PRELOAD wrapper: replace COCCL's native
 *                          compression with NVIDIA GPU–accelerated ZFP.
 *
 * Uses cuZFP (cuda_compress / cuda_decompress) — bitstream points directly
 * at GPU memory, no host ping-pong copies.
 *
 * Per-chunk compressed layout (GPU output buffer):
 *   [0..3]   uint32_t  zfp_stream_size  — actual ZFP bitstream bytes
 *   [4..]    uint8_t[] ZFP bitstream    — written by cuda_compress
 *
 * Build:
 *   ZFP_DIR=/path/to/zfp
 *   nvcc -shared -O3 -lnccl -lnvToolsExt -std=c++17              \
 *        -I${ZFP_DIR}/include -I${ZFP_DIR}/src/cuda_zfp          \
 *        -o libzfp_for_gpu_nvidia.so zfp_for_gpu_nvidia.cu        \
 *        -L${ZFP_DIR}/lib -lzfp
 *
 *   NOTE: zfp must be built WITH CUDA support.
 *   (cmake -DBUILD_CUDA=ON ..)
 *
 * Environment:
 *   ZFP_PRECISION     fixed-precision bit count (default: 14)
 *   ZFP_VERBOSE       print debug info if set
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>

#include <zfp.h>
#include <zfp/bitstream.h>
#include <cuZFP.h>

/*-------------------------------------------------------------------------*
 *  ZFP-specific compressed-data layout                                    *
 *-------------------------------------------------------------------------*/
struct ChunkHead {
  uint32_t stream_bytes;   /* actual ZFP bitstream byte count */
};
static constexpr size_t kHeadBytes = sizeof(ChunkHead);

static constexpr int kDefaultPrecision = 14;

static int getZfpPrecision() {
  const char* env = std::getenv("ZFP_PRECISION");
  if (!env) return kDefaultPrecision;
  int p = std::atoi(env);
  return (p > 0 && p <= 64) ? p : kDefaultPrecision;
}

static zfp_type ncclTypeToZfp(ncclDataType_t t) {
  switch (t) {
    case ncclFloat32: return zfp_type_float;
    case ncclFloat64: return zfp_type_double;
    case ncclInt32:   return zfp_type_int32;
    case ncclInt64:   return zfp_type_int64;
    default:          return zfp_type_none;
  }
}

static zfp_stream* makeZfpStream() {
  zfp_stream* zstr = zfp_stream_open(nullptr);
  if (!zstr) return nullptr;
  zfp_stream_set_execution(zstr, zfp_exec_cuda);
  zfp_stream_set_precision(zstr, static_cast<uint>(getZfpPrecision()));
  return zstr;
}

/*-------------------------------------------------------------------------*
 *  Pull in coccl wrapper template — 提供 4 个入口函数                      *
 *  下面需实现:  maxCompSize / compressOne / decompressOne                 *
 *-------------------------------------------------------------------------*/
#include "../coccl_wrapper_template.cu"

/*===========================================================================*
 *  Callback 1 — 最大压缩后字节数（含 kHeadBytes 头部）                      *
 *===========================================================================*/
static size_t maxCompSize(size_t numElems, ncclDataType_t type) {
  zfp_type zt = ncclTypeToZfp(type);
  if (zt == zfp_type_none) return 0;

  zfp_stream* zstr = makeZfpStream();
  if (!zstr) return 0;

  zfp_field* f = zfp_field_1d(nullptr, zt, numElems);
  if (!f) { zfp_stream_close(zstr); return 0; }

  size_t maxBytes = zfp_stream_maximum_size(zstr, f);
  zfp_field_free(f);
  zfp_stream_close(zstr);

  return kHeadBytes + maxBytes;
}

/*===========================================================================*
 *  Callback 2 — 压缩一个 chunk                                              *
 *===========================================================================*/
static ncclResult_t compressOne(const void* src, void* dst,
    size_t numElems, ncclDataType_t type, cudaStream_t stream) {

  zfp_type zt = ncclTypeToZfp(type);
  if (zt == zfp_type_none) return ncclInternalError;

  zfp_stream* zstr = makeZfpStream();
  if (!zstr) return ncclInternalError;

  /* Query max bitstream bytes for this config + size */
  zfp_field* tmpf = zfp_field_1d(nullptr, zt, numElems);
  if (!tmpf) { zfp_stream_close(zstr); return ncclInternalError; }
  size_t maxZfpBytes = zfp_stream_maximum_size(zstr, tmpf);
  zfp_field_free(tmpf);

  /* field → GPU input data */
  zfp_field* field = zfp_field_1d(
      const_cast<void*>(static_cast<const void*>(src)), zt, numElems);
  if (!field) { zfp_stream_close(zstr); return ncclInternalError; }

  /* bitstream → GPU output (skip header space) */
  bitstream* bs = stream_open(
      static_cast<char*>(dst) + kHeadBytes, maxZfpBytes);
  if (!bs) { zfp_field_free(field); zfp_stream_close(zstr); return ncclInternalError; }
  zfp_stream_set_bit_stream(zstr, bs);
  zfp_stream_rewind(zstr);

  /* cuda_compress — runs entirely on GPU when both field + bitstream are GPU */
  size_t actualBytes = cuda_compress(zstr, field);
  if (actualBytes == 0) {
    stream_close(bs); zfp_field_free(field);
    zfp_stream_close(zstr); return ncclInternalError;
  }

  /* Write header (4 bytes: actual stream size) */
  ChunkHead head;
  head.stream_bytes = static_cast<uint32_t>(actualBytes);
  cudaMemcpyAsync(dst, &head, kHeadBytes, cudaMemcpyHostToDevice, stream);

  stream_close(bs);
  zfp_field_free(field);
  zfp_stream_close(zstr);
  return ncclSuccess;
}

/*===========================================================================*
 *  Callback 3 — 解压缩一个 chunk                                            *
 *===========================================================================*/
static ncclResult_t decompressOne(const void* src, void* dst,
    size_t numElems, ncclDataType_t type, cudaStream_t stream) {

  zfp_type zt = ncclTypeToZfp(type);
  if (zt == zfp_type_none) return ncclInternalError;

  /* Read header from GPU */
  ChunkHead head;
  cudaMemcpyAsync(&head, src, kHeadBytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (head.stream_bytes == 0) return ncclInternalError;

  zfp_stream* zstr = makeZfpStream();
  if (!zstr) return ncclInternalError;

  /* bitstream → directly into GPU compressed data (after header) */
  bitstream* bs = stream_open(
      const_cast<char*>(static_cast<const char*>(src) + kHeadBytes),
      head.stream_bytes);
  if (!bs) { zfp_stream_close(zstr); return ncclInternalError; }
  zfp_stream_set_bit_stream(zstr, bs);
  zfp_stream_rewind(zstr);

  /* field → GPU output buffer */
  zfp_field* field = zfp_field_1d(dst, zt, numElems);
  if (!field) { stream_close(bs); zfp_stream_close(zstr); return ncclInternalError; }

  /* cuda_decompress — all on GPU */
  cuda_decompress(zstr, field);

  stream_close(bs);
  zfp_field_free(field);
  zfp_stream_close(zstr);
  return ncclSuccess;
}
