/**
 * sdp4bit.cu — LD_PRELOAD wrapper: replace COCCL's native compression
 *              with SDP4Bit (DeepSpeed) int4 quantization.
 *
 * Per-chunk compressed layout (GPU):
 *   [ChunkHeader: 16 bytes]
 *   [scales: num_groups * sizeof(float)]
 *   [quantized_data: num_groups * (elems_per_group * num_bits / 8)]
 *
 * Build:
 *   SDP4BIT_DIR=~/SDP4Bit/tools/jet_quant_cuda
 *   TORCH_INC=$(python -c 'import torch; ...')
 *
 *   nvcc -shared -O3 -lnccl -std=c++17                           \
 *        -I${SDP4BIT_DIR}/includes -I${TORCH_INC}                \
 *        -o libsdp4bit_wrapper.so sdp4bit.cu                     \
 *        ${SDP4BIT_DIR}/quantization/stochastic_quantize.cu      \
 *        ${SDP4BIT_DIR}/quantization/dequantize.cu               \
 *        ${SDP4BIT_DIR}/quantization/quant_reduce.cu
 *
 * Environment:
 *   SDP4BIT_BITS       quantization bits — 4 or 8  (default: 4)
 *   SDP4BIT_GROUP      elements per group          (default: 256)
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdlib>
#include <cstdint>

/*===========================================================================*
 *  SDP4Bit kernel declarations (no ATen dependency)                          *
 *===========================================================================*/
namespace quantize {
enum class Type { Symmetric, Asymmetric };
}

void launch_quant(int8_t* output_data, float* params,
    const float* input_data, int groups, int elems_per_group,
    int num_bits, quantize::Type quant_type, cudaStream_t stream);

void launch_quant(int8_t* output_data, float* params,
    const __half* input_data, int groups, int elems_per_group,
    int num_bits, quantize::Type quant_type, cudaStream_t stream);

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
    const int8_t* q_data, const float* q_params,
    quantize::Type q_type, int num_bits,
    int elems_per_group, int64_t total_elems,
    cudaStream_t stream);

/*===========================================================================*
 *  SDP4Bit compressed-data layout                                           *
 *===========================================================================*/
struct ChunkHeader {
  int num_groups;
  int num_bits;
  int elems_per_group;
  int quant_type;  // 0 = Symmetric
};

static constexpr int  kDefaultBits  = 4;
static constexpr int  kDefaultGroup = 256;

static int getEnvBits() {
  const char* e = std::getenv("SDP4BIT_BITS");
  if (!e) return kDefaultBits;
  int v = std::atoi(e);
  return (v == 4 || v == 8) ? v : kDefaultBits;
}

static int getEnvGroup() {
  const char* e = std::getenv("SDP4BIT_GROUP");
  if (!e) return kDefaultGroup;
  int v = std::atoi(e);
  return (v > 0 && v <= 4096) ? v : kDefaultGroup;
}

static inline int numGroups(size_t numElems, int elemsPerGroup) {
  return static_cast<int>((numElems + elemsPerGroup - 1) / elemsPerGroup);
}

static inline size_t packedSize(int numElems, int numBits) {
  return (static_cast<size_t>(numElems) * numBits + 7) / 8;
}

/*===========================================================================*
 *  拉入 coccl wrapper 模板 — 提供 4 个入口函数                              *
 *  下面需要实现:  maxCompSize / compressOne / decompressOne                  *
 *===========================================================================*/
#include "../coccl_wrapper_template.cu"

/*===========================================================================*
 *  Callback 1 — 最大压缩后字节数                                             *
 *===========================================================================*/
static size_t maxCompSize(size_t numElems, ncclDataType_t type) {
  (void)type;
  int bits  = getEnvBits();
  int group = getEnvGroup();
  int gps   = numGroups(numElems, group);
  return sizeof(ChunkHeader)
       + gps * sizeof(float)
       + packedSize(gps * group, bits);
}

/*===========================================================================*
 *  Callback 2 — 压缩一个 chunk                                              *
 *===========================================================================*/
static ncclResult_t compressOne(const void* src, void* dst,
    size_t numElems, ncclDataType_t type, cudaStream_t stream) {

  if (type != ncclFloat32 && type != ncclFloat16)
    return ncclInternalError;

  int bits  = getEnvBits();
  int group = getEnvGroup();
  int gps   = numGroups(numElems, group);

  ChunkHeader hdr;
  hdr.num_groups      = gps;
  hdr.num_bits        = bits;
  hdr.elems_per_group = group;
  hdr.quant_type      = 0;
  cudaMemcpyAsync(dst, &hdr, sizeof(hdr), cudaMemcpyHostToDevice, stream);

  float*  scales = reinterpret_cast<float*>(
      static_cast<char*>(dst) + sizeof(ChunkHeader));
  int8_t* qdata  = reinterpret_cast<int8_t*>(
      static_cast<char*>(dst) + sizeof(ChunkHeader) + gps * sizeof(float));

  quantize::Type qtype = quantize::Type::Symmetric;

  if (type == ncclFloat32) {
    launch_quant(qdata, scales,
        static_cast<const float*>(src),
        gps, group, bits, qtype, stream);
  } else {
    launch_quant(qdata, scales,
        static_cast<const __half*>(src),
        gps, group, bits, qtype, stream);
  }
  return ncclSuccess;
}

/*===========================================================================*
 *  Callback 3 — 解压缩一个 chunk                                            *
 *===========================================================================*/
static ncclResult_t decompressOne(const void* src, void* dst,
    size_t numElems, ncclDataType_t type, cudaStream_t stream) {

  if (type != ncclFloat32 && type != ncclFloat16)
    return ncclInternalError;

  ChunkHeader hdr;
  cudaMemcpyAsync(&hdr, src, sizeof(hdr), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int bits  = hdr.num_bits;
  int group = hdr.elems_per_group;
  int gps   = hdr.num_groups;
  int64_t total = static_cast<int64_t>(numElems);

  const float*  scales = reinterpret_cast<const float*>(
      static_cast<const char*>(src) + sizeof(ChunkHeader));
  const int8_t* qdata  = reinterpret_cast<const int8_t*>(
      static_cast<const char*>(src) + sizeof(ChunkHeader) + gps * sizeof(float));

  quantize::Type qtype = quantize::Type::Symmetric;

  if (type == ncclFloat32) {
    launch_dequantize_kernel<float>(
        static_cast<float*>(dst),
        qdata, const_cast<float*>(scales),
        qtype, bits, group, total, stream);
  } else {
    launch_dequantize_kernel<__half>(
        static_cast<__half*>(dst),
        qdata, const_cast<float*>(scales),
        qtype, bits, group, total, stream);
  }
  return ncclSuccess;
}
