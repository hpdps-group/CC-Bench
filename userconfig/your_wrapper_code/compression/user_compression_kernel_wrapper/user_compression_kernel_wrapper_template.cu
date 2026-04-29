/**
 * user_compression_kernel_wrapper_template.cu
 *
 *  ██  指  导  文  档  ████████████████████████████████████████████████████
 *
 *  这个文件是你【自己实现压缩算法】的入口模板。
 *  把里面的三个函数换成你自己的压缩库调用，编译成 .so，
 *  通过 LD_PRELOAD 挂载到 coccl 上层，即可劫持压缩通路。
 *
 *  三个函数要做什么：
 *
 *    1. compress       —  把 GPU buffer 压缩成更小的表示
 *    2. decompress     —  逆向操作，解回原始数据
 *    3. decompress+add —  解压后立刻与另一个 buffer 逐元素相加
 *                         （用于 AllReduce 的归约阶段）
 *
 *  你的压缩库可能有自己的 API 命名（compress/decompress/encode/decode 等），
 *  没关系——你在函数体里调你的，export 出的符号名保持下面这三个就好。
 *
 *  ██  接  口  约  定  ████████████████████████████████████████████████████
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │  · 所有函数 extern "C"（C/C++ 混合调用需要）                        │
 *   │  · 参数里的 buffer 都是 device pointer                              │
 *   │  · 返回 ncclResult_t（ncclSuccess / ncclInternalError 等）          │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 *  compress(in, inCount, inType, &outBuf, &outCount, &outType, nChunks, stream)
 *    ├─ in:       待压缩数据（device）
 *    ├─ inCount:  每个 chunk 的元素个数
 *    ├─ inType:   元素类型（ncclFloat32 / ncclFloat16 / ncclBFloat16）
 *    ├─ outBuf:   输出 buffer——首次传 nullptr，由你负责 cudaMalloc
 *    ├─ outCount: [输出] 每个 chunk 压缩后的"元素个数"（上层用它算 stride）
 *    ├─ outType:  [输出] 压缩后的数据类型（通常设 ncclUint8）
 *    ├─ nChunks:  把 in 切成多少段独立压缩
 *    └─ stream:   所有操作提交到这个 CUDA stream
 *
 *  decompress(in, inCount, inType, out, outCount, outType, nChunks, stream)
 *    └─ 与 compress 对称。inCount/inType 由 compress 写入，你按同样 stride 读取
 *
 *  (and, if you need):
 *  decompress_reduce(in, ..., reduceInput, output, ...)
 *    ├─ 解压后立即与 reduceInput 做 element-wise 加法，写入 output
 *    ├─ 用于 ReduceScatter 的"本地归约"步骤
 *    └─ 如果你的库没有 fused kernel，就先解压到临时 buffer 再加
 *
 *   for a complete example , please see wrappers/compression/nccl/dietgpu.cu
 * 
 *
 *  ██  编  译  方  式  ████████████████████████████████████████████████████
 *
 *  nvcc -shared -O3 -lnccl -ldl                                    \
 *       -I/path/to/your/compression/library/include                \
 *       -o libmy_compression_wrapper.so                            \
 *       user_compression_kernel_wrapper_template.cu
 *
 *  LD_PRELOAD=./libmy_compression_wrapper.so ./your_app
 *
 *  ██████████████████████████████████████████████████████████████████████
 */

/*===========================================================================*
 *  Step 0 — include 你自己的压缩库头文件                                    *
 *                                                                           *
 *  示例:                                                                    *
 *    #include "my_compression/encoder.h"                                    *
 *    #include "my_compression/decoder.h"                                    *
 *                                                                           *
 *  也可以用已有的 GPU 库，比如 dietgpu:                                    *
 *    #include <dietgpu/float/GpuFloatCodec.h>                               *
 *    #include <dietgpu/utils/StackDeviceMemory.h>                           *
 *                                                                           *
 *  必需的 system header（已提供）：                                         *
 *    <cuda_runtime.h>   — cudaMallocAsync, cudaStream_t 等                  *
 *    <nccl.h>           — ncclResult_t, ncclDataType_t 等                   *
 *===========================================================================*/

// #include "your_compression_library.h"

#include <cuda_runtime.h>
#include <nccl.h>

/*===========================================================================*
 *  Step 1 — 你自定义的辅助函数                                               *
 *                                                                           *
 *  完全取决于你的库需要什么，这里只是举例：                                  *
 *    - 类型尺寸映射                                                         *
 *    - 压缩后最大尺寸预估                                                   *
 *    - 自定义 CUDA kernel 声明                                              *
 *                                                                           *
 *  这些函数纯粹供你内部使用，没有固定接口约束。                             *
 *===========================================================================*/

/**
 * 示例：把 ncclDataType_t 映射成你库里的类型 ID（如果你的库需要）
 */
static int your_type_to_enum(ncclDataType_t t) {
  switch (t) {
    case ncclFloat32:  return 0;  /* 改成你库里的常量 */
    case ncclFloat16:  return 1;
    case ncclBFloat16: return 2;
    default:           return -1;
  }
}

/**
 * 示例：元素字节数（遍历 chunk 时算地址偏移用）
 */
static size_t your_type_size(ncclDataType_t t) {
  switch (t) {
    case ncclFloat32: return 4;
    case ncclFloat16:
    case ncclBFloat16: return 2;
    default:           return 4;
  }
}

/**
 * 示例：预估压缩后的最大字节数
 *
 * 你的库很可能有 getMaxCompressedSize() 之类的函数。
 * 这里包装一下，供上层的 compress 函数分配输出 buffer 时使用。
 */
static size_t your_max_compressed_size(/* 你的库需要的参数 */) {
  /* TODO: return your_library_max_compressed_size(...); */
  return 0;  // 改成你的实现
}

/*===========================================================================*
 *  Step 2 — 你自己的 CUDA kernel（如果需要）                                 *
 *                                                                           *
 *  第三个函数 decompress_reduce 在解压后需要做"逐元素加法"。                 *
 *  如果你的库有 fused kernel，直接调用即可，不需要自己写 kernel。            *
 *  如果没有，这里提供了一个简单的 add kernel 示例。                         *
 *===========================================================================*/

/**
 * 示例：逐元素加法 kernel
 */
template <typename T>
__global__ void your_add_kernel(const T* a, const T* b, T* out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

/*===========================================================================*
 *  Step 3 — 实现你的压缩函数                                               *
 *                                                                           *
 *  你要做的事：                                                              *
 *    1. 把 in 按 nChunks 分段（每段 inCount 个元素）                        *
 *    2. 对每一段调用你的压缩库                                              *
 *    3. 把压缩结果写入 outBuf                                              *
 *    4. 通过 outCount 告知上层每段压缩后的尺寸（用于 stride 计算）          *
 *===========================================================================*/

extern "C" ncclResult_t the_compress_func_you_want_to_hijak(
    const void* in,
    size_t      inCount,
    ncclDataType_t inType,
    void**      outBuf,
    size_t*     outCount,
    ncclDataType_t* outType,
    size_t      nChunks,
    cudaStream_t   stream)
{
  /* ---- YOUR CODE HERE ---- */

  /*
   * 伪代码模板：
   *
   * // 1. 确定输出类型（压缩后通常是裸字节流，设为 ncclUint8）
   * *outType = ncclUint8;
   *
   * // 2. 计算每个 chunk 压缩后的最大尺寸
   * size_t maxChunkBytes = your_max_compressed_size(...);
   * *outCount = maxChunkBytes;
   *
   * // 3. 首次调用时 outBuf == nullptr，由你分配
   * if (*outBuf == nullptr) {
   *     size_t total = maxChunkBytes * nChunks;
   *     cudaMallocAsync(outBuf, total, stream);
   * }
   *
   * // 4. 拆 chunk 并调用你的压缩库
   * size_t elemSize = your_type_size(inType);
   * for (size_t i = 0; i < nChunks; ++i) {
   *     char* chunk_in  = (char*)in + i * inCount * elemSize;
   *     char* chunk_out = (char*)(*outBuf) + i * maxChunkBytes;
   *     // your_compress(chunk_in, inCount, chunk_out, &actualSize, stream);
   * }
   *
   * // 5. 同步检查（可选）
   * cudaGetLastError();
   * return ncclSuccess;
   */

  (void)in;
  (void)inCount;
  (void)inType;
  (void)outBuf;
  (void)outCount;
  (void)outType;
  (void)nChunks;
  (void)stream;

  /* 删掉这行再开始写代码——这只是让编译器闭嘴 */
  return ncclInternalError;
}

/*===========================================================================*
 *  Step 4 — 实现你的解压缩函数                                              *
 *                                                                           *
 *  把 Step 3 压缩的数据解回原始类型。                                       *
 *  注意 outCount 是 Step 3 输出的每 chunk 压缩后字节数，                    *
 *  你按同样 stride 读取每个 chunk 即可。                                    *
 *===========================================================================*/

extern "C" ncclResult_t the_decompress_func_you_want_to_hijak(
    const void* in,
    size_t      inCount,
    ncclDataType_t inType,
    void*       out,
    size_t      outCount,
    ncclDataType_t outType,
    size_t      nChunks,
    cudaStream_t   stream)
{
  /* ---- YOUR CODE HERE ---- */

  /*
   * 伪代码模板：
   *
   * size_t elemSize = your_type_size(outType);
   * for (size_t i = 0; i < nChunks; ++i) {
   *     const char* chunk_in  = (const char*)in + i * inCount;
   *     char*       chunk_out = (char*)out + i * outCount * elemSize;
   *     // your_decompress(chunk_in, inCount, chunk_out, outCount, stream);
   * }
   * return ncclSuccess;
   */

  (void)in;
  (void)inCount;
  (void)inType;
  (void)out;
  (void)outCount;
  (void)outType;
  (void)nChunks;
  (void)stream;

  return ncclInternalError;
}

/*===========================================================================*
 *  Step 5 — 实现解压 + 归约                                                 *
 *                                                                           *
 *  解压后与 reduceInput 逐元素相加，写入 output。                           *
 *                                                                           *
 *  如果你的库有 fused kernel，直接调：                                      *
 *      your_fused_decompress_and_reduce(...);                              *
 *                                                                           *
 *  否则两步走：解压到 temp → launch add kernel。                           *
 *===========================================================================*/

extern "C" ncclResult_t the_decompress_reduce_func_you_want_to_hijak(
    const void* in,
    size_t      inCount,
    ncclDataType_t inType,
    const void* reduceInput,
    void*       output,
    size_t      outputCount,
    ncclDataType_t outputType,
    size_t      nChunks,
    cudaStream_t   stream)
{
  /* ---- YOUR CODE HERE ---- */

  /*
   * 伪代码模板（两步法）：
   *
   * // A. 分配临时 buffer
   * size_t elemSize   = your_type_size(outputType);
   * size_t totalBytes = outputCount * elemSize * nChunks;
   * void* temp = nullptr;
   * cudaMallocAsync(&temp, totalBytes, stream);
   *
   * // B. 解压到 temp（复用上面的 decompress 函数）
   * the_decompress_func_you_want_to_hijak(
   *     in, inCount, inType,
   *     temp, outputCount, outputType,
   *     nChunks, stream);
   *
   * // C. 逐元素相加
   * size_t totalElems = outputCount * nChunks;
   * int block = 256;
   * int grid  = (totalElems + block - 1) / block;
   * your_add_kernel<<<grid, block, 0, stream>>>(
   *     (const float*)temp, (const float*)reduceInput,
   *     (float*)output, totalElems);
   *
   * // D. 清理
   * cudaFreeAsync(temp, stream);
   * return ncclSuccess;
   */

  (void)in;
  (void)inCount;
  (void)inType;
  (void)reduceInput;
  (void)output;
  (void)outputCount;
  (void)outputType;
  (void)nChunks;
  (void)stream;

  return ncclInternalError;
}