/**
 * plain_nccl_compress.cu — Plain NCCL collectives with built-in GPU compression
 *
 * LD_PRELOAD library that intercepts NCCL calls and wraps them with simple
 * symmetric int8 quantization: compress GPU data before send, decompress after
 * receive.  The quantization kernel and all collective algorithms are
 * self-contained — no external compression library required.
 *
 * Collective implementations (all on-GPU, no CPU data download in hot path):
 *   ncclAllReduce   — ring reduce-scatter + ring allgather
 *   ncclBroadcast   — binomial tree
 *   ncclReduce      — binomial tree (GPU reduction)
 *   ncclAllGather   — ring
 *   ncclReduceScatter — recursive halving
 *   ncclAllToAll    — pairwise exchange
 *
 * Compression format (symmetric int8, per-group):
 *   scales_f32[num_groups] | qdata_i8[num_elems]
 *   group_size defaults to 256, configurable via env var.
 *
 * CCBench integration (edit userconfig/config_in_jsonc/communication_lib_selection.jsonc):
 *
 *   {
 *     "mode": 3,
 *     "source_dirs": ["userconfig/wrapper_code_examples/communication/plain_nccl_compress"],
 *     "include_dirs": [],
 *     "libraries": [],
 *     "output_name": "libplain_nccl_compress.so"
 *   }
 *
 * Then rebuild: ./scripts/build_script.sh --rebuild-bench
 *
 * Usage:
 *   LD_PRELOAD=./libplain_nccl_compress.so <your_nccl_app>
 *
 * Environment variables (all optional):
 *   NCCL_COMPRESS_DISABLE=1     — bypass compression (pass-through to real NCCL)
 *   NCCL_COMPRESS_BITS=8        — quantization bits (only 8 supported currently)
 *   NCCL_COMPRESS_GROUP=256     — elements per quantization group
 *   NCCL_DEBUG=INFO|WARN        — enable debug logging
 */

#define _GNU_SOURCE   /* for RTLD_NEXT from dlfcn.h */

#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

/*===========================================================================*
 * Configuration & helpers                                                    *
 *===========================================================================*/

/* Debug logging */
static int nccl_debug() {
    static int checked = 0, val = 0;
    if (!checked) {
        const char* e = getenv("NCCL_DEBUG");
        if (e && (!strcmp(e, "INFO") || !strcmp(e, "WARN") || !strcmp(e, "TRACE")))
            val = 1;
        checked = 1;
    }
    return val;
}

#define LOG(...) do { \
    if (nccl_debug()) { \
        fprintf(stderr, "[plain_nccl_compress] "); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } \
} while (0)

/* Compression config (one-time read from env) */
struct CompressConfig {
    int enabled;
    int bits;        /* quantization bits (8 only) */
    int group_size;  /* elements per group */
};

static const CompressConfig& get_config() {
    static int inited = 0;
    static CompressConfig cfg;
    if (!inited) {
        cfg.enabled   = 1;
        cfg.bits      = 8;
        cfg.group_size = 256;
        const char* e;
        e = getenv("NCCL_COMPRESS_DISABLE");
        if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y'))
            cfg.enabled = 0;
        e = getenv("NCCL_COMPRESS_BITS");
        if (e) { int v = atoi(e); if (v == 4 || v == 8) cfg.bits = v; }
        e = getenv("NCCL_COMPRESS_GROUP");
        if (e) { int v = atoi(e); if (v > 0 && v <= 4096) cfg.group_size = v; }
        inited = 1;
        LOG("config: enabled=%d bits=%d group_size=%d",
            cfg.enabled, cfg.bits, cfg.group_size);
    }
    return cfg;
}

static inline size_t dtype_bytes(ncclDataType_t dtype) {
    switch (dtype) {
        case ncclInt8:    case ncclUint8:    return 1;
        case ncclFloat16: case ncclBFloat16: return 2;
        case ncclFloat32: case ncclInt32:    return 4;
        case ncclFloat64: case ncclInt64:    return 8;
        default:                             return 4;
    }
}

/* Number of quantization groups for a given element count */
static inline int num_groups(size_t count, int group_size) {
    return (int)((count + group_size - 1) / group_size);
}

/* Compressed buffer size in bytes for a given element count and dtype */
static inline size_t compressed_buf_size(size_t count, ncclDataType_t dtype) {
    const auto& cfg = get_config();
    (void)dtype;
    int ng = num_groups(count, cfg.group_size);
    return (size_t)ng * sizeof(float) + count * ((size_t)cfg.bits / 8);
}

/* Whether this data type is compressible */
static inline int is_compressible_type(ncclDataType_t dtype) {
    return (dtype == ncclFloat32 || dtype == ncclFloat16 || dtype == ncclFloat64);
}

/*===========================================================================*
 * CUDA Quantization / Dequantization Kernels                                 *
 *===========================================================================*/

/* ── float32 → int8 symmetric quantize (tree reduce in shared memory) ── */
__global__ void quantize_f32_i8(const float* __restrict__ input,
                                 int8_t* __restrict__ output,
                                 float* __restrict__ scales,
                                 int num_elems, int group_size) {
    int group = blockIdx.x;
    int tid   = threadIdx.x;
    int idx   = group * group_size + tid;

    extern __shared__ float s_tmp[];

    /* Load |val| */
    s_tmp[tid] = (idx < num_elems) ? fabsf(input[idx]) : 0.0f;
    __syncthreads();

    /* Tree reduction for max */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_tmp[tid] = fmaxf(s_tmp[tid], s_tmp[tid + s]);
        __syncthreads();
    }

    /* Thread 0 computes scale and writes it */
    __shared__ float s_inv_scale;
    if (tid == 0) {
        float m = s_tmp[0];
        s_inv_scale = (m < 1e-10f) ? 1.0f : 127.0f / m;
        scales[group] = s_inv_scale;
    }
    __syncthreads();

    /* Quantize */
    float inv = s_inv_scale;
    if (idx < num_elems) {
        float q = input[idx] * inv;
        q = fmaxf(-128.0f, fminf(127.0f, roundf(q)));
        output[idx] = (int8_t)(q);
    }
}

/* ── int8 → float32 dequantize ── */
__global__ void dequantize_i8_f32(const int8_t* __restrict__ input,
                                   float* __restrict__ output,
                                   const float* __restrict__ scales,
                                   int num_elems, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int group = idx / group_size;
    float inv_scale = scales[group];
    output[idx] = (float)input[idx] * (1.0f / inv_scale);
    /* Note: inv_scale = 127/max, so 1/inv_scale = max/127 */
}

/* ── float16 → int8 quantize ── */
__global__ void quantize_f16_i8(const __half* __restrict__ input,
                                 int8_t* __restrict__ output,
                                 float* __restrict__ scales,
                                 int num_elems, int group_size) {
    int group = blockIdx.x;
    int tid   = threadIdx.x;
    int idx   = group * group_size + tid;

    extern __shared__ float s_tmp[];
    s_tmp[tid] = (idx < num_elems) ? (float)__half2float(__habs(input[idx])) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_tmp[tid] = fmaxf(s_tmp[tid], s_tmp[tid + s]);
        __syncthreads();
    }

    __shared__ float s_inv_scale;
    if (tid == 0) {
        float m = s_tmp[0];
        s_inv_scale = (m < 1e-10f) ? 1.0f : 127.0f / m;
        scales[group] = s_inv_scale;
    }
    __syncthreads();

    float inv = s_inv_scale;
    if (idx < num_elems) {
        float q = __half2float(input[idx]) * inv;
        q = fmaxf(-128.0f, fminf(127.0f, roundf(q)));
        output[idx] = (int8_t)(q);
    }
}

/* ── int8 → float16 dequantize ── */
__global__ void dequantize_i8_f16(const int8_t* __restrict__ input,
                                   __half* __restrict__ output,
                                   const float* __restrict__ scales,
                                   int num_elems, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int group = idx / group_size;
    float inv_scale = scales[group];
    output[idx] = __float2half((float)input[idx] * (1.0f / inv_scale));
}

/* ── float64 → int8 quantize (truncate to f32, quantize) ── */
__global__ void quantize_f64_i8(const double* __restrict__ input,
                                 int8_t* __restrict__ output,
                                 float* __restrict__ scales,
                                 int num_elems, int group_size) {
    int group = blockIdx.x;
    int tid   = threadIdx.x;
    int idx   = group * group_size + tid;

    extern __shared__ float s_tmp[];
    s_tmp[tid] = (idx < num_elems) ? (float)fabs(input[idx]) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_tmp[tid] = fmaxf(s_tmp[tid], s_tmp[tid + s]);
        __syncthreads();
    }

    __shared__ float s_inv_scale;
    if (tid == 0) {
        float m = s_tmp[0];
        s_inv_scale = (m < 1e-10f) ? 1.0f : 127.0f / m;
        scales[group] = s_inv_scale;
    }
    __syncthreads();

    float inv = s_inv_scale;
    if (idx < num_elems) {
        float q = (float)input[idx] * inv;
        q = fmaxf(-128.0f, fminf(127.0f, roundf(q)));
        output[idx] = (int8_t)(q);
    }
}

/* ── int8 → float64 dequantize ── */
__global__ void dequantize_i8_f64(const int8_t* __restrict__ input,
                                   double* __restrict__ output,
                                   const float* __restrict__ scales,
                                   int num_elems, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int group = idx / group_size;
    float inv_scale = scales[group];
    output[idx] = (double)((float)input[idx] * (1.0f / inv_scale));
}

/*===========================================================================*
 * GPU reduction kernels (element-wise)                                       *
 *===========================================================================*/

__global__ void reduce_sum_f32(const float* __restrict__ src,
                                float* __restrict__ dst, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] += src[i];
}

__global__ void reduce_prod_f32(const float* __restrict__ src,
                                 float* __restrict__ dst, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] *= src[i];
}

__global__ void reduce_max_f32(const float* __restrict__ src,
                                float* __restrict__ dst, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] = fmaxf(dst[i], src[i]);
}

__global__ void reduce_min_f32(const float* __restrict__ src,
                                float* __restrict__ dst, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] = fminf(dst[i], src[i]);
}

__global__ void reduce_sum_f16(const __half* __restrict__ src,
                                __half* __restrict__ dst, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] = __hadd(dst[i], src[i]);
}

__global__ void reduce_sum_f64(const double* __restrict__ src,
                                double* __restrict__ dst, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] += src[i];
}

static void launch_reduce(void* dst, const void* src, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op,
                           cudaStream_t stream) {
    constexpr int BS = 256;
    int grid = (int)((count + BS - 1) / BS);
    if (grid > 65535) grid = 65535;

    switch (dtype) {
    case ncclFloat32: {
        float* d = (float*)dst;
        const float* s = (const float*)src;
        switch (op) {
        case ncclSum:  reduce_sum_f32<<<grid, BS, 0, stream>>>(s, d, count); break;
        case ncclProd: reduce_prod_f32<<<grid, BS, 0, stream>>>(s, d, count); break;
        case ncclMax:  reduce_max_f32<<<grid, BS, 0, stream>>>(s, d, count); break;
        case ncclMin:  reduce_min_f32<<<grid, BS, 0, stream>>>(s, d, count); break;
        default: break;
        }
        break;
    }
    case ncclFloat16: {
        __half* d = (__half*)dst;
        const __half* s = (const __half*)src;
        if (op == ncclSum)
            reduce_sum_f16<<<grid, BS, 0, stream>>>(s, d, count);
        break;
    }
    case ncclFloat64: {
        double* d = (double*)dst;
        const double* s = (const double*)src;
        if (op == ncclSum)
            reduce_sum_f64<<<grid, BS, 0, stream>>>(s, d, count);
        break;
    }
    default: break;
    }
}

/*===========================================================================*
 * Wrapper communicator struct                                                *
 *===========================================================================*/
struct nccl_comm_wrap {
    ncclComm_t real_comm;         /* Underlying real NCCL communicator */
    int        rank;
    int        nranks;

    /* Reusable scratch buffer for compressed data (per-communicator) */
    void*      scratch;
    size_t     scratch_cap;
};

static inline nccl_comm_wrap* get_wrap(ncclComm_t comm) {
    return (nccl_comm_wrap*)comm;
}

/* Ensure scratch buffer is at least `need` bytes (plain malloc/free for safety) */
static int ensure_scratch(nccl_comm_wrap* w, size_t need) {
    if (w->scratch_cap >= need) return 0;
    if (w->scratch) {
        cudaFree(w->scratch);
        w->scratch = nullptr;
        w->scratch_cap = 0;
    }
    cudaError_t e = cudaMalloc(&w->scratch, need);
    if (e != cudaSuccess) return -1;
    w->scratch_cap = need;
    return 0;
}

/*===========================================================================*
 * Real NCCL function pointers (loaded once via dlsym)                        *
 *===========================================================================*/
#define LOAD_REAL(name) \
    static auto real_##name = (decltype(&name))dlsym(RTLD_NEXT, #name)

/*===========================================================================*
 * ncclGetUniqueId — forward to real NCCL                                     *
 *===========================================================================*/
extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    LOAD_REAL(ncclGetUniqueId);
    if (!real_ncclGetUniqueId) return ncclInternalError;
    LOG("ncclGetUniqueId");
    return real_ncclGetUniqueId(uniqueId);
}

/*===========================================================================*
 * ncclCommInitRank — init real NCCL + wrap in our struct                     *
 *===========================================================================*/
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks,
                                          ncclUniqueId commId, int myrank) {
    LOAD_REAL(ncclCommInitRank);
    if (!real_ncclCommInitRank) return ncclInternalError;

    /* First init real NCCL communicator */
    ncclComm_t real_comm;
    ncclResult_t r = real_ncclCommInitRank(&real_comm, nranks, commId, myrank);
    if (r != ncclSuccess) return r;

    /* Wrap it */
    nccl_comm_wrap* w = new nccl_comm_wrap();
    w->real_comm   = real_comm;
    w->rank        = myrank;
    w->nranks      = nranks;
    w->scratch     = nullptr;
    w->scratch_cap = 0;

    LOG("ncclCommInitRank rank=%d/%d", myrank, nranks);
    *newcomm = (ncclComm_t)w;
    return ncclSuccess;
}

/*===========================================================================*
 * ncclCommDestroy — destroy real comm + free wrapper                         *
 *===========================================================================*/
extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;

    /* Destroy real NCCL communicator */
    LOAD_REAL(ncclCommDestroy);
    if (real_ncclCommDestroy)
        real_ncclCommDestroy(w->real_comm);

    /* Clean up scratch buffer */
    if (w->scratch) {
        cudaFree(w->scratch);
        w->scratch = nullptr;
    }

    LOG("ncclCommDestroy rank=%d", w->rank);
    delete w;
    return ncclSuccess;
}

/*===========================================================================*
 * ncclCommUserRank / ncclCommCount — forward to real or use our wrapper      *
 *===========================================================================*/
extern "C" ncclResult_t ncclCommUserRank(ncclComm_t comm, int* rank) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w || !rank) return ncclInvalidArgument;
    *rank = w->rank;
    return ncclSuccess;
}

extern "C" ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    nccl_comm_wrap* w = get_wrap((ncclComm_t)comm);
    if (!w || !count) return ncclInvalidArgument;
    *count = w->nranks;
    return ncclSuccess;
}

/*===========================================================================*
 * ncclSend — quantize + send via real NCCL                                   *
 *===========================================================================*/
extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count,
                                  ncclDataType_t datatype, int peer,
                                  ncclComm_t comm, cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    LOAD_REAL(ncclSend);
    if (!real_ncclSend) return ncclInternalError;

    const auto& cfg = get_config();

    /* If compression disabled or not a float type, forward directly */
    if (!cfg.enabled || !is_compressible_type(datatype)) {
        return real_ncclSend(sendbuff, count, datatype, peer,
                             w->real_comm, stream);
    }

    /* Compute compressed buffer size and ensure scratch is large enough */
    size_t comp_sz = compressed_buf_size(count, datatype);
    if (ensure_scratch(w, comp_sz) != 0)
        return ncclInternalError;
    void* comp_buf = w->scratch;

    size_t comp_elems = comp_sz; /* send as bytes */

    /* Quantize: input GPU buffer → compressed scratch buffer */
    int ng = num_groups(count, cfg.group_size);
    float* scales = (float*)comp_buf;
    int8_t* qdata = (int8_t*)(scales + ng);
    int grid = ng;
    int block = cfg.group_size < 1024 ? cfg.group_size : 256;

    switch (datatype) {
    case ncclFloat32:
        quantize_f32_i8<<<grid, block, block * sizeof(float), stream>>>(
            (const float*)sendbuff, qdata, scales, (int)count, cfg.group_size);
        break;
    case ncclFloat16:
        quantize_f16_i8<<<grid, block, block * sizeof(float), stream>>>(
            (const __half*)sendbuff, qdata, scales, (int)count, cfg.group_size);
        break;
    case ncclFloat64:
        quantize_f64_i8<<<grid, block, block * sizeof(float), stream>>>(
            (const double*)sendbuff, qdata, scales, (int)count, cfg.group_size);
        break;
    default:
        return real_ncclSend(sendbuff, count, datatype, peer,
                             w->real_comm, stream);
    }

    /* Send compressed data via real NCCL */
    return real_ncclSend(comp_buf, comp_elems, ncclUint8, peer,
                         w->real_comm, stream);
}

/*===========================================================================*
 * ncclRecv — receive via real NCCL + dequantize                              *
 *===========================================================================*/
extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count,
                                  ncclDataType_t datatype, int peer,
                                  ncclComm_t comm, cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    LOAD_REAL(ncclRecv);
    if (!real_ncclRecv) return ncclInternalError;

    const auto& cfg = get_config();

    if (!cfg.enabled || !is_compressible_type(datatype)) {
        return real_ncclRecv(recvbuff, count, datatype, peer,
                             w->real_comm, stream);
    }

    /* Compute compressed buffer size */
    size_t comp_sz = compressed_buf_size(count, datatype);
    if (ensure_scratch(w, comp_sz) != 0)
        return ncclInternalError;
    void* comp_buf = w->scratch;
    size_t comp_elems = comp_sz;

    /* Receive compressed data via real NCCL */
    ncclResult_t r = real_ncclRecv(comp_buf, comp_elems, ncclUint8, peer,
                                    w->real_comm, stream);
    if (r != ncclSuccess) return r;

    /* Dequantize: compressed scratch buffer → output buffer */
    int ng = num_groups(count, cfg.group_size);
    const float* scales = (const float*)comp_buf;
    const int8_t* qdata = (const int8_t*)(scales + ng);

    constexpr int BS = 256;
    int grid = (int)((count + BS - 1) / BS);
    if (grid > 65535) grid = 65535;

    switch (datatype) {
    case ncclFloat32:
        dequantize_i8_f32<<<grid, BS, 0, stream>>>(
            qdata, (float*)recvbuff, scales, (int)count, cfg.group_size);
        break;
    case ncclFloat16:
        dequantize_i8_f16<<<grid, BS, 0, stream>>>(
            qdata, (__half*)recvbuff, scales, (int)count, cfg.group_size);
        break;
    case ncclFloat64:
        dequantize_i8_f64<<<grid, BS, 0, stream>>>(
            qdata, (double*)recvbuff, scales, (int)count, cfg.group_size);
        break;
    default:
        break;
    }

    return ncclSuccess;
}

/*===========================================================================*
 * Real NCCL collective function pointers (for fallback non-compressed ops)   *
 *===========================================================================*/
static ncclResult_t real_nccl_allreduce(const void* sbuf, void* rbuf,
                                         size_t count, ncclDataType_t dtype,
                                         ncclRedOp_t op, ncclComm_t comm,
                                         cudaStream_t stream) {
    static auto fn = (decltype(&ncclAllReduce))dlsym(RTLD_NEXT, "ncclAllReduce");
    if (!fn) return ncclInternalError;
    nccl_comm_wrap* w = get_wrap(comm);
    return fn(sbuf, rbuf, count, dtype, op, w->real_comm, stream);
}

static ncclResult_t real_nccl_broadcast(const void* sbuf, void* rbuf,
                                          size_t count, ncclDataType_t dtype,
                                          int root, ncclComm_t comm,
                                          cudaStream_t stream) {
    static auto fn = (decltype(&ncclBroadcast))dlsym(RTLD_NEXT, "ncclBroadcast");
    if (!fn) return ncclInternalError;
    nccl_comm_wrap* w = get_wrap(comm);
    return fn(sbuf, rbuf, count, dtype, root, w->real_comm, stream);
}

static ncclResult_t real_nccl_reduce(const void* sbuf, void* rbuf,
                                       size_t count, ncclDataType_t dtype,
                                       ncclRedOp_t op, int root,
                                       ncclComm_t comm, cudaStream_t stream) {
    static auto fn = (decltype(&ncclReduce))dlsym(RTLD_NEXT, "ncclReduce");
    if (!fn) return ncclInternalError;
    nccl_comm_wrap* w = get_wrap(comm);
    return fn(sbuf, rbuf, count, dtype, op, root, w->real_comm, stream);
}

static ncclResult_t real_nccl_allgather(const void* sbuf, void* rbuf,
                                          size_t count, ncclDataType_t dtype,
                                          ncclComm_t comm, cudaStream_t stream) {
    static auto fn = (decltype(&ncclAllGather))dlsym(RTLD_NEXT, "ncclAllGather");
    if (!fn) return ncclInternalError;
    nccl_comm_wrap* w = get_wrap(comm);
    return fn(sbuf, rbuf, count, dtype, w->real_comm, stream);
}

static ncclResult_t real_nccl_reduce_scatter(const void* sbuf, void* rbuf,
                                               size_t count, ncclDataType_t dtype,
                                               ncclRedOp_t op,
                                               ncclComm_t comm,
                                               cudaStream_t stream) {
    static auto fn = (decltype(&ncclReduceScatter))dlsym(RTLD_NEXT, "ncclReduceScatter");
    if (!fn) return ncclInternalError;
    nccl_comm_wrap* w = get_wrap(comm);
    return fn(sbuf, rbuf, count, dtype, op, w->real_comm, stream);
}

/*===========================================================================*
 * ncclAllReduce — ring reduce-scatter + ring allgather                       *
 *===========================================================================*/
extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                                       size_t count, ncclDataType_t datatype,
                                       ncclRedOp_t op, ncclComm_t comm,
                                       cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    const auto& cfg = get_config();
    if (!cfg.enabled || !is_compressible_type(datatype))
        return real_nccl_allreduce(sendbuff, recvbuff, count, datatype,
                                    op, comm, stream);

    int rank = w->rank, nranks = w->nranks;
    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, count * dtype_bytes(datatype),
                            cudaMemcpyDeviceToDevice, stream);
        return ncclSuccess;
    }

    size_t elem_sz = dtype_bytes(datatype);
    size_t chunk_bytes = (count / nranks) * elem_sz;
    int rem = (int)(count % nranks);

    /* Total buffer: size = nranks * chunk_bytes (rounded up for remainder) */
    size_t total_bytes = count * elem_sz;

    /* Temporary GPU buffer for all-to-all data */
    void* gpu_all = nullptr;
    cudaError_t ce = cudaMallocAsync(&gpu_all, total_bytes, stream);
    if (ce != cudaSuccess) return ncclInternalError;

    /* Copy local data into our slot (in-place if same buffer) */
    if (sendbuff == recvbuff)
        cudaMemcpyAsync(gpu_all, recvbuff, total_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    else
        cudaMemcpyAsync(gpu_all, sendbuff, total_bytes,
                        cudaMemcpyDeviceToDevice, stream);

    /* ── Phase 1: Reduce-scatter (ring) ── */
    for (int step = 0; step < nranks - 1; step++) {
        int send_rank = (rank - step + nranks) % nranks;
        int recv_rank = (rank - step - 1 + nranks) % nranks;

        size_t send_off = send_rank * chunk_bytes
                        + (size_t)(send_rank < rem ? elem_sz : 0);
        size_t send_sz  = chunk_bytes
                        + (size_t)(send_rank < rem ? elem_sz : 0);
        size_t recv_off = recv_rank * chunk_bytes
                        + (size_t)(recv_rank < rem ? elem_sz : 0);
        size_t recv_sz  = chunk_bytes
                        + (size_t)(recv_rank < rem ? elem_sz : 0);

        int dst = (rank + 1) % nranks;
        int src = (rank - 1 + nranks) % nranks;

        /* Allocate temporary buffer for received chunk */
        void* tmp = nullptr;
        ce = cudaMallocAsync(&tmp, recv_sz, stream);
        if (ce != ncclSuccess) { cudaFreeAsync(gpu_all, stream); return ncclInternalError; }

        /* Compress-send our chunk */
        ncclResult_t r = ncclSend((char*)gpu_all + send_off,
                                   send_sz / elem_sz, datatype,
                                   dst, (ncclComm_t)w, stream);
        if (r != ncclSuccess) { cudaFreeAsync(tmp, stream); cudaFreeAsync(gpu_all, stream); return r; }

        /* Receive-compress from peer */
        r = ncclRecv(tmp, recv_sz / elem_sz, datatype,
                      src, (ncclComm_t)w, stream);
        if (r != ncclSuccess) { cudaFreeAsync(tmp, stream); cudaFreeAsync(gpu_all, stream); return r; }

        /* Ensure NCCL operations complete before reducing */
        cudaStreamSynchronize(stream);

        /* GPU reduce into our chunk */
        launch_reduce((char*)gpu_all + recv_off, tmp,
                       recv_sz / elem_sz, datatype, op, stream);

        cudaFreeAsync(tmp, stream);
    }

    /* ── Phase 2: Allgather (ring) ── */
    for (int step = 0; step < nranks - 1; step++) {
        int send_rank = (rank - step + 1 + nranks) % nranks;
        int recv_rank = (rank - step + nranks) % nranks;

        size_t send_off = send_rank * chunk_bytes
                        + (size_t)(send_rank < rem ? elem_sz : 0);
        size_t send_sz  = chunk_bytes
                        + (size_t)(send_rank < rem ? elem_sz : 0);
        size_t recv_off = recv_rank * chunk_bytes
                        + (size_t)(recv_rank < rem ? elem_sz : 0);
        size_t recv_sz  = chunk_bytes
                        + (size_t)(recv_rank < rem ? elem_sz : 0);

        int dst = (rank + 1) % nranks;
        int src = (rank - 1 + nranks) % nranks;

        ncclResult_t r = ncclSend((char*)gpu_all + send_off,
                                   send_sz / elem_sz, datatype,
                                   dst, (ncclComm_t)w, stream);
        if (r != ncclSuccess) { cudaFreeAsync(gpu_all, stream); return r; }

        r = ncclRecv((char*)gpu_all + recv_off,
                      recv_sz / elem_sz, datatype,
                      src, (ncclComm_t)w, stream);
        if (r != ncclSuccess) { cudaFreeAsync(gpu_all, stream); return r; }

        cudaStreamSynchronize(stream);
    }

    /* Copy result back */
    cudaMemcpyAsync(recvbuff, gpu_all, total_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaFreeAsync(gpu_all, stream);

    return ncclSuccess;
}

/*===========================================================================*
 * ncclBroadcast — binomial tree                                              *
 *===========================================================================*/
extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
                                       size_t count, ncclDataType_t datatype,
                                       int root, ncclComm_t comm,
                                       cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    const auto& cfg = get_config();
    if (!cfg.enabled || !is_compressible_type(datatype))
        return real_nccl_broadcast(sendbuff, recvbuff, count, datatype,
                                    root, comm, stream);

    int rank = w->rank, nranks = w->nranks;
    if (nranks == 1) {
        if (rank == root && sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff,
                            count * dtype_bytes(datatype),
                            cudaMemcpyDeviceToDevice, stream);
        return ncclSuccess;
    }

    size_t bytes = count * dtype_bytes(datatype);
    void* buf = recvbuff;

    if (rank == root && sendbuff != recvbuff)
        cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                        cudaMemcpyDeviceToDevice, stream);

    int tr = rank ^ root; /* tree-space: root becomes 0 */

    /* Receive from parent */
    for (int mask = 1; mask < nranks; mask <<= 1) {
        if (tr & mask) {
            int parent = (tr ^ mask) ^ root;
            ncclResult_t r = ncclRecv(buf, count, datatype, parent,
                                       (ncclComm_t)w, stream);
            if (r != ncclSuccess) return r;
            cudaStreamSynchronize(stream);
            break;
        }
    }

    /* Send to children */
    for (int mask = 1; mask < nranks; mask <<= 1) {
        int child = (tr | mask) ^ root;
        if (child < nranks) {
            ncclResult_t r = ncclSend(buf, count, datatype, child,
                                       (ncclComm_t)w, stream);
            if (r != ncclSuccess) return r;
            cudaStreamSynchronize(stream);
        }
    }

    return ncclSuccess;
}

/*===========================================================================*
 * ncclReduce — binomial tree with GPU reduction                              *
 *===========================================================================*/
extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff,
                                    size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, int root,
                                    ncclComm_t comm, cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    const auto& cfg = get_config();
    if (!cfg.enabled || !is_compressible_type(datatype))
        return real_nccl_reduce(sendbuff, recvbuff, count, datatype,
                                 op, root, comm, stream);

    int rank = w->rank, nranks = w->nranks;
    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, count * dtype_bytes(datatype),
                            cudaMemcpyDeviceToDevice, stream);
        return ncclSuccess;
    }

    size_t bytes = count * dtype_bytes(datatype);

    /* Temp GPU buffer for accumulation */
    void* acc = nullptr;
    cudaError_t ce = cudaMallocAsync(&acc, bytes, stream);
    if (ce != ncclSuccess) return ncclInternalError;

    cudaMemcpyAsync(acc, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream);

    int tr = rank ^ root;

    /* Receive from children and reduce */
    for (int mask = 1; mask < nranks; mask <<= 1) {
        if (!(tr & mask)) {
            int child = (tr | mask) ^ root;
            if (child < nranks) {
                void* tmp = nullptr;
                ce = cudaMallocAsync(&tmp, bytes, stream);
                if (ce != ncclSuccess) { cudaFreeAsync(acc, stream); return ncclInternalError; }

                ncclResult_t r = ncclRecv(tmp, count, datatype, child,
                                           (ncclComm_t)w, stream);
                if (r != ncclSuccess) { cudaFreeAsync(tmp, stream); cudaFreeAsync(acc, stream); return r; }
                cudaStreamSynchronize(stream);

                launch_reduce(acc, tmp, count, datatype, op, stream);
                cudaFreeAsync(tmp, stream);
            }
        }
    }

    /* Send result to parent */
    for (int mask = 1; mask < nranks; mask <<= 1) {
        if (tr & mask) {
            int parent = (tr ^ mask) ^ root;

            ncclResult_t r = ncclSend(acc, count, datatype, parent,
                                       (ncclComm_t)w, stream);
            if (r != ncclSuccess) { cudaFreeAsync(acc, stream); return r; }
            cudaStreamSynchronize(stream);

            cudaFreeAsync(acc, stream);
            return ncclSuccess; /* non-root: done after sending */
        }
    }

    /* Root copies accumulator to recvbuff */
    if (rank == root)
        cudaMemcpyAsync(recvbuff, acc, bytes, cudaMemcpyDeviceToDevice, stream);

    cudaFreeAsync(acc, stream);
    return ncclSuccess;
}

/*===========================================================================*
 * ncclAllGather — ring                                                       *
 *===========================================================================*/
extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                                       size_t count, ncclDataType_t datatype,
                                       ncclComm_t comm, cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    const auto& cfg = get_config();
    if (!cfg.enabled || !is_compressible_type(datatype))
        return real_nccl_allgather(sendbuff, recvbuff, count, datatype,
                                    comm, stream);

    int rank = w->rank, nranks = w->nranks;
    size_t elem_sz = dtype_bytes(datatype);
    size_t chunk_bytes = count * elem_sz;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, chunk_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        return ncclSuccess;
    }

    /* Place local data */
    cudaMemcpyAsync((char*)recvbuff + rank * chunk_bytes, sendbuff,
                     chunk_bytes, cudaMemcpyDeviceToDevice, stream);

    /* Ring allgather */
    for (int step = 0; step < nranks - 1; step++) {
        int send_rank = (rank - step + nranks) % nranks;
        int recv_rank = (rank - step - 1 + nranks) % nranks;
        int dst = (rank + 1) % nranks;
        int src = (rank - 1 + nranks) % nranks;

        ncclResult_t r = ncclSend((char*)recvbuff + send_rank * chunk_bytes,
                                   count, datatype, dst,
                                   (ncclComm_t)w, stream);
        if (r != ncclSuccess) return r;

        r = ncclRecv((char*)recvbuff + recv_rank * chunk_bytes,
                      count, datatype, src,
                      (ncclComm_t)w, stream);
        if (r != ncclSuccess) return r;

        cudaStreamSynchronize(stream);
    }

    return ncclSuccess;
}

/*===========================================================================*
 * ncclReduceScatter — recursive halving                                      *
 *===========================================================================*/
extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                                            size_t recvcount,
                                            ncclDataType_t datatype,
                                            ncclRedOp_t op,
                                            ncclComm_t comm,
                                            cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;
    if (recvcount == 0) return ncclSuccess;

    const auto& cfg = get_config();
    if (!cfg.enabled || !is_compressible_type(datatype))
        return real_nccl_reduce_scatter(sendbuff, recvbuff, recvcount,
                                         datatype, op, comm, stream);

    int rank = w->rank, nranks = w->nranks;
    size_t elem_sz = dtype_bytes(datatype);
    size_t chunk = recvcount * elem_sz;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, chunk,
                            cudaMemcpyDeviceToDevice, stream);
        return ncclSuccess;
    }

    size_t total_bytes = (size_t)nranks * recvcount * elem_sz;

    /* Temporary buffer for the full data */
    void* buf = nullptr;
    cudaError_t ce = cudaMallocAsync(&buf, total_bytes, stream);
    if (ce != ncclSuccess) return ncclInternalError;

    if (sendbuff != recvbuff)
        cudaMemcpyAsync(buf, sendbuff, total_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    else
        cudaMemcpyAsync(buf, recvbuff, total_bytes,
                        cudaMemcpyDeviceToDevice, stream);

    /* Recursive halving */
    int remain = nranks;
    int mask = 1;
    while (mask < nranks) {
        int new_remain = (remain + 1) / 2;

        if (rank < remain) {
            int peer = rank ^ mask;
            if (peer < remain) {
                size_t half_sz = chunk * (size_t)(remain / 2);
                size_t half_elems = half_sz / elem_sz;

                if (rank < peer) {
                    /* Keep first half, send second half, reduce peer's first half */
                    void* tmp = nullptr;
                    ce = cudaMallocAsync(&tmp, half_sz, stream);
                    if (ce != ncclSuccess) { cudaFreeAsync(buf, stream); return ncclInternalError; }

                    ncclResult_t r = ncclSend((char*)buf + half_sz,
                                               half_elems, datatype,
                                               peer, (ncclComm_t)w, stream);
                    if (r != ncclSuccess) { cudaFreeAsync(tmp, stream); cudaFreeAsync(buf, stream); return r; }

                    r = ncclRecv(tmp, half_elems, datatype,
                                  peer, (ncclComm_t)w, stream);
                    if (r != ncclSuccess) { cudaFreeAsync(tmp, stream); cudaFreeAsync(buf, stream); return r; }
                    cudaStreamSynchronize(stream);

                    launch_reduce(buf, tmp, half_elems, datatype, op, stream);
                    cudaFreeAsync(tmp, stream);
                } else {
                    /* Receive first half from peer, reduce with our second half */
                    void* tmp = nullptr;
                    ce = cudaMallocAsync(&tmp, half_sz, stream);
                    if (ce != ncclSuccess) { cudaFreeAsync(buf, stream); return ncclInternalError; }

                    ncclResult_t r = ncclRecv(tmp, half_elems, datatype,
                                               peer, (ncclComm_t)w, stream);
                    if (r != ncclSuccess) { cudaFreeAsync(tmp, stream); cudaFreeAsync(buf, stream); return r; }

                    r = ncclSend((char*)buf, half_elems, datatype,
                                  peer, (ncclComm_t)w, stream);
                    if (r != ncclSuccess) { cudaFreeAsync(tmp, stream); cudaFreeAsync(buf, stream); return r; }
                    cudaStreamSynchronize(stream);

                    launch_reduce((char*)buf + half_sz, tmp,
                                   half_elems, datatype, op, stream);
                    cudaMemcpyAsync(buf, (char*)buf + half_sz, half_sz,
                                    cudaMemcpyDeviceToDevice, stream);
                    cudaFreeAsync(tmp, stream);
                }
            } else if (rank >= new_remain) {
                /* Shift data if needed */
                cudaMemcpyAsync((char*)buf + chunk * (size_t)(rank - new_remain),
                                 (char*)buf + chunk * (size_t)rank,
                                 chunk, cudaMemcpyDeviceToDevice, stream);
            }
        }

        remain = new_remain;
        mask <<= 1;
    }

    cudaMemcpyAsync(recvbuff, buf, chunk, cudaMemcpyDeviceToDevice, stream);
    cudaFreeAsync(buf, stream);
    return ncclSuccess;
}

/*===========================================================================*
 * ncclAllToAll — pairwise exchange                                           *
 *===========================================================================*/
extern "C" ncclResult_t ncclAllToAll(const void* sendbuff, size_t sendcount,
                                      ncclDataType_t sendtype,
                                      void* recvbuff, size_t recvcount,
                                      ncclDataType_t recvtype,
                                      ncclComm_t comm, cudaStream_t stream) {
    nccl_comm_wrap* w = get_wrap(comm);
    if (!w) return ncclInvalidArgument;

    const auto& cfg = get_config();
    if (!cfg.enabled || !is_compressible_type(sendtype)) {
        /* Fall through to real NCCL when compression disabled or non-float */
        static auto fn = (ncclResult_t(*)(const void*, size_t, ncclDataType_t,
                                           void*, size_t, ncclDataType_t,
                                           ncclComm_t, cudaStream_t))
            dlsym(RTLD_NEXT, "ncclAllToAll");
        if (fn) return fn(sendbuff, sendcount, sendtype,
                           recvbuff, recvcount, recvtype,
                           w->real_comm, stream);
    }

    int rank = w->rank, nranks = w->nranks;
    size_t snd_sz = sendcount * dtype_bytes(sendtype);
    size_t rcv_sz = recvcount * dtype_bytes(recvtype);

    /* Local copy */
    if (sendtype == recvtype)
        cudaMemcpyAsync((char*)recvbuff + rank * rcv_sz,
                         (const char*)sendbuff + rank * snd_sz,
                         (snd_sz < rcv_sz ? snd_sz : rcv_sz),
                         cudaMemcpyDeviceToDevice, stream);

    /* Coordinated pairwise exchange:
     * To avoid NCCL P2P deadlock, each rank pair agrees on operation order.
     * Lower rank sends-first / receives-second, higher rank reverse.
     * This ensures matching send/recv pairs across both ranks. */
    for (int i = 0; i < nranks; i++) {
        if (i == rank) continue;
        if (rank < i) {
            /* Lower rank: send first, then recv */
            ncclResult_t r = ncclSend((const char*)sendbuff + i * snd_sz,
                                       sendcount, sendtype,
                                       i, (ncclComm_t)w, stream);
            if (r != ncclSuccess) return r;
            cudaStreamSynchronize(stream);

            r = ncclRecv((char*)recvbuff + i * rcv_sz,
                          recvcount, recvtype,
                          i, (ncclComm_t)w, stream);
            if (r != ncclSuccess) return r;
        } else {
            /* Higher rank: recv first, then send */
            ncclResult_t r = ncclRecv((char*)recvbuff + i * rcv_sz,
                                       recvcount, recvtype,
                                       i, (ncclComm_t)w, stream);
            if (r != ncclSuccess) return r;
            cudaStreamSynchronize(stream);

            r = ncclSend((const char*)sendbuff + i * snd_sz,
                          sendcount, sendtype,
                          i, (ncclComm_t)w, stream);
            if (r != ncclSuccess) return r;
        }
        cudaStreamSynchronize(stream);
    }

    return ncclSuccess;
}

/*===========================================================================*
 * ncclGroupStart / ncclGroupEnd — pass-through to real NCCL                  *
 *===========================================================================*/
extern "C" ncclResult_t ncclGroupStart() {
    static auto real_fn = (ncclResult_t(*)())dlsym(RTLD_NEXT, "ncclGroupStart");
    return real_fn ? real_fn() : ncclSuccess;
}

extern "C" ncclResult_t ncclGroupEnd() {
    static auto real_fn = (ncclResult_t(*)())dlsym(RTLD_NEXT, "ncclGroupEnd");
    return real_fn ? real_fn() : ncclSuccess;
}

/*===========================================================================*
 * ncclGetErrorString                                                          *
 *===========================================================================*/
extern "C" const char* ncclGetErrorString(ncclResult_t error) {
    switch (error) {
    case ncclSuccess:             return "ncclSuccess";
    case ncclUnhandledCudaError:  return "ncclUnhandledCudaError";
    case ncclSystemError:         return "ncclSystemError";
    case ncclInternalError:       return "ncclInternalError";
    case ncclInvalidArgument:     return "ncclInvalidArgument";
    case ncclInvalidUsage:        return "ncclInvalidUsage";
    case ncclNumResults:          return "ncclNumResults";
    default:                      return "ncclUnknownError";
    }
}
