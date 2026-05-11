/*
 * perf_helper_nccl.h — LD_PRELOAD wrapper common types and helpers.
 *
 * LD_PRELOAD wrappers that intercept NCCL functions cannot link against
 * libnccl (self-reference).  This header provides the stub definitions
 * (enums, typedefs, type-size lookup) so each wrapper doesn't duplicate
 * them, plus a dlsym(RTLD_NEXT) helper.
 */
#ifndef PERF_HELPER_NCCL_H
#define PERF_HELPER_NCCL_H

#include <stddef.h>
#include <dlfcn.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── CUDA stream stub (no cudart dependency) ──────────────────── */
struct CUStream_st;
typedef struct CUStream_st *cudaStream_t;

/* ── NCCL type stubs (no libnccl dependency) ──────────────────── */
typedef struct ncclComm *ncclComm_t;

typedef enum {
    ncclSuccess             = 0,
    ncclUnhandledCudaError  = 1,
    ncclSystemError         = 2,
    ncclInternalError       = 3,
    ncclInvalidArgument     = 4,
    ncclInvalidUsage        = 5,
    ncclRemoteError         = 6,
    ncclInProgress          = 7,
} ncclResult_t;

typedef enum {
    ncclInt8    = 0,
    ncclUint8   = 1,
    ncclInt32   = 2,
    ncclUint32  = 3,
    ncclInt64   = 4,
    ncclUint64  = 5,
    ncclFloat16 = 6,
    ncclFloat32 = 7,
    ncclFloat64 = 8,
    ncclBfloat16 = 9,
} ncclDataType_t;

typedef enum {
    ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4
} ncclRedOp_t;

/* ── Element size lookup ──────────────────────────────────────── */
static inline size_t nccl_type_size(ncclDataType_t t) {
    switch (t) {
    case ncclInt8:    case ncclUint8:    return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    case ncclInt32:   case ncclUint32:
    case ncclFloat32:                    return 4;
    case ncclInt64:   case ncclUint64:
    case ncclFloat64:                    return 8;
    default:                             return 4;
    }
}

/* ── dlsym(RTLD_NEXT) — find the next symbol in the chain ──────── */
static inline void *perf_nccl_get_real(const char *name) {
    void *p = dlsym(RTLD_NEXT, name);
    if (!p)
        fprintf(stderr, "[perf_nccl] dlsym(RTLD_NEXT, %s) failed: %s\n",
                name, dlerror());
    return p;
}

/* ── Macro: declare + lazy-resolve a real function pointer,
 *    then call it with the given args.  Example usage:
 *
 *     ncclResult_t ncclAllReduce(...) {
 *         PERF_NCCL_REAL(ncclAllReduce);
 *         double t0 = perf_get_time();
 *         ncclResult_t ret = real_ncclAllReduce(...);
 *         ...
 *     }
 */
#define PERF_NCCL_REAL(name) \
    static __typeof__(&name) real_##name = NULL; \
    if (!real_##name) real_##name = (__typeof__(&name))perf_nccl_get_real(#name); \
    if (!real_##name) return ncclInternalError

/* ── TLS perf state (backed by perf_helper_nccl.c) ─────────────────── */
perf_state_t *perf_nccl_get_tls(void);
void          perf_nccl_flush(void);

/* ── GPU topology helpers (backed by perf_helper_nccl.c) ──────────── */

/**
 * Initialise the GPU node map.
 *
 * Reads the current GPU's PCI bus ID and exchanges it across all ranks
 * using NCCL AllGather (via ncclComm_t).  After this call,
 * perf_nccl_is_intra() can be queried.
 *
 * Safe to call multiple times — second call is a no-op.
 * Requires an initialised ncclComm_t (any communicator).
 */
void perf_nccl_init_node_map(ncclComm_t comm);

/**
 * Return non-zero when @p rank is on the same physical node
 * (same GPU PCI bus topology — same host).
 * Returns 0 if the node map is not yet initialised or rank is out of range.
 */
int  perf_nccl_is_intra(int rank);

/**
 * Return the number of ranks on the same node as the current rank.
 */
int  perf_nccl_local_size(void);

#ifdef __cplusplus
}
#endif

#endif /* PERF_HELPER_NCCL_H */
