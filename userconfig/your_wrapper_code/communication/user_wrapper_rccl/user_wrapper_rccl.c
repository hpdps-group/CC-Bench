/**
 * RCCL wrapper template — implement your own RCCL kernels here.
 *
 * This template INTERCEPTS ncclAllReduce / ncclReduce / etc. directly
 * (NOT MPI functions).  Use it when you want to hook RCCL calls from
 * any application that links against librccl.
 *
 * RCCL (ROCm Collective Communications Library) uses the SAME nccl*
 * API names as NCCL for drop-in compatibility (ncclAllReduce, ncclComm_t,
 * ncclDataType_t, ncclRedOp_t, etc.).  The differences are:
 *   - Device memory:    hipMalloc / hipFree  (instead of cudaMalloc)
 *   - Stream type:      hipStream_t          (instead of cudaStream_t)
 *   - Header:           <rccl.h>             (instead of <nccl.h>)
 *   - Link:             -lrccl               (instead of -lnccl)
 *
 * RCCL has NO profiling interface (no equivalent of MPI's PMPI).
 * To call the original library function, use dlsym(RTLD_NEXT, ...):
 *
 *     static typeof(ncclAllReduce)* real_ncclAllReduce = NULL;
 *     if (!real_ncclAllReduce)
 *         real_ncclAllReduce = dlsym(RTLD_NEXT, "ncclAllReduce");
 *     return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
 *
 * RULES:
 *   1. Manage GPU memory explicitly: hipMalloc / hipFree.
 *   2. Use dlsym(RTLD_NEXT, ...) for fallback — NOT a PMPI-style interface.
 *   3. RCCL functions return ncclResult_t (same as NCCL).
 *   4. RCCL ops are async (stream-ordered) — sync with hipStreamSynchronize.
 *   5. Read config from environment variables via getenv().
 *
 * Compile with ROCm:
 *   hipcc -shared -O3 -lrccl -ldl -o librccl_wrapper.so \
 *       user_wrapper_template.c
 *
 * Load:
 *   LD_PRELOAD=./librccl_wrapper.so ./your_app
 *
 * Environment variables (define your own):
 *   RCCL_DEBUG        Debug output (0/1)
 *   RCCL_THRESHOLD    Only intercept calls above this message size
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <rccl.h>

/*===========================================================================*
 *  dlsym helpers — resolve the original librccl functions                   *
 *                                                                           *
 *  RTLD_NEXT walks the shared-library lookup order and returns the next     *
 *  occurrence of the symbol after the current library — i.e. the real       *
 *  ncclAllReduce in librccl.so, not this wrapper's version.                 *
 *===========================================================================*/

static ncclResult_t real_ncclAllReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, hipStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclRedOp_t, ncclComm_t, hipStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclAllReduce");
    return fn(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

static ncclResult_t real_ncclReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    int root, ncclComm_t comm, hipStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclRedOp_t, int, ncclComm_t, hipStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclReduce");
    return fn(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

static ncclResult_t real_ncclBcast(void *buff, size_t count,
    ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {
    static ncclResult_t (*fn)(void*, size_t, ncclDataType_t,
                              int, ncclComm_t, hipStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclBcast");
    return fn(buff, count, datatype, root, comm, stream);
}

static ncclResult_t real_ncclAllGather(const void *sendbuff, void *recvbuff,
    size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, hipStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclComm_t, hipStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclAllGather");
    return fn(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

static ncclResult_t real_ncclReduceScatter(const void *sendbuff, void *recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, hipStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclRedOp_t, ncclComm_t, hipStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclReduceScatter");
    return fn(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

/*===========================================================================*
 *  ncclAllReduce — EXAMPLE                                                  *
 *                                                                           *
 *  Pattern:                                                                 *
 *    1. Optionally read config from environment                             *
 *    2. If message is large enough, apply custom compression                *
 *    3. Otherwise, fall through to original ncclAllReduce via dlsym         *
 *===========================================================================*/

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, hipStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    /*
     * Example: conditionally apply custom logic for large messages.
     *
     *   // 1. Read threshold from environment
     *   size_t threshold = 1024 * 1024;  // 1 MB default
     *   char *env = getenv("RCCL_THRESHOLD");
     *   if (env) threshold = atol(env);
     *
     *   // 2. Debug
     *   env = getenv("RCCL_DEBUG");
     *   if (env && atoi(env)) {
     *       fprintf(stderr, "[RCCL] ncclAllReduce count=%zu\n", count);
     *   }
     *
     *   // 3. For large messages, apply your custom algorithm
     *   if (count * sizeof(float) > threshold) {
     *       // ... your custom compressed allreduce here ...
     *       // return custom_result;
     *   }
     *
     *   // 4. Fallback to original RCCL
     */

    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

/*===========================================================================*
 *  ncclReduce                                                                *
 *===========================================================================*/

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    int root, ncclComm_t comm, hipStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

/*===========================================================================*
 *  ncclBcast                                                                *
 *===========================================================================*/

ncclResult_t ncclBcast(void *buff, size_t count,
    ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclBcast(buff, count, datatype, root, comm, stream);
}

/*===========================================================================*
 *  ncclAllGather                                                             *
 *===========================================================================*/

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
    size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, hipStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

/*===========================================================================*
 *  ncclReduceScatter                                                        *
 *===========================================================================*/

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, hipStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}
