/**
 * NCCL wrapper template — implement your own NCCL kernels here.
 *
 * This template INTERCEPTS ncclAllReduce / ncclReduce / etc. directly
 * (NOT MPI functions).  Use it when you want to hook NCCL calls from
 * any application that links against libnccl.
 *
 * NCCL has NO profiling interface (no equivalent of MPI's PMPI).
 * To call the original library function, use dlsym(RTLD_NEXT, ...):
 *
 *     static typeof(ncclAllReduce)* real_ncclAllReduce = NULL;
 *     if (!real_ncclAllReduce)
 *         real_ncclAllReduce = dlsym(RTLD_NEXT, "ncclAllReduce");
 *     return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
 *
 * RULES:
 *   1. Manage GPU memory explicitly: cudaMalloc / cudaFree.
 *   2. Use dlsym(RTLD_NEXT, ...) for fallback — NOT a PMPI-style interface.
 *   3. NCCL functions return ncclResult_t, not int.
 *   4. NCCL ops are async (stream-ordered) — sync with cudaStreamSynchronize.
 *   5. Read config from environment variables via getenv().
 *
 * Compile:
 *   nvcc -shared -O3 -lnccl -ldl -o libnccl_wrapper.so \
 *       user_wrapper_template.c
 *
 * Load:
 *   LD_PRELOAD=./libnccl_wrapper.so ./your_app
 *
 * Or without LD_PRELOAD, link directly:
 *   nvcc -o your_app your_app.c -L. -lnccl_wrapper
 *
 * Environment variables (define your own):
 *   NCCL_DEBUG        Debug output (0/1)
 *   NCCL_THRESHOLD    Only intercept calls above this message size
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <nccl.h>

/*===========================================================================*
 *  dlsym helpers — resolve the original libnccl functions                   *
 *                                                                           *
 *  RTLD_NEXT walks the shared-library lookup order and returns the next     *
 *  occurrence of the symbol after the current library — i.e. the real       *
 *  ncclAllReduce in libnccl.so, not this wrapper's version.                 *
 *===========================================================================*/

static ncclResult_t real_ncclAllReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclAllReduce");
    return fn(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

static ncclResult_t real_ncclReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    int root, ncclComm_t comm, cudaStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclRedOp_t, int, ncclComm_t, cudaStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclReduce");
    return fn(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

static ncclResult_t real_ncclBcast(void *buff, size_t count,
    ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
    static ncclResult_t (*fn)(void*, size_t, ncclDataType_t,
                              int, ncclComm_t, cudaStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclBcast");
    return fn(buff, count, datatype, root, comm, stream);
}

static ncclResult_t real_ncclAllGather(const void *sendbuff, void *recvbuff,
    size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclComm_t, cudaStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclAllGather");
    return fn(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

static ncclResult_t real_ncclReduceScatter(const void *sendbuff, void *recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream) {
    static ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t,
                              ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
    if (!fn) fn = (typeof(fn))dlsym(RTLD_NEXT, "ncclReduceScatter");
    return fn(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

/*===========================================================================*
 *  ncclAllReduce — EXAMPLE                                                  *
 *                                                                           *
 *  Pattern:                                                                 *
 *    1. Optionally read config from environment                             *
 *    2. If message is small enough (threshold), do custom compression       *
 *    3. Otherwise, fall through to original ncclAllReduce via dlsym         *
 *===========================================================================*/

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    /*
     * Example: conditionally apply compression for large messages.
     *
     *   // 1. Read threshold from environment
     *   size_t threshold = 1024 * 1024;  // 1 MB default
     *   char *env = getenv("NCCL_THRESHOLD");
     *   if (env) threshold = atol(env);
     *
     *   // 2. Debug
     *   env = getenv("NCCL_DEBUG");
     *   if (env && atoi(env)) {
     *       fprintf(stderr, "[NCCL] ncclAllReduce count=%zu size=%zu\n",
     *               count, count * sizeof(float));  // adjust for type
     *   }
     *
     *   // 3. For large messages, apply your custom algorithm
     *   if (count * sizeof(float) > threshold) {
     *       // ... your custom compressed allreduce here ...
     *       // return custom_result;
     *   }
     *
     *   // 4. Fallback to original NCCL
     */

    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

/*===========================================================================*
 *  ncclReduce                                                                *
 *===========================================================================*/

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    int root, ncclComm_t comm, cudaStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

/*===========================================================================*
 *  ncclBcast                                                                *
 *===========================================================================*/

ncclResult_t ncclBcast(void *buff, size_t count,
    ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclBcast(buff, count, datatype, root, comm, stream);
}

/*===========================================================================*
 *  ncclAllGather                                                             *
 *===========================================================================*/

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
    size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

/*===========================================================================*
 *  ncclReduceScatter                                                        *
 *===========================================================================*/

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream) {

    /* ---- YOUR CODE HERE ---- */

    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}
