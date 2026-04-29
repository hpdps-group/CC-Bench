/**
 * NCCL: load base (reference) implementation.
 *
 * Reads the .so path saved by findso.c, dlopen's it, and resolves
 * all NCCL collective entry points.  The wrappers declared in
 * nccl/nccl_base.h forward calls to the original library.
 */

#include "base_impl.h"
#include "nccl/nccl_base.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

/* ── dlopen handle and resolved function pointers ─────────────────────── */
static void *g_handle = NULL;

static ncclResult_t (*real_ncclAllReduce)(const void *, void *, size_t,
                                          ncclDataType_t, ncclRedOp_t,
                                          ncclComm_t, cudaStream_t) = NULL;

static ncclResult_t (*real_ncclBroadcast)(const void *, void *, size_t,
                                          ncclDataType_t, int,
                                          ncclComm_t, cudaStream_t) = NULL;

static ncclResult_t (*real_ncclReduce)(const void *, void *, size_t,
                                       ncclDataType_t, ncclRedOp_t, int,
                                       ncclComm_t, cudaStream_t) = NULL;

static ncclResult_t (*real_ncclAllGather)(const void *, void *, size_t,
                                          ncclDataType_t,
                                          ncclComm_t, cudaStream_t) = NULL;

static ncclResult_t (*real_ncclReduceScatter)(const void *, void *, size_t,
                                              ncclDataType_t, ncclRedOp_t,
                                              ncclComm_t, cudaStream_t) = NULL;

static ncclResult_t (*real_ncclSend)(const void *, size_t, ncclDataType_t,
                                     int, ncclComm_t, cudaStream_t) = NULL;

static ncclResult_t (*real_ncclRecv)(void *, size_t, ncclDataType_t,
                                     int, ncclComm_t, cudaStream_t) = NULL;

/* ── Macro to resolve one symbol ──────────────────────────────────────── */
#define RESOLVE(name) do {                                              \
    real_##name = dlsym(g_handle, #name);                               \
    if (!real_##name) {                                                 \
        fprintf(stderr, "[base_impl] symbol \"%s\" not found: %s\n",    \
                #name, dlerror());                                      \
        return -1;                                                      \
    }                                                                   \
} while (0)

/* ── Public load function ─────────────────────────────────────────────── */
int load_base_impl(const char *state_path)
{
    FILE *f = fopen(state_path, "r");
    if (!f) {
        perror("[base_impl] fopen");
        return -1;
    }

    char so_path[4096];
    if (!fgets(so_path, sizeof(so_path), f)) {
        fprintf(stderr, "[base_impl] state file is empty\n");
        fclose(f);
        return -1;
    }
    fclose(f);

    /* Strip trailing newline */
    size_t len = strlen(so_path);
    while (len > 0 && (so_path[len - 1] == '\n' || so_path[len - 1] == '\r'))
        so_path[--len] = '\0';

    g_handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!g_handle) {
        fprintf(stderr, "[base_impl] dlopen(\"%s\") failed: %s\n",
                so_path, dlerror());
        return -1;
    }

    /* Resolve all collective symbols */
    RESOLVE(ncclAllReduce);
    RESOLVE(ncclBroadcast);
    RESOLVE(ncclReduce);
    RESOLVE(ncclAllGather);
    RESOLVE(ncclReduceScatter);
    RESOLVE(ncclSend);
    RESOLVE(ncclRecv);

    printf("[base_impl] NCCL reference loaded from: %s\n", so_path);
    return 0;
}

/* ── Wrapper functions ────────────────────────────────────────────────── */

ncclResult_t base_ncclAllReduce(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream)
{
    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t base_ncclBroadcast(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                int root, ncclComm_t comm,
                                cudaStream_t stream)
{
    return real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t base_ncclReduce(const void *sendbuff, void *recvbuff,
                             size_t count, ncclDataType_t datatype,
                             ncclRedOp_t op, int root,
                             ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t base_ncclAllGather(const void *sendbuff, void *recvbuff,
                                size_t sendcount, ncclDataType_t datatype,
                                ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t base_ncclReduceScatter(const void *sendbuff, void *recvbuff,
                                    size_t recvcount, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm_t comm,
                                    cudaStream_t stream)
{
    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t base_ncclSend(const void *sendbuff, size_t count,
                           ncclDataType_t datatype, int peer,
                           ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t base_ncclRecv(void *recvbuff, size_t count,
                           ncclDataType_t datatype, int peer,
                           ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}