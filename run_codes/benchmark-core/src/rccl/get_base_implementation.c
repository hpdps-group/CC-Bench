/**
 * RCCL: load base (reference) implementation.
 *
 * Reads the .so path saved by findso.c, dlopen's it, and resolves
 * all collective entry points.  Wrappers are declared with the same
 * nccl-prefixed names as NCCL (RCCL implements the NCCL API).
 *
 * The rccl.h header provides the same types and function signatures
 * as nccl.h (ncclDataType_t, ncclRedOp_t, ncclComm_t, etc.).
 */

#include "base_impl.h"
#include "rccl/rccl_base.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

/* ── dlopen handle and resolved function pointers ─────────────────────── */
static void *g_handle = NULL;

/* Separate base communicator — see nccl_base.h for rationale. */
static ncclComm_t g_base_comm = NULL;

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
#define RESOLVE_SYM(name) do {                                          \
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
    RESOLVE_SYM(ncclAllReduce);
    RESOLVE_SYM(ncclBroadcast);
    RESOLVE_SYM(ncclReduce);
    RESOLVE_SYM(ncclAllGather);
    RESOLVE_SYM(ncclReduceScatter);
    RESOLVE_SYM(ncclSend);
    RESOLVE_SYM(ncclRecv);

    printf("[base_impl] RCCL reference loaded from: %s\n", so_path);
    return 0;
}

/* ── Wrapper functions ────────────────────────────────────────────────── */
/* Note: when g_base_comm is set, use it instead of the caller-provided
 * comm to avoid struct-layout mismatches with custom implementations. */

void base_nccl_set_comm(ncclComm_t comm) { g_base_comm = comm; }

static inline ncclComm_t base_comm(ncclComm_t user_comm) {
  return g_base_comm ? g_base_comm : user_comm;
}

ncclResult_t base_ncclAllReduce(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream)
{
    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op,
                              base_comm(comm), stream);
}

ncclResult_t base_ncclBroadcast(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                int root, ncclComm_t comm,
                                cudaStream_t stream)
{
    return real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root,
                              base_comm(comm), stream);
}

ncclResult_t base_ncclReduce(const void *sendbuff, void *recvbuff,
                             size_t count, ncclDataType_t datatype,
                             ncclRedOp_t op, int root,
                             ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root,
                           base_comm(comm), stream);
}

ncclResult_t base_ncclAllGather(const void *sendbuff, void *recvbuff,
                                size_t sendcount, ncclDataType_t datatype,
                                ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype,
                              base_comm(comm), stream);
}

ncclResult_t base_ncclReduceScatter(const void *sendbuff, void *recvbuff,
                                    size_t recvcount, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm_t comm,
                                    cudaStream_t stream)
{
    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                  base_comm(comm), stream);
}

ncclResult_t base_ncclSend(const void *sendbuff, size_t count,
                           ncclDataType_t datatype, int peer,
                           ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclSend(sendbuff, count, datatype, peer, base_comm(comm), stream);
}

ncclResult_t base_ncclRecv(void *recvbuff, size_t count,
                           ncclDataType_t datatype, int peer,
                           ncclComm_t comm, cudaStream_t stream)
{
    return real_ncclRecv(recvbuff, count, datatype, peer, base_comm(comm), stream);
}