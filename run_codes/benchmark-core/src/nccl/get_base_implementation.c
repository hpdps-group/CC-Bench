/**
 * NCCL: load base (reference) implementation.
 *
 * Uses dl_iterate_phdr to find the already-loaded real libnccl.so
 * (excluding LD_PRELOAD'd custom implementations like COCCL/ZCCL),
 * then resolves all NCCL collective entry points via dlsym.
 *
 * If the base library is not yet loaded, falls back to searching
 * common sonames and LD_LIBRARY_PATH, then to the findso state file.
 */

#define _GNU_SOURCE
#include "base_impl.h"
#include "nccl/nccl_base.h"
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
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

/* ── Substrings identifying custom NCCL implementations to exclude ──── */
static const char *excluded_patterns[] = {
    "coccl",
    "zccl",
    NULL  /* sentinel */
};

static int is_excluded(const char *path)
{
    if (!path) return 0;
    for (int i = 0; excluded_patterns[i]; i++) {
        if (strstr(path, excluded_patterns[i]))
            return 1;
    }
    return 0;
}

/* ── Check if a library path matches an LD_PRELOAD entry by inode ───── */
static int is_preloaded(const char *path)
{
    const char *preload = getenv("LD_PRELOAD");
    if (!preload || !path || !path[0]) return 0;

    struct stat path_st;
    if (stat(path, &path_st) != 0) return 0;

    int found = 0;
    char *copy = strdup(preload);
    if (!copy) return 0;

    char *tok = strtok(copy, " :");
    while (tok) {
        struct stat tok_st;
        if (stat(tok, &tok_st) == 0 &&
            path_st.st_dev == tok_st.st_dev &&
            path_st.st_ino == tok_st.st_ino) {
            found = 1;
            break;
        }
        tok = strtok(NULL, " :");
    }
    free(copy);
    return found;
}

/* ── dl_iterate_phdr callback: find first valid base libnccl.so ────── */
struct find_ctx {
    char found_path[4096];
};

static int find_callback(struct dl_phdr_info *info, size_t size, void *data)
{
    (void)size;
    struct find_ctx *ctx = (struct find_ctx *)data;

    if (!info->dlpi_name || !info->dlpi_name[0])
        return 0;

    /* Must contain libnccl.so */
    if (!strstr(info->dlpi_name, "libnccl.so"))
        return 0;

    /* Skip known custom implementations */
    if (is_excluded(info->dlpi_name))
        return 0;

    /* Skip LD_PRELOAD'd copies */
    if (is_preloaded(info->dlpi_name))
        return 0;

    strncpy(ctx->found_path, info->dlpi_name, sizeof(ctx->found_path) - 1);
    ctx->found_path[sizeof(ctx->found_path) - 1] = '\0';
    return 1;  /* stop iteration */
}

/* ── dlopen a candidate path and verify it gives real NCCL symbols ──── */
static int try_load(const char *candidate)
{
    void *h = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
    if (!h) return -1;

    /* Verify this is the real NCCL by resolving and checking the symbol */
    void *s = dlsym(h, "ncclAllReduce");
    if (s) {
        Dl_info info;
        if (dladdr(s, &info) && info.dli_fname) {
            if (!is_excluded(info.dli_fname)) {
                g_handle = h;
                return 0;
            }
        }
    }
    dlclose(h);
    return -1;
}

/* ── Macro to resolve one symbol ────────────────────────────────────── */
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
    const char *sonames[] = {
        "libnccl.so.2",
        "libnccl.so",
        "libnccl.so.1",
        NULL
    };

    /* ── Phase 1: Search already-loaded modules via dl_iterate_phdr ──── */
    /*     Uses RTLD_NOLOAD — returns existing handle, never re-loads.  */
    struct find_ctx fctx = { {0} };
    dl_iterate_phdr(find_callback, &fctx);

    if (fctx.found_path[0]) {
        g_handle = dlopen(fctx.found_path, RTLD_NOLOAD | RTLD_LAZY);
        if (g_handle) {
            printf("[base_impl] Found loaded base NCCL: %s\n", fctx.found_path);
            goto resolve;
        }
    }

    /* ── Phase 2: Try common sonames via dlopen ──────────────────────── */
    for (int i = 0; sonames[i]; i++) {
        if (try_load(sonames[i]) == 0) {
            printf("[base_impl] Loaded base NCCL via soname: %s\n", sonames[i]);
            goto resolve;
        }
    }

    /* ── Phase 3: Scan LD_LIBRARY_PATH manually ──────────────────────── */
    const char *llp = getenv("LD_LIBRARY_PATH");
    if (llp) {
        char *copy = strdup(llp);
        if (copy) {
            char *dir = copy, *next;
            do {
                next = strchr(dir, ':');
                if (next) *next = '\0';

                if (!is_excluded(dir)) {
                    for (int i = 0; sonames[i]; i++) {
                        char full[4096];
                        int n = snprintf(full, sizeof(full), "%s/%s",
                                         dir, sonames[i]);
                        if (n < 0 || (size_t)n >= sizeof(full)) continue;

                        if (try_load(full) == 0) {
                            printf("[base_impl] Loaded base NCCL: %s\n", full);
                            free(copy);
                            goto resolve;
                        }
                    }
                }
                dir = next ? next + 1 : NULL;
            } while (dir);
            free(copy);
        }
    }

    /* ── Phase 4: Fallback to findso state file if it exists ─────────── */
    FILE *f = fopen(state_path ? state_path : BASE_SO_FILE, "r");
    if (f) {
        char so_path[4096];
        if (fgets(so_path, sizeof(so_path), f)) {
            size_t len = strlen(so_path);
            while (len > 0 && (so_path[len - 1] == '\n' || so_path[len - 1] == '\r'))
                so_path[--len] = '\0';

            if (try_load(so_path) == 0) {
                printf("[base_impl] Loaded base NCCL from findso file: %s\n", so_path);
                fclose(f);
                goto resolve;
            }
        }
        fclose(f);
    }

    fprintf(stderr, "[base_impl] ERROR: cannot locate base libnccl.so "
            "(ncclAllReduce not found)\n");
    fprintf(stderr, "[base_impl] Tried: dl_iterate_phdr, sonames, "
            "LD_LIBRARY_PATH, and findso file.\n");
    fprintf(stderr, "[base_impl] Run ./bin/nccl/findso to generate "
            "the state file as last resort.\n");
    return -1;

resolve:
    /* Resolve all collective symbols */
    RESOLVE(ncclAllReduce);
    RESOLVE(ncclBroadcast);
    RESOLVE(ncclReduce);
    RESOLVE(ncclAllGather);
    RESOLVE(ncclReduceScatter);
    RESOLVE(ncclSend);
    RESOLVE(ncclRecv);

    printf("[base_impl] NCCL reference loaded successfully\n");
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
