/*
 * perf_helper_nccl.c — NCCL perf state management + GPU topology discovery.
 *
 * Replaces perf_helper_mpi.c for LD_PRELOAD wrappers that don't use MPI.
 *
 * Node map: uses the real ncclAllGather (loaded via dlopen + dlsym, same
 * pattern as get_base_implementation.c) to exchange GPU PCI bus IDs across
 * all ranks.  perf_nccl_is_intra() then compares bus IDs to determine
 * whether a peer is on the same physical node.
 *
 * The real libnccl.so path is read from the file written by findso
 * (bin/libs/base_so_name), falling back to common sonames if unavailable.
 */
#define _GNU_SOURCE
#include "perf_helper.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <pthread.h>

/* ── Per-process state (thread-safe via pthread_once) ─────────── */
static perf_state_t   g_state;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void do_init(void)
{
    perf_init(&g_state);
}

perf_state_t *perf_nccl_get_tls(void)
{
    pthread_once(&g_once, do_init);
    return &g_state;
}

/* ── Load real NCCL via dl_iterate_phdr + fallback chain ──────────────── */
#define BASE_SO_FILE "bin/libs/base_so_name"

static void *g_nccl_handle = NULL;
static ncclResult_t (*real_ncclAllGather)(const void *, void *, size_t,
                                          ncclDataType_t, ncclComm_t,
                                          cudaStream_t) = NULL;
static ncclResult_t (*real_ncclCommUserRank)(ncclComm_t, int *) = NULL;
static ncclResult_t (*real_ncclCommCount)(ncclComm_t, int *) = NULL;

/* ── Excluded patterns (custom NCCL implementations) ───────────────── */
static const char *nccl_excluded[] = {
    "coccl", "zccl", NULL
};

static int nccl_is_excluded(const char *path)
{
    if (!path) return 0;
    for (int i = 0; nccl_excluded[i]; i++)
        if (strstr(path, nccl_excluded[i])) return 1;
    return 0;
}

/* ── Check if path matches an LD_PRELOAD entry by inode ────────────── */
static int nccl_is_preloaded(const char *path)
{
    const char *preload = getenv("LD_PRELOAD");
    if (!preload || !path || !path[0]) return 0;

    struct stat pst;
    if (stat(path, &pst) != 0) return 0;

    char *copy = strdup(preload);
    if (!copy) return 0;
    int found = 0;
    char *tok = strtok(copy, " :");
    while (tok) {
        struct stat tst;
        if (stat(tok, &tst) == 0 &&
            pst.st_dev == tst.st_dev && pst.st_ino == tst.st_ino) {
            found = 1; break;
        }
        tok = strtok(NULL, " :");
    }
    free(copy);
    return found;
}

/* ── dl_iterate_phdr callback ───────────────────────────────────────── */
struct nccl_find_ctx {
    char found_path[4096];
};

static int nccl_find_cb(struct dl_phdr_info *info, size_t size, void *data)
{
    (void)size;
    struct nccl_find_ctx *ctx = (struct nccl_find_ctx *)data;
    if (!info->dlpi_name || !info->dlpi_name[0]) return 0;
    if (!strstr(info->dlpi_name, "libnccl.so")) return 0;
    if (nccl_is_excluded(info->dlpi_name)) return 0;
    if (nccl_is_preloaded(info->dlpi_name)) return 0;
    strncpy(ctx->found_path, info->dlpi_name, sizeof(ctx->found_path) - 1);
    ctx->found_path[sizeof(ctx->found_path) - 1] = '\0';
    return 1;
}

static int ensure_real_nccl(void)
{
    if (g_nccl_handle)
        return 0;

    /* ── Phase 1: Search already-loaded modules via dl_iterate_phdr ── */
    struct nccl_find_ctx fctx = { {0} };
    dl_iterate_phdr(nccl_find_cb, &fctx);

    if (fctx.found_path[0]) {
        g_nccl_handle = dlopen(fctx.found_path, RTLD_NOLOAD | RTLD_LAZY);
        if (g_nccl_handle)
            goto resolve;
    }

    /* ── Phase 2: Try the path saved by findso ────────────────────── */
    {
        FILE *f = fopen(BASE_SO_FILE, "r");
        if (f) {
            char path[4096];
            if (fgets(path, sizeof(path), f)) {
                size_t len = strlen(path);
                while (len > 0 && (path[len - 1] == '\n' || path[len - 1] == '\r'))
                    path[--len] = '\0';
                g_nccl_handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            }
            fclose(f);
        }
    }

    /* ── Phase 3: Fallback — try common sonames ───────────────────── */
    if (!g_nccl_handle) {
        const char *candidates[] = {
            "libnccl.so.2", "libnccl.so", "libnccl.so.1", NULL
        };
        for (int i = 0; candidates[i]; i++) {
            g_nccl_handle = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
            if (g_nccl_handle) break;
        }
    }

    if (!g_nccl_handle)
        return -1;

resolve:
    real_ncclAllGather      = dlsym(g_nccl_handle, "ncclAllGather");
    real_ncclCommUserRank   = dlsym(g_nccl_handle, "ncclCommUserRank");
    real_ncclCommCount      = dlsym(g_nccl_handle, "ncclCommCount");

    if (!real_ncclAllGather || !real_ncclCommUserRank || !real_ncclCommCount) {
        dlclose(g_nccl_handle);
        g_nccl_handle = NULL;
        return -1;
    }

    return 0;
}

/* ── GPU node map ──────────────────────────────────────────────── */
#define PCI_ID_LEN 64

static char   *s_pci_ids = NULL;   /* [nranks][PCI_ID_LEN] */
static int     s_nranks  = 0;
static int     s_my_rank = -1;
static int     s_node_map_ready = 0;

void perf_nccl_init_node_map(ncclComm_t comm)
{
    if (s_node_map_ready)
        return;

    if (ensure_real_nccl() != 0)
        return;

    int rank, nranks;
    if (real_ncclCommUserRank(comm, &rank) != ncclSuccess)
        return;
    if (real_ncclCommCount(comm, &nranks) != ncclSuccess)
        return;

    /* Get the PCI bus ID of the current GPU */
    int dev;
    cudaGetDevice(&dev);
    char my_pci[PCI_ID_LEN];
    if (cudaDeviceGetPCIBusId(my_pci, PCI_ID_LEN, dev) != cudaSuccess)
        return;

    /* Allocate receive buffer */
    char *all_pci = (char *)calloc((size_t)nranks, PCI_ID_LEN);
    if (!all_pci) return;

    /* Exchange PCI bus IDs via the real ncclAllGather */
    if (real_ncclAllGather(my_pci, all_pci, PCI_ID_LEN, ncclUint8,
                           comm, NULL) != ncclSuccess) {
        free(all_pci);
        return;
    }

    s_my_rank      = rank;
    s_nranks       = nranks;
    s_pci_ids      = all_pci;
    s_node_map_ready = 1;
}

int perf_nccl_is_intra(int rank)
{
    if (!s_node_map_ready || rank < 0 || rank >= s_nranks)
        return 0;
    return strncmp(s_pci_ids + (size_t)rank * PCI_ID_LEN,
                   s_pci_ids + (size_t)s_my_rank * PCI_ID_LEN,
                   PCI_ID_LEN) == 0;
}

int perf_nccl_local_size(void)
{
    if (!s_node_map_ready)
        return 0;
    int count = 0;
    for (int i = 0; i < s_nranks; i++) {
        if (perf_nccl_is_intra(i))
            count++;
    }
    return count;
}

/* ── Flush ──────────────────────────────────────────────────────── */
void perf_nccl_flush(void)
{
    if (g_state.total_records == 0) return;

    perf_flush_to_rank_file(&g_state, s_my_rank, "perf_nccl");
    perf_destroy(&g_state);
}

/* ── Auto-flush at exit ─────────────────────────────────────────── */
__attribute__((destructor))
static void perf_nccl_on_exit(void)
{
    perf_nccl_flush();
    free(s_pci_ids);
    s_pci_ids = NULL;
    if (g_nccl_handle) {
        dlclose(g_nccl_handle);
        g_nccl_handle = NULL;
    }
}
