/*
 * perf_helper_nccl.c — NCCL perf state management + GPU topology discovery.
 *
 * Replaces perf_helper_mpi.c for LD_PRELOAD wrappers that don't use MPI.
 *
 * Node map: file-based hostname exchange (no NCCL collective).
 * perf_nccl_is_intra() compares hostnames to determine whether a peer
 * is on the same physical node.
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
#include <unistd.h>
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
#define HOSTNAME_LEN 64

static char   *s_hostnames = NULL;   /* [nranks][HOSTNAME_LEN] */
static int     s_nranks    = 0;
static int     s_node_map_ready = 0;

/* Separate comm for node-map queries, set via perf_nccl_set_node_map_comm().
 * When a custom NCCL wraps ncclComm_t, this must be a real (unwrapped) comm
 * so rank/nranks queries work without struct-layout mismatch. */
static ncclComm_t g_node_map_comm = NULL;

/* ── Rank from benchmark binary via env var ──────────────────────
 *
 * nccl_test_init() in nccl_utils.c sets PERF_NCCL_RANK from the
 * process rank.  No -rdynamic or symbol interposition needed.             */
static int perf_get_rank(void)
{
    static int rank = -1;
    static int resolved = 0;
    if (!resolved) {
        const char *r = getenv("PERF_NCCL_RANK_998244353");
        if (r) rank = atoi(r);
        resolved = 1;
    }
    return rank;
}

void perf_nccl_set_node_map_comm(ncclComm_t comm)
{
    g_node_map_comm = comm;
}

/* ── Init node map (file-based hostname exchange) ──────────────
 *
 * Each rank writes its hostname to nccl_id_file/hostname_<rank>,
 * then all ranks read every file.  No NCCL collective involved, so this
 * avoids CUDA 700 on setups where two NCCL comms' proxy threads collide.
 *
 * Stale hostname_* files from previous runs are removed at the start.
 * Also removes stale pci_id_* files from the old format.                     */

void perf_nccl_init_node_map(ncclComm_t comm)
{
    if (s_node_map_ready)
        return;

    if (ensure_real_nccl() != 0)
        return;

    /* Use registered comm if available — only for rank/nranks queries
     * (no collectives performed here). */
    ncclComm_t info_comm = g_node_map_comm ? g_node_map_comm : comm;

    int rank, nranks;
    if (real_ncclCommUserRank(info_comm, &rank) != ncclSuccess)
        return;
    if (real_ncclCommCount(info_comm, &nranks) != ncclSuccess)
        return;

    /* Get the hostname — all GPUs on the same node share the same hostname,
     * which is how we determine intra-node communication. */
    char my_hostname[HOSTNAME_LEN];
    if (gethostname(my_hostname, HOSTNAME_LEN) != 0)
        return;
    my_hostname[HOSTNAME_LEN - 1] = '\0';

    /* Remove stale pci_id_* files from the old (buggy) format */
    char fname[256];
    snprintf(fname, sizeof(fname), "nccl_id_file/pci_id_%d", rank);
    remove(fname);

    /* Allocate receive buffer */
    char *all_hostnames = (char *)calloc((size_t)nranks, HOSTNAME_LEN);
    if (!all_hostnames) return;

    /* ── Clean our own stale file from previous runs ────────────
     * Each rank removes only its own file — removing other ranks'
     * files races with their writes (they may have already written). */

    /* ── Write our own hostname ────────────────────────────────── */
    snprintf(fname, sizeof(fname), "nccl_id_file/hostname_%d", rank);
    FILE *f = fopen(fname, "wb");
    if (!f) { free(all_hostnames); return; }
    fwrite(my_hostname, 1, HOSTNAME_LEN, f);
    fclose(f);

    /* Our own entry — no need to read the file */
    memcpy(all_hostnames + (size_t)rank * HOSTNAME_LEN, my_hostname, HOSTNAME_LEN);

    /* ── Wait for and read all other ranks' files ──────────────── */
    for (int r = 0; r < nranks; r++) {
        if (r == rank) continue;

        snprintf(fname, sizeof(fname), "nccl_id_file/hostname_%d", r);
        struct stat st;
        int waited = 0;
        while (stat(fname, &st) != 0 || st.st_size == 0) {
            usleep(10000);
            waited++;
            if (waited > 3000) {
                fprintf(stderr, "[perf_nccl] timeout waiting for hostname_%d\n", r);
                free(all_hostnames);
                return;
            }
        }
        FILE *rf = fopen(fname, "rb");
        if (!rf) { free(all_hostnames); return; }
        size_t n = fread(all_hostnames + (size_t)r * HOSTNAME_LEN, 1, HOSTNAME_LEN, rf);
        fclose(rf);
        if (n != HOSTNAME_LEN) {
            fprintf(stderr, "[perf_nccl] short read on hostname_%d\n", r);
            free(all_hostnames);
            return;
        }
    }

    s_nranks       = nranks;
    s_hostnames    = all_hostnames;
    s_node_map_ready = 1;
}

int perf_nccl_is_intra(int rank)
{
    if (!s_node_map_ready || rank < 0 || rank >= s_nranks)
        return 0;
    return strncmp(s_hostnames + (size_t)rank * HOSTNAME_LEN,
                   s_hostnames + (size_t)perf_get_rank() * HOSTNAME_LEN,
                   HOSTNAME_LEN) == 0;
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

    perf_flush_to_rank_file(&g_state, perf_get_rank(), "perf_nccl");
    perf_destroy(&g_state);
}

/* ── Auto-flush at exit ─────────────────────────────────────────── */
__attribute__((destructor))
static void perf_nccl_on_exit(void)
{
    perf_nccl_flush();
    free(s_hostnames);
    s_hostnames = NULL;
    if (g_nccl_handle) {
        dlclose(g_nccl_handle);
        g_nccl_handle = NULL;
    }
}
