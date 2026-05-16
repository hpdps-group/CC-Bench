/**
 * NCCL benchmark utilities — no MPI dependency.
 *
 * Bootstrap: rank/size from environment variables, NCCL unique ID
 * via file-based exchange.  Timing via CUDA events.
 * Results aggregated across ranks using file-based exchange.
 */

#define _GNU_SOURCE
#include "nccl/nccl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

/* Fixed path used for NCCL unique-ID bootstrap across processes */
#define NCCL_ID_FILE "nccl_id_file/nccl_bench_id"

/* File for TCP barrier address exchange (same shared directory) */
#define BARRIER_ADDR_FILE "nccl_id_file/nccl_barrier_addr"

#ifndef NCCLCHECK
#define NCCLCHECK(call) do { \
    ncclResult_t _r = (call); \
    if (_r != ncclSuccess) { \
        fprintf(stderr, "[nccl_utils] NCCLCHECK error at %s:%d: %s\n", \
                __FILE__, __LINE__, ncclGetErrorString(_r)); \
        exit(1); \
    } \
} while (0)
#endif

/* ── Global rank accessor for LD_PRELOAD wrappers (perf, etc.) ─ */
static int g_nccl_my_rank = -1;

int nccl_get_my_rank(void) { return g_nccl_my_rank; }

/* ── Rank / size from environment (no MPI) ────────────────────── */
static void get_rank_size(int *rank, int *size)
{
    const char *r, *s;

    r = getenv("OMPI_COMM_WORLD_RANK"); s = getenv("OMPI_COMM_WORLD_SIZE");
    if (!r) { r = getenv("PMI_RANK");       s = getenv("PMI_SIZE");       }
    if (!r) { r = getenv("SLURM_PROCID");    s = getenv("SLURM_NPROCS");   }
    if (!r) { r = getenv("MV2_COMM_WORLD_RANK"); s = getenv("MV2_COMM_WORLD_SIZE"); }

    *rank = r ? atoi(r) : 0;
    *size = s ? atoi(s) : 1;
}

/* ── NCCL communicator bootstrap ──────────────────────────────── */
static ncclComm_t init_nccl_comm(int rank, int size)
{
    ncclUniqueId id;

    if (size == 1) {
        ncclGetUniqueId(&id);
    } else if (rank == 0) {
        printf("[nccl] Generating NCCL unique ID...\n");
        NCCLCHECK(ncclGetUniqueId(&id));
        FILE *f = fopen(NCCL_ID_FILE, "wb");
        if (!f) { perror("fopen NCCL_ID_FILE"); exit(1); }
        fwrite(&id, sizeof(id), 1, f);
        fclose(f);
    }

    if (size > 1 && rank != 0) {
        printf("[nccl] Waiting for NCCL unique ID from rank 0...\n");
        struct stat st;
        int waited = 0;
        while (stat(NCCL_ID_FILE, &st) != 0 || st.st_size == 0) {
            usleep(10000); /* 10 ms */
            waited++;
            if (waited > 3000) { /* 30 s timeout */
                fprintf(stderr, "Rank %d: timeout waiting for NCCL id\n", rank);
                exit(1);
            }
            if (waited % 100 == 0) /* every ~1 s */
                printf("[nccl]   still waiting... (%d s)\n", waited / 100);
        }
        FILE *f = fopen(NCCL_ID_FILE, "rb");
        if (!f) { perror("fopen NCCL_ID_FILE for read"); exit(1); }
        size_t n = fread(&id, 1, sizeof(id), f);
        fclose(f);
        if (n != sizeof(id)) {
            fprintf(stderr, "Rank %d: short read on NCCL id\n", rank);
            exit(1);
        }
    }

    printf("[nccl] Creating NCCL communicator (%d ranks)...\n", size);
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    printf("[nccl] NCCL communicator created\n");
    return comm;
}

/* ── TCP barrier init (defined below, called from nccl_test_init) ── */
static void nccl_barrier_init(nccl_test_context_t *ctx);

/* ── Public API ───────────────────────────────────────────────── */

nccl_test_context_t nccl_test_init(int argc, char **argv,
                                   const char *test_name)
{
    nccl_test_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));

    get_rank_size(&ctx.rank, &ctx.size);
    g_nccl_my_rank = ctx.rank;
    {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", ctx.rank);
        setenv("PERF_NCCL_RANK_998244353", buf, 1);
    }

    printf("[nccl] Initializing CUDA...\n");
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    int dev = ctx.rank % (ndev > 0 ? ndev : 1);
    cudaSetDevice(dev);
    cudaStreamCreate(&ctx.stream);

    /* Allocate one int for barrier */
    cudaMalloc(&ctx.d_barrier, sizeof(int));

    /* TCP barrier — init BEFORE ncclCommInitRank so that all ranks
     * synchronise before UCCL's cross-node connection phase.
     * This prevents a fast rank from sending OOB connection requests
     * before a slower peer's Endpoint TCP server is listening. */
    nccl_barrier_init(&ctx);

    // /* Pre-init barrier disabled — letting ranks stagger into
    //  * ncclCommInitRank to avoid thundering-herd connection attempts
    //  * that cause UCCL QP state-machine races. */
    // nccl_barrier(&ctx);

    /* NCCL communicator — ranks start staggered */
    ctx.comm = init_nccl_comm(ctx.rank, ctx.size);

    /* Second barrier: ensure all ranks completed ncclCommInitRank
     * before anyone proceeds to the first benchmark iteration. */
    nccl_barrier(&ctx);

    /* Parse config */
    ctx.config = parse_arguments(argc, argv);

    /* Banner (rank 0) */
    if (ctx.rank == 0) {
        printf("=== %s Test ===\n", test_name);
        printf("Ranks: %d\n", ctx.size);
        if (ctx.config.use_size_list) {
            printf("Message sizes (list): ");
            for (int i = 0; i < ctx.config.num_sizes; i++) {
                if (i > 0) printf(", ");
                printf("%zu", ctx.config.size_list[i]);
            }
            printf("\n");
        } else {
            printf("Message sizes: %zu to %zu (x%d)\n",
                   ctx.config.min_message_size,
                   ctx.config.max_message_size,
                   ctx.config.message_size_incr);
        }
        printf("Iterations: %d (warmup: %d)\n",
               ctx.config.iterations, ctx.config.warmup_iterations);
        printf("Validation: %s\n",
               ctx.config.validate ? "enabled" : "disabled");
        if (ctx.config.input_file)
            printf("Input file: %s\n", ctx.config.input_file);
        printf("Tolerance: %e\n\n", ctx.config.tolerance);
    }

    return ctx;
}

void nccl_test_fini(nccl_test_context_t *ctx)
{
    if (ctx->rank == 0)
        printf("=== Test Complete ===\n");

    /* Close TCP barrier connections */
    for (int i = 0; i < ctx->size; i++) {
        if (ctx->barrier_peers[i] >= 0) close(ctx->barrier_peers[i]);
    }
    if (ctx->barrier_listen_fd >= 0) close(ctx->barrier_listen_fd);

    if (ctx->d_barrier) cudaFree(ctx->d_barrier);
    ncclCommDestroy(ctx->comm);
    cudaStreamDestroy(ctx->stream);

    /* Clean up NCCL id file (best-effort) */
    if (ctx->rank == 0) {
        remove(NCCL_ID_FILE);
        remove(BARRIER_ADDR_FILE);
    }
}

/* ── TCP barrier initialization ────────────────────────────────── */
static void nccl_barrier_init(nccl_test_context_t *ctx)
{
    ctx->barrier_listen_fd = -1;
    for (int i = 0; i < 256; i++) ctx->barrier_peers[i] = -1;

    if (ctx->size <= 1) return;

    if (ctx->rank == 0) {
        /* Remove stale barrier address file from previous runs so that peers
         * don't connect to a defunct port before rank 0 writes the new one. */
        remove(BARRIER_ADDR_FILE);

        /* Create listen socket on kernel-assigned port */
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { perror("barrier socket"); exit(1); }
        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = 0;
        if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("barrier bind"); exit(1);
        }
        listen(fd, ctx->size - 1);

        /* Get assigned port */
        socklen_t slen = sizeof(addr);
        getsockname(fd, (struct sockaddr*)&addr, &slen);
        int port = ntohs(addr.sin_port);

        /* Get hostname and write to shared file */
        char host[256];
        if (gethostname(host, sizeof(host)) < 0) {
            perror("gethostname"); exit(1);
        }
        FILE *f = fopen(BARRIER_ADDR_FILE, "w");
        if (!f) { perror("fopen barrier_addr"); exit(1); }
        fprintf(f, "%s:%d\n", host, port);
        fclose(f);

        /* Accept connections from all peers */
        int accepted = 0;
        while (accepted < ctx->size - 1) {
            struct sockaddr_in peer_addr;
            socklen_t peer_len = sizeof(peer_addr);
            int peer_fd = accept(fd, (struct sockaddr*)&peer_addr, &peer_len);
            if (peer_fd < 0) { perror("barrier accept"); exit(1); }
            /* Read peer rank */
            uint8_t peer_rank;
            if (read(peer_fd, &peer_rank, 1) != 1) {
                fprintf(stderr, "barrier: failed to read peer rank\n");
                exit(1);
            }
            ctx->barrier_peers[peer_rank] = peer_fd;
            accepted++;
        }
        close(fd); /* listen socket no longer needed */
        ctx->barrier_listen_fd = -1;
        remove(BARRIER_ADDR_FILE);
    } else {
        /* Wait for rank 0's address file */
        struct stat st;
        while (stat(BARRIER_ADDR_FILE, &st) != 0) {
            usleep(10000);
        }
        /* Read address */
        FILE *f = fopen(BARRIER_ADDR_FILE, "r");
        if (!f) { perror("fopen barrier_addr for read"); exit(1); }
        char host[256]; int port;
        if (fscanf(f, "%255[^:]:%d", host, &port) != 2) {
            fprintf(stderr, "barrier: failed to parse address\n");
            exit(1);
        }
        fclose(f);
        /* Connect to rank 0 */
        struct hostent *he = gethostbyname(host);
        if (!he) { fprintf(stderr, "barrier: unknown host %s\n", host); exit(1); }
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { perror("barrier connect socket"); exit(1); }
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("barrier connect"); exit(1);
        }
        /* Send my rank */
        uint8_t my_rank = ctx->rank;
        write(fd, &my_rank, 1);
        ctx->barrier_peers[0] = fd;
    }
}


void nccl_load_input(const nccl_test_context_t *ctx, void *buf,
                     size_t msg_size, ncclDataType_t datatype)
{
    if (ctx->config.input_file) {
        FILE *fp = fopen(ctx->config.input_file, "rb");
        if (!fp) {
            fprintf(stderr, "Rank %d: failed to open %s\n",
                    ctx->rank, ctx->config.input_file);
            exit(1);
        }
        fseek(fp, 0, SEEK_END);
        size_t fsize = ftell(fp);
        rewind(fp);

        size_t chunk = msg_size;
        size_t off = (size_t)ctx->rank * chunk;

        const char *custom = getenv("DS_CUSTOM_OFFSETS");
        if (custom && custom[0]) {
            int n = 0;
            int *offsets = nccl_parse_env_int_array("DS_CUSTOM_OFFSETS", 0, &n);
            if (offsets && ctx->rank < n)
                off = (size_t)offsets[ctx->rank];
            free(offsets);
        } else {
            const char *base_env = getenv("DS_BASE_OFFSET");
            if (base_env) off = (size_t)atol(base_env);
            const char *per_rank = getenv("DS_PER_RANK_OFFSET");
            if (!per_rank || strcmp(per_rank, "true") == 0)
                off += (size_t)ctx->rank * chunk;
        }

        if (fsize > 0) {
            off %= fsize;
            if (off + chunk > fsize)
                chunk = fsize - off;
        }

        fseek(fp, off, SEEK_SET);
        size_t got = fread(buf, 1, chunk, fp);
        fclose(fp);

        if (got < msg_size)
            memset((char *)buf + got, 0, msg_size - got);
    } else {
        size_t esz = nccl_dtype_size(datatype);
        int count = (msg_size > 0) ? (int)(msg_size / esz) : 1;
        if (count < 1) count = 1;
        init_buffer_pattern(buf, count, nccl_to_data_type(datatype),
                            ctx->config.pattern_type, ctx->rank);
    }
}

/*
 * ── File-based cross-rank aggregation ──────────────────────────────
 *
 * Each rank writes its timing + error data to a unique file, then all
 * ranks stat-wait for every file and read them independently.
 *
 * No NCCL collective involved, so this works with any NCCL plugin
 * (COCCL, UCCL, etc.) without struct-layout issues or library conflicts.
 *
 * A static step counter keeps successive calls (one per message size)
 * from racing on the same file names.
 */
#define AGG_DIR "runtime_files/nccl_agg"

void nccl_report_results(const nccl_test_context_t *ctx,
                         size_t msg_size, int count,
                         float total_time_ms, int local_errors,
                         const validation_result_t *metrics,
                         double bw)
{
    double avg_us = ctx->config.iterations > 0
        ? (double)total_time_ms / ctx->config.iterations * 1000.0
        : 0.0;

    static int agg_step = 0;
    int rank = ctx->rank;
    int nranks = ctx->size;

    if (nranks == 1) {
        /* Single rank — no exchange needed, just print */
        if (rank == 0) {
            printf("Size: %8zu bytes (%6d elements)\n", msg_size, count);
            printf("  User: avg=%8.2f us\n", avg_us);
            printf("  Bandwidth: %8.2f GB/s\n", bw);
            if (ctx->config.validate) {
                printf("  Correct: %s\n", local_errors == 0 ? "YES" : "NO");
                if (metrics && ctx->config.compute_metrics && metrics->num_elements > 0) {
                    for (int i = 0; i < g_metric_registry_count && i < MAX_METRICS; i++) {
                        if (ctx->config.metrics_mask & (1u << i))
                            printf("  %s: %.6e\n", g_metric_registry[i].name, metrics->values[i]);
                    }
                }
            }
            printf("\n");
        }
        return;
    }

    /* ── Ensure directory exists (rank 0) ────────────────────── */
    if (rank == 0) {
        mkdir("runtime_files", 0755);
        mkdir(AGG_DIR, 0755);
    }

    /* ── Write our data ──────────────────────────────────────── */
    double agg_data[3] = { avg_us, (double)local_errors, bw };
    char fname[256];
    snprintf(fname, sizeof(fname), AGG_DIR "/s%d_r%d", agg_step, rank);

    remove(fname);
    FILE *f = fopen(fname, "wb");
    if (!f) { perror("fopen agg write"); return; }
    fwrite(agg_data, sizeof(double), 3, f);
    fclose(f);

    /* ── Wait for and read all other ranks' files ────────────── */
    double *all_data = (double *)calloc((size_t)nranks, 3 * sizeof(double));
    if (!all_data) { perror("calloc agg"); return; }

    for (int r = 0; r < nranks; r++) {
        if (r == rank) {
            all_data[(size_t)r * 3]     = avg_us;
            all_data[(size_t)r * 3 + 1] = (double)local_errors;
            all_data[(size_t)r * 3 + 2] = bw;
            continue;
        }

        char rfname[256];
        snprintf(rfname, sizeof(rfname), AGG_DIR "/s%d_r%d", agg_step, r);

        struct stat st;
        int waited = 0;
        while (stat(rfname, &st) != 0 || st.st_size == 0) {
            usleep(10000);
            if (++waited > 3000) {
                fprintf(stderr, "[agg] timeout waiting for %s\n", rfname);
                free(all_data);
                return;
            }
        }

        FILE *rf = fopen(rfname, "rb");
        if (!rf) { free(all_data); return; }
        size_t n = fread(&all_data[(size_t)r * 3], sizeof(double), 3, rf);
        fclose(rf);
        if (n != 3) {
            fprintf(stderr, "[agg] short read on %s\n", rfname);
            free(all_data);
            return;
        }
    }

    /* ── Compute aggregated stats ────────────────────────────── */
    double min_us = avg_us, max_us = avg_us, sum_us = avg_us;
    double bw_min = bw, bw_max = bw, bw_sum = bw;
    int total_errors = local_errors;
    for (int r = 0; r < nranks; r++) {
        double v = all_data[(size_t)r * 3];
        int    e = (int)all_data[(size_t)r * 3 + 1];
        double b = all_data[(size_t)r * 3 + 2];
        if (v < min_us) min_us = v;
        if (v > max_us) max_us = v;
        sum_us += v;
        total_errors += e;
        if (b < bw_min) bw_min = b;
        if (b > bw_max) bw_max = b;
        bw_sum += b;
    }
    double avg_global = sum_us / nranks;
    double bw_avg = bw_sum / nranks;

    free(all_data);
    agg_step++;

    /* ── Print results (rank 0 only) ─────────────────────────── */
    if (rank == 0) {
        printf("Size: %8zu bytes (%6d elements)\n", msg_size, count);
        printf("  User: avg=%8.2f us, min=%8.2f us, max=%8.2f us\n",
               avg_global, min_us, max_us);
        printf("  Bandwidth: avg=%8.2f GB/s, min=%8.2f GB/s, max=%8.2f GB/s\n",
               bw_avg, bw_min, bw_max);

        if (ctx->config.validate) {
            printf("  Correct: %s\n", total_errors == 0 ? "YES" : "NO");
            if (metrics && ctx->config.compute_metrics && metrics->num_elements > 0) {
                for (int i = 0; i < g_metric_registry_count && i < MAX_METRICS; i++) {
                    if (ctx->config.metrics_mask & (1u << i))
                        printf("  %s: %.6e\n", g_metric_registry[i].name, metrics->values[i]);
                }
            }
        }
        printf("\n");
    }
}

/* ── Type conversion ──────────────────────────────────────────── */

data_type_t nccl_to_data_type(ncclDataType_t nccl_type)
{
    switch (nccl_type) {
    case ncclInt8:    return TYPE_CHAR;
    case ncclUint8:   return TYPE_CHAR;
    case ncclInt32:   return TYPE_INT;
    case ncclUint32:  return TYPE_INT;
    case ncclFloat32: return TYPE_FLOAT;
    case ncclInt64:   return TYPE_DOUBLE;
    case ncclUint64:  return TYPE_DOUBLE;
    case ncclFloat64: return TYPE_DOUBLE;
    default:          return TYPE_FLOAT;
    }
}

ncclDataType_t data_type_to_nccl(data_type_t type)
{
    switch (type) {
    case TYPE_INT:    return ncclInt32;
    case TYPE_FLOAT:  return ncclFloat32;
    case TYPE_DOUBLE: return ncclFloat64;
    case TYPE_CHAR:   return ncclInt8;
    default:          return ncclFloat32;
    }
}

size_t nccl_dtype_size(ncclDataType_t dtype)
{
    switch (dtype) {
    case ncclInt8:    return 1;
    case ncclUint8:   return 1;
    case ncclFloat16: return 2;
    case ncclInt32:   return 4;
    case ncclUint32:  return 4;
    case ncclFloat32: return 4;
    case ncclInt64:   return 8;
    case ncclUint64:  return 8;
    case ncclFloat64: return 8;
    default:          return 4;
    }
}

/* ── Barrier across all ranks (TCP-based) ──────────────────────── */
void nccl_barrier(nccl_test_context_t *ctx)
{
    if (ctx->size <= 1) return;

    char c;
    if (ctx->rank == 0) {
        /* Wait for all peers to arrive */
        for (int r = 1; r < ctx->size; r++) {
            if (read(ctx->barrier_peers[r], &c, 1) != 1) {
                fprintf(stderr, "[rank=0] barrier: lost connection from rank %d\n", r);
                exit(1);
            }
        }
        /* Broadcast release */
        for (int r = 1; r < ctx->size; r++) {
            write(ctx->barrier_peers[r], "1", 1);
        }
    } else {
        /* Signal rank 0 that we've arrived */
        write(ctx->barrier_peers[0], "1", 1);
        /* Wait for release */
        if (read(ctx->barrier_peers[0], &c, 1) != 1) {
            fprintf(stderr, "[rank=%d] barrier: lost connection to rank 0\n", ctx->rank);
            exit(1);
        }
    }
}

/* ── Parse comma-separated int array from env var ─────────────── */
int *nccl_parse_env_int_array(const char *env_name, int expected_len, int *out_len)
{
    const char *s = getenv(env_name);
    if (!s) { *out_len = 0; return NULL; }

    char *copy = strdup(s);
    if (!copy) { fprintf(stderr, "strdup failed\n"); exit(1); }

    int cap = 8, len = 0;
    int *arr = malloc(cap * sizeof(int));
    if (!arr) { free(copy); exit(1); }

    char *tok = strtok(copy, ",");
    while (tok) {
        if (len >= cap) {
            cap *= 2;
            int *tmp = realloc(arr, cap * sizeof(int));
            if (!tmp) { free(arr); free(copy); exit(1); }
            arr = tmp;
        }
        arr[len++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    free(copy);

    if (expected_len > 0 && len != expected_len) {
        fprintf(stderr, "Error: %s has %d values, expected %d\n",
                env_name, len, expected_len);
        free(arr);
        exit(1);
    }

    *out_len = len;
    return arr;
}
