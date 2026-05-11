/**
 * NCCL benchmark utilities — no MPI dependency.
 *
 * Bootstrap: rank/size from environment variables, NCCL unique ID
 * via file-based exchange.  Timing via CUDA events.
 * Results aggregated across ranks using base_ncclAllReduce
 * (real NCCL loaded by findso + get_base_implementation.c).
 */

#include "nccl/nccl_utils.h"
#include "nccl/nccl_base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

/* Fixed path used for NCCL unique-ID bootstrap across processes */
#define NCCL_ID_FILE "nccl_id_file/nccl_bench_id"

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
        NCCLCHECK(ncclGetUniqueId(&id));
        FILE *f = fopen(NCCL_ID_FILE, "wb");
        if (!f) { perror("fopen NCCL_ID_FILE"); exit(1); }
        fwrite(&id, sizeof(id), 1, f);
        fclose(f);
    }

    if (size > 1 && rank != 0) {
        struct stat st;
        int waited = 0;
        while (stat(NCCL_ID_FILE, &st) != 0 || st.st_size == 0) {
            usleep(10000); /* 10 ms */
            waited++;
            if (waited > 3000) { /* 30 s timeout */
                fprintf(stderr, "Rank %d: timeout waiting for NCCL id\n", rank);
                exit(1);
            }
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

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    return comm;
}

/* ── Public API ───────────────────────────────────────────────── */

nccl_test_context_t nccl_test_init(int argc, char **argv,
                                   const char *test_name)
{
    nccl_test_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));

    get_rank_size(&ctx.rank, &ctx.size);

    /* Pick GPU: rank % available devices */
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    cudaSetDevice(ctx.rank % (ndev > 0 ? ndev : 1));
    cudaStreamCreate(&ctx.stream);

    /* Allocate one int for barrier */
    cudaMalloc(&ctx.d_barrier, sizeof(int));

    /* NCCL communicator */
    ctx.comm = init_nccl_comm(ctx.rank, ctx.size);

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

    if (ctx->d_barrier) cudaFree(ctx->d_barrier);
    ncclCommDestroy(ctx->comm);
    cudaStreamDestroy(ctx->stream);

    /* Clean up NCCL id file (best-effort) */
    if (ctx->rank == 0)
        remove(NCCL_ID_FILE);
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

void nccl_report_results(const nccl_test_context_t *ctx,
                         size_t msg_size, int count,
                         float total_time_ms, int local_errors,
                         const validation_result_t *metrics)
{
    double avg_us = (double)total_time_ms / ctx->config.iterations * 1000.0;

    /* Allocate GPU buffers for cross-rank aggregation */
    double *d_in, *d_out;
    size_t sz = sizeof(double);
    cudaMalloc(&d_in, sz);
    cudaMalloc(&d_out, sz);

    /* Min */
    cudaMemcpy(d_in, &avg_us, sz, cudaMemcpyHostToDevice);
    base_ncclAllReduce(d_in, d_out, 1, ncclFloat64, ncclMin,
                       ctx->comm, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    double min_us;
    cudaMemcpy(&min_us, d_out, sz, cudaMemcpyDeviceToHost);

    /* Max */
    cudaMemcpy(d_in, &avg_us, sz, cudaMemcpyHostToDevice);
    base_ncclAllReduce(d_in, d_out, 1, ncclFloat64, ncclMax,
                       ctx->comm, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    double max_us;
    cudaMemcpy(&max_us, d_out, sz, cudaMemcpyDeviceToHost);

    /* Sum → global average */
    cudaMemcpy(d_in, &avg_us, sz, cudaMemcpyHostToDevice);
    base_ncclAllReduce(d_in, d_out, 1, ncclFloat64, ncclSum,
                       ctx->comm, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    double sum_us;
    cudaMemcpy(&sum_us, d_out, sz, cudaMemcpyDeviceToHost);
    double avg_global = sum_us / ctx->size;

    cudaFree(d_in);
    cudaFree(d_out);

    /* Total errors across ranks */
    int total_errors;
    int *d_err;
    cudaMalloc(&d_err, sizeof(int));
    cudaMemcpy(d_err, &local_errors, sizeof(int), cudaMemcpyHostToDevice);
    base_ncclAllReduce(d_err, d_err, 1, ncclInt32, ncclSum,
                       ctx->comm, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(&total_errors, d_err, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_err);

    if (ctx->rank == 0) {
        printf("Size: %8zu bytes (%6d elements)\n", msg_size, count);
        printf("  User: avg=%8.2f us, min=%8.2f us, max=%8.2f us\n",
               avg_global, min_us, max_us);

        if (ctx->config.validate) {
            printf("  Correct: %s\n",
                   total_errors == 0 ? "YES" : "NO");
            if (metrics && ctx->config.compute_metrics &&
                metrics->num_elements > 0) {
                for (int i = 0; i < g_metric_registry_count && i < MAX_METRICS; i++) {
                    if (ctx->config.metrics_mask & (1u << i)) {
                        printf("  %s: %.6e\n", g_metric_registry[i].name,
                               metrics->values[i]);
                    }
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

/* ── Barrier using ncclAllReduce (bypasses wrapper) ────────────── */
void nccl_barrier(nccl_test_context_t *ctx)
{
    int zero = 0;
    cudaMemcpyAsync(ctx->d_barrier, &zero, sizeof(int),
                    cudaMemcpyHostToDevice, ctx->stream);
    base_ncclAllReduce(ctx->d_barrier, ctx->d_barrier, 1,
                       ncclInt32, ncclSum, ctx->comm, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
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
