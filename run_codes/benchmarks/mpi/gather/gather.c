/**
 * MPI_Gather correctness and performance test.
 *
 * Tests user's MPI_Gather implementation against PMPI reference.
 * All ranks send data to root, which gathers them.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "validation.h"
#include "base_impl.h"
#include "mpi/mpi_utils.h"
#include "binary_output.h"

/* PMPI declarations */
extern int PMPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       int root, MPI_Comm comm);
extern int PMPI_Barrier(MPI_Comm comm);
extern int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
extern int PMPI_Type_size(MPI_Datatype datatype, int *size);

#define ROOT 0

static void run_test_size(const mpi_test_context_t *ctx, size_t msg_size,
                          MPI_Datatype datatype,
                          void *user_sendbuf, void *user_recvbuf,
                          void *ref_sendbuf, void *ref_recvbuf) {
    int dtype_size;
    PMPI_Type_size(datatype, &dtype_size);
    int count_per_rank = msg_size / dtype_size;
    if (count_per_rank <= 0) count_per_rank = 1;

    /* 1. Load input data (all ranks have their piece) */
    mpi_load_input(ctx, user_sendbuf, msg_size, datatype);
    memcpy(ref_sendbuf, user_sendbuf, msg_size);

    /* 2. Warmup */
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        MPI_Gather(user_sendbuf, count_per_rank, datatype,
                   user_recvbuf, count_per_rank, datatype,
                   ROOT, MPI_COMM_WORLD);
        PMPI_Gather(ref_sendbuf, count_per_rank, datatype,
                    ref_recvbuf, count_per_rank, datatype,
                    ROOT, MPI_COMM_WORLD);
    }

    /* 3. Reference result */
    PMPI_Gather(ref_sendbuf, count_per_rank, datatype,
                ref_recvbuf, count_per_rank, datatype,
                ROOT, MPI_COMM_WORLD);

    /* 3a. Allocate user accumulator for binary output */
    data_type_t dtype_gen = mpi_to_data_type(datatype);
    size_t total_recv_bytes = msg_size * ctx->size;
    void *user_accum = ctx->config.save_binary ? calloc(1, total_recv_bytes) : NULL;

    /* 4. Main timing and validation loop */
    double total_time = 0.0;
    int iter_errors = 0;
    validation_result_t metrics_acc = {0};
    metrics_acc.num_elements = count_per_rank * ctx->size;

    for (int iter = 0; iter < ctx->config.iterations; iter++) {
        PMPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        MPI_Gather(user_sendbuf, count_per_rank, datatype,
                   user_recvbuf, count_per_rank, datatype,
                   ROOT, MPI_COMM_WORLD);
        PMPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        total_time += (end - start);

        /* Accumulate user result on root for binary output */
        if (user_accum && ctx->rank == ROOT) binary_accumulate(user_accum, user_recvbuf, count_per_rank * ctx->size, dtype_gen);

        /* Validate on root (only root has the full gathered result) */
        if (ctx->config.validate && ctx->rank == ROOT) {
            validation_result_t iter_metrics = validate_result(
                user_recvbuf, ref_recvbuf, count_per_rank * ctx->size,
                mpi_to_data_type(datatype), ctx->config.tolerance,
                ctx->config.metrics_mask);
            if (!iter_metrics.correct) iter_errors++;
            for (int m = 0; m < MAX_METRICS; m++) {
                if (ctx->config.metrics_mask & (1u << m))
                    metrics_acc.values[m] += iter_metrics.values[m];
            }
        }
    }

    /* 5. Average metrics */
    if (ctx->config.validate && ctx->config.iterations > 0) {
        if (ctx->rank == ROOT) {
            metrics_acc.correct = (iter_errors == 0);
            for (int m = 0; m < MAX_METRICS; m++) {
                if (ctx->config.metrics_mask & (1u << m))
                    metrics_acc.values[m] /= ctx->config.iterations;
            }
            if (iter_errors > 0) {
                printf("  Validation FAILED at size %zu (%d/%d iterations failed)\n",
                       msg_size, iter_errors, ctx->config.iterations);
            }
        }
    }

    /* 5a. Average and write binary output */
    if (user_accum) {
        if (ctx->config.iterations > 0 && ctx->rank == ROOT)
            binary_average(user_accum, count_per_rank * ctx->size, dtype_gen, ctx->config.iterations);
        write_binary_single(ctx->config.bin_path, "reference", ref_recvbuf, total_recv_bytes, ctx->rank, ROOT);
        write_binary_single(ctx->config.bin_path, "user", user_accum, total_recv_bytes, ctx->rank, ROOT);
        free(user_accum);
    }

    /* 6. Bandwidth: each rank sends msg_size to root */
    double avg_sec = total_time / ctx->config.iterations;
    double bw = 0.0;
    if (avg_sec > 0.0)
        bw = msg_size / avg_sec;
    bw /= 1.0e9; /* GB/s */

    double bw_min = 0, bw_max = 0, bw_sum = 0;
    PMPI_Reduce(&bw, &bw_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&bw, &bw_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&bw, &bw_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    bw_sum /= ctx->size;

    if (ctx->rank == 0)
        printf("  Bandwidth: avg=%8.2f GB/s, min=%8.2f GB/s, max=%8.2f GB/s\n",
               bw_sum, bw_min, bw_max);

    /* 7. Report */
    mpi_report_results(ctx, msg_size, count_per_rank,
                       total_time, iter_errors,
                       ctx->config.validate ? &metrics_acc : NULL);
}

int main(int argc, char **argv) {
    mpi_test_context_t ctx = mpi_test_init(argc, argv, "MPI_Gather");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        mpi_test_fini(&ctx);
        return 1;
    }

    if (load_base_impl(BASE_SO_FILE) != 0) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: base impl not found — run ./bin/mpi/findso first\n");
        mpi_test_fini(&ctx);
        return 1;
    }

    MPI_Datatype datatype = data_type_to_mpi(ctx.config.data_type);

    size_t max_size = ctx.config.max_message_size;
    size_t root_max_bytes = max_size * ctx.size;

    /* All ranks need sendbuf (one piece), root needs larger recvbuf (all pieces) */
    void *user_sendbuf = allocate_aligned_buffer(max_size, 64);
    void *user_recvbuf = allocate_aligned_buffer(root_max_bytes, 64);
    void *ref_sendbuf  = allocate_aligned_buffer(max_size, 64);
    void *ref_recvbuf  = allocate_aligned_buffer(root_max_bytes, 64);

    if (!user_sendbuf || !user_recvbuf || !ref_sendbuf || !ref_recvbuf) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Memory allocation failed\n");
        free_aligned_buffer(user_sendbuf);
        free_aligned_buffer(user_recvbuf);
        free_aligned_buffer(ref_sendbuf);
        free_aligned_buffer(ref_recvbuf);
        mpi_test_fini(&ctx);
        return 1;
    }

    size_iter_t msg_iter;
    size_iter_init(&msg_iter, &ctx.config);
    size_t msg_size;
    while (size_iter_next(&msg_iter, &msg_size)) {
        run_test_size(&ctx, msg_size, datatype,
                     user_sendbuf, user_recvbuf,
                     ref_sendbuf, ref_recvbuf);
    }

    free_aligned_buffer(user_sendbuf);
    free_aligned_buffer(user_recvbuf);
    free_aligned_buffer(ref_sendbuf);
    free_aligned_buffer(ref_recvbuf);

    mpi_test_fini(&ctx);
    return 0;
}
