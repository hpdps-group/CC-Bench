/**
 * MPI_Reduce_scatter correctness and performance test.
 *
 * Tests user's MPI_Reduce_scatter against PMPI reference.
 * Each rank contributes data; result is reduced and scattered so each rank
 * gets a distinct chunk of the final reduction (equal-sized blocks).
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "validation.h"
#include "mpi/mpi_utils.h"
#include "binary_output.h"

/* PMPI declarations */
extern int PMPI_Reduce_scatter(const void *sendbuf, void *recvbuf,
                               const int recvcounts[], MPI_Datatype datatype,
                               MPI_Op op, MPI_Comm comm);
extern int PMPI_Barrier(MPI_Comm comm);
extern int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
extern int PMPI_Type_size(MPI_Datatype datatype, int *size);

static void run_test_size(const mpi_test_context_t *ctx, size_t msg_size,
                          MPI_Datatype datatype, MPI_Op op,
                          const int *recvcounts, int sendcount,
                          void *user_sendbuf, void *user_recvbuf,
                          void *ref_sendbuf, void *ref_recvbuf)
{
    int dtype_size;
    PMPI_Type_size(datatype, &dtype_size);
    int recvcount = recvcounts[ctx->rank];
    size_t send_bytes = (size_t)sendcount * dtype_size;
    size_t recv_bytes = (size_t)recvcount * dtype_size;

    /* 1. Load input data */
    mpi_load_input(ctx, user_sendbuf, send_bytes, datatype);
    memcpy(ref_sendbuf, user_sendbuf, send_bytes);

    /* 2. Warmup */
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        int _ret = MPI_Reduce_scatter(user_sendbuf, user_recvbuf, recvcounts,
                                      datatype, op, MPI_COMM_WORLD);
        if (_ret != MPI_SUCCESS) {
            fprintf(stderr, "[rank=%d] MPI_Reduce_scatter FAILED -- aborting\n", ctx->rank);
            exit(1);
        }
        PMPI_Reduce_scatter(ref_sendbuf, ref_recvbuf, recvcounts,
                            datatype, op, MPI_COMM_WORLD);
    }

    /* 3. Reference result */
    PMPI_Reduce_scatter(ref_sendbuf, ref_recvbuf, recvcounts,
                        datatype, op, MPI_COMM_WORLD);

    /* 3a. Allocate user accumulator for binary output */
    data_type_t dtype_gen = mpi_to_data_type(datatype);
    void *user_accum = ctx->config.save_binary ? calloc(1, recv_bytes) : NULL;

    /* 4. Main timing and validation loop */
    double total_time = 0.0;
    int iter_errors = 0;
    validation_result_t metrics_acc = {0};
    metrics_acc.num_elements = recvcount;

    for (int iter = 0; iter < ctx->config.iterations; iter++) {
        PMPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        int _ret = MPI_Reduce_scatter(user_sendbuf, user_recvbuf, recvcounts,
                                      datatype, op, MPI_COMM_WORLD);
        if (_ret != MPI_SUCCESS) {
            fprintf(stderr, "[rank=%d] MPI_Reduce_scatter FAILED -- aborting\n", ctx->rank);
            exit(1);
        }
        PMPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        total_time += (end - start);

        /* Accumulate user result across iterations for binary output */
        if (user_accum) binary_accumulate(user_accum, user_recvbuf, recvcount, dtype_gen);

        if (ctx->config.validate) {
            validation_result_t iter_metrics = validate_result(
                user_recvbuf, ref_recvbuf, recvcount,
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
        metrics_acc.correct = (iter_errors == 0);
        for (int m = 0; m < MAX_METRICS; m++) {
            if (ctx->config.metrics_mask & (1u << m))
                metrics_acc.values[m] /= ctx->config.iterations;
        }
        if (iter_errors > 0 && ctx->rank == 0) {
            printf("  Validation FAILED at size %zu (%d/%d iterations failed)\n",
                   msg_size, iter_errors, ctx->config.iterations);
        }
    }

    /* 5a. Average and write binary output */
    if (user_accum) {
        if (ctx->config.iterations > 0)
            binary_average(user_accum, recvcount, dtype_gen, ctx->config.iterations);
        write_binary_multi(ctx->config.bin_path, "reference",
                           ref_recvbuf, recv_bytes, ctx->rank, ctx->size);
        write_binary_multi(ctx->config.bin_path, "user",
                           user_accum, recv_bytes, ctx->rank, ctx->size);
        free(user_accum);
    }

    /* 6. Bandwidth: each rank recvs recv_bytes */
    double avg_sec = total_time / ctx->config.iterations;
    double bw = 0.0;
    if (avg_sec > 0.0)
        bw = recv_bytes / avg_sec;
    bw /= 1.0e9;

    double bw_min = 0, bw_max = 0, bw_sum = 0;
    PMPI_Reduce(&bw, &bw_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&bw, &bw_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&bw, &bw_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    bw_sum /= ctx->size;

    if (ctx->rank == 0)
        printf("  Bandwidth: avg=%8.2f GB/s, min=%8.2f GB/s, max=%8.2f GB/s\n",
               bw_sum, bw_min, bw_max);

    /* 7. Report */
    mpi_report_results(ctx, recv_bytes, recvcount, total_time, iter_errors,
                       ctx->config.validate ? &metrics_acc : NULL);
}

int main(int argc, char **argv)
{
    mpi_test_context_t ctx = mpi_test_init(argc, argv, "MPI_Reduce_scatter");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        mpi_test_fini(&ctx);
        return 1;
    }


    MPI_Datatype datatype = data_type_to_mpi(ctx.config.data_type);
    MPI_Op op = MPI_SUM;

    /* Equal recvcounts for each rank */
    int *recvcounts = malloc(ctx.size * sizeof(int));
    if (!recvcounts) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Memory allocation failed\n");
        mpi_test_fini(&ctx);
        return 1;
    }

    size_t max_size = ctx.config.max_message_size;
    /* sendbuf needs recvcount * size elements worst-case */
    size_t send_max = max_size * ctx.size;
    size_t recv_max = max_size;

    void *user_sendbuf = allocate_aligned_buffer(send_max, 64);
    void *user_recvbuf = allocate_aligned_buffer(recv_max, 64);
    void *ref_sendbuf  = allocate_aligned_buffer(send_max, 64);
    void *ref_recvbuf  = allocate_aligned_buffer(recv_max, 64);

    if (!user_sendbuf || !user_recvbuf || !ref_sendbuf || !ref_recvbuf) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Memory allocation failed\n");
        free_aligned_buffer(user_sendbuf);
        free_aligned_buffer(user_recvbuf);
        free_aligned_buffer(ref_sendbuf);
        free_aligned_buffer(ref_recvbuf);
        free(recvcounts);
        mpi_test_fini(&ctx);
        return 1;
    }

    size_iter_t msg_iter;
    size_iter_init(&msg_iter, &ctx.config);
    size_t msg_size;
    while (size_iter_next(&msg_iter, &msg_size)) {
        int dtype_size;
        MPI_Type_size(datatype, &dtype_size);
        int recvcount = msg_size / dtype_size;
        if (recvcount <= 0) recvcount = 1;

        /* All ranks get equal recvcounts */
        for (int i = 0; i < ctx.size; i++)
            recvcounts[i] = recvcount;

        int sendcount = recvcount * ctx.size;
        run_test_size(&ctx, msg_size, datatype, op,
                      recvcounts, sendcount,
                      user_sendbuf, user_recvbuf,
                      ref_sendbuf, ref_recvbuf);
    }

    free_aligned_buffer(user_sendbuf);
    free_aligned_buffer(user_recvbuf);
    free_aligned_buffer(ref_sendbuf);
    free_aligned_buffer(ref_recvbuf);
    free(recvcounts);

    mpi_test_fini(&ctx);
    return 0;
}
