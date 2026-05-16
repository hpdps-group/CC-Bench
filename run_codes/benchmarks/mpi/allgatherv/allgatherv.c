/**
 * MPI_Allgatherv correctness and performance test.
 *
 * Tests user's MPI_Allgatherv against PMPI reference.
 * Each rank sends a potentially different number of elements.
 * Distribution controlled by env ALLGATHERV_WEIGHTS (comma-separated,
 * one per rank).  Falls back to uniform distribution if unset.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "validation.h"
#include "mpi/mpi_utils.h"

/* PMPI declarations */
extern int PMPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, const int recvcounts[], const int displs[],
                           MPI_Datatype recvtype, MPI_Comm comm);
extern int PMPI_Barrier(MPI_Comm comm);
extern int PMPI_Type_size(MPI_Datatype datatype, int *size);
extern int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

/* Build recvcounts and displs from weights and total_elems.
 * Returns total elements in recvbuf (= sum of recvcounts). */
static int build_distribution(int *recvcounts, int *displs,
                               const int *weights, int total_weight,
                               int nranks, int total_elems) {
    int offset = 0;
    for (int i = 0; i < nranks; i++) {
        if (weights) {
            recvcounts[i] = weights[i] * total_elems / total_weight;
            if (recvcounts[i] < 1) recvcounts[i] = 1;
        } else {
            recvcounts[i] = total_elems / nranks;
            if (recvcounts[i] < 1) recvcounts[i] = 1;
        }
        displs[i] = offset;
        offset += recvcounts[i];
    }
    return offset;  /* total_elems in recvbuf */
}

static void run_test_size(const mpi_test_context_t *ctx, size_t msg_size,
                          MPI_Datatype datatype,
                          const int *weights, int total_weight,
                          void *user_sendbuf, void *user_recvbuf,
                          void *ref_sendbuf, void *ref_recvbuf) {
    int dtype_size;
    PMPI_Type_size(datatype, &dtype_size);
    int total_elems = msg_size / dtype_size;
    if (total_elems < ctx->size) total_elems = ctx->size;

    int *recvcounts = malloc(ctx->size * sizeof(int));
    int *displs     = malloc(ctx->size * sizeof(int));
    int total_recv = build_distribution(recvcounts, displs,
                                         weights, total_weight,
                                         ctx->size, total_elems);
    int sendcount = recvcounts[ctx->rank];

    /* 1. Load input data */
    mpi_load_input(ctx, user_sendbuf, msg_size, datatype);
    memcpy(ref_sendbuf, user_sendbuf, msg_size);

    /* 2. Warmup */
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        int _ret = MPI_Allgatherv(user_sendbuf, sendcount, datatype,
                       user_recvbuf, recvcounts, displs, datatype,
                       MPI_COMM_WORLD);
        if (_ret != MPI_SUCCESS) {
            fprintf(stderr, "[rank=%d] MPI_Allgatherv FAILED -- aborting\n", ctx->rank);
            exit(1);
        }
        PMPI_Allgatherv(ref_sendbuf, sendcount, datatype,
                        ref_recvbuf, recvcounts, displs, datatype,
                        MPI_COMM_WORLD);
    }

    /* 3. Reference result */
    PMPI_Allgatherv(ref_sendbuf, sendcount, datatype,
                    ref_recvbuf, recvcounts, displs, datatype,
                    MPI_COMM_WORLD);

    /* 4. Main timing and validation loop */
    double total_time = 0.0;
    int iter_errors = 0;
    validation_result_t metrics_acc = {0};
    metrics_acc.num_elements = total_recv;

    for (int iter = 0; iter < ctx->config.iterations; iter++) {
        PMPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        int _ret = MPI_Allgatherv(user_sendbuf, sendcount, datatype,
                       user_recvbuf, recvcounts, displs, datatype,
                       MPI_COMM_WORLD);
        if (_ret != MPI_SUCCESS) {
            fprintf(stderr, "[rank=%d] MPI_Allgatherv FAILED -- aborting\n", ctx->rank);
            exit(1);
        }
        PMPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        total_time += (end - start);

        if (ctx->config.validate) {
            validation_result_t iter_metrics = validate_result(
                user_recvbuf, ref_recvbuf, total_recv,
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

    /* 6. Bandwidth: each rank receives (total_recv - sendcount) from others */
    double avg_sec = total_time / ctx->config.iterations;
    size_t recv_other = (size_t)(total_recv - sendcount) * dtype_size;
    double bw = 0.0;
    if (avg_sec > 0.0)
        bw = (double)recv_other / avg_sec;
    bw /= 1.0e9; /* GB/s */

    double bw_min = 0, bw_max = 0, bw_sum = 0;
    PMPI_Reduce(&bw, &bw_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&bw, &bw_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&bw, &bw_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    bw_sum /= ctx->size;

    if (ctx->rank == 0)
        printf("  Bandwidth: avg=%8.2f GB/s, min=%8.2f GB/s, max=%8.2f GB/s\n",
               bw_sum, bw_min, bw_max);

    /* 7. Report (sendcount per rank, total across all = total_recv) */
    mpi_report_results(ctx, msg_size, sendcount, total_time, iter_errors,
                       ctx->config.validate ? &metrics_acc : NULL);

    free(recvcounts);
    free(displs);
}

int main(int argc, char **argv) {
    mpi_test_context_t ctx = mpi_test_init(argc, argv, "MPI_Allgatherv");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        mpi_test_fini(&ctx);
        return 1;
    }


    /* Parse weights from env (fallback: uniform = NULL) */
    int n_weights = 0;
    int *weights = parse_env_int_array("ALLGATHERV_WEIGHTS", ctx.size, &n_weights);
    int total_weight = 0;
    if (weights) {
        for (int i = 0; i < n_weights; i++) total_weight += weights[i];
    }

    MPI_Datatype datatype = data_type_to_mpi(ctx.config.data_type);

    size_t max_size = ctx.config.max_message_size;
    /* Worst-case recvbuf: all data goes to one rank = size * max_size */
    size_t worst_case_bytes = max_size * ctx.size;

    void *user_sendbuf = allocate_aligned_buffer(max_size, 64);
    void *user_recvbuf = allocate_aligned_buffer(worst_case_bytes, 64);
    void *ref_sendbuf  = allocate_aligned_buffer(max_size, 64);
    void *ref_recvbuf  = allocate_aligned_buffer(worst_case_bytes, 64);

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
                      weights, total_weight,
                      user_sendbuf, user_recvbuf,
                      ref_sendbuf, ref_recvbuf);
    }

    free_aligned_buffer(user_sendbuf);
    free_aligned_buffer(user_recvbuf);
    free_aligned_buffer(ref_sendbuf);
    free_aligned_buffer(ref_recvbuf);
    free(weights);

    mpi_test_fini(&ctx);
    return 0;
}
