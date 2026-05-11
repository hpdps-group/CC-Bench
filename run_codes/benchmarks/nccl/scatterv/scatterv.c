/**
 * NCCL Scatterv correctness and performance test.
 *
 * Root scatters potentially different-sized chunks to each rank.
 * Distribution controlled by env SCATTERV_WEIGHTS (comma-separated).
 * Falls back to uniform distribution if unset.
 * No base_ncclScatterv — validated against CPU-computed reference.
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "validation.h"
#include "base_impl.h"
#include "nccl/nccl_utils.h"

#define ROOT 0

/* ── Forward declaration (from plain_nccl_compress.c) ─────────────────── */
extern int ncclScatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                        ncclDataType_t sendtype,
                        void *recvbuf, int recvcount, ncclDataType_t recvtype,
                        int root, const int *d_sendcounts, const int *d_displs,
                        ncclComm_t comm, cudaStream_t stream);

static int build_distribution(int *sendcounts, int *displs,
                               const int *weights, int total_weight,
                               int nranks, int total_elems)
{
    int off = 0;
    for (int i = 0; i < nranks; i++) {
        if (weights) {
            sendcounts[i] = weights[i] * total_elems / total_weight;
            if (sendcounts[i] < 1) sendcounts[i] = 1;
        } else {
            sendcounts[i] = total_elems / nranks;
            if (sendcounts[i] < 1) sendcounts[i] = 1;
        }
        displs[i] = off;
        off += sendcounts[i];
    }
    return off;
}

static void run_test_size(const nccl_test_context_t *ctx, size_t msg_size,
                          ncclDataType_t datatype,
                          const int *weights, int total_weight,
                          void *h_send, void *h_recv, void *h_ref,
                          void *d_send, void *d_recv,
                          int *d_sendcounts, int *d_displs)
{
    size_t esz = nccl_dtype_size(datatype);
    int total_elems = (int)(msg_size / esz);
    if (total_elems < ctx->size) total_elems = ctx->size;

    int *sendcounts = malloc(ctx->size * sizeof(int));
    int *displs     = malloc(ctx->size * sizeof(int));
    int total_send  = build_distribution(sendcounts, displs,
                                          weights, total_weight,
                                          ctx->size, total_elems);
    int recvcount = sendcounts[ctx->rank];
    size_t root_send_bytes = (size_t)total_send * esz;
    size_t recv_bytes = (size_t)recvcount * esz;

    /* Upload arrays to GPU */
    cudaMemcpy(d_sendcounts, sendcounts, ctx->size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_displs, displs, ctx->size * sizeof(int), cudaMemcpyHostToDevice);

    /* 1. Load input on root */
    if (ctx->rank == ROOT) {
        nccl_load_input(ctx, h_send, root_send_bytes, datatype);
        cudaMemcpy(d_send, h_send, root_send_bytes, cudaMemcpyHostToDevice);
    }
    nccl_barrier((nccl_test_context_t *)ctx);

    /* 2. CPU reference: each rank expects data generated with its own rank */
    if (ctx->config.validate) {
        data_type_t dt = nccl_to_data_type(datatype);
        init_buffer_pattern(h_ref, recvcount, dt, ctx->config.pattern_type, ctx->rank);
    }

    /* 3. Warmup */
    cudaStreamSynchronize(ctx->stream);
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        ncclScatterv(d_send, sendcounts, displs, datatype,
                     d_recv, recvcount, datatype,
                     ROOT, d_sendcounts, d_displs,
                     ctx->comm, ctx->stream);
        cudaStreamSynchronize(ctx->stream);
    }

    /* 4. Main timing loop */
    float total_time_ms = 0.0f;
    int iter_errors = 0;
    validation_result_t metrics_acc = {0};
    metrics_acc.num_elements = recvcount;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < ctx->config.iterations; iter++) {
        nccl_barrier((nccl_test_context_t *)ctx);

        cudaEventRecord(start, ctx->stream);
        ncclScatterv(d_send, sendcounts, displs, datatype,
                     d_recv, recvcount, datatype,
                     ROOT, d_sendcounts, d_displs,
                     ctx->comm, ctx->stream);
        cudaEventRecord(stop, ctx->stream);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_ms += ms;

        if (ctx->config.validate) {
            cudaMemcpy(h_recv, d_recv, recv_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(ctx->stream);
            validation_result_t vm = validate_result(
                h_recv, h_ref, recvcount,
                nccl_to_data_type(datatype), ctx->config.tolerance,
                ctx->config.metrics_mask);
            if (!vm.correct) iter_errors++;
            for (int m = 0; m < MAX_METRICS; m++) {
                if (ctx->config.metrics_mask & (1u << m))
                    metrics_acc.values[m] += vm.values[m];
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (ctx->config.validate && ctx->config.iterations > 0) {
        metrics_acc.correct = (iter_errors == 0);
        for (int m = 0; m < MAX_METRICS; m++) {
            if (ctx->config.metrics_mask & (1u << m))
                metrics_acc.values[m] /= ctx->config.iterations;
        }
        if (iter_errors > 0 && ctx->rank == ROOT)
            printf("  Validation FAILED at size %zu (%d/%d iterations failed)\n",
                   msg_size, iter_errors, ctx->config.iterations);
    }

    nccl_report_results(ctx, recv_bytes, recvcount, total_time_ms, iter_errors,
                        ctx->config.validate ? &metrics_acc : NULL);

    free(sendcounts);
    free(displs);
}

int main(int argc, char **argv)
{
    nccl_test_context_t ctx = nccl_test_init(argc, argv, "NCCL_Scatterv");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        nccl_test_fini(&ctx);
        return 1;
    }

    /* Parse weights */
    int n_weights = 0;
    int *weights = nccl_parse_env_int_array("SCATTERV_WEIGHTS", ctx.size, &n_weights);
    int total_weight = 0;
    if (weights) {
        for (int i = 0; i < n_weights; i++) total_weight += weights[i];
    }

    ncclDataType_t datatype = data_type_to_nccl(ctx.config.data_type);

    size_t max_sz = ctx.config.max_message_size;
    size_t root_max = max_sz * ctx.size;

    void *h_send = malloc(root_max);
    void *h_recv = malloc(max_sz);
    void *h_ref  = malloc(max_sz);
    void *d_send = NULL; cudaMalloc(&d_send, root_max);
    void *d_recv = NULL; cudaMalloc(&d_recv, max_sz);
    int *d_sendcounts = NULL; cudaMalloc(&d_sendcounts, ctx.size * sizeof(int));
    int *d_displs     = NULL; cudaMalloc(&d_displs, ctx.size * sizeof(int));

    if (!h_send || !h_recv || !h_ref || !d_send || !d_recv ||
        !d_sendcounts || !d_displs) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(h_send); free(h_recv); free(h_ref);
        cudaFree(d_send); cudaFree(d_recv);
        cudaFree(d_sendcounts); cudaFree(d_displs);
        free(weights);
        nccl_test_fini(&ctx);
        return 1;
    }

    size_iter_t it;
    size_iter_init(&it, &ctx.config);
    size_t sz;
    while (size_iter_next(&it, &sz))
        run_test_size(&ctx, sz, datatype,
                      weights, total_weight,
                      h_send, h_recv, h_ref,
                      d_send, d_recv, d_sendcounts, d_displs);

    free(h_send); free(h_recv); free(h_ref);
    cudaFree(d_send); cudaFree(d_recv);
    cudaFree(d_sendcounts); cudaFree(d_displs);
    free(weights);
    nccl_test_fini(&ctx);
    return 0;
}
