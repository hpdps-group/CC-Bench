/**
 * NCCL AllGatherv correctness and performance test.
 *
 * Each rank sends a potentially different number of elements.
 * Distribution controlled by env ALLGATHERV_WEIGHTS (comma-separated,
 * one per rank).  Falls back to uniform if unset.
 *
 * No base_ncclAllGatherv — validated against CPU-computed reference.
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "validation.h"
#include "nccl/nccl_utils.h"

/* Build recvcounts/displs from weights.  Returns total recv elements. */
static int build_distribution(int *recvcounts, int *displs,
                               const int *weights, int total_weight,
                               int nranks, int total_elems)
{
    int off = 0;
    for (int i = 0; i < nranks; i++) {
        if (weights) {
            recvcounts[i] = weights[i] * total_elems / total_weight;
            if (recvcounts[i] < 1) recvcounts[i] = 1;
        } else {
            recvcounts[i] = total_elems / nranks;
            if (recvcounts[i] < 1) recvcounts[i] = 1;
        }
        displs[i] = off;
        off += recvcounts[i];
    }
    return off;
}

static void run_test_size(const nccl_test_context_t *ctx, size_t msg_size,
                          ncclDataType_t datatype,
                          const int *weights, int total_weight,
                          void *h_send, void *h_recv, void *h_ref,
                          void *d_send, void *d_recv,
                          int *d_recvcounts, int *d_displs)
{
    size_t esz = nccl_dtype_size(datatype);
    int total_elems = (int)(msg_size / esz);
    if (total_elems < ctx->size) total_elems = ctx->size;

    int *recvcounts = malloc(ctx->size * sizeof(int));
    int *displs     = malloc(ctx->size * sizeof(int));
    int total_recv  = build_distribution(recvcounts, displs,
                                          weights, total_weight,
                                          ctx->size, total_elems);
    int sendcount = recvcounts[ctx->rank];
    size_t send_bytes = (size_t)sendcount * esz;
    size_t recv_bytes = (size_t)total_recv * esz;

    /* Upload recvcounts/displs to GPU for ncclAllGatherv */
    cudaMemcpy(d_recvcounts, recvcounts, ctx->size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_displs, displs, ctx->size * sizeof(int), cudaMemcpyHostToDevice);

    /* 1. Load input */
    nccl_load_input(ctx, h_send, send_bytes, datatype);
    cudaMemcpy(d_send, h_send, send_bytes, cudaMemcpyHostToDevice);

    /* 2. CPU reference: generate each rank's chunk with init_buffer_pattern */
    if (ctx->config.validate) {
        data_type_t dt = nccl_to_data_type(datatype);
        void *tmp = malloc((size_t)total_recv * esz);
        for (int r = 0; r < ctx->size; r++) {
            init_buffer_pattern(tmp, recvcounts[r], dt, ctx->config.pattern_type, r);
            memcpy((char*)h_ref + (size_t)displs[r] * esz, tmp, (size_t)recvcounts[r] * esz);
        }
        free(tmp);
    }

    /* 3. Warmup: need a barrier to ensure recvcounts/displs are ready */
    nccl_barrier((nccl_test_context_t *)ctx);
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        ncclResult_t _ret = ncclAllGatherv(d_send, sendcount, datatype,
                                           d_recv, recvcounts, displs, datatype,
                                           d_recvcounts, d_displs,
                                           ctx->comm, ctx->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[rank=%d] ncclAllGatherv FAILED at size=%zu iter=%d "
                            "error=%d -- aborting\n",
                    ctx->rank, msg_size, i, (int)_ret);
            exit(1);
        }
        cudaStreamSynchronize(ctx->stream);
    }

    /* 4. Reference: just use the CPU-computed h_ref (already done above) */

    /* 5. Main timing loop */
    float total_time_ms = 0.0f;
    int iter_errors = 0;
    validation_result_t metrics_acc = {0};
    metrics_acc.num_elements = total_recv;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < ctx->config.iterations; iter++) {
        /* Re-upload our send data */
        cudaMemcpy(d_send, h_send, send_bytes, cudaMemcpyHostToDevice);
        nccl_barrier((nccl_test_context_t *)ctx);

        cudaEventRecord(start, ctx->stream);
        ncclResult_t _ret = ncclAllGatherv(d_send, sendcount, datatype,
                                           d_recv, recvcounts, displs, datatype,
                                           d_recvcounts, d_displs,
                                           ctx->comm, ctx->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[rank=%d] ncclAllGatherv FAILED at size=%zu iter=%d "
                            "error=%d -- aborting\n",
                    ctx->rank, msg_size, iter, (int)_ret);
            exit(1);
        }
        cudaEventRecord(stop, ctx->stream);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_ms += ms;

        if (ctx->config.validate) {
            cudaMemcpy(h_recv, d_recv, recv_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(ctx->stream);
            validation_result_t vm = validate_result(
                h_recv, h_ref, total_recv,
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
        if (iter_errors > 0 && ctx->rank == 0)
            printf("  Validation FAILED at size %zu (%d/%d iterations failed)\n",
                   msg_size, iter_errors, ctx->config.iterations);
    }

    /* 6. Bandwidth: each rank receives (recv_bytes - send_bytes) from others */
    double avg_sec = (double)total_time_ms / ctx->config.iterations / 1000.0;
    double bw = 0.0;
    if (avg_sec > 0.0)
        bw = (recv_bytes - send_bytes) / avg_sec;
    bw /= 1.0e9;
    nccl_report_results(ctx, send_bytes, sendcount, total_time_ms, iter_errors,
                        ctx->config.validate ? &metrics_acc : NULL, bw);

    free(recvcounts);
    free(displs);
}

/* ── Forward declaration of ncclAllGatherv (from plain_nccl_compress) ──── */
extern int ncclAllGatherv(const void *sendbuf, int sendcount, ncclDataType_t sendtype,
                          void *recvbuf, const int recvcounts[], const int displs[],
                          ncclDataType_t recvtype,
                          const int *d_recvcounts, const int *d_displs,
                          ncclComm_t comm, cudaStream_t stream);

int main(int argc, char **argv)
{
    nccl_test_context_t ctx = nccl_test_init(argc, argv, "NCCL_AllGatherv");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        nccl_test_fini(&ctx);
        return 1;
    }

    /* Parse weights from env */
    int n_weights = 0;
    int *weights = nccl_parse_env_int_array("ALLGATHERV_WEIGHTS", ctx.size, &n_weights);
    int total_weight = 0;
    if (weights) {
        for (int i = 0; i < n_weights; i++) total_weight += weights[i];
    }

    ncclDataType_t datatype = data_type_to_nccl(ctx.config.data_type);

    size_t max_sz = ctx.config.max_message_size;
    size_t worst_case = max_sz * ctx.size;

    void *h_send = malloc(max_sz);
    void *h_recv = malloc(worst_case);
    void *h_ref  = malloc(worst_case);
    void *d_send = NULL; cudaMalloc(&d_send, max_sz);
    void *d_recv = NULL; cudaMalloc(&d_recv, worst_case);
    int *d_recvcounts = NULL; cudaMalloc(&d_recvcounts, ctx.size * sizeof(int));
    int *d_displs     = NULL; cudaMalloc(&d_displs, ctx.size * sizeof(int));

    if (!h_send || !h_recv || !h_ref || !d_send || !d_recv ||
        !d_recvcounts || !d_displs) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(h_send); free(h_recv); free(h_ref);
        cudaFree(d_send); cudaFree(d_recv);
        cudaFree(d_recvcounts); cudaFree(d_displs);
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
                      d_send, d_recv, d_recvcounts, d_displs);

    free(h_send); free(h_recv); free(h_ref);
    cudaFree(d_send); cudaFree(d_recv);
    cudaFree(d_recvcounts); cudaFree(d_displs);
    free(weights);
    nccl_test_fini(&ctx);
    return 0;
}
