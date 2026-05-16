/**
 * NCCL ReduceScatter correctness and performance test.
 *
 * Tests ncclReduceScatter for correctness and performance.
 * Each rank contributes data; result is reduced and scattered so each rank
 * gets a distinct chunk of the final reduction.
 * Reference computed on CPU via dummy_reduce_scatter.
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "validation.h"
#include "nccl/nccl_utils.h"
#include "binary_output.h"
#include "dummy_collectives.h"

static void run_test_size(const nccl_test_context_t *ctx, size_t msg_size,
                          ncclDataType_t datatype, ncclRedOp_t op,
                          void *h_send, void *h_recv, void *h_ref,
                          void *d_send, void *d_recv)
{
    size_t esz = nccl_dtype_size(datatype);
    int recvcount = (int)(msg_size / esz);
    if (recvcount <= 0) recvcount = 1;
    int sendcount = recvcount * ctx->size;
    size_t send_bytes = (size_t)sendcount * esz;
    size_t recv_bytes = (size_t)recvcount * esz;

    /* 1. Load input data (sendbuf has recvcount * size elements) */
    nccl_load_input(ctx, h_send, send_bytes, datatype);
    cudaMemcpy(d_send, h_send, send_bytes, cudaMemcpyHostToDevice);

    /* 2. CPU reference (no NCCL call) */
    dummy_reduce_scatter(h_ref,
                         ctx->config.input_file,
                         nccl_to_data_type(datatype),
                         ctx->config.pattern_type,
                         recvcount, ctx->size, ctx->rank);

    /* 3. Warmup */
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        ncclResult_t _ret = ncclReduceScatter(d_send, d_recv, recvcount, datatype, op,
                                              ctx->comm, ctx->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[rank=%d] ncclReduceScatter FAILED at size=%zu iter=%d "
                            "error=%d — aborting\n",
                    ctx->rank, recv_bytes, i, (int)_ret);
            exit(1);
        }
        cudaStreamSynchronize(ctx->stream);
    }

    /* 4. Allocate user accumulator for binary output */
    data_type_t dtype_gen = nccl_to_data_type(datatype);
    void *user_accum = ctx->config.save_binary ? calloc(1, recv_bytes) : NULL;

    /* 5. Main timing and validation loop */
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
        ncclResult_t _ret = ncclReduceScatter(d_send, d_recv, recvcount, datatype, op,
                                              ctx->comm, ctx->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[rank=%d] ncclReduceScatter FAILED at size=%zu iter=%d "
                            "error=%d — aborting\n",
                    ctx->rank, recv_bytes, iter, (int)_ret);
            exit(1);
        }
        cudaEventRecord(stop, ctx->stream);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_ms += ms;

        /* Accumulate user result for binary output */
        if (user_accum) {
            if (!ctx->config.validate) {
                cudaMemcpy(h_recv, d_recv, recv_bytes, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(ctx->stream);
            }
            binary_accumulate(user_accum, h_recv, recvcount, dtype_gen);
        }

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

    /* 6. Average metrics */
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

    /* 6a. Average and write binary output */
    if (user_accum) {
        if (ctx->config.iterations > 0)
            binary_average(user_accum, recvcount, dtype_gen, ctx->config.iterations);
        write_binary_multi(ctx->config.bin_path, "reference",
                           h_ref, recv_bytes, ctx->rank, ctx->size);
        write_binary_multi(ctx->config.bin_path, "user",
                           user_accum, recv_bytes, ctx->rank, ctx->size);
        free(user_accum);
    }

    /* 7. Bandwidth: each rank recvs recv_bytes */
    double avg_sec = (double)total_time_ms / ctx->config.iterations / 1000.0;
    double bw = (avg_sec > 0.0) ? (recv_bytes / avg_sec) / 1.0e9 : 0.0;

    /* 8. Report */
    nccl_report_results(ctx, recv_bytes, recvcount, total_time_ms, iter_errors,
                        ctx->config.validate ? &metrics_acc : NULL, bw);
}

int main(int argc, char **argv)
{
    nccl_test_context_t ctx = nccl_test_init(argc, argv, "NCCL_ReduceScatter");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        nccl_test_fini(&ctx);
        return 1;
    }

    ncclDataType_t datatype = data_type_to_nccl(ctx.config.data_type);
    ncclRedOp_t op = ncclSum;

    size_t max_sz = ctx.config.max_message_size;
    size_t send_max = max_sz * ctx.size;
    void *h_send = malloc(send_max);
    void *h_recv = malloc(max_sz);
    void *h_ref  = malloc(max_sz);
    void *d_send = NULL; cudaMalloc(&d_send, send_max);
    void *d_recv = NULL; cudaMalloc(&d_recv, max_sz);

    if (!h_send || !h_recv || !h_ref || !d_send || !d_recv) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(h_send); free(h_recv); free(h_ref);
        cudaFree(d_send); cudaFree(d_recv);
        nccl_test_fini(&ctx);
        return 1;
    }

    size_iter_t it;
    size_iter_init(&it, &ctx.config);
    size_t sz;
    while (size_iter_next(&it, &sz))
        run_test_size(&ctx, sz, datatype, op,
                      h_send, h_recv, h_ref,
                      d_send, d_recv);

    free(h_send); free(h_recv); free(h_ref);
    cudaFree(d_send); cudaFree(d_recv);
    nccl_test_fini(&ctx);
    return 0;
}
