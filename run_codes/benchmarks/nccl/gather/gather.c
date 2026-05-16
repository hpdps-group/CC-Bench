/**
 * NCCL Gather correctness and performance test.
 *
 * All ranks send data to root, which gathers them.
 * No base_ncclGather — validated against CPU-computed reference.
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

#define ROOT 0

/* ── Forward declaration (from plain_nccl_compress.c) ─────────────────── */
extern int ncclGather(const void *sendbuf, size_t sendcount, ncclDataType_t sendtype,
                      void *recvbuf, size_t recvcount, ncclDataType_t recvtype,
                      int root, ncclComm_t comm, cudaStream_t stream);

static void run_test_size(const nccl_test_context_t *ctx, size_t msg_size,
                          ncclDataType_t datatype,
                          void *h_send, void *h_recv, void *h_ref,
                          void *d_send, void *d_recv)
{
    size_t esz = nccl_dtype_size(datatype);
    int count_per = (int)(msg_size / esz);
    if (count_per <= 0) count_per = 1;
    int total_count = count_per * ctx->size;
    size_t send_bytes = (size_t)count_per * esz;
    size_t recv_bytes = (size_t)total_count * esz;

    /* 1. Load input */
    nccl_load_input(ctx, h_send, send_bytes, datatype);
    cudaMemcpy(d_send, h_send, send_bytes, cudaMemcpyHostToDevice);

    /* 2. CPU reference (dummy_gather loads all ranks' data from file or pattern) */
    if ((ctx->config.validate || ctx->config.save_binary) && ctx->rank == ROOT) {
        data_type_t dt = nccl_to_data_type(datatype);
        dummy_gather(h_ref, ctx->config.input_file, dt, ctx->config.pattern_type,
                     count_per, ctx->size, ROOT);
    }

    /* 2a. Allocate user accumulator for binary output */
    data_type_t dtype_gen = nccl_to_data_type(datatype);
    void *user_accum = ctx->config.save_binary || ctx->config.validate ? calloc(1, recv_bytes) : NULL;

    /* 3. Warmup */
    cudaStreamSynchronize(ctx->stream);
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        ncclResult_t _ret = ncclGather(d_send, count_per, datatype,
                                       d_recv, count_per, datatype,
                                       ROOT, ctx->comm, ctx->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[rank=%d] ncclGather FAILED at size=%zu iter=%d "
                            "error=%d -- aborting\n",
                    ctx->rank, msg_size, i, (int)_ret);
            exit(1);
        }
        cudaStreamSynchronize(ctx->stream);
    }

    /* 4. Main timing loop */
    float total_time_ms = 0.0f;
    int iter_errors = 0;
    validation_result_t metrics_acc = {0};
    metrics_acc.num_elements = total_count;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < ctx->config.iterations; iter++) {
        nccl_barrier((nccl_test_context_t *)ctx);

        cudaEventRecord(start, ctx->stream);
        ncclResult_t _ret = ncclGather(d_send, count_per, datatype,
                                       d_recv, count_per, datatype,
                                       ROOT, ctx->comm, ctx->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[rank=%d] ncclGather FAILED at size=%zu iter=%d "
                            "error=%d -- aborting\n",
                    ctx->rank, msg_size, iter, (int)_ret);
            exit(1);
        }
        cudaEventRecord(stop, ctx->stream);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_ms += ms;

        /* Accumulate user result on root for binary output */
        if (user_accum && ctx->rank == ROOT) {
            if (!(ctx->config.validate && ctx->rank == ROOT)) {
                cudaMemcpy(h_recv, d_recv, recv_bytes, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(ctx->stream);
            }
            binary_accumulate(user_accum, h_recv, total_count, dtype_gen);
        }

        if (ctx->config.validate && ctx->rank == ROOT) {
            cudaMemcpy(h_recv, d_recv, recv_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(ctx->stream);
            validation_result_t vm = validate_result(
                h_recv, h_ref, total_count,
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
        if (ctx->rank == ROOT) {
            metrics_acc.correct = (iter_errors == 0);
            for (int m = 0; m < MAX_METRICS; m++) {
                if (ctx->config.metrics_mask & (1u << m))
                    metrics_acc.values[m] /= ctx->config.iterations;
            }
            if (iter_errors > 0)
                printf("  Validation FAILED at size %zu (%d/%d iterations failed)\n",
                       msg_size, iter_errors, ctx->config.iterations);
        }
    }

    /* 5a. Average and write binary output */
    if (user_accum) {
        if (ctx->config.iterations > 0 && ctx->rank == ROOT)
            binary_average(user_accum, total_count, dtype_gen, ctx->config.iterations);
        write_binary_single(ctx->config.bin_path, "reference", h_ref, recv_bytes, ctx->rank, ROOT);
        write_binary_single(ctx->config.bin_path, "user", user_accum, recv_bytes, ctx->rank, ROOT);
        free(user_accum);
    }

    /* 6. Bandwidth: each rank sends msg_size to root */
    double avg_sec = (double)total_time_ms / ctx->config.iterations / 1000.0;
    double bw = (avg_sec > 0.0) ? (msg_size / avg_sec) / 1.0e9 : 0.0;

    /* Root reports */
    if (ctx->rank == ROOT)
        nccl_report_results(ctx, msg_size, count_per, total_time_ms, iter_errors,
                            ctx->config.validate ? &metrics_acc : NULL, bw);
}

int main(int argc, char **argv)
{
    nccl_test_context_t ctx = nccl_test_init(argc, argv, "NCCL_Gather");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        nccl_test_fini(&ctx);
        return 1;
    }

    ncclDataType_t datatype = data_type_to_nccl(ctx.config.data_type);

    size_t max_sz = ctx.config.max_message_size;
    size_t root_max = max_sz * ctx.size;

    void *h_send = malloc(max_sz);
    void *h_recv = malloc(root_max);
    void *h_ref  = malloc(root_max);
    void *d_send = NULL; cudaMalloc(&d_send, max_sz);
    void *d_recv = NULL; cudaMalloc(&d_recv, root_max);

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
        run_test_size(&ctx, sz, datatype,
                      h_send, h_recv, h_ref,
                      d_send, d_recv);

    free(h_send); free(h_recv); free(h_ref);
    cudaFree(d_send); cudaFree(d_recv);
    nccl_test_fini(&ctx);
    return 0;
}
