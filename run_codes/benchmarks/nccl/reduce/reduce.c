/**
 * NCCL Reduce correctness and performance test.
 *
 * Tests ncclReduce against base_ncclReduce (real NCCL via dlopen).
 * All ranks send data, root receives the reduced result.
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "validation.h"
#include "base_impl.h"
#include "nccl/nccl_base.h"
#include "nccl/nccl_utils.h"
#include "binary_output.h"

#define ROOT 0

static void run_test_size(const nccl_test_context_t *ctx, size_t msg_size,
                          ncclDataType_t datatype, ncclRedOp_t op,
                          void *h_send, void *h_recv, void *h_ref,
                          void *d_send, void *d_recv,
                          void *d_ref_send, void *d_ref_recv)
{
    size_t esz = nccl_dtype_size(datatype);
    int count = (int)(msg_size / esz);
    if (count <= 0) count = 1;
    size_t bytes = (size_t)count * esz;

    /* 1. Load input data */
    nccl_load_input(ctx, h_send, bytes, datatype);
    memcpy(h_ref, h_send, bytes);

    cudaMemcpy(d_send, h_send, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_send, h_send, bytes, cudaMemcpyHostToDevice);

    /* 2. Warmup */
    for (int i = 0; i < ctx->config.warmup_iterations; i++) {
        ncclReduce(d_send, d_recv, count, datatype, op, ROOT,
                   ctx->comm, ctx->stream);
        cudaStreamSynchronize(ctx->stream);
    }

    /* 3. Reference result */
    base_ncclReduce(d_ref_send, d_ref_recv, count, datatype, op, ROOT,
                    ctx->comm, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    if (ctx->rank == ROOT)
        cudaMemcpy(h_ref, d_ref_recv, bytes, cudaMemcpyDeviceToHost);

    /* 3a. Allocate user accumulator for binary output */
    data_type_t dtype_gen = nccl_to_data_type(datatype);
    void *user_accum = ctx->config.save_binary ? calloc(1, bytes) : NULL;

    /* 4. Main timing and validation loop */
    float total_time_ms = 0.0f;
    int iter_errors = 0;
    validation_result_t metrics_acc = {0};
    metrics_acc.num_elements = count;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < ctx->config.iterations; iter++) {
        nccl_barrier((nccl_test_context_t *)ctx);

        cudaEventRecord(start, ctx->stream);
        ncclReduce(d_send, d_recv, count, datatype, op, ROOT,
                   ctx->comm, ctx->stream);
        cudaEventRecord(stop, ctx->stream);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_ms += ms;

        /* Accumulate user result on root for binary output */
        if (user_accum && ctx->rank == ROOT) {
            if (!(ctx->config.validate && ctx->rank == ROOT)) {
                cudaMemcpy(h_recv, d_recv, bytes, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(ctx->stream);
            }
            binary_accumulate(user_accum, h_recv, count, dtype_gen);
        }

        if (ctx->config.validate && ctx->rank == ROOT) {
            cudaMemcpy(h_recv, d_recv, bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(ctx->stream);
            validation_result_t vm = validate_result(
                h_recv, h_ref, count,
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

    /* 5. Average metrics */
    if (ctx->config.validate && ctx->config.iterations > 0) {
        if (ctx->rank == ROOT) {
            metrics_acc.correct = (iter_errors == 0);
            for (int m = 0; m < MAX_METRICS; m++) {
                if (ctx->config.metrics_mask & (1u << m))
                    metrics_acc.values[m] /= ctx->config.iterations;
            }
            if (iter_errors > 0)
                printf("  Validation FAILED at size %zu (%d/%d iterations failed)\n",
                       bytes, iter_errors, ctx->config.iterations);
        }
    }

    /* 5a. Average and write binary output */
    if (user_accum) {
        if (ctx->config.iterations > 0 && ctx->rank == ROOT)
            binary_average(user_accum, count, dtype_gen, ctx->config.iterations);
        write_binary_single(ctx->config.bin_path, "reference", h_ref, bytes, ctx->rank, ROOT);
        write_binary_single(ctx->config.bin_path, "user", user_accum, bytes, ctx->rank, ROOT);
        free(user_accum);
    }

    /* 6. Report */
    if (ctx->rank == ROOT) {
        nccl_report_results(ctx, bytes, count, total_time_ms, iter_errors,
                            ctx->config.validate ? &metrics_acc : NULL);
    }
}

int main(int argc, char **argv)
{
    nccl_test_context_t ctx = nccl_test_init(argc, argv, "NCCL_Reduce");

    if (ctx.size < 2) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        nccl_test_fini(&ctx);
        return 1;
    }

    if (load_base_impl(BASE_SO_FILE) != 0) {
        if (ctx.rank == 0)
            fprintf(stderr, "Error: base impl not found\n");
        nccl_test_fini(&ctx);
        return 1;
    }

    ncclDataType_t datatype = data_type_to_nccl(ctx.config.data_type);
    ncclRedOp_t op = ncclSum;

    size_t max_sz = ctx.config.max_message_size;
    void *h_send    = malloc(max_sz);
    void *h_recv    = malloc(max_sz);
    void *h_ref     = malloc(max_sz);
    void *d_send    = NULL; cudaMalloc(&d_send, max_sz);
    void *d_recv    = NULL; cudaMalloc(&d_recv, max_sz);
    void *d_ref_send = NULL; cudaMalloc(&d_ref_send, max_sz);
    void *d_ref_recv = NULL; cudaMalloc(&d_ref_recv, max_sz);

    if (!h_send || !h_recv || !h_ref || !d_send || !d_recv ||
        !d_ref_send || !d_ref_recv) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(h_send); free(h_recv); free(h_ref);
        cudaFree(d_send); cudaFree(d_recv);
        cudaFree(d_ref_send); cudaFree(d_ref_recv);
        nccl_test_fini(&ctx);
        return 1;
    }

    size_iter_t it;
    size_iter_init(&it, &ctx.config);
    size_t sz;
    while (size_iter_next(&it, &sz))
        run_test_size(&ctx, sz, datatype, op,
                      h_send, h_recv, h_ref,
                      d_send, d_recv, d_ref_send, d_ref_recv);

    free(h_send); free(h_recv); free(h_ref);
    cudaFree(d_send); cudaFree(d_recv);
    cudaFree(d_ref_send); cudaFree(d_ref_recv);
    nccl_test_fini(&ctx);
    return 0;
}
