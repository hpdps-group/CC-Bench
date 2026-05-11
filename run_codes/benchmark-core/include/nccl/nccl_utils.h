/**
 * NCCL benchmark utilities — replaces mpi_utils.h for NCCL-only tests.
 *
 * No MPI dependency. Rank/size are obtained from environment variables.
 * Reference calls use base_nccl* (real NCCL loaded via dlopen by findso).
 */

#ifndef NCCL_UTILS_H
#define NCCL_UTILS_H

#include <nccl.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include "utils.h"
#include "validation.h"

/* ── Error checking macro (not in public nccl.h) ──────────────── */
#ifndef NCCLCHECK
#define NCCLCHECK(call) do { \
    ncclResult_t _r = (call); \
    if (_r != ncclSuccess) { \
        fprintf(stderr, "[NCCLCHECK error at %s:%d] %s\n", \
                __FILE__, __LINE__, ncclGetErrorString(_r)); \
        exit(1); \
    } \
} while (0)
#endif

/* ── NCCL test context ────────────────────────────────────────── */
typedef struct {
    int rank;
    int size;
    test_config_t config;
    ncclComm_t comm;
    cudaStream_t stream;
    int *d_barrier;    /* pre-allocated GPU buffer for barrier */
} nccl_test_context_t;

/* ── Lifecycle ────────────────────────────────────────────────── */
nccl_test_context_t nccl_test_init(int argc, char **argv,
                                   const char *test_name);
void nccl_test_fini(nccl_test_context_t *ctx);

/* ── Input data loading (CPU buffer) ──────────────────────────── */
void nccl_load_input(const nccl_test_context_t *ctx, void *buf,
                     size_t msg_size, ncclDataType_t datatype);

/* ── Reporting ────────────────────────────────────────────────── */
void nccl_report_results(const nccl_test_context_t *ctx,
                         size_t msg_size, int count,
                         float total_time_ms, int local_errors,
                         const validation_result_t *metrics);

/* ── Type conversion ──────────────────────────────────────────── */
data_type_t      nccl_to_data_type(ncclDataType_t nccl_type);
ncclDataType_t   data_type_to_nccl(data_type_t type);
size_t           nccl_dtype_size(ncclDataType_t dtype);

/* ── Barrier across all ranks (uses base_ncclAllReduce) ───────── */
void nccl_barrier(nccl_test_context_t *ctx);

/* ── Parse comma-separated int array from env var ─────────────── */
int *nccl_parse_env_int_array(const char *env_name, int expected_len, int *out_len);

#endif /* NCCL_UTILS_H */
