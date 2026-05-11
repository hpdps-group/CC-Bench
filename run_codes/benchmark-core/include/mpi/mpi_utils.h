/**
 * MPI bridge functions — convert between generic data_type_t
 * and MPI-specific types.  Only used by MPI-backend code.
 */

#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>
#include "utils.h"

/**
 * Convert MPI_Datatype to generic data_type_t.
 */
data_type_t mpi_to_data_type(MPI_Datatype mpi_type);

/**
 * Convert generic data_type_t to MPI_Datatype.
 */
MPI_Datatype data_type_to_mpi(data_type_t type);

/**
 * Get MPI_Op from string (e.g., "sum", "max", "min").
 */
MPI_Op parse_mpi_op(const char *name);

/* ── Common MPI test lifecycle ────────────────────────────────── */

#include "validation.h"

typedef struct {
    int rank;
    int size;
    test_config_t config;
} mpi_test_context_t;

/**
 * Initialize MPI test: MPI_Init, parse args, print banner.
 */
mpi_test_context_t mpi_test_init(int argc, char **argv, const char *test_name);

/**
 * Print footer and MPI_Finalize.
 */
void mpi_test_fini(const mpi_test_context_t *ctx);

/**
 * Load input data into buf — from file (per-rank offset) or init_buffer_pattern.
 */
void mpi_load_input(const mpi_test_context_t *ctx, void *buf, size_t msg_size,
                    MPI_Datatype datatype);

/**
 * Collect timing across ranks and print results.
 * For per-rank metrics (MSE/MAE/PSNR/SSIM), rank 0's values are printed.
 * Pass NULL for metrics when validation is disabled.
 */
void mpi_report_results(const mpi_test_context_t *ctx, size_t msg_size, int count,
                        double total_time_user, int local_errors,
                        const validation_result_t *metrics);

/**
 * Parse a comma-separated integer array from an environment variable.
 * Returns a malloc'd array (caller must free) and sets *out_len.
 * Returns NULL if env var is not set (caller handles fallback).
 * If expected_len > 0 and parsed length doesn't match, prints error and aborts.
 */
int *parse_env_int_array(const char *env_name, int expected_len, int *out_len);

/**
 * CSV output utility.
 * Only rank 0 writes. If header is provided and the file does not exist,
 * the header line is written first. Each fmt call appends one data line.
 */
void mpi_csv_write(const char *path, const char *header, const char *fmt, ...);

/*
 * size_iter_t moved to utils.h (no MPI dependency) — keep the include
 * so existing MPI code still picks up the declarations.
 */
#include "utils.h"

#endif /* MPI_UTILS_H */
