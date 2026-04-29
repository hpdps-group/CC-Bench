/**
 * MPI bridge implementation.
 */

#include "mpi/mpi_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <strings.h>

/* PMPI declarations */
extern int PMPI_Init(int *argc, char ***argv);
extern int PMPI_Finalize(void);
extern int PMPI_Abort(MPI_Comm comm, int errorcode);
extern int PMPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                      int root, MPI_Comm comm);
extern int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
extern int PMPI_Comm_rank(MPI_Comm comm, int *rank);
extern int PMPI_Comm_size(MPI_Comm comm, int *size);
extern int PMPI_Type_size(MPI_Datatype datatype, int *size);

/* ── Common test lifecycle ──────────────────────────────────────────── */

mpi_test_context_t mpi_test_init(int argc, char **argv, const char *test_name) {
    mpi_test_context_t ctx;

    PMPI_Init(&argc, &argv);
    PMPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
    ctx.config = parse_arguments(argc, argv);

    if (ctx.rank == 0) {
        printf("=== %s Test ===\n", test_name);
        printf("Processes: %d\n", ctx.size);
        printf("Message sizes: %zu to %zu (x%d)\n",
               ctx.config.min_message_size,
               ctx.config.max_message_size,
               ctx.config.message_size_incr);
        printf("Iterations: %d (warmup: %d)\n",
               ctx.config.iterations, ctx.config.warmup_iterations);
        printf("Validation: %s\n", ctx.config.validate ? "enabled" : "disabled");
        if (ctx.config.input_file)
            printf("Input file: %s\n", ctx.config.input_file);
        printf("Tolerance: %e\n\n", ctx.config.tolerance);
    }

    return ctx;
}

void mpi_test_fini(const mpi_test_context_t *ctx) {
    if (ctx->rank == 0)
        printf("=== Test Complete ===\n");
    PMPI_Finalize();
}

void mpi_load_input(const mpi_test_context_t *ctx, void *buf, size_t msg_size,
                    MPI_Datatype datatype) {
    if (ctx->config.input_file) {
        FILE *fp = fopen(ctx->config.input_file, "rb");
        if (!fp) {
            fprintf(stderr, "Rank %d: Failed to open file %s\n",
                    ctx->rank, ctx->config.input_file);
            PMPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        size_t total_file_size = 0;
        if (ctx->rank == 0) {
            fseek(fp, 0, SEEK_END);
            total_file_size = ftell(fp);
            fseek(fp, 0, SEEK_SET);
        }
        PMPI_Bcast(&total_file_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        size_t chunk_size = msg_size;
        size_t offset = ctx->rank * chunk_size;
        if (total_file_size > 0) {
            offset = offset % total_file_size;
            if (offset + chunk_size > total_file_size)
                chunk_size = total_file_size - offset;
        }

        fseek(fp, offset, SEEK_SET);
        size_t read_bytes = fread(buf, 1, chunk_size, fp);
        fclose(fp);

        if (read_bytes < msg_size)
            memset((char *)buf + read_bytes, 0, msg_size - read_bytes);
    } else {
        int dtype_size;
        PMPI_Type_size(datatype, &dtype_size);
        int count = msg_size / dtype_size;
        if (count <= 0) count = 1;
        init_buffer_pattern(buf, count, mpi_to_data_type(datatype), ctx->config.pattern_type, ctx->rank);
    }
}

void mpi_report_results(const mpi_test_context_t *ctx, size_t msg_size, int count,
                        double total_time_user, int local_errors,
                        const validation_result_t *metrics) {
    double avg_time = total_time_user / ctx->config.iterations * 1e6;
    double min_time, max_time, avg_global;

    PMPI_Reduce(&avg_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&avg_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    PMPI_Reduce(&avg_time, &avg_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_global /= ctx->size;

    int total_errors;
    PMPI_Reduce(&local_errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ctx->rank == 0) {
        printf("Size: %8zu bytes (%6d elements)\n", msg_size, count);
        printf("  User: avg=%8.2f us, min=%8.2f us, max=%8.2f us\n",
               avg_global, min_time, max_time);

        if (ctx->config.validate) {
            printf("  Correct: %s\n", total_errors == 0 ? "YES" : "NO");
            if (metrics && ctx->config.compute_metrics && metrics->num_elements > 0) {
                for (int i = 0; i < g_metric_registry_count && i < MAX_METRICS; i++) {
                    if (ctx->config.metrics_mask & (1u << i)) {
                        printf("  %s: %.6e\n", g_metric_registry[i].name,
                               metrics->values[i]);
                    }
                }
            }
        }
        printf("\n");
    }
}

/* ── Type conversion ───────────────────────────────────────────────── */

data_type_t mpi_to_data_type(MPI_Datatype mpi_type) {
    if (mpi_type == MPI_INT)   return TYPE_INT;
    if (mpi_type == MPI_FLOAT) return TYPE_FLOAT;
    if (mpi_type == MPI_DOUBLE) return TYPE_DOUBLE;
    if (mpi_type == MPI_CHAR || mpi_type == MPI_SIGNED_CHAR ||
        mpi_type == MPI_UNSIGNED_CHAR) return TYPE_CHAR;
    return TYPE_DOUBLE;  /* safest default */
}

MPI_Datatype data_type_to_mpi(data_type_t type) {
    switch (type) {
        case TYPE_INT:    return MPI_INT;
        case TYPE_FLOAT:  return MPI_FLOAT;
        case TYPE_DOUBLE: return MPI_DOUBLE;
        case TYPE_CHAR:   return MPI_CHAR;
        default:          return MPI_DOUBLE;
    }
}

MPI_Op parse_mpi_op(const char *name) {
    if (!name) return MPI_SUM;
    if (strcasecmp(name, "sum") == 0)  return MPI_SUM;
    if (strcasecmp(name, "max") == 0)  return MPI_MAX;
    if (strcasecmp(name, "min") == 0)  return MPI_MIN;
    if (strcasecmp(name, "prod") == 0) return MPI_PROD;
    return MPI_SUM;
}
