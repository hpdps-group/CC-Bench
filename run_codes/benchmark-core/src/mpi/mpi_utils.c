/**
 * MPI bridge implementation.
 */

#include "mpi/mpi_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <strings.h>
#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include "job_device_mapper.h"

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

    /* ── Job-device mapper (optional, loaded via dlopen) ─────────── */
    device_map_ctx_t dm_ctx;
    device_map_init(&dm_ctx, ctx.rank, ctx.size);

    void *dm_handle = dlopen("bin/libs/libjob_device_mapper.so",
                             RTLD_LAZY | RTLD_LOCAL);
    if (dm_handle) {
        void (*dm_func)(device_map_ctx_t *) =
            (void (*)(device_map_ctx_t *))dlsym(dm_handle, "map_job_to_device");
        if (dm_func)
            dm_func(&dm_ctx);
        dlclose(dm_handle);
    }

    device_map_flush_to_file(&dm_ctx);
    device_map_destroy(&dm_ctx);
    /* ────────────────────────────────────────────────────────────── */

    ctx.config = parse_arguments(argc, argv);

    if (ctx.rank == 0) {
        printf("=== %s Test ===\n", test_name);
        printf("Processes: %d\n", ctx.size);
        if (ctx.config.use_size_list) {
            printf("Message sizes (list): ");
            for (int i = 0; i < ctx.config.num_sizes; i++) {
                if (i > 0) printf(", ");
                printf("%zu", ctx.config.size_list[i]);
            }
            printf("\n");
        } else {
            printf("Message sizes: %zu to %zu (%s%d)\n",
                   ctx.config.min_message_size,
                   ctx.config.max_message_size,
                   ctx.config.message_size_incr_mode ? "+" : "x",
                   ctx.config.message_size_incr);
        }
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

/* ── Helper: load data from folder (each rank reads its own file) ─── */
static void mpi_load_from_folder(const mpi_test_context_t *ctx, void *buf,
                                  size_t msg_size)
{
    const char *folder = ctx->config.input_file;
    const char *suffix_env = getenv("DS_SUFFIX");
    const char *format_env = getenv("DS_FORMAT");
    char path[1024];
    char default_suffix[16] = "";

    /* Determine suffix: explicit DS_SUFFIX, or infer from file_format */
    const char *suffix = suffix_env;
    if (!suffix || suffix[0] == '\0') {
        if (format_env && strcasecmp(format_env, "text") == 0)
            snprintf(default_suffix, sizeof(default_suffix), ".txt");
        else
            snprintf(default_suffix, sizeof(default_suffix), ".bin");
        suffix = default_suffix;
    }

    /* Try rank_<id><suffix> first */
    snprintf(path, sizeof(path), "%s/rank_%d%s", folder, ctx->rank, suffix);
    FILE *fp = fopen(path, "rb");

    /* Fallback: scan folder for any file starting with rank_<id> */
    if (!fp) {
        DIR *dir = opendir(folder);
        if (!dir) {
            fprintf(stderr, "Rank %d: cannot open folder '%s': %s\n",
                    ctx->rank, folder, strerror(errno));
            PMPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        struct dirent *entry;
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "rank_%d", ctx->rank);
        int prefix_len = strlen(prefix);
        char found[1024] = "";
        while ((entry = readdir(dir)) != NULL) {
            const char *name = entry->d_name;
            if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
                continue;
            if (strncmp(name, prefix, prefix_len) != 0)
                continue;
            char after = name[prefix_len];
            if (after != '.' && after != '\0')
                continue;
            snprintf(found, sizeof(found), "%s/%s", folder, name);
            if (suffix[0] && strcmp(name + prefix_len, suffix) != 0)
                fprintf(stderr, "Rank %d: warning — expected '%s' but found '%s'\n",
                        ctx->rank, path, found);
            break;
        }
        closedir(dir);

        if (found[0] == '\0') {
            fprintf(stderr, "Rank %d: no file matching 'rank_%d*' in '%s'\n",
                    ctx->rank, ctx->rank, folder);
            PMPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fp = fopen(found, "rb");
        if (!fp) {
            fprintf(stderr, "Rank %d: failed to open '%s': %s\n",
                    ctx->rank, found, strerror(errno));
            PMPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    size_t got = fread(buf, 1, msg_size, fp);
    fclose(fp);

    if (got < msg_size)
        memset((char *)buf + got, 0, msg_size - got);
}

void mpi_load_input(const mpi_test_context_t *ctx, void *buf, size_t msg_size,
                    MPI_Datatype datatype) {
    if (ctx->config.input_file) {
        const char *ds_type = getenv("DS_TYPE");
        if (ds_type && strcmp(ds_type, "folder") == 0) {
            mpi_load_from_folder(ctx, buf, msg_size);
            return;
        }

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
        size_t offset = 0;

        /* Offset mode: env vars set by build_script.sh from JSONC offset_config */
        const char *custom_env = getenv("DS_CUSTOM_OFFSETS");
        if (custom_env && custom_env[0]) {
            /* Custom per-rank offsets via comma-separated array */
            int n = 0;
            int *offsets = parse_env_int_array("DS_CUSTOM_OFFSETS", 0, &n);
            if (offsets && ctx->rank < n)
                offset = (size_t)offsets[ctx->rank];
            free(offsets);
        } else {
            const char *base_env = getenv("DS_BASE_OFFSET");
            if (base_env) offset = (size_t)atol(base_env);
            const char *per_rank_env = getenv("DS_PER_RANK_OFFSET");
            if (!per_rank_env || strcmp(per_rank_env, "true") == 0)
                offset += (size_t)ctx->rank * chunk_size;
        }

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
                for (int i = 0; i < validation_registry_count() && i < MAX_METRICS; i++) {
                    if (ctx->config.metrics_mask & (1u << i)) {
                        printf("  %s: %.6e\n", validation_metric_name(i),
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

/* ── Env-var array parsing ───────────────────────────────────────── */

int *parse_env_int_array(const char *env_name, int expected_len, int *out_len) {
    const char *s = getenv(env_name);
    if (!s) {
        *out_len = 0;
        return NULL;
    }

    char *copy = strdup(s);
    if (!copy) {
        fprintf(stderr, "Error: strdup failed for %s\n", env_name);
        exit(1);
    }

    int cap = 8, len = 0;
    int *arr = malloc(cap * sizeof(int));
    if (!arr) {
        fprintf(stderr, "Error: malloc failed for %s\n", env_name);
        free(copy);
        exit(1);
    }

    char *tok = strtok(copy, ",");
    while (tok) {
        if (len >= cap) {
            cap *= 2;
            int *tmp = realloc(arr, cap * sizeof(int));
            if (!tmp) { free(arr); free(copy); exit(1); }
            arr = tmp;
        }
        arr[len++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    free(copy);

    if (expected_len > 0 && len != expected_len) {
        fprintf(stderr, "Error: %s has %d values, expected %d\n",
                env_name, len, expected_len);
        free(arr);
        exit(1);
    }

    *out_len = len;
    return arr;
}

/* ── CSV output ───────────────────────────────────────────────── */

void mpi_csv_write(const char *path, const char *header, const char *fmt, ...) {
    int rank;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;

    int exists = 0;
    FILE *fp = fopen(path, "r");
    if (fp) { exists = 1; fclose(fp); }

    fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "Error: cannot open CSV file %s\n", path);
        return;
    }

    if (header && !exists) {
        fprintf(fp, "%s\n", header);
    }

    if (fmt) {
        va_list args;
        va_start(args, fmt);
        vfprintf(fp, fmt, args);
        fprintf(fp, "\n");
        va_end(args);
    }

    fclose(fp);
}

/* ── Message size iterator ────────────────────────────────────── */

/* size_iter moved to utils.c */
