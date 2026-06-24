/**
 * Implementation of utility functions.
 */

#include "utils.h"
#include "validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <strings.h>

/* Element size lookup */
static size_t type_size(data_type_t t) {
    switch (t) {
        case TYPE_INT:    return sizeof(int);
        case TYPE_FLOAT:  return sizeof(float);
        case TYPE_DOUBLE: return sizeof(double);
        case TYPE_CHAR:   return sizeof(char);
        default:          return 1;
    }
}

/* Default configuration */
#define DEFAULT_MIN_SIZE 1
#define DEFAULT_MAX_SIZE 65536
#define DEFAULT_INCR 2
#define DEFAULT_ITERATIONS 100
#define DEFAULT_WARMUP 10
#define DEFAULT_TOLERANCE 1e-6

/* parse_metric_mask lives in validation.c (needs access to registry) */

int read_from_file(
    const char *filename,
    void *buffer,
    size_t buffer_size,
    file_format_t format,
    data_type_t datatype
) {
    if (!filename || !buffer) return -1;

    FILE *fp = fopen(filename, format == FILE_FORMAT_BINARY ? "rb" : "r");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    if (format == FILE_FORMAT_BINARY) {
        size_t read = fread(buffer, 1, buffer_size, fp);
        fclose(fp);
        return (read == buffer_size) ? 0 : -1;
    } else {
        /* Text format: read based on datatype */
        int count = buffer_size / type_size(datatype);
        if (count < 1) count = 1;
        if (datatype == TYPE_INT) {
            int *ibuf = (int *)buffer;
            for (int i = 0; i < count; i++) {
                if (fscanf(fp, "%d", &ibuf[i]) != 1) {
                    fclose(fp);
                    return -1;
                }
            }
        } else if (datatype == TYPE_FLOAT) {
            float *fbuf = (float *)buffer;
            for (int i = 0; i < count; i++) {
                if (fscanf(fp, "%f", &fbuf[i]) != 1) {
                    fclose(fp);
                    return -1;
                }
            }
        } else if (datatype == TYPE_DOUBLE) {
            double *dbuf = (double *)buffer;
            for (int i = 0; i < count; i++) {
                if (fscanf(fp, "%lf", &dbuf[i]) != 1) {
                    fclose(fp);
                    return -1;
                }
            }
        } else {
            /* Default to char */
            char *cbuf = (char *)buffer;
            for (int i = 0; i < count; i++) {
                if (fscanf(fp, "%c", &cbuf[i]) != 1) {
                    fclose(fp);
                    return -1;
                }
            }
        }
        fclose(fp);
        return 0;
    }
}

int write_to_file(
    const char *filename,
    const void *buffer,
    size_t buffer_size,
    file_format_t format,
    data_type_t datatype
) {
    if (!filename || !buffer) return -1;

    FILE *fp = fopen(filename, format == FILE_FORMAT_BINARY ? "wb" : "w");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    if (format == FILE_FORMAT_BINARY) {
        size_t written = fwrite(buffer, 1, buffer_size, fp);
        fclose(fp);
        return (written == buffer_size) ? 0 : -1;
    } else {
        /* Text format */
        int count = buffer_size / type_size(datatype);
        if (count < 1) count = 1;
        if (datatype == TYPE_INT) {
            const int *ibuf = (const int *)buffer;
            for (int i = 0; i < count; i++) {
                fprintf(fp, "%d\n", ibuf[i]);
            }
        } else if (datatype == TYPE_FLOAT) {
            const float *fbuf = (const float *)buffer;
            for (int i = 0; i < count; i++) {
                fprintf(fp, "%.6f\n", fbuf[i]);
            }
        } else if (datatype == TYPE_DOUBLE) {
            const double *dbuf = (const double *)buffer;
            for (int i = 0; i < count; i++) {
                fprintf(fp, "%.10lf\n", dbuf[i]);
            }
        } else {
            const char *cbuf = (const char *)buffer;
            for (int i = 0; i < count; i++) {
                fprintf(fp, "%c\n", cbuf[i]);
            }
        }
        fclose(fp);
        return 0;
    }
}

void *allocate_aligned_buffer(size_t size, size_t alignment) {
    void *ptr = NULL;
#ifdef _POSIX_C_SOURCE
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
#else
    ptr = malloc(size);
#endif
    return ptr;
}

void free_aligned_buffer(void *ptr) {
    free(ptr);
}

void init_buffer_pattern(
    void *buffer,
    int count,
    data_type_t datatype,
    pattern_type_t pattern_type,
    int rank
) {
    if (!buffer) return;

    if (datatype == TYPE_INT) {
        int *ibuf = (int *)buffer;
        for (int i = 0; i < count; i++) {
            switch (pattern_type) {
                case PATTERN_RANK_LINEAR:
                    ibuf[i] = rank + i; break;
                case PATTERN_RANK_PRODUCT:
                    ibuf[i] = (rank + 1) * (i + 1); break;
                case PATTERN_SEQUENTIAL:
                    ibuf[i] = rank * count + i; break;
                case PATTERN_PLAIN:
                default:
                    ibuf[i] = 1; break;
            }
        }
    } else if (datatype == TYPE_FLOAT) {
        float *fbuf = (float *)buffer;
        for (int i = 0; i < count; i++) {
            switch (pattern_type) {
                case PATTERN_RANK_LINEAR:
                    fbuf[i] = (float)(rank + i); break;
                case PATTERN_RANK_PRODUCT:
                    fbuf[i] = (float)((rank + 1) * (i + 1)); break;
                case PATTERN_SEQUENTIAL:
                    fbuf[i] = (float)(rank * count + i); break;
                case PATTERN_PLAIN:
                default:
                    fbuf[i] = 1.0f; break;
            }
        }
    } else if (datatype == TYPE_DOUBLE) {
        double *dbuf = (double *)buffer;
        for (int i = 0; i < count; i++) {
            switch (pattern_type) {
                case PATTERN_RANK_LINEAR:
                    dbuf[i] = (double)(rank + i); break;
                case PATTERN_RANK_PRODUCT:
                    dbuf[i] = (double)((rank + 1) * (i + 1)); break;
                case PATTERN_SEQUENTIAL:
                    dbuf[i] = (double)(rank * count + i); break;
                case PATTERN_PLAIN:
                default:
                    dbuf[i] = 1.0; break;
            }
        }
    } else {
        /* Default to char */
        char *cbuf = (char *)buffer;
        for (int i = 0; i < count; i++) {
            cbuf[i] = (char)((rank + i) % 256);
        }
    }
}

void print_buffer(
    const void *buffer,
    int count,
    data_type_t datatype,
    int max_to_print
) {
    if (max_to_print <= 0 || count <= 0) return;

    int limit = (count < max_to_print) ? count : max_to_print;
    printf("Buffer (first %d of %d): ", limit, count);

    if (datatype == TYPE_INT) {
        const int *ibuf = (const int *)buffer;
        for (int i = 0; i < limit; i++) printf("%d ", ibuf[i]);
    } else if (datatype == TYPE_FLOAT) {
        const float *fbuf = (const float *)buffer;
        for (int i = 0; i < limit; i++) printf("%.3f ", fbuf[i]);
    } else if (datatype == TYPE_DOUBLE) {
        const double *dbuf = (const double *)buffer;
        for (int i = 0; i < limit; i++) printf("%.6lf ", dbuf[i]);
    } else {
        const char *cbuf = (const char *)buffer;
        for (int i = 0; i < limit; i++) printf("%d ", (int)cbuf[i]);
    }
    printf("\n");
}

pattern_type_t parse_pattern_type(const char *name) {
    if (strcasecmp(name, "plain") == 0)         return PATTERN_PLAIN;
    if (strcasecmp(name, "rank_linear") == 0)   return PATTERN_RANK_LINEAR;
    if (strcasecmp(name, "rank_product") == 0)  return PATTERN_RANK_PRODUCT;
    if (strcasecmp(name, "sequential") == 0)    return PATTERN_SEQUENTIAL;
    fprintf(stderr, "Unknown pattern type: '%s' (options: plain, rank_linear, rank_product, sequential)\n", name);
    exit(1);
}

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -d TYPE          Data type: float, double, int, char (default: double)\n");
    printf("  -m MIN:MAX:INCR   Message size range (default: 1:65536:2)\n");
    printf("  -p PATTERN        Data pattern: plain, rank_linear, rank_product, sequential (default: rank_linear)\n");
    printf("  -i ITER           Number of iterations (default: 100)\n");
    printf("  -w WARMUP         Warmup iterations (default: 10)\n");
    printf("  -f FILE           Input file (optional)\n");
    printf("  -t TOLERANCE      Tolerance for validation (default: 1e-6)\n");
    printf("  -v                Enable validation\n");
    printf("  -M                Compute all metrics\n");
    printf("  -e METRICS        Comma-separated metrics to compute (mse,mae,psnr,ssim,all)\n");
    printf("  -c                Enable CSV output\n");
    printf("  -o PATH           CSV output path (default: results/<test_name>.csv)\n");
    printf("  -b                Enable binary output (reference + user data dump)\n");
    printf("  -B PATH           Binary output base path (used as basename)\n");
    printf("  -A                Add mode (INCR is added each step, default: multiply)\n");
    printf("  -L X,Y,Z          Explicit message size list (overrides -m)\n");
    printf("  -S DIR            Save reference: run once, save result to DIR, exit\n");
    printf("  -R DIR            Load reference from DIR instead of computing internally\n");
    printf("  -h                Print this help\n");
}

test_config_t parse_arguments(int argc, char **argv) {
    test_config_t config = {
        .min_message_size = DEFAULT_MIN_SIZE,
        .max_message_size = DEFAULT_MAX_SIZE,
        .message_size_incr = DEFAULT_INCR,
        .message_size_incr_mode = 0,
        .iterations = DEFAULT_ITERATIONS,
        .warmup_iterations = DEFAULT_WARMUP,
        .input_file = NULL,
        .file_format = FILE_FORMAT_BINARY,
        .tolerance = DEFAULT_TOLERANCE,
        .validate = 0,
        .compute_metrics = 0,
        .metrics_mask = 0,
        .save_binary = 0,
        .bin_path = "",
        .pattern_type = PATTERN_RANK_LINEAR,
        .data_type = TYPE_DOUBLE
    };

    int opt;
    while ((opt = getopt(argc, argv, "d:m:i:w:f:t:vMhe:p:co:L:bB:S:R:A")) != -1) {
        switch (opt) {
            case 'm': {
                char *token = strtok(optarg, ":");
                if (token) config.min_message_size = atoi(token);
                token = strtok(NULL, ":");
                if (token) config.max_message_size = atoi(token);
                token = strtok(NULL, ":");
                if (token) config.message_size_incr = atoi(token);
                break;
            }
            case 'i':
                config.iterations = atoi(optarg);
                break;
            case 'w':
                config.warmup_iterations = atoi(optarg);
                break;
            case 'f':
                config.input_file = optarg;
                break;
            case 't':
                config.tolerance = atof(optarg);
                break;
            case 'v':
                config.validate = 1;
                break;
            case 'M':
                config.compute_metrics = 1;
                config.metrics_mask = (1u << validation_registry_count()) - 1;
                break;
            case 'e':
                config.compute_metrics = 1;
                config.metrics_mask = parse_metric_mask(optarg);
                break;
            case 'p':
                config.pattern_type = parse_pattern_type(optarg);
                break;
            case 'd':
                if (strcasecmp(optarg, "float") == 0)
                    config.data_type = TYPE_FLOAT;
                else if (strcasecmp(optarg, "double") == 0)
                    config.data_type = TYPE_DOUBLE;
                else if (strcasecmp(optarg, "int") == 0)
                    config.data_type = TYPE_INT;
                else if (strcasecmp(optarg, "char") == 0)
                    config.data_type = TYPE_CHAR;
                else {
                    fprintf(stderr, "Unknown data type: '%s' (options: float, double, int, char)\n", optarg);
                    exit(1);
                }
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            case 'c':
                config.save_csv = 1;
                break;
            case 'o':
                strncpy(config.csv_path, optarg, sizeof(config.csv_path) - 1);
                config.csv_path[sizeof(config.csv_path) - 1] = '\0';
                break;
            case 'b':
                config.save_binary = 1;
                break;
            case 'B':
                strncpy(config.bin_path, optarg, sizeof(config.bin_path) - 1);
                config.bin_path[sizeof(config.bin_path) - 1] = '\0';
                break;
            case 'S':
                strncpy(config.save_ref_path, optarg, sizeof(config.save_ref_path) - 1);
                config.save_ref_path[sizeof(config.save_ref_path) - 1] = '\0';
                break;
            case 'R':
                strncpy(config.ref_dir, optarg, sizeof(config.ref_dir) - 1);
                config.ref_dir[sizeof(config.ref_dir) - 1] = '\0';
                config.validate = 1;  /* -R implies validation */
                break;
            case 'A':
                config.message_size_incr_mode = 1;
                break;
            case 'L': {
                config.use_size_list = 1;
                config.num_sizes = 0;
                size_t min_seen = (size_t)-1, max_seen = 0;
                char *tok = strtok(optarg, ",");
                while (tok && config.num_sizes < 256) {
                    size_t sz = (size_t)atol(tok);
                    config.size_list[config.num_sizes++] = sz;
                    if (sz < min_seen) min_seen = sz;
                    if (sz > max_seen) max_seen = sz;
                    tok = strtok(NULL, ",");
                }
                config.min_message_size = min_seen;
                config.max_message_size = max_seen;
                break;
            }
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }

    return config;
}

/* ── Message-size iterator ───────────────────────────────────────── */

void size_iter_init(size_iter_t *it, const test_config_t *config) {
    it->use_list  = config->use_size_list;
    it->index     = 0;
    it->min_size  = config->min_message_size;
    it->max_size  = config->max_message_size;
    it->incr      = config->message_size_incr;
    it->incr_mode = config->message_size_incr_mode;
    it->num_sizes = config->num_sizes;
    if (it->use_list) {
        for (int i = 0; i < it->num_sizes; i++)
            it->size_list[i] = config->size_list[i];
    }
}

int size_iter_next(size_iter_t *it, size_t *size) {
    if (it->use_list) {
        if (it->index >= it->num_sizes) return 0;
        *size = it->size_list[it->index++];
        return 1;
    }

    size_t s;
    if (it->incr_mode == 0) {
        /* multiply: s = min * incr^index */
        s = it->min_size;
        for (int i = 0; i < it->index; i++)
            s *= it->incr;
    } else {
        /* add: s = min + index * incr */
        s = it->min_size + (size_t)it->index * it->incr;
    }
    it->index++;
    if (s > it->max_size) return 0;
    *size = s;
    return 1;
}