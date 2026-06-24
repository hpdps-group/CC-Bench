/**
 * Validation framework — correctness check + metric dispatch via registry.
 *
 * The metric registry is loaded at runtime from bin/libs/validation.so
 * via dlopen/dlsym.  Adding a metric only requires re-running
 * scripts/register_metrics.sh (no benchmark binary rebuild).
 */

#include "validation.h"
#include <dlfcn.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── Runtime-loaded metric registry ───────────────────────────── */
static void          *g_so_handle   = NULL;
static metric_entry_t *g_registry    = NULL;
static int            g_registry_count = 0;

static int load_validation_so(void)
{
    if (g_so_handle) return 0;

    const char *bench_dir = getenv("BENCH_DIR");
    char path[4096];

    /* Try BENCH_DIR-relative path first (for non-cwd runs) */
    if (bench_dir) {
        snprintf(path, sizeof(path), "%s/bin/libs/validation.so", bench_dir);
        g_so_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    }
    /* Fallback to cwd-relative path */
    if (!g_so_handle)
        g_so_handle = dlopen("bin/libs/validation.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_so_handle) {
        fprintf(stderr, "[validation] dlopen(bin/libs/validation.so) failed: %s\n", dlerror());
        return -1;
    }

    g_registry = (metric_entry_t *)dlsym(g_so_handle, "g_metric_registry");
    int *count_ptr = (int *)dlsym(g_so_handle, "g_metric_registry_count");
    if (!g_registry || !count_ptr) {
        fprintf(stderr, "[validation] dlsym failed: %s\n", dlerror());
        dlclose(g_so_handle);
        g_so_handle = NULL;
        return -1;
    }
    g_registry_count = *count_ptr;
    return 0;
}

/* ── Public accessors ─────────────────────────────────────────── */

int init_validation_so(void)
{
    return load_validation_so();
}

int validation_registry_count(void)
{
    if (!g_so_handle) load_validation_so();
    return g_registry_count;
}

const char *validation_metric_name(int idx)
{
    if (!g_so_handle) load_validation_so();
    if (idx >= 0 && idx < g_registry_count)
        return g_registry[idx].name;
    return NULL;
}

/* ── Helpers (used by metric implementations in deviation_metrics.c) ─ */

size_t get_element_size(data_type_t dtype) {
    switch (dtype) {
        case TYPE_INT:    return sizeof(int);
        case TYPE_FLOAT:  return sizeof(float);
        case TYPE_DOUBLE: return sizeof(double);
        case TYPE_CHAR:   return sizeof(char);
        default:          return 1;
    }
}

double to_double(const void *ptr, data_type_t dtype, int index) {
    size_t elem_size = get_element_size(dtype);
    const char *byte_ptr = (const char *)ptr + index * elem_size;

    switch (dtype) {
        case TYPE_INT:    return (double)*(const int *)byte_ptr;
        case TYPE_FLOAT:  return (double)*(const float *)byte_ptr;
        case TYPE_DOUBLE: return *(const double *)byte_ptr;
        case TYPE_CHAR:   return (double)*(const char *)byte_ptr;
        default:          return 0.0;
    }
}

/* ── Correctness check ────────────────────────────────────────── */

int buffers_equal(
    const void *buf1,
    const void *buf2,
    int count,
    data_type_t datatype,
    double tolerance
) {
    data_type_t dtype = datatype;
    size_t elem_size = get_element_size(dtype);

    if (dtype == TYPE_FLOAT || dtype == TYPE_DOUBLE) {
        for (int i = 0; i < count; i++) {
            double v1 = to_double(buf1, dtype, i);
            double v2 = to_double(buf2, dtype, i);
            if (fabs(v1 - v2) > tolerance) return 0;
        }
        return 1;
    } else {
        return memcmp(buf1, buf2, count * elem_size) == 0;
    }
}

/* ── Metric name → bitmask ────────────────────────────────────── */

uint32_t parse_metric_mask(const char *str) {
    if (!str || str[0] == '\0') return 0;
    if (load_validation_so() != 0) return 0;

    uint32_t mask = 0;
    char *copy = strdup(str);
    if (!copy) return 0;

    char *token = strtok(copy, ",");
    while (token) {
        /* Trim whitespace */
        while (*token == ' ') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && *end == ' ') *end-- = '\0';

        if (strcasecmp(token, "all") == 0) {
            mask = (1u << g_registry_count) - 1;
            break;
        }

        for (int i = 0; i < g_registry_count; i++) {
            if (strcasecmp(token, g_registry[i].name) == 0) {
                mask |= (1u << i);
                break;
            }
        }
        token = strtok(NULL, ",");
    }

    free(copy);
    return mask;
}

/* ── Validation dispatch ──────────────────────────────────────── */

validation_result_t validate_result(
    const void *user_buf,
    const void *reference_buf,
    int count,
    data_type_t dtype,
    double tolerance,
    uint32_t metrics_mask
) {
    validation_result_t result = {0};
    if (load_validation_so() != 0) return result;

    result.num_elements = count;
    result.correct = buffers_equal(user_buf, reference_buf, count, dtype, tolerance);

    /* Compute requested metrics via registry */
    for (int i = 0; i < g_registry_count && i < MAX_METRICS; i++) {
        if (metrics_mask & (1u << i)) {
            result.values[i] = g_registry[i].func(
                user_buf, reference_buf, count, dtype);
        }
    }

    return result;
}