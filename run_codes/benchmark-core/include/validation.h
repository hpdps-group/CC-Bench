/**
 * Validation module for correctness and error metrics.
 *
 * Metrics are pluggable: each metric is a separate .c file in
 * userconfig/deviation_metric_examples/ (or your_deviation_metrics/).
 * Run scripts/register_metrics.sh to regenerate the registry.
 *
 * Architecture-independent — operates on raw memory with data_type_t.
 */

#ifndef VALIDATION_H
#define VALIDATION_H

#include <stddef.h>
#include <stdint.h>
#include "utils.h"

/* Maximum number of metrics (bitmask = uint32_t) */
#define MAX_METRICS 30

/* ── Metric registry ──────────────────────────────────────────── */

/** Signature for a metric function. Extra params read via getenv(). */
typedef double (*metric_func_t)(const void *buf1, const void *buf2,
                                int count, data_type_t dtype);

/** A single entry in the metric registry. */
typedef struct {
    const char *name;       /* registration name (from filename) */
    metric_func_t func;     /* the compute function */
} metric_entry_t;

/** Global registry — loaded at runtime from bin/libs/validation.so. */
int init_validation_so(void);
int validation_registry_count(void);
const char *validation_metric_name(int idx);

/* ── Helpers (non-static so metric implementations can use them) ─ */

/** Return sizeof one element of the given data type. */
size_t get_element_size(data_type_t dtype);

/** Convert element at index to double for arithmetic. */
double to_double(const void *ptr, data_type_t dtype, int index);

/* ── Validation result ────────────────────────────────────────── */

/** Validation result with dynamic metric values (indexed same as registry). */
typedef struct {
    int correct;                 /* 1 if buffers equal within tolerance */
    size_t num_elements;
    double values[MAX_METRICS];  /* metric results, indexed by registry slot */
} validation_result_t;

/**
 * Parse comma-separated metric names into a bitmask.
 * Names are matched against g_metric_registry (case-insensitive).
 * Whitespace around tokens is trimmed.
 * Returns 0 on empty/null input.
 */
uint32_t parse_metric_mask(const char *str);

/**
 * Validate communication result against reference.
 * Computes only the metrics whose bits are set in metrics_mask.
 */
validation_result_t validate_result(
    const void *user_buf,
    const void *reference_buf,
    int count,
    data_type_t dtype,
    double tolerance,
    uint32_t metrics_mask
);

/**
 * Check if two buffers are equal within tolerance.
 */
int buffers_equal(
    const void *buf1,
    const void *buf2,
    int count,
    data_type_t datatype,
    double tolerance
);

#endif /* VALIDATION_H */