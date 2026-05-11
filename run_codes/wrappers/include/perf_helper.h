#ifndef PERF_HELPER_H
#define PERF_HELPER_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── tunables ────────────────────────────────────────────────── */
#define PERF_MAX_FUNCS    64         /* max unique functions tracked   */
#define PERF_NAME_MAX     64         /* max length of a name string    */
#define PERF_COL_MAX      32         /* max columns (attrs) per func   */
#define PERF_INIT_CAP     1024       /* initial records per function   */

/** Output directory: respects $PERF_OUTPUT_DIR env, falls back to "perf_files". */
const char *perf_output_dir(void);

/* ── one attribute (key-value pair) ──────────────────────────── */
typedef struct {
    const char *key;
    double value;
} perf_attr_t;

/* ── records for one function: dynamic schema (col-major) ────── */
typedef struct {
    char name[PERF_NAME_MAX];
    char col_names[PERF_COL_MAX][PERF_NAME_MAX];
    int num_cols;
    double **col_data;          /* [num_cols] arrays, each [capacity]     */
    int count;
    int entry_capacity;
} perf_func_t;

/* ── per-process state (thread-safe via internal lock) ─────────── */
typedef struct {
    perf_func_t funcs[PERF_MAX_FUNCS];
    int num_funcs;              /* how many unique functions seen       */
    int64_t total_records;      /* total records across all funcs       */
    double base_time;           /* CLOCK_MONOTONIC @ init (seconds)    */
    pthread_mutex_t lock;       /* guards all mutable fields            */
} perf_state_t;

/* ── API ─────────────────────────────────────────────────────── */

/**
 * Initialise perf state (record base_time, zero counters).
 */
void perf_init(perf_state_t *s);

/**
 * Record one call with dynamic attributes.
 *
 *   func_name  – name of the function called
 *   start      – perf_get_time() just before the call (seconds)
 *   num_attrs  – number of key-value pairs
 *   attrs      – array of {key, value}
 *
 * Column 0 is always "timestamp_us" (auto-computed from start).
 * Each unique attr key becomes a new column on first sight.
 * Missing columns for existing entries are filled with NaN.
 * Pass "duration" as an attr: {"duration", t1 - t0}.
 */
void perf_notedown(perf_state_t *s, const char *func_name,
                   double start, int num_attrs, const perf_attr_t *attrs);

/**
 * Write all recorded data to a CSV file.
 *
 * The header is the union of all column names across all functions.
 * Missing attr values appear as empty fields.
 *
 * Returns 0 on success, -1 on error.
 */
int perf_flush_to_file(perf_state_t *s, const char *filepath);

/**
 * Convenience: builds path "prefix_<rank>.csv" under PERF_OUTPUT_DIR.
 */
int perf_flush_to_rank_file(perf_state_t *s, int rank, const char *prefix);

/**
 * Create perf_output_dir() if it doesn't exist.
 * $PERF_OUTPUT_DIR env var overrides the default ("perf_files").
 */
void perf_ensure_output_dir(void);

/**
 * Monotonic high-resolution time in seconds.
 */
double perf_get_time(void);

/**
 * Free all dynamically allocated memory in the perf state.
 */
void perf_destroy(perf_state_t *s);

#ifdef __cplusplus
}
#endif

#endif /* PERF_HELPER_H */
