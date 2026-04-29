#ifndef PERF_HELPER_H
#define PERF_HELPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── tunables ────────────────────────────────────────────────── */
#define PERF_MAX_FUNCS    64         /* max unique functions tracked   */
#define PERF_NAME_MAX     64         /* max length of a function name  */
#define PERF_INIT_CAP     1024       /* initial records per function   */

/** Output directory: respects $PERF_OUTPUT_DIR env, falls back to "perf_files". */
const char *perf_output_dir(void);

/* ── one recorded call ───────────────────────────────────────── */
typedef struct {
    double timestamp_us;    /* start time (µs, relative to init)    */
    double duration_us;     /* duration (µs)                        */
} perf_entry_t;

/* ── records for one function ────────────────────────────────── */
typedef struct {
    char name[PERF_NAME_MAX];
    perf_entry_t *entries;  /* dynamic array of entries             */
    int count;              /* how many so far                      */
    int capacity;           /* allocated capacity                   */
} perf_func_t;

/* ── per-thread / per-process state ──────────────────────────── */
typedef struct {
    perf_func_t funcs[PERF_MAX_FUNCS];
    int num_funcs;          /* how many unique functions seen       */
    int64_t total_records;  /* total records across all funcs      */
    double base_time;       /* CLOCK_MONOTONIC @ init (seconds)    */
} perf_state_t;

/* ── API ─────────────────────────────────────────────────────── */

/**
 * Initialise perf state (record base_time, zero counters).
 */
void perf_init(perf_state_t *s);

/**
 * Record one call.
 *   func_name  – human-readable name of the function called
 *   start      – time just before the call (seconds, e.g. from perf_get_time())
 *   duration   – how long the call took (seconds)
 *
 * Internally hashes func_name to find-or-create a slot, then appends
 * a perf_entry_t with µs-precision values (relative to init).
 */
void perf_notedown(perf_state_t *s, const char *func_name,
                   double start, double duration);

/**
 * Write all recorded data to a CSV file.
 * Format:
 *   timestamp_us,duration_us,func_name
 *   0.000,12.345,MPI_Send
 *   100.456,8.901,MPI_Send
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
 * Safe to call multiple times.
 */
void perf_ensure_output_dir(void);

/**
 * Monotonic high-resolution time in seconds (like MPI_Wtime).
 * No MPI dependency — uses clock_gettime(CLOCK_MONOTONIC).
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