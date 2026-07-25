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

    /* ── Gendata (app_gendata buffer dump) ──────────────────────── */
    int gc_loaded;                      /* PERF_GENDATA_TARGETS parsed?    */
    char gc_ops[64][PERF_NAME_MAX];     /* operation names from targets    */
    int  gc_occs[64][64];              /* occurrence lists per entry      */
    int  gc_occ_cnt[64];               /* # of occurrences per entry      */
    int  gc_num_entries;                /* total target entries            */
    int  gc_func_counters[PERF_MAX_FUNCS]; /* per-function call counters   */
    int  gc_matched[64][64];            /* per (entry,occ_idx): dumped?    */
    int  gc_rank;                       /* cached process rank             */
    int  gc_nranks;                     /* cached world size               */
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

/* ── Gendata: app_gendata buffer dump ────────────────────────── */

/**
 * Load gendata targets from PERF_GENDATA_TARGETS env var.
 * Format: "op:occ1,occ2;op:occ1"  e.g. "all:50,100;MPI_Allreduce:25"
 * Reads PERF_GENDATA_OUTPUT_DIR and PERF_GENDATA_DONE_DIR from env.
 * Returns number of entries loaded, or 0 if none.
 */
int perf_gendata_load_targets(perf_state_t *s);

/**
 * Dump buffer to {PERF_GENDATA_OUTPUT_DIR}/{func_name}_{occurrence}/rank_{rank}.bin
 * if the current call counter for func_name matches a gendata target.
 *
 * When this rank completes all its targets, writes a done signal file to
 * PERF_GENDATA_DONE_DIR and polls for peer signals (300s timeout).
 * If all peers finish in time the process exits early; otherwise it
 * continues to natural termination.
 */
void perf_gendata_dump_if_target(perf_state_t *s, const char *func_name,
                                 const void *buf, size_t buf_size);

/**
 * Get rank from env var chain: OMPI_COMM_WORLD_RANK → PMI_RANK →
 * SLURM_PROCID → MV2_COMM_WORLD_RANK → 0.
 */
int perf_gendata_get_rank(void);

/**
 * Get nranks from env var chain: OMPI_COMM_WORLD_SIZE → PMI_SIZE →
 * SLURM_NPROCS → MV2_COMM_WORLD_SIZE → 1.
 */
int perf_gendata_get_nranks(void);

#ifdef __cplusplus
}
#endif

#endif /* PERF_HELPER_H */
