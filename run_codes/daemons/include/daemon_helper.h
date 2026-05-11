#ifndef DAEMON_HELPER_H
#define DAEMON_HELPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── tunables ────────────────────────────────────────────────── */
#define DAEMON_MAX_METRICS  10         /* named metrics per poll      */
#define DAEMON_NAME_MAX     64
#define DAEMON_UNIT_MAX     16
#define DAEMON_INIT_SAMPLES 1024       /* initial ring capacity       */

/** Output directory: respects $PERF_OUTPUT_DIR env, falls back to "perf_files". */
const char *daemon_output_dir(void);

/* ── metric definition (set once in setup) ───────────────────── */
typedef struct {
    char name[DAEMON_NAME_MAX];
    char unit[DAEMON_UNIT_MAX];        /* e.g. "%", "MB/s", "GT/s"   */
} daemon_metric_def_t;

/* ── one poll sample ─────────────────────────────────────────── */
typedef struct {
    double  timestamp;                  /* elapsed seconds (monotonic)*/
    double  values[DAEMON_MAX_METRICS];
    int     count;                      /* how many values valid      */
} daemon_sample_t;

/* ── daemon state ────────────────────────────────────────────── */
typedef struct {
    char device_name[DAEMON_NAME_MAX];  /* "ib", "cpu", "gpu", "pcie"*/
    daemon_metric_def_t metrics[DAEMON_MAX_METRICS];
    int  num_metrics;
    daemon_sample_t *samples;           /* dynamic array              */
    int  num_samples;
    int  capacity;
    double time_zero;                   /* CLOCK_MONOTONIC at init    */
} daemon_state_t;

/* ── core API ────────────────────────────────────────────────── */
void daemon_init(daemon_state_t *s, const char *device_name);

int daemon_add_metric(daemon_state_t *s, const char *name, const char *unit);

daemon_sample_t *daemon_new_sample(daemon_state_t *s);

/**
 * Flush all samples to a CSV file.
 * Header row: timestamp_sec,metric1,metric2,...
 * Data rows:  seconds,val1,val2,...
 *
 * File is created under PERF_OUTPUT_DIR, with flock(2) during write.
 */
int daemon_flush_to_file(daemon_state_t *s, const char *filepath);

/**
 * Create daemon_output_dir() if it doesn't exist.  Safe to call repeatedly.
 * The directory is determined by $PERF_OUTPUT_DIR env var (default "perf_files").
 */
void daemon_ensure_output_dir(void);

void daemon_destroy(daemon_state_t *s);

/* ── utilities ───────────────────────────────────────────────── */
double daemon_get_time(void);
int    daemon_interruptible_sleep(double seconds);
void   daemon_get_hostname(char *buf, size_t size);

/* ── signal handling ─────────────────────────────────────────── */
void daemon_setup_signal_handler(void);
int  daemon_should_stop(void);

/* ── file-based signal ─────────────────────────────────────────
 *
 * Instead of (or in addition to) OS signals, a daemon can periodically
 * check a "signal file" written by the controlling script.  Content:
 *
 *   <empty> / absent  → normal operation
 *   PAUSE             → skip sampling this interval
 *   EXIT / FLUSH_EXIT → flush data and exit gracefully
 *
 * The path is relative to the daemon's CWD (typically the project root).
 */
void daemon_set_signal_file(const char *path);
int  daemon_check_signal_file(void);   /* 0=normal, 1=PAUSE, 2=EXIT */

/* ── sysfs helpers ───────────────────────────────────────────── */
int daemon_read_sysfs_u64(const char *path, unsigned long long *val);
int daemon_read_sysfs_str(const char *path, char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* DAEMON_HELPER_H */