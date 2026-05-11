/*
 * daemon_stress_cpu.c — CPU stress daemon (multi-core)
 *
 * Spawns one worker thread per online CPU core; each thread runs an
 * identical work/sleep duty cycle so that every core experiences the
 * same level of contention.  Runs alongside Phase 2 benchmarks to
 * test performance under computational load.
 *
 * Environment:
 *   STRESS_CPU_PERCENT  — per-core target utilisation 1-100 (default 50)
 *
 * Signal / file control:  standard daemon helper mechanism.
 *
 * Build (automatic via register_daemons.sh):
 *   ./scripts/register_daemons.sh
 *
 * Output: bin/daemons/daemon_stress_cpu
 */

#include "daemon_helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

/* ── Busy-work tunable ─────────────────────────────────────────── */
#define CYCLE_SEC  0.100   /* 100 ms duty-cycle granularity */

/* ── Shared stop flag (set by main thread on EXIT signal) ──────── */
static volatile sig_atomic_t g_worker_stop = 0;

/* ── Per-thread parameters ─────────────────────────────────────── */
typedef struct {
    double work_sec;
    double sleep_sec;
} worker_params_t;

/* ── Busy computation — guaranteed not to be optimized away ────── */
static void burn_cpu(double seconds)
{
    volatile double x = 1.0;
    double t0 = daemon_get_time();
    while (daemon_get_time() - t0 < seconds) {
        x = x * 1.0000001 + 0.0000001;   /* mul + add */
        x = x / 1.0000001 - 0.0000001;   /* div + sub */
    }
    (void)x;
}

/* ── Clamp utility ─────────────────────────────────────────────── */
static int clamp(int val, int lo, int hi)
{
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/* ── Worker thread — one per core ──────────────────────────────── */
static void *worker_routine(void *arg)
{
    const worker_params_t *p = (const worker_params_t *)arg;

    while (!g_worker_stop && !daemon_should_stop()) {
        burn_cpu(p->work_sec);
        if (p->sleep_sec > 0.0)
            daemon_interruptible_sleep(p->sleep_sec);
    }
    return NULL;
}

/* ── Main ──────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    /* Read target CPU% from environment */
    int cpu_pct = 50;
    const char *env = getenv("STRESS_CPU_PERCENT");
    if (env) cpu_pct = clamp(atoi(env), 1, 100);

    /* Detect online CPU count */
    int num_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cores < 1) num_cores = 1;

    double work_sec  = CYCLE_SEC * cpu_pct / 100.0;
    double sleep_sec = CYCLE_SEC - work_sec;

    /* Signals */
    daemon_setup_signal_handler();
    daemon_set_signal_file("daemon_signals/daemon_stress_cpu.signal");

    fprintf(stderr, "[daemon_stress_cpu] starting, %d cores @ %d%% duty each\n",
            num_cores, cpu_pct);

    /* Spawn one worker per core */
    pthread_t *workers = malloc((size_t)num_cores * sizeof(pthread_t));
    if (!workers) {
        fprintf(stderr, "[daemon_stress_cpu] malloc failed\n");
        return 1;
    }

    worker_params_t params = { work_sec, sleep_sec };
    for (int i = 0; i < num_cores; i++)
        pthread_create(&workers[i], NULL, worker_routine, &params);

    /* Main thread monitors signal file */
    while (!daemon_should_stop()) {
        int sig = daemon_check_signal_file();
        if (sig == 2) {          /* EXIT / FLUSH_EXIT */
            g_worker_stop = 1;
            break;
        }
        if (sig == 1) {          /* PAUSE */
            daemon_interruptible_sleep(CYCLE_SEC);
            continue;
        }
        daemon_interruptible_sleep(CYCLE_SEC);
    }

    /* Join all workers */
    for (int i = 0; i < num_cores; i++)
        pthread_join(workers[i], NULL);

    free(workers);
    fprintf(stderr, "[daemon_stress_cpu] stopped\n");
    return 0;
}
