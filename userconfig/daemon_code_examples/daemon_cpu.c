/*
 * daemon_cpu.c  —  Per-core CPU + memory utilization poller.
 *
 * Discovers all cpuN lines from /proc/stat and monitors each
 * core individually plus the aggregate.
 *
 * Usage   ./daemon_cpu [interval_sec]
 *         SIGINT/SIGTERM to stop gracefully.
 *
 * Output  perf_cpu_<hostname>.txt
 *         perf_cpu0_<hostname>.txt  …  perf_cpuN_<hostname>.txt
 */

#include "daemon_helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_CORES 512

/* ── per-core context (keep previous for delta) ──────────────── */
typedef struct {
    daemon_state_t state;
    char           core_tag[16];    /* "cpu" for aggregate, "cpu0", … */
    unsigned long long prev_user;
    unsigned long long prev_nice;
    unsigned long long prev_system;
    unsigned long long prev_idle;
    unsigned long long prev_iowait;
    unsigned long long prev_irq;
    unsigned long long prev_softirq;
    unsigned long long prev_steal;
    int first;
} core_ctx_t;

static core_ctx_t  ctxs[MAX_CORES];
static int         nctx = 0;
static char        hostname[64];

/* ── parse one /proc/stat line (starting with "cpu" or "cpuN") ─ */
static int parse_stat_line(FILE *fp, const char *prefix,
                           unsigned long long *user,
                           unsigned long long *nice,
                           unsigned long long *system,
                           unsigned long long *idle,
                           unsigned long long *iowait,
                           unsigned long long *irq,
                           unsigned long long *softirq,
                           unsigned long long *steal)
{
    rewind(fp);
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, prefix, strlen(prefix)) != 0) continue;
        /* make sure it's "cpu" or "cpuN ", not "cpuN_anything" */
        int len = (int)strlen(prefix);
        if (line[len] != ' ' && line[len] != '\t') continue;

        return sscanf(line + len,
            "%llu %llu %llu %llu %llu %llu %llu %llu",
            user, nice, system, idle, iowait, irq, softirq, steal);
    }
    return -1;
}

/* ── read /proc/meminfo ──────────────────────────────────────── */
static double read_mem_util(void)
{
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0.0;

    unsigned long long total = 0, free_mem = 0, buffers = 0, cached = 0;
    char line[128];
    while (fgets(line, sizeof(line), fp)) {
        if      (sscanf(line, "MemTotal: %llu", &total)   == 1) {}
        else if (sscanf(line, "MemFree: %llu", &free_mem) == 1) {}
        else if (sscanf(line, "Buffers: %llu", &buffers)  == 1) {}
        else if (sscanf(line, "Cached: %llu", &cached)    == 1) {}
    }
    fclose(fp);

    if (total == 0) return 0.0;
    unsigned long long used = total - free_mem - buffers - cached;
    return (double)used / (double)total * 100.0;
}

/* ── discover cores from /proc/stat ──────────────────────────── */
static int discover_cores(void)
{
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) return -1;

    /* Aggregate "cpu" */
    if (nctx < MAX_CORES) {
        core_ctx_t *c = &ctxs[nctx];
        char name[64];
        snprintf(name, sizeof(name), "cpu_%s", hostname);
        daemon_init(&c->state, name);
        snprintf(c->core_tag, sizeof(c->core_tag), "cpu");
        daemon_add_metric(&c->state, "cpu_util",   "%");
        daemon_add_metric(&c->state, "cpu_user",   "%");
        daemon_add_metric(&c->state, "cpu_sys",    "%");
        daemon_add_metric(&c->state, "cpu_iowait", "%");
        daemon_add_metric(&c->state, "mem_util",   "%");
        c->first = 1;
        nctx++;
    }

    /* Per-core lines */
    char line[256];
    while (fgets(line, sizeof(line), fp) && nctx < MAX_CORES) {
        if (strncmp(line, "cpu", 3) != 0) continue;
        /* check for cpu0, cpu1, … (but not the aggregate "cpu ") */
        if (line[3] < '0' || line[3] > '9') continue;

        char core_name[16];
        int n = 0;
        sscanf(line, "%15s%n", core_name, &n);
        if (n <= 0) continue;

        core_ctx_t *c = &ctxs[nctx];
        char state_name[64];
        snprintf(state_name, sizeof(state_name), "%s_%s", core_name, hostname);
        daemon_init(&c->state, state_name);
        snprintf(c->core_tag, sizeof(c->core_tag), "%s", core_name);
        daemon_add_metric(&c->state, "cpu_util",   "%");
        daemon_add_metric(&c->state, "cpu_user",   "%");
        daemon_add_metric(&c->state, "cpu_sys",    "%");
        daemon_add_metric(&c->state, "cpu_iowait", "%");
        c->first = 1;
        nctx++;
    }

    fclose(fp);
    return nctx;
}

/* ── poll one core ───────────────────────────────────────────── */
static void poll_core(core_ctx_t *c, FILE *fp, double mem_util)
{
    unsigned long long user=0, nice=0, system=0, idle=0;
    unsigned long long iowait=0, irq=0, softirq=0, steal=0;

    if (parse_stat_line(fp, c->core_tag,
                        &user, &nice, &system, &idle,
                        &iowait, &irq, &softirq, &steal) < 4)
        return;

    daemon_sample_t *sp = daemon_new_sample(&c->state);
    if (!sp) return;

    if (c->first) {
        sp->values[0] = 0.0;
        sp->values[1] = 0.0;
        sp->values[2] = 0.0;
        sp->values[3] = 0.0;
        if (c->state.num_metrics > 4)
            sp->values[4] = mem_util;   /* only aggregate has mem_util */
    } else {
        unsigned long long d_total =
            (user - c->prev_user) + (nice - c->prev_nice) +
            (system - c->prev_system) + (idle - c->prev_idle) +
            (iowait - c->prev_iowait) + (irq - c->prev_irq) +
            (softirq - c->prev_softirq) + (steal - c->prev_steal);
        unsigned long long d_idle = idle - c->prev_idle;

        if (d_total > 0) {
            sp->values[0] = (double)(d_total - d_idle) / (double)d_total * 100.0;
            sp->values[1] = (double)(user - c->prev_user) / (double)d_total * 100.0;
            sp->values[2] = (double)(system - c->prev_system) / (double)d_total * 100.0;
            sp->values[3] = (double)(iowait - c->prev_iowait) / (double)d_total * 100.0;
        }
        if (c->state.num_metrics > 4)
            sp->values[4] = mem_util;
    }

    c->prev_user    = user;
    c->prev_nice    = nice;
    c->prev_system  = system;
    c->prev_idle    = idle;
    c->prev_iowait  = iowait;
    c->prev_irq     = irq;
    c->prev_softirq = softirq;
    c->prev_steal   = steal;
    c->first = 0;
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    double interval = 1.0;
    if (argc > 1) interval = atof(argv[1]);
    if (interval <= 0.0) interval = 1.0;

    daemon_get_hostname(hostname, sizeof(hostname));

    if (discover_cores() <= 0) {
        fprintf(stderr, "[daemon_cpu] no cores found\n");
        return 1;
    }
    printf("[daemon_cpu] monitoring %d core(s) on %s, interval %.2fs\n",
           nctx, hostname, interval);

    daemon_setup_signal_handler();
    daemon_set_signal_file("daemon_signals/daemon_cpu.signal");
    daemon_ensure_output_dir();

    while (!daemon_should_stop()) {
        int sig = daemon_check_signal_file();
        if (sig == 2) break;           /* EXIT → flush and stop */
        if (sig == 1) {                /* PAUSE → skip sampling */
            daemon_interruptible_sleep(interval);
            continue;
        }

        double t0 = daemon_get_time();

        double mem_util = read_mem_util();

        /* open /proc/stat once and reuse across cores */
        FILE *fp = fopen("/proc/stat", "r");
        if (fp) {
            for (int i = 0; i < nctx; i++)
                poll_core(&ctxs[i], fp, mem_util);
            fclose(fp);
        }

        double elapsed = daemon_get_time() - t0;
        if (elapsed < interval)
            daemon_interruptible_sleep(interval - elapsed);
    }

    for (int i = 0; i < nctx; i++) {
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/perf_%s.csv",
                 daemon_output_dir(), ctxs[i].state.device_name);
        daemon_flush_to_file(&ctxs[i].state, fname);
        daemon_destroy(&ctxs[i].state);
    }

    printf("[daemon_cpu] stopped, wrote %d core(s)\n", nctx);
    return 0;
}