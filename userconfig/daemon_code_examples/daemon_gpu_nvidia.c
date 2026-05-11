/*
 * daemon_gpu.c  —  GPU utilisation poller (NVIDIA, via nvidia-smi).
 *
 * Usage   ./daemon_gpu [interval_sec]
 *         SIGINT/SIGTERM to stop gracefully.
 *
 * Output  perf_gpu_<idx>.txt   (one file per GPU)
 *
 * Notes
 * -----
 * Requires nvidia-smi on PATH.
 * Only polls GPUs visible at start — no hot-plug.
 */

#define _GNU_SOURCE
#include "daemon_helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_GPUS 16

/* ── per-GPU context ─────────────────────────────────────────── */
typedef struct {
    int            idx;              /* GPU index                   */
    daemon_state_t state;
} gpu_ctx_t;

static gpu_ctx_t ctxs[MAX_GPUS];
static int       ngpu = 0;

/* ── discover GPUs via nvidia-smi ────────────────────────────── */
static int discover_gpus(void)
{
    FILE *fp = popen(
        "nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null",
        "r");
    if (!fp) return -1;

    char buf[64];
    while (fgets(buf, sizeof(buf), fp) && ngpu < MAX_GPUS) {
        int idx = atoi(buf);
        gpu_ctx_t *c = &ctxs[ngpu];
        c->idx = idx;

        char devname[32];
        snprintf(devname, sizeof(devname), "gpu_%d", idx);
        daemon_init(&c->state, devname);

        daemon_add_metric(&c->state, "gpu_util",  "%");
        daemon_add_metric(&c->state, "mem_util",  "%");
        daemon_add_metric(&c->state, "mem_used",  "MB");
        daemon_add_metric(&c->state, "temp",      "C");
        daemon_add_metric(&c->state, "pcie_gen",  "");
        daemon_add_metric(&c->state, "pcie_width","");
        ngpu++;
    }
    pclose(fp);
    return ngpu;
}

/* ── poll ────────────────────────────────────────────────────── */
static void poll_gpu(gpu_ctx_t *c)
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
        "nvidia-smi --id=%d "
        "--query-gpu=utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,"
        "pcie.link.gen.current,pcie.link.width.current "
        "--format=csv,noheader,nounits 2>/dev/null",
        c->idx);

    FILE *fp = popen(cmd, "r");
    if (!fp) return;

    double gpu_u=0, mem_u=0, mem_used=0, mem_total=0, temp=0;
    double pcie_gen=0, pcie_w=0;
    int n = fscanf(fp, "%lf, %lf, %lf, %lf, %lf, %lf, %lf",
                   &gpu_u, &mem_u, &mem_used, &mem_total,
                   &temp, &pcie_gen, &pcie_w);
    pclose(fp);

    if (n < 1) return;

    daemon_sample_t *sp = daemon_new_sample(&c->state);
    if (!sp) return;
    sp->values[0] = gpu_u;
    sp->values[1] = mem_u;
    sp->values[2] = mem_used;
    sp->values[3] = temp;
    sp->values[4] = pcie_gen;
    sp->values[5] = pcie_w;
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    double interval = 1.0;
    if (argc > 1) interval = atof(argv[1]);
    if (interval <= 0.0) interval = 1.0;

    char hostname[64];
    daemon_get_hostname(hostname, sizeof(hostname));

    if (discover_gpus() <= 0) {
        fprintf(stderr, "[daemon_gpu] no GPUs found (nvidia-smi ok?)\n");
        return 1;
    }

    /* append hostname */
    for (int i = 0; i < ngpu; i++) {
        char merged[DAEMON_NAME_MAX];
        snprintf(merged, sizeof(merged), "%s_%s",
                 ctxs[i].state.device_name, hostname);
        strncpy(ctxs[i].state.device_name, merged, DAEMON_NAME_MAX - 1);
    }

    printf("[daemon_gpu] monitoring %d GPU(s) on %s, interval %.2fs\n",
           ngpu, hostname, interval);

    daemon_setup_signal_handler();
    daemon_set_signal_file("daemon_signals/daemon_gpu_nvidia.signal");
    daemon_ensure_output_dir();

    while (!daemon_should_stop()) {
        int sig = daemon_check_signal_file();
        if (sig == 2) break;           /* EXIT → flush and stop */
        if (sig == 1) {                /* PAUSE → skip sampling */
            daemon_interruptible_sleep(interval);
            continue;
        }

        double t0 = daemon_get_time();

        for (int i = 0; i < ngpu; i++)
            poll_gpu(&ctxs[i]);

        double elapsed = daemon_get_time() - t0;
        if (elapsed < interval)
            daemon_interruptible_sleep(interval - elapsed);
    }

    for (int i = 0; i < ngpu; i++) {
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/perf_%s.csv",
                 daemon_output_dir(), ctxs[i].state.device_name);
        daemon_flush_to_file(&ctxs[i].state, fname);
        daemon_destroy(&ctxs[i].state);
    }

    printf("[daemon_gpu] stopped\n");
    return 0;
}