/*
 * daemon_ib.c  —  InfiniBand port-counter poller.
 *
 * Discovers IB devices/ports from sysfs, polls hardware counters
 * at a fixed interval, computes RX/TX bandwidth, and writes all
 * samples on graceful shutdown.
 *
 * Usage   ./daemon_ib [interval_sec]
 *         Default interval: 1.0 second.
 *         SIGINT/SIGTERM to stop gracefully.
 *
 * Output  perf_ib_<dev>_<port>.txt  (one file per port)
 */

#include "daemon_helper.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ── per-port context ────────────────────────────────────────── */
typedef struct {
    char              dev[256];
    int               port;
    daemon_state_t    state;          /* owns its samples            */
    unsigned long long prev_rcv_data;
    unsigned long long prev_xmit_data;
    double             prev_time;
    int                first;         /* 1 before first real sample  */
} ib_ctx_t;

#define MAX_CTX 16
static ib_ctx_t  ctxs[MAX_CTX];
static int       nctx = 0;

/* ── discover ────────────────────────────────────────────────── */
static int discover_ports(void)
{
    DIR *d = opendir("/sys/class/infiniband");
    if (!d) { perror("/sys/class/infiniband"); return -1; }

    struct dirent *e;
    while ((e = readdir(d)) && nctx < MAX_CTX) {
        if (e->d_name[0] == '.') continue;

        char pdir[512];
        snprintf(pdir, sizeof(pdir),
                 "/sys/class/infiniband/%s/ports", e->d_name);
        DIR *pd = opendir(pdir);
        if (!pd) continue;

        struct dirent *pe;
        while ((pe = readdir(pd)) && nctx < MAX_CTX) {
            if (pe->d_name[0] == '.') continue;
            ib_ctx_t *c = &ctxs[nctx];
            snprintf(c->dev, sizeof(c->dev), "%s", e->d_name);
            c->port  = atoi(pe->d_name);
            c->first = 1;

            char devname[256];
            snprintf(devname, sizeof(devname), "ib_%s_p%d", c->dev, c->port);
            daemon_init(&c->state, devname);

            daemon_add_metric(&c->state, "rcv_data_mb",  "MB");
            daemon_add_metric(&c->state, "xmit_data_mb", "MB");
            daemon_add_metric(&c->state, "rcv_packets",  "pkt");
            daemon_add_metric(&c->state, "xmit_packets", "pkt");
            daemon_add_metric(&c->state, "rcv_errors",   "err");
            daemon_add_metric(&c->state, "xmit_discards","pkt");
            daemon_add_metric(&c->state, "rcv_bw_Bps",   "B/s");
            daemon_add_metric(&c->state, "xmit_bw_Bps",  "B/s");
            nctx++;
        }
        closedir(pd);
    }
    closedir(d);
    return nctx;
}

/* ── poll one port ───────────────────────────────────────────── */
static void poll_port(ib_ctx_t *c)
{
    unsigned long long rcv_d = 0, xmit_d = 0;
    unsigned long long rcv_p = 0, xmit_p = 0;
    unsigned long long rcv_e = 0, xmit_disc = 0;
    char path[256];

#define READ_COUNTER(n, v) do { \
    snprintf(path, sizeof(path), \
        "/sys/class/infiniband/%s/ports/%d/counters/%s", \
        c->dev, c->port, n); \
    daemon_read_sysfs_u64(path, &v); \
} while(0)

    READ_COUNTER("port_rcv_data",   rcv_d);
    READ_COUNTER("port_xmit_data",  xmit_d);
    READ_COUNTER("port_rcv_packets", rcv_p);
    READ_COUNTER("port_xmit_packets", xmit_p);
    READ_COUNTER("port_rcv_errors",  rcv_e);
    READ_COUNTER("port_xmit_discards", xmit_disc);

    double now = daemon_get_time();
    double dt  = c->first ? 0.0 : now - c->prev_time;
    double rcv_bw = 0.0, xmit_bw = 0.0;
    if (dt > 0.0) {
        /* Counters are in 4-byte units */
        rcv_bw = (double)(rcv_d - c->prev_rcv_data) * 4.0 / dt;
        xmit_bw = (double)(xmit_d - c->prev_xmit_data) * 4.0 / dt;
    }
    c->prev_rcv_data = rcv_d;
    c->prev_xmit_data = xmit_d;
    c->prev_time = now;
    c->first = 0;

    daemon_sample_t *sp = daemon_new_sample(&c->state);
    if (!sp) return;
    /* rcv_data/xmit_data in MB for readability */
    double rcv_mb = (double)rcv_d * 4.0 / (1024.0 * 1024.0);
    double xmit_mb = (double)xmit_d * 4.0 / (1024.0 * 1024.0);
    sp->values[0] = rcv_mb;
    sp->values[1] = xmit_mb;
    sp->values[2] = (double)rcv_p;
    sp->values[3] = (double)xmit_p;
    sp->values[4] = (double)rcv_e;
    sp->values[5] = (double)xmit_disc;
    sp->values[6] = rcv_bw;
    sp->values[7] = xmit_bw;
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    double interval = 1.0;
    if (argc > 1) interval = atof(argv[1]);
    if (interval <= 0.0) interval = 1.0;

    char hostname[64];
    daemon_get_hostname(hostname, sizeof(hostname));

    if (discover_ports() <= 0) {
        fprintf(stderr, "[daemon_ib] no IB ports found\n");
        return 1;
    }

    /* append hostname to each device name */
    for (int i = 0; i < nctx; i++) {
        char merged[DAEMON_NAME_MAX];
        snprintf(merged, sizeof(merged), "%s_%s",
                 ctxs[i].state.device_name, hostname);
        strncpy(ctxs[i].state.device_name, merged, DAEMON_NAME_MAX - 1);
    }

    printf("[daemon_ib] monitoring %d port(s) on %s, interval %.2fs\n",
           nctx, hostname, interval);

    daemon_setup_signal_handler();
    daemon_set_signal_file("daemon_signals/daemon_ib.signal");
    daemon_ensure_output_dir();

    while (!daemon_should_stop()) {
        int sig = daemon_check_signal_file();
        if (sig == 2) break;           /* EXIT → flush and stop */
        if (sig == 1) {                /* PAUSE → skip sampling */
            daemon_interruptible_sleep(interval);
            continue;
        }

        double t0 = daemon_get_time();

        for (int i = 0; i < nctx; i++)
            poll_port(&ctxs[i]);

        double elapsed = daemon_get_time() - t0;
        if (elapsed < interval)
            daemon_interruptible_sleep(interval - elapsed);
    }

    /* flush all ports */
    for (int i = 0; i < nctx; i++) {
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/perf_%s.csv",
                 daemon_output_dir(), ctxs[i].state.device_name);
        daemon_flush_to_file(&ctxs[i].state, fname);
        daemon_destroy(&ctxs[i].state);
    }

    printf("[daemon_ib] stopped, wrote %d port(s)\n", nctx);
    return 0;
}