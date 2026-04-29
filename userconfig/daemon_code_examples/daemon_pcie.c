/*
 * daemon_pcie.c  —  PCIe device link-status poller.
 *
 * Scans /sys/bus/pci/devices/, reads link speed/width + available
 * hardware counters for each device at a fixed interval.
 *
 * Usage   ./daemon_pcie [interval_sec]
 *         SIGINT/SIGTERM to stop gracefully.
 *
 * Output  perf_pcie_<bdf>.txt   (one file per device, BDF address)
 *
 * Notes
 * -----
 * Standard sysfs exposes link speed/width but not data traffic
 * counters.  Metric values beyond link info will be 0 unless the
 * kernel/driver exposes additional per-device counters.
 */

#include "daemon_helper.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_PCIE 64

/* ── per-device context ──────────────────────────────────────── */
typedef struct {
    char           bdf[256];         /* bus:device.function         */
    daemon_state_t state;
} pcie_ctx_t;

static pcie_ctx_t ctxs[MAX_PCIE];
static int        nctx = 0;

/* ── discover PCIe devices ───────────────────────────────────── */
static int discover_devices(void)
{
    const char *sysfs = "/sys/bus/pci/devices";
    DIR *d = opendir(sysfs);
    if (!d) { perror(sysfs); return -1; }

    struct dirent *e;
    while ((e = readdir(d)) && nctx < MAX_PCIE) {
        if (e->d_name[0] == '.') continue;

        /* Check if this device has link attributes */
        char path[512];
        snprintf(path, sizeof(path), "%s/%s/current_link_speed",
                 sysfs, e->d_name);
        FILE *fp = fopen(path, "r");
        if (!fp) continue;   /* not a PCIe endpoint with link info */
        fclose(fp);

        pcie_ctx_t *c = &ctxs[nctx];
        snprintf(c->bdf, sizeof(c->bdf), "%s", e->d_name);

        char devname[512];
        snprintf(devname, sizeof(devname), "pcie_%s", e->d_name);
        daemon_init(&c->state, devname);

        /* fixed metrics */
        daemon_add_metric(&c->state, "link_speed", "GT/s");
        daemon_add_metric(&c->state, "link_width", "lanes");
        /* vendor/class info — changes rarely but useful */
        daemon_add_metric(&c->state, "vendor_id",  "hex");
        daemon_add_metric(&c->state, "device_id",  "hex");

        /* Try to find any unsigned counter files in the device dir.
         * We don't know the names, so we probe a common set. */
        daemon_add_metric(&c->state, "counter_1",  "");
        daemon_add_metric(&c->state, "counter_2",  "");
        daemon_add_metric(&c->state, "counter_3",  "");
        daemon_add_metric(&c->state, "counter_4",  "");
        daemon_add_metric(&c->state, "counter_5",  "");
        nctx++;
    }
    closedir(d);
    return nctx;
}

/* ── read link speed (GT/s) from sysfs string ────────────────── */
static double parse_link_speed(const char *s)
{
    /* typical: "2.5 GT/s", "8 GT/s", "16 GT/s" */
    double val = 0.0;
    sscanf(s, "%lf", &val);
    return val;
}

/* ── poll one device ─────────────────────────────────────────── */
static void poll_device(pcie_ctx_t *c)
{
    char path[256], buf[64];
    unsigned long long vendor = 0, device = 0;
    unsigned long long c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;

    /* Link speed */
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/current_link_speed", c->bdf);
    double speed = 0.0;
    if (daemon_read_sysfs_str(path, buf, sizeof(buf)) == 0)
        speed = parse_link_speed(buf);

    /* Link width */
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/current_link_width", c->bdf);
    unsigned long long width = 0;
    daemon_read_sysfs_u64(path, &width);

    /* Vendor & device IDs */
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/vendor", c->bdf);
    daemon_read_sysfs_u64(path, &vendor);
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/device", c->bdf);
    daemon_read_sysfs_u64(path, &device);

    /* Probe for counters — best-effort */
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/counters/tx_bytes", c->bdf);
    daemon_read_sysfs_u64(path, &c1);
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/counters/rx_bytes", c->bdf);
    daemon_read_sysfs_u64(path, &c2);

    /* Some drivers expose AER counters */
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/aer_dev_correctable", c->bdf);
    daemon_read_sysfs_u64(path, &c3);
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/aer_dev_nonfatal", c->bdf);
    daemon_read_sysfs_u64(path, &c4);
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/aer_dev_fatal", c->bdf);
    daemon_read_sysfs_u64(path, &c5);

    daemon_sample_t *sp = daemon_new_sample(&c->state);
    if (!sp) return;
    sp->values[0] = speed;
    sp->values[1] = (double)width;
    sp->values[2] = (double)vendor;
    sp->values[3] = (double)device;
    sp->values[4] = (double)c1;
    sp->values[5] = (double)c2;
    sp->values[6] = (double)c3;
    sp->values[7] = (double)c4;
    sp->values[8] = (double)c5;
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    double interval = 1.0;
    if (argc > 1) interval = atof(argv[1]);
    if (interval <= 0.0) interval = 1.0;

    char hostname[64];
    daemon_get_hostname(hostname, sizeof(hostname));

    if (discover_devices() <= 0) {
        fprintf(stderr, "[daemon_pcie] no PCIe devices with link info\n");
        return 1;
    }

    /* append hostname */
    for (int i = 0; i < nctx; i++) {
        char merged[DAEMON_NAME_MAX];
        snprintf(merged, sizeof(merged), "%s_%s",
                 ctxs[i].state.device_name, hostname);
        strncpy(ctxs[i].state.device_name, merged, DAEMON_NAME_MAX - 1);
    }

    printf("[daemon_pcie] monitoring %d PCIe device(s) on %s, interval %.2fs\n",
           nctx, hostname, interval);

    daemon_setup_signal_handler();
    daemon_set_signal_file("daemon_signals/daemon_pcie.signal");
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
            poll_device(&ctxs[i]);

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

    printf("[daemon_pcie] stopped\n");
    return 0;
}