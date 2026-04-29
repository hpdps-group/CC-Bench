#define _GNU_SOURCE
#include "daemon_helper.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ==================================================================
 * Signal handling (global — each process runs one daemon)
 * ================================================================*/
static volatile sig_atomic_t g_stop_flag = 0;

static void handle_signal(int sig)
{
    (void)sig;
    g_stop_flag = 1;
}

void daemon_setup_signal_handler(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

int daemon_should_stop(void)
{
    return g_stop_flag;
}

/* ==================================================================
 * Time
 * ================================================================*/
double daemon_get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int daemon_interruptible_sleep(double seconds)
{
    struct timespec req, rem;
    req.tv_sec  = (time_t)seconds;
    req.tv_nsec = (long)((seconds - (double)req.tv_sec) * 1e9);

    while (nanosleep(&req, &rem) == -1 && errno == EINTR) {
        if (daemon_should_stop()) return 1;
        req = rem;
    }
    return 0;
}

void daemon_get_hostname(char *buf, size_t size)
{
    if (gethostname(buf, size) != 0)
        snprintf(buf, size, "unknown");
    buf[size - 1] = '\0';
}

const char *daemon_output_dir(void)
{
    static char buf[256];
    static int done = 0;
    if (!done) {
        const char *env = getenv("PERF_OUTPUT_DIR");
        if (env && env[0])
            snprintf(buf, sizeof(buf), "%s", env);
        else
            snprintf(buf, sizeof(buf), "perf_files");
        done = 1;
    }
    return buf;
}

void daemon_ensure_output_dir(void)
{
    mkdir(daemon_output_dir(), 0755);
}

/* ==================================================================
 * File-based signal
 * ================================================================*/
static char g_signal_path[256] = "";

void daemon_set_signal_file(const char *path)
{
    if (path)
        strncpy(g_signal_path, path, sizeof(g_signal_path) - 1);
    else
        g_signal_path[0] = '\0';
}

int daemon_check_signal_file(void)
{
    if (g_signal_path[0] == '\0') return 0;

    FILE *f = fopen(g_signal_path, "r");
    if (!f) return 0;       /* absent = normal */

    char buf[16] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);

    /* Trim trailing whitespace */
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r' || buf[n-1] == ' '))
        buf[--n] = '\0';

    if (n == 0) return 0;                       /* empty = normal */
    if (strcmp(buf, "EXIT") == 0)     return 2;
    if (strcmp(buf, "FLUSH_EXIT") == 0) return 2;
    if (strcmp(buf, "PAUSE") == 0)    return 1;
    return 0;                                    /* unknown → normal */
}

/* ==================================================================
 * Sysfs helpers
 * ================================================================*/
int daemon_read_sysfs_u64(const char *path, unsigned long long *val)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    int r = fscanf(fp, "%llu", val);
    fclose(fp);
    return (r == 1) ? 0 : -1;
}

int daemon_read_sysfs_str(const char *path, char *buf, size_t buf_size)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    if (!fgets(buf, (int)buf_size, fp)) {
        fclose(fp);
        return -1;
    }
    /* strip trailing newline */
    size_t n = strlen(buf);
    if (n > 0 && buf[n - 1] == '\n') buf[n - 1] = '\0';
    fclose(fp);
    return 0;
}

/* ==================================================================
 * Core API
 * ================================================================*/
void daemon_init(daemon_state_t *s, const char *device_name)
{
    memset(s, 0, sizeof(*s));
    strncpy(s->device_name, device_name, DAEMON_NAME_MAX - 1);
}

int daemon_add_metric(daemon_state_t *s, const char *name, const char *unit)
{
    if (s->num_metrics >= DAEMON_MAX_METRICS) return -1;
    daemon_metric_def_t *m = &s->metrics[s->num_metrics];
    strncpy(m->name, name, DAEMON_NAME_MAX - 1);
    strncpy(m->unit, unit, DAEMON_UNIT_MAX - 1);
    s->num_metrics++;
    return 0;
}

daemon_sample_t *daemon_new_sample(daemon_state_t *s)
{
    if (s->num_samples >= s->capacity) {
        int new_cap = s->capacity == 0
                      ? DAEMON_INIT_SAMPLES
                      : s->capacity * 2;
        daemon_sample_t *tmp = realloc(s->samples,
                                (size_t)new_cap * sizeof(daemon_sample_t));
        if (!tmp) {
            fprintf(stderr, "[daemon] realloc failed\n");
            return NULL;
        }
        s->samples  = tmp;
        s->capacity = new_cap;
    }

    daemon_sample_t *sp = &s->samples[s->num_samples];
    sp->timestamp = daemon_get_time();
    sp->count     = s->num_metrics;
    memset(sp->values, 0, sizeof(sp->values[0]) * (size_t)s->num_metrics);
    s->num_samples++;
    return sp;
}

int daemon_flush_to_file(daemon_state_t *s, const char *filepath)
{
    if (!s || !filepath || s->num_samples == 0) return -1;

    /* Open with file lock */
    int fd = open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("[daemon] open");
        return -1;
    }

    flock(fd, LOCK_EX);

    FILE *fp = fdopen(fd, "w");
    if (!fp) {
        perror("[daemon] fdopen");
        flock(fd, LOCK_UN);
        close(fd);
        return -1;
    }

    /* CSV header row: timestamp_sec,metric1,metric2,... */
    fprintf(fp, "timestamp_sec");
    for (int i = 0; i < s->num_metrics; i++)
        fprintf(fp, ",%s", s->metrics[i].name);
    fprintf(fp, "\n");

    /* Data rows */
    for (int i = 0; i < s->num_samples; i++) {
        daemon_sample_t *sp = &s->samples[i];
        fprintf(fp, "%.6f", sp->timestamp);
        for (int j = 0; j < sp->count; j++)
            fprintf(fp, ",%.6f", sp->values[j]);
        fprintf(fp, "\n");
    }

    fclose(fp);   /* also releases flock and closes fd */
    return 0;
}

void daemon_destroy(daemon_state_t *s)
{
    if (s) {
        free(s->samples);
        memset(s, 0, sizeof(*s));
    }
}