#include "perf_helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* ── hash: djb2 ──────────────────────────────────────────────── */
static unsigned long hash_str(const char *s)
{
    unsigned long h = 5381;
    int c;
    while ((c = *s++))
        h = ((h << 5) + h) + (unsigned long)c;
    return h;
}

/* ── init ────────────────────────────────────────────────────── */
void perf_init(perf_state_t *s)
{
    if (!s) return;
    memset(s, 0, sizeof(*s));
    s->base_time = perf_get_time();
}

/* ── notedown ────────────────────────────────────────────────── */
void perf_notedown(perf_state_t *s, const char *func_name,
                   double start, double duration)
{
    if (!s || !func_name) return;

    /* Find or create the slot for this function name */
    int idx = -1;
    for (int i = 0; i < s->num_funcs; i++) {
        if (strcmp(s->funcs[i].name, func_name) == 0) {
            idx = i;
            break;
        }
    }

    /* Not found → create a new slot */
    if (idx < 0) {
        if (s->num_funcs >= PERF_MAX_FUNCS) {
            fprintf(stderr, "[perf_helper] WARNING: too many unique functions "
                    "(max %d), dropping '%s'\n", PERF_MAX_FUNCS, func_name);
            return;
        }
        idx = s->num_funcs;
        perf_func_t *f = &s->funcs[idx];
        strncpy(f->name, func_name, PERF_NAME_MAX - 1);
        f->name[PERF_NAME_MAX - 1] = '\0';
        f->entries = NULL;
        f->count   = 0;
        f->capacity = 0;
        s->num_funcs++;
    }

    perf_func_t *f = &s->funcs[idx];

    /* Grow if needed */
    if (f->count >= f->capacity) {
        int new_cap = f->capacity == 0
                      ? PERF_INIT_CAP
                      : f->capacity * 2;
        perf_entry_t *tmp = realloc(f->entries,
                                    (size_t)new_cap * sizeof(perf_entry_t));
        if (!tmp) {
            fprintf(stderr, "[perf_helper] ERROR: realloc failed for '%s'\n",
                    func_name);
            return;
        }
        f->entries  = tmp;
        f->capacity = new_cap;
    }

    /* Append the record (store µs values) */
    double ts = (start - s->base_time) * 1e6;
    double dur = duration * 1e6;
    f->entries[f->count].timestamp_us = ts;
    f->entries[f->count].duration_us  = dur;
    f->count++;
    s->total_records++;
}

/* ── flush to file (CSV) ─────────────────────────────────────── */

const char *perf_output_dir(void)
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

void perf_ensure_output_dir(void)
{
    mkdir(perf_output_dir(), 0755);
}

int perf_flush_to_file(perf_state_t *s, const char *filepath)
{
    if (!s || !filepath) return -1;

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        perror("[perf_helper] fopen");
        return -1;
    }

    /* CSV header */
    fprintf(fp, "timestamp_us,duration_us,func_name\n");

    /* Data rows */
    for (int i = 0; i < s->num_funcs; i++) {
        perf_func_t *f = &s->funcs[i];
        for (int j = 0; j < f->count; j++) {
            fprintf(fp, "%.3f,%.3f,%s\n",
                    f->entries[j].timestamp_us,
                    f->entries[j].duration_us,
                    f->name);
        }
    }

    fclose(fp);
    return 0;
}

/* ── flush to rank file (CSV, under perf_output_dir()) ───────── */
int perf_flush_to_rank_file(perf_state_t *s, int rank, const char *prefix)
{
    if (!s || !prefix) return -1;

    perf_ensure_output_dir();

    char path[512];
    int n = snprintf(path, sizeof(path), "%s/%s_%d.csv",
                     perf_output_dir(), prefix, rank);
    if (n < 0 || (size_t)n >= sizeof(path)) return -1;

    return perf_flush_to_file(s, path);
}

/* ── monotonic time ──────────────────────────────────────────── */
double perf_get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── destroy ──────────────────────────────────────────────────── */
void perf_destroy(perf_state_t *s)
{
    if (!s) return;
    for (int i = 0; i < s->num_funcs; i++) {
        free(s->funcs[i].entries);
        s->funcs[i].entries  = NULL;
        s->funcs[i].count    = 0;
        s->funcs[i].capacity = 0;
    }
    s->num_funcs     = 0;
    s->total_records = 0;
}