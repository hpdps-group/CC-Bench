#include "perf_helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <math.h>

/* ── helpers ────────────────────────────────────────────────── */
static unsigned long hash_str(const char *s)
{
    unsigned long h = 5381;
    int c;
    while ((c = *s++))
        h = ((h << 5) + h) + (unsigned long)c;
    return h;
}

/* Find column index in func, or -1. */
static int find_col(const perf_func_t *f, const char *name)
{
    for (int i = 0; i < f->num_cols; i++)
        if (strcmp(f->col_names[i], name) == 0)
            return i;
    return -1;
}

static int find_attr(const perf_attr_t *attrs, int num_attrs, const char *key)
{
    for (int i = 0; i < num_attrs; i++)
        if (strcmp(attrs[i].key, key) == 0)
            return i;
    return -1;
}

/* ── init ────────────────────────────────────────────────────── */
void perf_init(perf_state_t *s)
{
    if (!s) return;
    memset(s, 0, sizeof(*s));
    s->base_time = perf_get_time();
    pthread_mutex_init(&s->lock, NULL);
}

/* ── notedown ────────────────────────────────────────────────── */
void perf_notedown(perf_state_t *s, const char *func_name,
                   double start, int num_attrs, const perf_attr_t *attrs)
{
    if (!s || !func_name) return;

    pthread_mutex_lock(&s->lock);

    /* ── Find or create function slot ── */
    int fidx = -1;
    for (int i = 0; i < s->num_funcs; i++) {
        if (strcmp(s->funcs[i].name, func_name) == 0) {
            fidx = i;
            break;
        }
    }
    if (fidx < 0) {
        if (s->num_funcs >= PERF_MAX_FUNCS) {
            fprintf(stderr, "[perf_helper] WARNING: too many unique functions "
                    "(max %d), dropping '%s'\n", PERF_MAX_FUNCS, func_name);
            pthread_mutex_unlock(&s->lock);
            return;
        }
        fidx = s->num_funcs;
        perf_func_t *f = &s->funcs[fidx];
        strncpy(f->name, func_name, PERF_NAME_MAX - 1);
        f->name[PERF_NAME_MAX - 1] = '\0';

        /* Column 0 is always timestamp_us */
        strncpy(f->col_names[0], "timestamp_us", PERF_NAME_MAX - 1);
        f->col_names[0][PERF_NAME_MAX - 1] = '\0';
        f->num_cols = 1;

        f->col_data = calloc(PERF_COL_MAX, sizeof(double *));
        f->count         = 0;
        f->entry_capacity = 0;
        s->num_funcs++;
    }

    perf_func_t *f = &s->funcs[fidx];

    /* ── Ensure all attrs have columns (auto-register new ones) ── */
    for (int a = 0; a < num_attrs; a++) {
        if (find_col(f, attrs[a].key) >= 0) continue;

        if (f->num_cols >= PERF_COL_MAX) {
            fprintf(stderr, "[perf_helper] WARNING: too many columns for '%s' "
                    "(max %d), dropping '%s'\n",
                    func_name, PERF_COL_MAX, attrs[a].key);
            continue;
        }

        int c = f->num_cols++;
        strncpy(f->col_names[c], attrs[a].key, PERF_NAME_MAX - 1);
        f->col_names[c][PERF_NAME_MAX - 1] = '\0';

        /* Allocate new column array and backfill existing entries with NaN */
        f->col_data[c] = malloc((size_t)f->entry_capacity * sizeof(double));
        for (int e = 0; e < f->count; e++)
            f->col_data[c][e] = NAN;
    }

    /* ── Grow entry arrays if needed ── */
    if (f->count >= f->entry_capacity) {
        int new_cap = f->entry_capacity == 0
                      ? PERF_INIT_CAP
                      : f->entry_capacity * 2;
        for (int c = 0; c < f->num_cols; c++) {
            double *tmp = realloc(f->col_data[c],
                                  (size_t)new_cap * sizeof(double));
            if (!tmp) {
                fprintf(stderr, "[perf_helper] ERROR: realloc failed for '%s'\n",
                        func_name);
                pthread_mutex_unlock(&s->lock);
                return;
            }
            f->col_data[c] = tmp;
        }
        f->entry_capacity = new_cap;
    }

    /* ── Append entry ── */
    int e = f->count;
    /* column 0: timestamp_us (auto) */
    f->col_data[0][e] = (start - s->base_time) * 1e6;

    /* remaining columns: fill from attrs, or NaN */
    for (int c = 1; c < f->num_cols; c++) {
        int a = find_attr(attrs, num_attrs, f->col_names[c]);
        if (a >= 0)
            f->col_data[c][e] = attrs[a].value;
        else
            f->col_data[c][e] = NAN;
    }

    f->count++;
    s->total_records++;

    pthread_mutex_unlock(&s->lock);
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

    pthread_mutex_lock(&s->lock);

    /* ── Build global column set (union across all functions) ── */
    char global_cols[PERF_COL_MAX][PERF_NAME_MAX];
    int num_global = 0;

    for (int i = 0; i < s->num_funcs; i++) {
        perf_func_t *f = &s->funcs[i];
        for (int c = 0; c < f->num_cols; c++) {
            int found = 0;
            for (int g = 0; g < num_global; g++) {
                if (strcmp(global_cols[g], f->col_names[c]) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                if (num_global >= PERF_COL_MAX) break;
                strncpy(global_cols[num_global], f->col_names[c],
                        PERF_NAME_MAX - 1);
                global_cols[num_global][PERF_NAME_MAX - 1] = '\0';
                num_global++;
            }
        }
    }

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        perror("[perf_helper] fopen");
        pthread_mutex_unlock(&s->lock);
        return -1;
    }

    /* ── CSV header ── */
    fprintf(fp, "func_name");
    for (int g = 0; g < num_global; g++)
        fprintf(fp, ",%s", global_cols[g]);
    fprintf(fp, "\n");

    /* ── Data rows ── */
    for (int i = 0; i < s->num_funcs; i++) {
        perf_func_t *f = &s->funcs[i];
        for (int e = 0; e < f->count; e++) {
            fprintf(fp, "%s", f->name);
            for (int g = 0; g < num_global; g++) {
                int ci = find_col(f, global_cols[g]);
                if (ci >= 0)
                    fprintf(fp, ",%.6f", f->col_data[ci][e]);
                else
                    fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
    pthread_mutex_unlock(&s->lock);
    return 0;
}

/* ── flush to rank file ──────────────────────────────────────── */
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

/* ── destroy ─────────────────────────────────────────────────── */
void perf_destroy(perf_state_t *s)
{
    if (!s) return;
    for (int i = 0; i < s->num_funcs; i++) {
        perf_func_t *f = &s->funcs[i];
        for (int c = 0; c < f->num_cols; c++)
            free(f->col_data[c]);
        free(f->col_data);
        f->col_data = NULL;
        f->num_cols = 0;
        f->count    = 0;
        f->entry_capacity = 0;
    }
    s->num_funcs     = 0;
    s->total_records = 0;
    pthread_mutex_destroy(&s->lock);
}
