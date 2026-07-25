/**
 * dummy_collectives.c — CPU-based collective simulation for reference generation.
 */
#include "dummy_collectives.h"
#include "validation.h"  /* get_element_size() */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <strings.h>
#include <dirent.h>
#include <errno.h>

/* ── Helper: load data from folder (each simulated rank reads its own file) ── */
static void dummy_load_from_folder(const char *folder,
                                    void *buf, size_t bytes,
                                    int sim_rank)
{
    const char *suffix_env = getenv("DS_SUFFIX");
    const char *format_env = getenv("DS_FORMAT");
    char path[1024];
    char default_suffix[16] = "";

    const char *suffix = suffix_env;
    if (!suffix || suffix[0] == '\0') {
        if (format_env && strcasecmp(format_env, "text") == 0)
            snprintf(default_suffix, sizeof(default_suffix), ".txt");
        else
            snprintf(default_suffix, sizeof(default_suffix), ".bin");
        suffix = default_suffix;
    }

    snprintf(path, sizeof(path), "%s/rank_%d%s", folder, sim_rank, suffix);
    FILE *fp = fopen(path, "rb");

    if (!fp) {
        DIR *dir = opendir(folder);
        if (!dir) {
            fprintf(stderr, "[dummy] cannot open folder '%s': %s\n",
                    folder, strerror(errno));
            exit(1);
        }
        struct dirent *entry;
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "rank_%d", sim_rank);
        int prefix_len = strlen(prefix);
        char found[1024] = "";
        while ((entry = readdir(dir)) != NULL) {
            const char *name = entry->d_name;
            if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
                continue;
            if (strncmp(name, prefix, prefix_len) != 0)
                continue;
            char after = name[prefix_len];
            if (after != '.' && after != '\0')
                continue;
            snprintf(found, sizeof(found), "%s/%s", folder, name);
            if (suffix[0] && strcmp(name + prefix_len, suffix) != 0)
                fprintf(stderr, "[dummy] warning — expected '%s' but found '%s'\n",
                        path, found);
            break;
        }
        closedir(dir);

        if (found[0] == '\0') {
            fprintf(stderr, "[dummy] no file matching 'rank_%d*' in '%s'\n",
                    sim_rank, folder);
            exit(1);
        }
        fp = fopen(found, "rb");
        if (!fp) {
            fprintf(stderr, "[dummy] failed to open '%s': %s\n",
                    found, strerror(errno));
            exit(1);
        }
    }

    size_t got = fread(buf, 1, bytes, fp);
    fclose(fp);

    if (got < bytes)
        memset((char *)buf + got, 0, bytes - got);
}

/* ── Load one simulated rank's input ────────────────────────────── */

void dummy_load_rank_data(const char *input_file,
                          data_type_t datatype,
                          pattern_type_t pattern_type,
                          void *buf, size_t bytes,
                          int sim_rank, int nranks)
{
    (void)nranks;

    if (input_file) {
        const char *ds_type = getenv("DS_TYPE");
        if (ds_type && strcmp(ds_type, "folder") == 0) {
            dummy_load_from_folder(input_file, buf, bytes, sim_rank);
            return;
        }

        FILE *fp = fopen(input_file, "rb");
        if (!fp) {
            fprintf(stderr, "[dummy] Error: cannot open '%s'\n", input_file);
            exit(1);
        }

        fseek(fp, 0, SEEK_END);
        size_t fsize = (size_t)ftell(fp);

        size_t chunk = bytes;
        size_t off = (size_t)sim_rank * chunk;

        /* Apply offset env vars (same logic as nccl_load_input) */
        const char *custom = getenv("DS_CUSTOM_OFFSETS");
        if (custom && custom[0]) {
            /* Parse comma-separated offsets — simplest approach */
            char *copy = strdup(custom);
            int r = 0;
            char *tok = strtok(copy, ",");
            while (tok && r <= sim_rank) {
                if (r == sim_rank) {
                    long v = atol(tok);
                    if (v >= 0) off = (size_t)v;
                    break;
                }
                r++;
                tok = strtok(NULL, ",");
            }
            free(copy);
        } else {
            const char *base_env = getenv("DS_BASE_OFFSET");
            if (base_env) off = (size_t)atol(base_env);
            const char *per_rank = getenv("DS_PER_RANK_OFFSET");
            if (!per_rank || strcmp(per_rank, "true") == 0)
                off += (size_t)sim_rank * chunk;
        }

        if (fsize > 0) {
            off %= fsize;
            if (off + chunk > fsize)
                chunk = fsize - off;
        }

        fseek(fp, (long)off, SEEK_SET);
        size_t got = fread(buf, 1, chunk, fp);
        fclose(fp);

        if (got < bytes)
            memset((char *)buf + got, 0, bytes - got);
    } else {
        size_t esz = get_element_size(datatype);
        int count = (bytes > 0) ? (int)(bytes / esz) : 1;
        if (count < 1) count = 1;
        init_buffer_pattern(buf, count, datatype, pattern_type, sim_rank);
    }
}

/* ── Internal helper: element-wise add src into dst ─────────────── */

static void accum_add(void *dst, const void *src, int count, data_type_t dtype)
{
    switch (dtype) {
        case TYPE_INT: {
            int *d = (int *)dst;
            const int *s = (const int *)src;
            for (int i = 0; i < count; i++) d[i] += s[i];
            break;
        }
        case TYPE_FLOAT: {
            float *d = (float *)dst;
            const float *s = (const float *)src;
            for (int i = 0; i < count; i++) d[i] += s[i];
            break;
        }
        case TYPE_DOUBLE: {
            double *d = (double *)dst;
            const double *s = (const double *)src;
            for (int i = 0; i < count; i++) d[i] += s[i];
            break;
        }
        case TYPE_CHAR: {
            char *d = (char *)dst;
            const char *s = (const char *)src;
            for (int i = 0; i < count; i++) d[i] += s[i];
            break;
        }
    }
}

/* ── dummy_allreduce ────────────────────────────────────────────── */

void dummy_allreduce(void *ref,
                     const char *input_file,
                     data_type_t datatype,
                     pattern_type_t pattern_type,
                     int count, int nranks)
{
    size_t esz = get_element_size(datatype);
    size_t bytes = (size_t)count * esz;

    /* Initialize ref to zero */
    memset(ref, 0, bytes);

    /* Temporary buffer for one rank's data */
    void *rank_buf = malloc(bytes);
    if (!rank_buf) { fprintf(stderr, "[dummy_allreduce] malloc failed\n"); exit(1); }

    for (int r = 0; r < nranks; r++) {
        dummy_load_rank_data(input_file, datatype, pattern_type,
                             rank_buf, bytes, r, nranks);
        accum_add(ref, rank_buf, count, datatype);
    }

    free(rank_buf);
}

/* ── dummy_alltoall ─────────────────────────────────────────────── */

void dummy_alltoall(void *ref,
                    const char *input_file,
                    data_type_t datatype,
                    pattern_type_t pattern_type,
                    int count_per, int nranks, int rank)
{
    size_t esz = get_element_size(datatype);
    size_t chunk_bytes = (size_t)count_per * esz;
    size_t per_rank_bytes = chunk_bytes * (size_t)nranks;

    void *rank_buf = malloc(per_rank_bytes);
    if (!rank_buf) { fprintf(stderr, "[dummy_alltoall] malloc failed\n"); exit(1); }

    for (int s = 0; s < nranks; s++) {
        /* Load source s's full send buffer */
        dummy_load_rank_data(input_file, datatype, pattern_type,
                             rank_buf, per_rank_bytes, s, nranks);
        /* Extract the chunk meant for this rank */
        memcpy((char *)ref + (size_t)s * chunk_bytes,
               (char *)rank_buf + (size_t)rank * chunk_bytes,
               chunk_bytes);
    }

    free(rank_buf);
}

/* ── dummy_allgather ───────────────────────────────────────────── */

void dummy_allgather(void *ref,
                     const char *input_file,
                     data_type_t datatype,
                     pattern_type_t pattern_type,
                     int count, int nranks)
{
    size_t esz = get_element_size(datatype);
    size_t bytes = (size_t)count * esz;

    for (int r = 0; r < nranks; r++) {
        dummy_load_rank_data(input_file, datatype, pattern_type,
                             (char *)ref + (size_t)r * bytes,
                             bytes, r, nranks);
    }
}

/* ── dummy_reduce ──────────────────────────────────────────────── */

void dummy_reduce(void *ref,
                  const char *input_file,
                  data_type_t datatype,
                  pattern_type_t pattern_type,
                  int count, int nranks, int root)
{
    (void)root;
    /* reduce is same as allreduce, only root holds the result */
    dummy_allreduce(ref, input_file, datatype, pattern_type, count, nranks);
}

/* ── dummy_scatter ────────────────────────────────────────────── */

void dummy_scatter(void *ref,
                   const char *input_file,
                   data_type_t datatype,
                   pattern_type_t pattern_type,
                   int count, int nranks, int root, int rank)
{
    size_t esz = get_element_size(datatype);
    size_t root_bytes = (size_t)count * (size_t)nranks * esz;
    size_t chunk_bytes = (size_t)count * esz;

    void *root_buf = malloc(root_bytes);
    if (!root_buf) { fprintf(stderr, "[dummy_scatter] malloc failed\n"); exit(1); }

    /* Root's data is the only source for scatter */
    dummy_load_rank_data(input_file, datatype, pattern_type,
                         root_buf, root_bytes, root, nranks);

    memcpy(ref, (char *)root_buf + (size_t)rank * chunk_bytes, chunk_bytes);
    free(root_buf);
}

/* ── dummy_gather ──────────────────────────────────────────────── */

void dummy_gather(void *ref,
                  const char *input_file,
                  data_type_t datatype,
                  pattern_type_t pattern_type,
                  int count, int nranks, int root)
{
    (void)root;
    /* gather is same as allgather — root gets concatenation */
    dummy_allgather(ref, input_file, datatype, pattern_type, count, nranks);
}

/* ── dummy_reduce_scatter ──────────────────────────────────────── */

void dummy_reduce_scatter(void *ref,
                          const char *input_file,
                          data_type_t datatype,
                          pattern_type_t pattern_type,
                          int count, int nranks, int rank)
{
    size_t esz = get_element_size(datatype);
    size_t full_count = (size_t)count * (size_t)nranks;
    size_t full_bytes = full_count * esz;
    size_t chunk_bytes = (size_t)count * esz;

    void *full_ref = malloc(full_bytes);
    if (!full_ref) { fprintf(stderr, "[dummy_reduce_scatter] malloc failed\n"); exit(1); }

    /* First compute the full allreduce result */
    dummy_allreduce(full_ref, input_file, datatype, pattern_type,
                    (int)full_count, nranks);

    /* Then scatter: each rank gets a chunk */
    memcpy(ref, (char *)full_ref + (size_t)rank * chunk_bytes, chunk_bytes);
    free(full_ref);
}

/* ── dummy_bcast ───────────────────────────────────────────────── */

void dummy_bcast(void *ref,
                 const char *input_file,
                 data_type_t datatype,
                 pattern_type_t pattern_type,
                 int count, int nranks, int root)
{
    size_t esz = get_element_size(datatype);
    size_t bytes = (size_t)count * esz;

    /* Load root's data and broadcast to all */
    dummy_load_rank_data(input_file, datatype, pattern_type,
                         ref, bytes, root, nranks);
}
