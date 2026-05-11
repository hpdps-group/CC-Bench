/**
 * binary_output.c — implementation of binary output helpers.
 *
 * Uses POSIX open/pwrite so it works without MPI (for NCCL benchmarks).
 * All ranks open the same file concurrently; on parallel file systems
 * (Lustre, NFS) pwrite to disjoint offsets is safe.
 */

#include "binary_output.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── internal: build full path from base + suffix ───────────────── */
static int build_path(char *dst, size_t dst_len,
                      const char *base, const char *suffix)
{
    return snprintf(dst, dst_len, "%s_%s.bin", base, suffix);
}

/* ── single-writer ──────────────────────────────────────────────── */
void write_binary_single(const char *base_path, const char *suffix,
                         const void *buf, size_t bytes,
                         int rank, int writer_rank)
{
    if (!base_path || !suffix || !buf) return;
    if (bytes == 0) return;

    char path[1024];
    build_path(path, sizeof(path), base_path, suffix);

    if (rank == writer_rank) {
        int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) {
            perror("[binary_output] open");
            return;
        }
        size_t written = (size_t)write(fd, buf, bytes);
        if (written != bytes)
            fprintf(stderr, "[binary_output] short write: %zu/%zu\n",
                    written, bytes);
        close(fd);
    }
}

/* ── multi-writer ──────────────────────────────────────────────── */
void write_binary_multi(const char *base_path, const char *suffix,
                        const void *buf, size_t chunk_size,
                        int rank, int nranks)
{
    if (!base_path || !suffix || !buf) return;
    if (chunk_size == 0 || nranks == 0) return;

    char path[1024];
    build_path(path, sizeof(path), base_path, suffix);

    /* Rank 0 creates / truncates the file */
    if (rank == 0) {
        int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) {
            perror("[binary_output] open");
            return;
        }
        /* Reserve space so parallel pwrite doesn't race on metadata */
        off_t total = (off_t)nranks * (off_t)chunk_size;
        if (ftruncate(fd, total) < 0)
            perror("[binary_output] ftruncate");
        close(fd);
    }

    /* All ranks open and pwrite their chunk */
    int fd = open(path, O_WRONLY, 0644);
    if (fd < 0) {
        perror("[binary_output] open");
        return;
    }

    off_t offset = (off_t)rank * (off_t)chunk_size;
    size_t written = (size_t)pwrite(fd, buf, chunk_size, offset);
    if (written != chunk_size)
        fprintf(stderr, "[binary_output] rank %d short pwrite: %zu/%zu\n",
                rank, written, chunk_size);
    close(fd);
}

/* ── accumulation helpers ────────────────────────────────── */

void binary_accumulate(void *accum, const void *buf, int count, data_type_t dtype)
{
    if (!accum || !buf) return;
    switch (dtype) {
        case TYPE_INT: {
            int *a = (int *)accum;
            const int *b = (const int *)buf;
            for (int i = 0; i < count; i++) a[i] += b[i];
            break;
        }
        case TYPE_FLOAT: {
            float *a = (float *)accum;
            const float *b = (const float *)buf;
            for (int i = 0; i < count; i++) a[i] += b[i];
            break;
        }
        case TYPE_DOUBLE: {
            double *a = (double *)accum;
            const double *b = (const double *)buf;
            for (int i = 0; i < count; i++) a[i] += b[i];
            break;
        }
        case TYPE_CHAR: {
            char *a = (char *)accum;
            const char *b = (const char *)buf;
            for (int i = 0; i < count; i++) a[i] += b[i];
            break;
        }
    }
}

void binary_average(void *accum, int count, data_type_t dtype, int divisor)
{
    if (!accum || divisor == 0) return;
    switch (dtype) {
        case TYPE_INT: {
            int *a = (int *)accum;
            for (int i = 0; i < count; i++) a[i] /= divisor;
            break;
        }
        case TYPE_FLOAT: {
            float *a = (float *)accum;
            float d = (float)divisor;
            for (int i = 0; i < count; i++) a[i] /= d;
            break;
        }
        case TYPE_DOUBLE: {
            double *a = (double *)accum;
            double d = (double)divisor;
            for (int i = 0; i < count; i++) a[i] /= d;
            break;
        }
        case TYPE_CHAR: {
            char *a = (char *)accum;
            for (int i = 0; i < count; i++) a[i] = (char)((int)a[i] / divisor);
            break;
        }
    }
}
