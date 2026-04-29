/*
 * perfmpi.c  —  LD_PRELOAD wrapper that profiles MPI_Send,
 *                mpi_compress, and mpi_decompress.
 *
 * The heavy lifting (TLS state, init, auto-flush at exit) lives
 * in perf_helper_mpi — this file ONLY wraps the target functions.
 *
 * Build
 * -----
 *   mpicc -O2 -fPIC -shared -o libperfmpi.so                         \
 *         -I/path/to/run_codes/wrappers/include                           \
 *         -I/path/to/run_codes/wrappers                                   \
 *         perfmpi.c                                                       \
 *         /path/to/run_codes/wrappers/src/perf_helper.c                   \
 *         /path/to/run_codes/wrappers/src/mpi/perf_helper_mpi.c           \
 *         -ldl
 *
 * Run
 * ---
 *   LD_PRELOAD=./libperfmpi.so mpirun -np 4 ./your_app
 *
 * Output
 * ------
 *   perf_mpi_<rank>.txt   — one per MPI process, auto-flushed at exit.
 *
 * Extending
 * ---------
 *   Follow the same pattern: declare the function with the original
 *   signature, dlsym(RTLD_NEXT) for the real impl, time + notedown.
 *   No MPI header needed for the function being intercepted.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include "mpi/perf_helper_mpi.h"

/* ── helpers ────────────────────────────────────────────────────── */
static perf_state_t *get_state(void)
{
    return perf_mpi_get_tls();
}

static void *get_real(const char *name)
{
    void *p = dlsym(RTLD_NEXT, name);
    if (!p)
        fprintf(stderr, "[perfmpi] dlsym(RTLD_NEXT, %s) failed: %s\n",
                name, dlerror());
    return p;
}

/* ── MPI_Send ───────────────────────────────────────────────────── */
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm)
{
    static int (*real)(const void *, int, MPI_Datatype, int, int, MPI_Comm) = NULL;
    if (!real) real = get_real("MPI_Send");
    if (!real) return MPI_ERR_INTERN;

    double t0 = perf_get_time();
    int ret = real(buf, count, datatype, dest, tag, comm);
    perf_notedown(get_state(), "MPI_Send", t0, perf_get_time() - t0);
    return ret;
}

/* ── mpi_compress ───────────────────────────────────────────────── */
int mpi_compress(void *input, size_t input_size,
                 void *output, size_t *output_size)
{
    static int (*real)(void *, size_t, void *, size_t *) = NULL;
    if (!real) real = get_real("mpi_compress");
    if (!real) return -1;

    double t0 = perf_get_time();
    int ret = real(input, input_size, output, output_size);
    perf_notedown(get_state(), "mpi_compress", t0, perf_get_time() - t0);
    return ret;
}

/* ── mpi_decompress ─────────────────────────────────────────────── */
int mpi_decompress(void *input, size_t input_size,
                   void *output, size_t output_size)
{
    static int (*real)(void *, size_t, void *, size_t) = NULL;
    if (!real) real = get_real("mpi_decompress");
    if (!real) return -1;

    double t0 = perf_get_time();
    int ret = real(input, input_size, output, output_size);
    perf_notedown(get_state(), "mpi_decompress", t0, perf_get_time() - t0);
    return ret;
}