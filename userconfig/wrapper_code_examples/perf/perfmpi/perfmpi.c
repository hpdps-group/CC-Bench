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
 *         -ldl -lm
 *
 * Run
 * ---
 *   LD_PRELOAD=./libperfmpi.so mpirun -np 4 ./your_app
 *
 * Output
 * ------
 *   perf_mpi_<rank>.csv   — one per MPI process, auto-flushed at exit.
 *
 * Extending
 * ---------
 *   Follow the same pattern: declare the function with the original
 *   signature, dlsym(RTLD_NEXT) for the real impl, time + notedown.
 *   Add whatever attrs you like — the schema auto-extends.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include "perf_helper.h"
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

    perf_mpi_init_node_map();
    double t0 = perf_get_time();
    int ret = real(buf, count, datatype, dest, tag, comm);
    double t1 = perf_get_time();

    perf_notedown(get_state(), "MPI_Send", t0, 3,
        (perf_attr_t[]){
            {"duration",  t1 - t0},
            {"msg_bytes", perf_mpi_msg_size(count, datatype)},
            {"intra",     perf_mpi_is_intra(dest) ? 1.0 : 0.0},
        });
    return ret;
}

/* ── MPI_Recv ───────────────────────────────────────────────────── */
int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status)
{
    static int (*real)(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *) = NULL;
    if (!real) real = get_real("MPI_Recv");
    if (!real) return MPI_ERR_INTERN;

    perf_mpi_init_node_map();
    double t0 = perf_get_time();
    int ret = real(buf, count, datatype, source, tag, comm, status);
    double t1 = perf_get_time();

    /* Resolve MPI_ANY_SOURCE from status */
    int actual_source = source;
    if (actual_source == MPI_ANY_SOURCE && status)
        actual_source = status->MPI_SOURCE;

    perf_notedown(get_state(), "MPI_Recv", t0, 3,
        (perf_attr_t[]){
            {"duration",  t1 - t0},
            {"msg_bytes", perf_mpi_msg_size(count, datatype)},
            {"intra",     perf_mpi_is_intra(actual_source) ? 1.0 : 0.0},
        });
    return ret;
}

/* ── MPI_Isend ──────────────────────────────────────────────────── */
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
    static int (*real)(const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *) = NULL;
    if (!real) real = get_real("MPI_Isend");
    if (!real) return MPI_ERR_INTERN;

    perf_mpi_init_node_map();
    double t0 = perf_get_time();
    int ret = real(buf, count, datatype, dest, tag, comm, request);
    double t1 = perf_get_time();

    perf_notedown(get_state(), "MPI_Isend", t0, 3,
        (perf_attr_t[]){
            {"duration",  t1 - t0},
            {"msg_bytes", perf_mpi_msg_size(count, datatype)},
            {"intra",     perf_mpi_is_intra(dest) ? 1.0 : 0.0},
        });
    return ret;
}

/* ── MPI_Irecv ──────────────────────────────────────────────────── */
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    static int (*real)(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *) = NULL;
    if (!real) real = get_real("MPI_Irecv");
    if (!real) return MPI_ERR_INTERN;

    perf_mpi_init_node_map();
    double t0 = perf_get_time();
    int ret = real(buf, count, datatype, source, tag, comm, request);
    double t1 = perf_get_time();

    int src = source;
    perf_notedown(get_state(), "MPI_Irecv", t0, 3,
        (perf_attr_t[]){
            {"duration",  t1 - t0},
            {"msg_bytes", perf_mpi_msg_size(count, datatype)},
            {"intra",     (src != MPI_ANY_SOURCE && perf_mpi_is_intra(src)) ? 1.0 : 0.0},
        });
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
    double t1 = perf_get_time();

    perf_attr_t attrs[] = {
        {"duration",          t1 - t0},
        {"input_size",        (double)input_size},
        {"output_size",       (double)(output_size ? *output_size : 0)},
        {"compression_ratio", output_size && *output_size > 0
                                ? (double)input_size / *output_size : 1.0},
    };
    perf_notedown(get_state(), "mpi_compress", t0, 4, attrs);
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
    double t1 = perf_get_time();

    perf_attr_t attrs[] = {
        {"duration",          t1 - t0},
        {"input_size",        (double)input_size},
        {"output_size",       (double)output_size},
    };
    perf_notedown(get_state(), "mpi_decompress", t0, 3, attrs);
    return ret;
}
