/*
 * perfmpi.c  —  Canonical CCBench LD_PRELOAD MPI profiling wrapper.
 *
 * Intercepts MPI point-to-point (Send/Recv/Isend/Irecv) and MPI
 * collective operations (Allreduce, Allgather) to record timing and
 * data-size metadata via perf_helper.  When benchmark_type is
 * "app_gendata", buffer contents are also dumped for dataset
 * construction — see perf_gendata_dump_if_target() below.
 *
 * The perf_gendata_dump_if_target() call is a no-op when no gendata
 * targets are configured, so it enables both app_trace and app_gendata
 * from a single wrapper.
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
 *   perf_function_<rank>.csv   — one per MPI process, auto-flushed at exit.
 *
 * Gendata (app_gendata mode)
 *   Dumps to {bin_path}/{func}_{occurrence}/rank_{rank}.bin.
 *   Configured via bench_basic_config.jsonc → gendata_capture.
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
    perf_gendata_dump_if_target(get_state(), "MPI_Send",
                                buf, perf_mpi_msg_size(count, datatype));
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
    perf_gendata_dump_if_target(get_state(), "MPI_Recv",
                                buf, perf_mpi_msg_size(count, datatype));
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
    perf_gendata_dump_if_target(get_state(), "MPI_Isend",
                                buf, perf_mpi_msg_size(count, datatype));
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
    perf_gendata_dump_if_target(get_state(), "MPI_Irecv",
                                buf, perf_mpi_msg_size(count, datatype));
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════
 * MPI collectives — commonly used by torch.distributed (DDP)
 * ═══════════════════════════════════════════════════════════════════ */

/* ── MPI_Allreduce ──────────────────────────────────────────────── */
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    static int (*real)(const void *, void *, int,
                       MPI_Datatype, MPI_Op, MPI_Comm) = NULL;
    if (!real) real = get_real("MPI_Allreduce");
    if (!real) return MPI_ERR_INTERN;

    double t0 = perf_get_time();
    int ret = real(sendbuf, recvbuf, count, datatype, op, comm);
    double t1 = perf_get_time();

    int dtype_size = 0;
    PMPI_Type_size(datatype, &dtype_size);

    perf_notedown(get_state(), "MPI_Allreduce", t0, 2,
        (perf_attr_t[]){
            {"duration",   t1 - t0},
            {"data_bytes", (double)count * dtype_size},
        });
    perf_gendata_dump_if_target(get_state(), "MPI_Allreduce",
                                recvbuf, (size_t)count * dtype_size);
    return ret;
}

/* ── MPI_Allgather ──────────────────────────────────────────────── */
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm)
{
    static int (*real)(const void *, int, MPI_Datatype,
                       void *, int, MPI_Datatype, MPI_Comm) = NULL;
    if (!real) real = get_real("MPI_Allgather");
    if (!real) return MPI_ERR_INTERN;

    double t0 = perf_get_time();
    int ret = real(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    double t1 = perf_get_time();

    int dtype_size = 0;
    PMPI_Type_size(sendtype, &dtype_size);

    perf_notedown(get_state(), "MPI_Allgather", t0, 2,
        (perf_attr_t[]){
            {"duration",   t1 - t0},
            {"data_bytes", (double)sendcount * dtype_size},
        });
    perf_gendata_dump_if_target(get_state(), "MPI_Allgather",
                                recvbuf, (size_t)sendcount * dtype_size);
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
