/*
 * perfzccl.c  —  LD_PRELOAD wrapper that profiles ZCCL compression /
 *                decompression and the MPI collectives ZCCL accelerates.
 *
 * Interception groups:
 *   1. ZCCL_float_single_thread_arg              — lossy compress (ST)
 *   2. ZCCL_float_openmp_threadblock_arg          — lossy compress (MT)
 *   3. ZCCL_float_decompress_*                    — lossy decompress (ST/MT)
 *   4. ZCCL_float_single_thread_arg_split_record  — lossy compress chunked
 *   5. ZCCL_float_homomophic_add_*                — compressed-domain reduce
 *   6. MPI_Allreduce / MPI_Bcast / MPI_Scatter / MPI_Allgather
 *      (end-to-end timing of ZCCL-accelerated collectives)
 *
 * Relies on dlsym(RTLD_NEXT) to chain to the real implementation
 * (in libZCCL.so or the zccl comm wrapper, respectively).
 *
 * Build
 * -----
 *   mpicc -O2 -fPIC -shared -o libperfzccl.so                                  \
 *         -I/path/to/run_codes/wrappers/include                                    \
 *         -I/path/to/run_codes/wrappers                                            \
 *         perfzccl.c                                                               \
 *         /path/to/run_codes/wrappers/src/perf_helper.c                            \
 *         /path/to/run_codes/wrappers/src/mpi/perf_helper_mpi.c                    \
 *         -ldl -lm
 *
 * Run
 * ---
 *   LD_PRELOAD=libperfzccl.so:libzccl_comm_wrapper.so mpirun -np 4 ./app
 *
 * Output
 * ------
 *   perf_mpi_<rank>.csv  — one per MPI process, auto-flushed at exit.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

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
        fprintf(stderr, "[perfzccl] dlsym(RTLD_NEXT, %s) failed: %s\n",
                name, dlerror());
    return p;
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 1 — ZCCL lossy compress (called inside ZCCL collectives)
 * ═══════════════════════════════════════════════════════════════════ */

void ZCCL_float_single_thread_arg(unsigned char *outputBytes,
                                   float *oriData,
                                   size_t *outSize,
                                   float absErrBound,
                                   size_t nbEle,
                                   int blockSize)
{
    static void (*real)(unsigned char *, float *, size_t *, float, size_t, int) = NULL;
    if (!real) real = get_real("ZCCL_float_single_thread_arg");
    if (!real) return;

    double t0 = perf_get_time();
    real(outputBytes, oriData, outSize, absErrBound, nbEle, blockSize);
    double t1 = perf_get_time();

    double in_bytes  = (double)nbEle * sizeof(float);
    double out_bytes = outSize ? (double)*outSize : 0;

    perf_notedown(get_state(), "ZCCL_compress", t0, 5,
        (perf_attr_t[]){
            {"duration",          t1 - t0},
            {"input_bytes",       in_bytes},
            {"output_bytes",      out_bytes},
            {"compression_ratio", in_bytes > 0 ? (double)out_bytes / (double)in_bytes : 1.0},
            {"nbEle",             (double)nbEle},
        });
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 2 — ZCCL lossy compress, multi-thread (called inside MT collectives)
 * ═══════════════════════════════════════════════════════════════════ */

void ZCCL_float_openmp_threadblock_arg(unsigned char *outputBytes,
                                        float *oriData,
                                        size_t *outSize,
                                        float absErrBound,
                                        size_t nbEle,
                                        int blockSize)
{
    static void (*real)(unsigned char *, float *, size_t *, float, size_t, int) = NULL;
    if (!real) real = get_real("ZCCL_float_openmp_threadblock_arg");
    if (!real) return;

    double t0 = perf_get_time();
    real(outputBytes, oriData, outSize, absErrBound, nbEle, blockSize);
    double t1 = perf_get_time();

    double in_bytes  = (double)nbEle * sizeof(float);
    double out_bytes = outSize ? (double)*outSize : 0;

    perf_notedown(get_state(), "ZCCL_compress_mt", t0, 5,
        (perf_attr_t[]){
            {"duration",          t1 - t0},
            {"input_bytes",       in_bytes},
            {"output_bytes",      out_bytes},
            {"compression_ratio", in_bytes > 0 ? (double)out_bytes / (double)in_bytes : 1.0},
            {"nbEle",             (double)nbEle},
        });
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 3 — ZCCL lossy decompress (single-thread + multi-thread)
 * ═══════════════════════════════════════════════════════════════════ */

void ZCCL_float_decompress_single_thread_arg(float *newData,
                                              size_t nbEle,
                                              float absErrBound,
                                              int blockSize,
                                              unsigned char *cmpBytes)
{
    static void (*real)(float *, size_t, float, int, unsigned char *) = NULL;
    if (!real) real = get_real("ZCCL_float_decompress_single_thread_arg");
    if (!real) return;

    double t0 = perf_get_time();
    real(newData, nbEle, absErrBound, blockSize, cmpBytes);
    double t1 = perf_get_time();

    perf_notedown(get_state(), "ZCCL_decompress_st", t0, 3,
        (perf_attr_t[]){
            {"duration",     t1 - t0},
            {"output_bytes", (double)nbEle * sizeof(float)},
            {"nbEle",        (double)nbEle},
        });
}

void ZCCL_float_decompress_openmp_threadblock_arg(float *newData,
                                                    size_t nbEle,
                                                    float absErrBound,
                                                    int blockSize,
                                                    unsigned char *cmpBytes)
{
    static void (*real)(float *, size_t, float, int, unsigned char *) = NULL;
    if (!real) real = get_real("ZCCL_float_decompress_openmp_threadblock_arg");
    if (!real) return;

    double t0 = perf_get_time();
    real(newData, nbEle, absErrBound, blockSize, cmpBytes);
    double t1 = perf_get_time();

    perf_notedown(get_state(), "ZCCL_decompress_mt", t0, 3,
        (perf_attr_t[]){
            {"duration",     t1 - t0},
            {"output_bytes", (double)nbEle * sizeof(float)},
            {"nbEle",        (double)nbEle},
        });
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 4 — ZCCL lossy compress, chunked (split-record variant)
 * ═══════════════════════════════════════════════════════════════════ */

void ZCCL_float_single_thread_arg_split_record(unsigned char *outputBytes,
                                                float *oriData,
                                                size_t *outSize,
                                                float absErrBound,
                                                size_t nbEle,
                                                int blockSize,
                                                unsigned char *chunk_arr,
                                                size_t chunk_iter)
{
    static void (*real)(unsigned char *, float *, size_t *, float, size_t, int,
                        unsigned char *, size_t) = NULL;
    if (!real) real = get_real("ZCCL_float_single_thread_arg_split_record");
    if (!real) return;

    double t0 = perf_get_time();
    real(outputBytes, oriData, outSize, absErrBound, nbEle, blockSize,
         chunk_arr, chunk_iter);
    double t1 = perf_get_time();

    double in_bytes  = (double)nbEle * sizeof(float);
    double out_bytes = outSize ? (double)*outSize : 0;

    perf_notedown(get_state(), "ZCCL_compress_chunk", t0, 6,
        (perf_attr_t[]){
            {"duration",          t1 - t0},
            {"input_bytes",       in_bytes},
            {"output_bytes",      out_bytes},
            {"compression_ratio", in_bytes > 0 ? (double)out_bytes / (double)in_bytes : 1.0},
            {"nbEle",             (double)nbEle},
            {"chunk_iter",        (double)chunk_iter},
        });
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 5 — ZCCL homomorphic addition (compressed-domain reduce)
 * ═══════════════════════════════════════════════════════════════════ */

void ZCCL_float_homomophic_add_single_thread(unsigned char *final_cmpBytes,
                                              size_t *final_cmpSize,
                                              size_t nbEle,
                                              float absErrBound,
                                              int blockSize,
                                              unsigned char *cmpBytes,
                                              unsigned char *cmpBytes2)
{
    static void (*real)(unsigned char *, size_t *, size_t, float, int,
                        unsigned char *, unsigned char *) = NULL;
    if (!real) real = get_real("ZCCL_float_homomophic_add_single_thread");
    if (!real) return;

    double t0 = perf_get_time();
    real(final_cmpBytes, final_cmpSize, nbEle, absErrBound, blockSize,
         cmpBytes, cmpBytes2);
    double t1 = perf_get_time();

    perf_notedown(get_state(), "ZCCL_homoadd_st", t0, 2,
        (perf_attr_t[]){
            {"duration", t1 - t0},
            {"nbEle",    (double)nbEle},
        });
}

void ZCCL_float_homomophic_add_openmp_threadblock(unsigned char *final_cmpBytes,
                                                   size_t *final_cmpSize,
                                                   size_t nbEle,
                                                   float absErrBound,
                                                   int blockSize,
                                                   unsigned char *cmpBytes,
                                                   unsigned char *cmpBytes2)
{
    static void (*real)(unsigned char *, size_t *, size_t, float, int,
                        unsigned char *, unsigned char *) = NULL;
    if (!real) real = get_real("ZCCL_float_homomophic_add_openmp_threadblock");
    if (!real) return;

    double t0 = perf_get_time();
    real(final_cmpBytes, final_cmpSize, nbEle, absErrBound, blockSize,
         cmpBytes, cmpBytes2);
    double t1 = perf_get_time();

    perf_notedown(get_state(), "ZCCL_homoadd_mt", t0, 2,
        (perf_attr_t[]){
            {"duration", t1 - t0},
            {"nbEle",    (double)nbEle},
        });
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 6 — MPI collectives accelerated by ZCCL (end-to-end timing)
 *           dlsym(RTLD_NEXT) → libzccl_comm_wrapper.so → ZCCL
 * ═══════════════════════════════════════════════════════════════════ */

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
    return ret;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
               int root, MPI_Comm comm)
{
    static int (*real)(void *, int, MPI_Datatype, int, MPI_Comm) = NULL;
    if (!real) real = get_real("MPI_Bcast");
    if (!real) return MPI_ERR_INTERN;

    double t0 = perf_get_time();
    int ret = real(buffer, count, datatype, root, comm);
    double t1 = perf_get_time();

    int dtype_size = 0;
    PMPI_Type_size(datatype, &dtype_size);

    perf_notedown(get_state(), "MPI_Bcast", t0, 2,
        (perf_attr_t[]){
            {"duration",   t1 - t0},
            {"data_bytes", (double)count * dtype_size},
        });
    return ret;
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
{
    static int (*real)(const void *, int, MPI_Datatype,
                       void *, int, MPI_Datatype, int, MPI_Comm) = NULL;
    if (!real) real = get_real("MPI_Scatter");
    if (!real) return MPI_ERR_INTERN;

    double t0 = perf_get_time();
    int ret = real(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    double t1 = perf_get_time();

    int dtype_size = 0;
    PMPI_Type_size(sendtype, &dtype_size);

    perf_notedown(get_state(), "MPI_Scatter", t0, 2,
        (perf_attr_t[]){
            {"duration",   t1 - t0},
            {"data_bytes", (double)recvcount * dtype_size},
        });
    return ret;
}

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
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 7 — MPI point-to-point (blocking + non-blocking)
 * ═══════════════════════════════════════════════════════════════════ */

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
