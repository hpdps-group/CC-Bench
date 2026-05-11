/*
 * mpi_zccl_wrapper.c — MPI collectives → ZCCL delegation
 *
 * For each MPI collective, calls ZCCL's compression-accelerated version
 * when available.  Falls through to PMPI when ZCCL has no equivalent.
 *
 * ZCCL collective API coverage:
 *   MPI_Bcast      → MPI_Bcast_ZCCL / MPI_Bcast_ZCCL_mt
 *   MPI_Scatter    → MPIR_Scatter_ZCCL / MPIR_Scatter_ZCCL_mt
 *   MPI_Allreduce  → MPI_Allreduce_ZCCL_RI2_st_oa_record / _mt_oa_record
 *   MPI_Allgather  → MPIR_Allgatherv_intra_ring_RI2_st_oa_record / _mt_oa_record
 *   MPI_Reduce     — not in ZCCL API → PMPI fallthrough
 *   MPI_Gather     — not in ZCCL API → PMPI fallthrough
 *   MPI_Alltoall   — not in ZCCL API → PMPI fallthrough
 *
 * Environment variables (all optional):
 *   ZCCL_CR        — compression ratio / abs error bound   (default: 0.001)
 *   ZCCL_TOL       — tolerance                             (default: 0.08)
 *   ZCCL_BLOCK     — block size for OpenMP threading       (default: 36)
 *   ZCCL_MODE      — 0=mt (multi-thread), 1=st (single-thread)  (default: 0)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* PMPI declarations for fallback + internal use */
extern int PMPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                       int root, MPI_Comm comm);
extern int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                        MPI_Datatype datatype, MPI_Op op, int root,
                        MPI_Comm comm);
extern int PMPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
extern int PMPI_Scatter(const void *sendbuf, int sendcount,
                         MPI_Datatype sendtype, void *recvbuf, int recvcount,
                         MPI_Datatype recvtype, int root, MPI_Comm comm);
extern int PMPI_Gather(const void *sendbuf, int sendcount,
                        MPI_Datatype sendtype, void *recvbuf, int recvcount,
                        MPI_Datatype recvtype, int root, MPI_Comm comm);
extern int PMPI_Allgather(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm);
extern int PMPI_Alltoall(const void *sendbuf, int sendcount,
                          MPI_Datatype sendtype, void *recvbuf, int recvcount,
                          MPI_Datatype recvtype, MPI_Comm comm);

/* ZCCL headers
 *
 * NOTE: ZCCL_scatter.h wrongly uses the include guard "ZCCL_BROADCAST_H",
 * which shadows ZCCL_broadcast.h when both are included via ZCCL.h.
 * We keep ZCCL.h (which gives us scatter/ring/ring_ho) and forward-declare
 * the broadcast functions ourselves.
 */
#include "ZCCL.h"

/* Broadcast forward declarations (shadowed by ZCCL_scatter.h's broken guard) */
extern int MPI_Bcast_ZCCL(void *buffer, float compressionRatio, float tolerance,
                           int blockSize, MPI_Aint count, MPI_Datatype datatype,
                           int root, MPI_Comm comm);
extern int MPI_Bcast_ZCCL_mt(void *buffer, float compressionRatio, float tolerance,
                              int blockSize, MPI_Aint count, MPI_Datatype datatype,
                              int root, MPI_Comm comm);

/*===========================================================================*
 * Configuration                                                              *
 *===========================================================================*/

static void get_config(float *cr, float *tol, int *block, int *use_mt) {
    static int init = 0;
    static float s_cr     = 0.001f;
    static float s_tol    = 0.08f;
    static int   s_block  = 36;
    static int   s_mt     = 1;  /* 1=multi-threaded, 0=single-threaded */

    if (!init) {
        const char *e;
        e = getenv("ZCCL_CR");     if (e) s_cr    = (float)atof(e);
        e = getenv("ZCCL_TOL");    if (e) s_tol   = (float)atof(e);
        e = getenv("ZCCL_BLOCK");  if (e) s_block = atoi(e);
        e = getenv("ZCCL_MODE");   if (e) s_mt    = atoi(e);
        init = 1;
    }

    *cr    = s_cr;
    *tol   = s_tol;
    *block = s_block;
    *use_mt = s_mt;
}

/*===========================================================================*
 * MPI_Init — upgrade to MPI_Init_thread for ZCCL's OpenMP                   *
 *===========================================================================*/

int MPI_Init(int *argc, char ***argv) {
    int provided;
    return PMPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
}

/*===========================================================================*
 * MPI_Bcast → MPI_Bcast_ZCCL / MPI_Bcast_ZCCL_mt                            *
 *===========================================================================*/

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
               int root, MPI_Comm comm) {
    if (datatype != MPI_FLOAT)
        return PMPI_Bcast(buffer, count, datatype, root, comm);

    float cr; float tol; int block; int mt;
    get_config(&cr, &tol, &block, &mt);

    if (mt)
        return MPI_Bcast_ZCCL_mt(buffer, cr, tol, block,
                                  (MPI_Aint)count, datatype, root, comm);
    else
        return MPI_Bcast_ZCCL(buffer, cr, tol, block,
                               (MPI_Aint)count, datatype, root, comm);
}

/*===========================================================================*
 * MPI_Scatter → MPIR_Scatter_ZCCL / MPIR_Scatter_ZCCL_mt                    *
 *===========================================================================*/

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm) {
    if (sendtype != MPI_FLOAT || recvtype != MPI_FLOAT)
        return PMPI_Scatter(sendbuf, sendcount, sendtype,
                            recvbuf, recvcount, recvtype, root, comm);

    float cr; float tol; int block; int mt;
    get_config(&cr, &tol, &block, &mt);

    if (mt)
        return MPIR_Scatter_ZCCL_mt(sendbuf, cr, tol, block,
                                     (MPI_Aint)sendcount, sendtype,
                                     recvbuf, (MPI_Aint)recvcount, recvtype,
                                     root, comm);
    else
        return MPIR_Scatter_ZCCL(sendbuf, cr, tol, block,
                                  (MPI_Aint)sendcount, sendtype,
                                  recvbuf, (MPI_Aint)recvcount, recvtype,
                                  root, comm);
}

/*===========================================================================*
 * MPI_Allreduce → MPI_Allreduce_ZCCL_RI2_*_oa_record                        *
 *===========================================================================*/

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    if (datatype != MPI_FLOAT)
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

    float cr; float tol; int block; int mt;
    get_config(&cr, &tol, &block, &mt);

    if (mt)
        return MPI_Allreduce_ZCCL_RI2_mt_oa_record(
            sendbuf, recvbuf, cr, tol, block,
            (MPI_Aint)count, datatype, op, comm);
    else
        return MPI_Allreduce_ZCCL_RI2_st_oa_record(
            sendbuf, recvbuf, cr, tol, block,
            (MPI_Aint)count, datatype, op, comm);
}

/*===========================================================================*
 * MPI_Allgather → MPIR_Allgatherv_intra_ring_RI2_*_oa_record                *
 *   (allgatherv with uniform recvcounts = recvcount)                        *
 *===========================================================================*/

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm) {
    if (sendtype != MPI_FLOAT || recvtype != MPI_FLOAT)
        return PMPI_Allgather(sendbuf, sendcount, sendtype,
                               recvbuf, recvcount, recvtype, comm);

    float cr; float tol; int block; int mt;
    get_config(&cr, &tol, &block, &mt);

    MPI_Aint n = (MPI_Aint)recvcount;
    MPI_Aint zero = 0;
    MPI_Aint *displs = NULL;

    if (mt) {
        size_t sz = (size_t)recvcount * sizeof(float), comp_cap = sz + 64 + sz / 16;
        unsigned char *out = (unsigned char *)malloc(comp_cap);
        if (!out) return MPI_ERR_NO_MEM;
        int rc = MPIR_Allgatherv_intra_ring_RI2_mt_oa_record(
            sendbuf, (MPI_Aint)sendcount, sendtype,
            recvbuf, &n, &zero, recvtype, comm,
            out, cr, tol, block);
        free(out);
        return rc;
    } else {
        return MPIR_Allgatherv_intra_ring_RI2_st_oa_record(
            sendbuf, (MPI_Aint)sendcount, sendtype,
            recvbuf, &n, &zero, recvtype, comm,
            NULL, cr, tol, block);
    }
}

/*===========================================================================*
 * Unsupported by ZCCL → PMPI fallthrough                                    *
 *===========================================================================*/

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm) {
    return PMPI_Gather(sendbuf, sendcount, sendtype,
                        recvbuf, recvcount, recvtype, root, comm);
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {
    return PMPI_Alltoall(sendbuf, sendcount, sendtype,
                          recvbuf, recvcount, recvtype, comm);
}
