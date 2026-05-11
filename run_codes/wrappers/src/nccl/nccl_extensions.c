#include "nccl/nccl_extensions.h"

/*
 * ncclCommUserRank is exported from libnccl.so but not always declared
 * in the public header.  Declare it here so the wrapper compiles against
 * any NCCL build.  At runtime the symbol resolves from the LD_PRELOAD
 * library (COCCL, ZCCL, etc.) or from the real NCCL.
 */
extern ncclResult_t ncclCommUserRank(ncclComm_t comm, int *rank);

/* ── Internal: element size lookup ───────────────────────────── */
static size_t esize(ncclDataType_t dtype)
{
    switch (dtype) {
    case ncclInt8:   case ncclUint8:   return 1;
    case ncclFloat16:                  return 2;
    case ncclInt32:  case ncclUint32:
    case ncclFloat32:                  return 4;
    case ncclInt64:  case ncclUint64:
    case ncclFloat64:                  return 8;
    default:                           return 4;
    }
}

/*===========================================================================*
 * ncclScatter — root distributes to all                                     *
 *===========================================================================*/
int ncclScatter(const void *sendbuf, size_t sendcount, ncclDataType_t sendtype,
                void *recvbuf, size_t recvcount, ncclDataType_t recvtype,
                int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);

    if (nranks == 1) {
        if (rank == root && sendbuf != recvbuf && sendcount > 0)
            cudaMemcpyAsync(recvbuf, sendbuf, sendcount * esize(sendtype),
                            cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    size_t chunk = recvcount * esize(recvtype);

    if (rank == root) {
        for (int i = 0; i < nranks; i++) {
            if (i == root) continue;
            if (ncclSend((const char*)sendbuf + (size_t)i * chunk, chunk,
                         ncclUint8, i, comm, stream) != ncclSuccess)
                return 1;
        }
        /* Root's own chunk */
        if (sendbuf != recvbuf && chunk > 0)
            cudaMemcpyAsync(recvbuf, (const char*)sendbuf + (size_t)root * chunk,
                            chunk, cudaMemcpyDeviceToDevice, stream);
    } else {
        if (ncclRecv(recvbuf, chunk, ncclUint8, root, comm, stream) != ncclSuccess)
            return 1;
    }
    return 0;
}

/*===========================================================================*
 * ncclGather — root collects from all                                       *
 *===========================================================================*/
int ncclGather(const void *sendbuf, size_t sendcount, ncclDataType_t sendtype,
               void *recvbuf, size_t recvcount, ncclDataType_t recvtype,
               int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);

    if (nranks == 1) {
        if (rank == root && sendbuf != recvbuf && sendcount > 0)
            cudaMemcpyAsync(recvbuf, sendbuf, sendcount * esize(sendtype),
                            cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    size_t chunk = sendcount * esize(sendtype);

    if (rank == root) {
        for (int i = 0; i < nranks; i++) {
            if (i == root) continue;
            if (ncclRecv((char*)recvbuf + (size_t)i * chunk, chunk,
                         ncclUint8, i, comm, stream) != ncclSuccess)
                return 1;
        }
        /* Root's own data */
        if (sendbuf != recvbuf && chunk > 0)
            cudaMemcpyAsync((char*)recvbuf + (size_t)root * chunk, sendbuf,
                            chunk, cudaMemcpyDeviceToDevice, stream);
    } else {
        if (ncclSend(sendbuf, chunk, ncclUint8, root, comm, stream) != ncclSuccess)
            return 1;
    }
    return 0;
}

/*===========================================================================*
 * ncclAllGatherv — all-gather variable-size                                 *
 *===========================================================================*/
int ncclAllGatherv(const void *sendbuf, int sendcount, ncclDataType_t sendtype,
                   void *recvbuf, const int recvcounts[], const int displs[],
                   ncclDataType_t recvtype,
                   const int *d_recvcounts, const int *d_displs,
                   ncclComm_t comm, cudaStream_t stream)
{
    (void)d_recvcounts;
    (void)d_displs;

    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);

    if (nranks == 1) {
        if (sendbuf != recvbuf && sendcount > 0)
            cudaMemcpyAsync(recvbuf, sendbuf, (size_t)sendcount * esize(sendtype),
                            cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    size_t snd_sz = (size_t)sendcount * esize(sendtype);

    for (int r = 0; r < nranks; r++) {
        size_t off = (size_t)displs[r] * esize(recvtype);
        size_t rcv_sz = (size_t)recvcounts[r] * esize(recvtype);

        if (rank == r) {
            for (int p = 0; p < nranks; p++) {
                if (p == r) continue;
                if (ncclSend(sendbuf, snd_sz, ncclUint8, p, comm, stream) != ncclSuccess)
                    return 1;
            }
            if (sendbuf != (const char*)recvbuf + off && rcv_sz > 0)
                cudaMemcpyAsync((char*)recvbuf + off, sendbuf, rcv_sz,
                                cudaMemcpyDeviceToDevice, stream);
        } else {
            if (ncclRecv((char*)recvbuf + off, rcv_sz, ncclUint8,
                         r, comm, stream) != ncclSuccess)
                return 1;
        }
    }
    return 0;
}

/*===========================================================================*
 * ncclScatterv — scatter variable-size                                      *
 *===========================================================================*/
int ncclScatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                 ncclDataType_t sendtype,
                 void *recvbuf, int recvcount, ncclDataType_t recvtype,
                 int root, const int *d_sendcounts, const int *d_displs,
                 ncclComm_t comm, cudaStream_t stream)
{
    (void)d_sendcounts;
    (void)d_displs;

    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);

    if (nranks == 1) {
        if (rank == root && sendbuf != recvbuf && recvcount > 0)
            cudaMemcpyAsync(recvbuf, sendbuf, (size_t)recvcount * esize(recvtype),
                            cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    size_t esz = esize(sendtype);

    if (rank == root) {
        for (int i = 0; i < nranks; i++) {
            if (i == root) continue;
            size_t sz  = (size_t)sendcounts[i] * esz;
            size_t off = (size_t)displs[i] * esz;
            if (sz > 0 &&
                ncclSend((const char*)sendbuf + off, sz, ncclUint8,
                         i, comm, stream) != ncclSuccess)
                return 1;
        }
        size_t root_sz  = (size_t)sendcounts[root] * esz;
        size_t root_off = (size_t)displs[root] * esz;
        if (sendbuf != recvbuf && root_sz > 0)
            cudaMemcpyAsync(recvbuf, (const char*)sendbuf + root_off, root_sz,
                            cudaMemcpyDeviceToDevice, stream);
    } else {
        size_t sz = (size_t)recvcount * esize(recvtype);
        if (sz > 0 &&
            ncclRecv(recvbuf, sz, ncclUint8, root, comm, stream) != ncclSuccess)
            return 1;
    }
    return 0;
}

/*===========================================================================*
 * ncclAllToAll — each rank exchanges with every other rank                   *
 *===========================================================================*/
int ncclAllToAll(const void *sendbuf, size_t sendcount, ncclDataType_t sendtype,
                 void *recvbuf, size_t recvcount, ncclDataType_t recvtype,
                 ncclComm_t comm, cudaStream_t stream)
{
    (void)sendtype;
    (void)recvtype;

    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);

    if (nranks == 1) {
        if (sendbuf != recvbuf && sendcount > 0)
            cudaMemcpyAsync(recvbuf, sendbuf, sendcount * esize(sendtype),
                            cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    size_t chunk = sendcount * esize(sendtype);
    size_t rchunk = recvcount * esize(recvtype);

    /*
     * Batch all p2p inside ncclGroupStart/End so NCCL's proxy sees every
     * send/recv before scheduling.  Without it every rank posts all recv()
     * first, creating a circular dependency: each recv waits for a matching
     * send that another rank hasn't posted yet (it is still in its recv
     * phase on the stream).
     */
    ncclGroupStart();
    for (int i = 0; i < nranks; i++) {
        if (i == rank) continue;
        if (ncclRecv((char*)recvbuf + (size_t)i * rchunk, rchunk,
                     ncclUint8, i, comm, stream) != ncclSuccess)
            return 1;
    }
    for (int i = 0; i < nranks; i++) {
        if (i == rank) continue;
        if (ncclSend((const char*)sendbuf + (size_t)i * chunk, chunk,
                     ncclUint8, i, comm, stream) != ncclSuccess)
            return 1;
    }
    ncclGroupEnd();

    /* Own chunk */
    if (sendbuf != recvbuf && chunk > 0)
        cudaMemcpyAsync((char*)recvbuf + (size_t)rank * rchunk,
                        (const char*)sendbuf + (size_t)rank * chunk,
                        chunk, cudaMemcpyDeviceToDevice, stream);

    return 0;
}
