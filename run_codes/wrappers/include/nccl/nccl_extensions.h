#ifndef NCCL_EXTENSIONS_H
#define NCCL_EXTENSIONS_H

/*
 * nccl_extensions.h — Extended NCCL collectives not in the standard API
 *
 * All operations are asynchronous on the given CUDA stream.
 * All pointers are GPU device pointers unless noted.
 *
 * Implementations use ncclSend/ncclRecv directly on GPU pointers —
 * no cudaMemcpy, no cudaStreamSynchronize.
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stddef.h>

/* ── ncclScatter — root distributes chunks to all ranks ─────────
 * Root sends sendcount elements to each rank from its sendbuf.
 * Each rank (including root) ends up with recvcount elements in recvbuf.
 * Linear pattern: root sends sequentially to each non-root rank. */
int ncclScatter(const void *sendbuf, size_t sendcount, ncclDataType_t sendtype,
                void *recvbuf, size_t recvcount, ncclDataType_t recvtype,
                int root, ncclComm_t comm, cudaStream_t stream);

/* ── ncclGather — root collects chunks from all ranks ───────────
 * Each rank sends sendcount elements to root.
 * Root receives into recvbuf with each rank's data at rank * recvcount.
 * Linear pattern: root recvs sequentially from each non-root rank. */
int ncclGather(const void *sendbuf, size_t sendcount, ncclDataType_t sendtype,
               void *recvbuf, size_t recvcount, ncclDataType_t recvtype,
               int root, ncclComm_t comm, cudaStream_t stream);

/* ── ncclAllGatherv — all-gather with variable per-rank sizes ──
 * Each rank contributes sendcount elements.
 * recvbuf receives recvcounts[r] elements from rank r at offset displs[r].
 * recvcounts/displs are CPU host pointers; d_recvcounts/d_displs are GPU
 * copies provided for potential kernel use.
 * Sequential broadcast: each rank broadcasts its chunk in round-robin. */
int ncclAllGatherv(const void *sendbuf, int sendcount, ncclDataType_t sendtype,
                   void *recvbuf, const int recvcounts[], const int displs[],
                   ncclDataType_t recvtype,
                   const int *d_recvcounts, const int *d_displs,
                   ncclComm_t comm, cudaStream_t stream);

/* ── ncclScatterv — scatter with variable per-rank sizes ────────
 * Root sends sendcounts[i] elements from offset displs[i] to each rank i.
 * Each non-root receives recvcount elements into recvbuf.
 * Linear pattern: root sends sequentially to each non-root rank. */
int ncclScatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                 ncclDataType_t sendtype,
                 void *recvbuf, int recvcount, ncclDataType_t recvtype,
                 int root, const int *d_sendcounts, const int *d_displs,
                 ncclComm_t comm, cudaStream_t stream);

/* ── ncclAllToAll — each rank exchanges with every other rank ────
 * Each rank sends sendcount elements to every other rank.
 * sendbuf layout: [chunk for rank 0][chunk for rank 1]...
 * recvbuf layout: [chunk from rank 0][chunk from rank 1]...
 * Point-to-point based (receives posted before sends). */
int ncclAllToAll(const void *sendbuf, size_t sendcount, ncclDataType_t sendtype,
                 void *recvbuf, size_t recvcount, ncclDataType_t recvtype,
                 ncclComm_t comm, cudaStream_t stream);

#endif /* NCCL_EXTENSIONS_H */
