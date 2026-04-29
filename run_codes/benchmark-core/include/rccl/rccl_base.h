/**
 * RCCL base (reference) implementation wrappers.
 *
 * These functions call the original librccl.so (discovered by findso.c),
 * bypassing any LD_PRELOAD interposer layers.
 *
 * RCCL implements the NCCL API — the wrapper function names are
 * the same as the NCCL base wrappers (base_ncclAllReduce, etc.).
 */

#ifndef RCCL_BASE_H
#define RCCL_BASE_H

#include <rccl.h>

ncclResult_t base_ncclAllReduce(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream);

ncclResult_t base_ncclBroadcast(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                int root, ncclComm_t comm,
                                cudaStream_t stream);

ncclResult_t base_ncclReduce(const void *sendbuff, void *recvbuff,
                             size_t count, ncclDataType_t datatype,
                             ncclRedOp_t op, int root,
                             ncclComm_t comm, cudaStream_t stream);

ncclResult_t base_ncclAllGather(const void *sendbuff, void *recvbuff,
                                size_t sendcount, ncclDataType_t datatype,
                                ncclComm_t comm, cudaStream_t stream);

ncclResult_t base_ncclReduceScatter(const void *sendbuff, void *recvbuff,
                                    size_t recvcount, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm_t comm,
                                    cudaStream_t stream);

ncclResult_t base_ncclSend(const void *sendbuff, size_t count,
                           ncclDataType_t datatype, int peer,
                           ncclComm_t comm, cudaStream_t stream);

ncclResult_t base_ncclRecv(void *recvbuff, size_t count,
                           ncclDataType_t datatype, int peer,
                           ncclComm_t comm, cudaStream_t stream);

#endif /* RCCL_BASE_H */