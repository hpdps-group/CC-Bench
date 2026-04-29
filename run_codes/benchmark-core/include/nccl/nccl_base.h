/**
 * NCCL base (reference) implementation wrappers.
 *
 * These functions call the original libnccl.so (discovered by findso.c),
 * bypassing any LD_PRELOAD interposer layers.
 */

#ifndef NCCL_BASE_H
#define NCCL_BASE_H

#include <nccl.h>
#include <cuda_runtime.h>

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

#endif /* NCCL_BASE_H */