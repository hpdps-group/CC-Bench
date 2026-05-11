#ifndef NCCL_COMPRESS_TOOLS_H
#define NCCL_COMPRESS_TOOLS_H

#include <nccl.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

/*===========================================================================*
 * Utilities                                                                  *
 *===========================================================================*/
size_t      nccl_type_size(ncclDataType_t dtype);
static inline size_t nccl_min_sz(size_t a, size_t b) { return a < b ? a : b; }

/* Max wire bytes needed to hold [comp_size | compressed_data] for a
 * given number of uncompressed bytes.  Both sides agree on this fixed
 * size so ncclSend / ncclRecv counts match without a header exchange. */
static inline size_t nccl_max_wire_size(size_t data_bytes)
{
    return data_bytes + sizeof(uint64_t) + 64 + data_bytes / 16;
}

/*===========================================================================*
 * Compression hooks — weak symbols, override at link time                    *
 *===========================================================================*/
int nccl_compress(void *input, size_t input_size,
                  void *output, size_t *output_size);
int nccl_decompress(void *input, size_t input_size,
                    void *output, size_t output_size);
int compression_enabled(void);

/*===========================================================================*
 * Compressed NCCL point-to-point primitives                                  *
 *                                                                           *
 * All work on CPU (host) buffers.  Internally the compressed data is        *
 * uploaded to a GPU temp buffer for ncclSend / ncclRecv transport, then     *
 * downloaded back to CPU for decompression.                                  *
 *                                                                           *
 * Functions return 0 on success, non-zero on error.                         *
 *===========================================================================*/
int nccl_send_compressed(const void *buf, size_t data_bytes,
                         int peer, ncclComm_t comm, cudaStream_t stream);

int nccl_recv_decompress(void *buf, size_t data_bytes,
                         int peer, ncclComm_t comm, cudaStream_t stream);

int nccl_exchange_compressed(const void *sendbuf, size_t send_bytes,
                             void *recvbuf,  size_t recv_bytes,
                             int dst, int src, ncclComm_t comm,
                             cudaStream_t stream);

/* CPU-side reduction for implementing collectives without real NCCL reduce */
int nccl_reduce_cpu(void *dst, const void *src, size_t count,
                    ncclDataType_t dtype, ncclRedOp_t op);

#endif /* NCCL_COMPRESS_TOOLS_H */
