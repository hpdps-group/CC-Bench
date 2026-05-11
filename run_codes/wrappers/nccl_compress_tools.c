#include "nccl_compress_tools.h"
#include <stdlib.h>
#include <string.h>

/*===========================================================================*
 * Size of one NCCL data element in bytes                                    *
 *===========================================================================*/
size_t nccl_type_size(ncclDataType_t dtype)
{
    switch (dtype) {
    case ncclInt8:    return 1;
    case ncclUint8:   return 1;
    case ncclFloat16: return 2;
    case ncclInt32:   return 4;
    case ncclUint32:  return 4;
    case ncclFloat32: return 4;
    case ncclInt64:   return 8;
    case ncclUint64:  return 8;
    case ncclFloat64: return 8;
    default:          return 4;
    }
}

/*===========================================================================*
 * Weak default: compression always enabled                                   *
 *===========================================================================*/
int __attribute__((weak)) compression_enabled(void) { return 1; }

/*===========================================================================*
 * Weak default: nccl_compress — no-op pass-through                           *
 *===========================================================================*/
int __attribute__((weak))
nccl_compress(void *input, size_t input_size,
              void *output, size_t *output_size)
{
    if (output != input)
        memcpy(output, input, input_size);
    *output_size = input_size;
    return 0;
}

/*===========================================================================*
 * Weak default: nccl_decompress — no-op pass-through                        *
 *===========================================================================*/
int __attribute__((weak))
nccl_decompress(void *input, size_t input_size,
                void *output, size_t output_size)
{
    (void)input_size;
    if (output != input)
        memcpy(output, input, output_size);
    return 0;
}

/*===========================================================================*
 * Internal: pack [compressed_size | compressed_data], pad to wire_size       *
 *===========================================================================*/
static void *pack_compressed(const void *data, size_t data_bytes,
                             size_t *out_wire_size)
{
    size_t wire_sz = nccl_max_wire_size(data_bytes);

    void *comp = malloc(wire_sz);
    if (!comp) return NULL;
    size_t comp_size = wire_sz;
    if (compression_enabled()) {
        if (nccl_compress((void*)data, data_bytes, comp, &comp_size) != 0) {
            free(comp);
            return NULL;
        }
    } else {
        memcpy(comp, data, data_bytes);
        comp_size = data_bytes;
    }

    void *wire = calloc(1, wire_sz);
    if (!wire) { free(comp); return NULL; }

    uint64_t hdr = (uint64_t)comp_size;
    memcpy(wire, &hdr, sizeof(hdr));
    memcpy((char*)wire + sizeof(hdr), comp, comp_size);
    free(comp);

    *out_wire_size = wire_sz;
    return wire;
}

/*===========================================================================*
 * Internal: unpack received wire buffer, decompress into output              *
 *===========================================================================*/
static int unpack_decompress(const void *wire, size_t wire_sz,
                             void *output, size_t output_bytes)
{
    if (wire_sz < sizeof(uint64_t)) return 1;

    uint64_t comp_size;
    memcpy(&comp_size, wire, sizeof(comp_size));
    if (comp_size > wire_sz - sizeof(uint64_t)) return 1;

    if (compression_enabled()) {
        nccl_decompress((const char*)wire + sizeof(uint64_t),
                        (size_t)comp_size, output, output_bytes);
    } else {
        size_t cp = nccl_min_sz(output_bytes, (size_t)comp_size);
        memcpy(output, (const char*)wire + sizeof(uint64_t), cp);
    }
    return 0;
}

/*===========================================================================*
 * nccl_send_compressed — compress CPU data, send via ncclSend                *
 *===========================================================================*/
int nccl_send_compressed(const void *buf, size_t data_bytes,
                          int peer, ncclComm_t comm, cudaStream_t stream)
{
    if (data_bytes == 0) return 0;

    /* Pack [hdr | compressed] from CPU buf */
    size_t wire_sz;
    void *wire = pack_compressed(buf, data_bytes, &wire_sz);
    if (!wire) return 1;

    /* Upload to GPU temp */
    void *gpu_temp = NULL;
    cudaMalloc(&gpu_temp, wire_sz);
    if (!gpu_temp) { free(wire); return 1; }
    cudaMemcpyAsync(gpu_temp, wire, wire_sz, cudaMemcpyHostToDevice, stream);
    free(wire);

    /* Wait for upload to finish, then send */
    cudaStreamSynchronize(stream);
    ncclSend(gpu_temp, wire_sz, ncclUint8, peer, comm, stream);

    cudaStreamSynchronize(stream);
    cudaFree(gpu_temp);
    return 0;
}

/*===========================================================================*
 * nccl_recv_decompress — receive via ncclRecv, decompress to CPU buffer      *
 *===========================================================================*/
int nccl_recv_decompress(void *buf, size_t data_bytes,
                          int peer, ncclComm_t comm, cudaStream_t stream)
{
    if (data_bytes == 0) return 0;

    size_t wire_sz = nccl_max_wire_size(data_bytes);

    void *gpu_temp = NULL;
    cudaMalloc(&gpu_temp, wire_sz);
    if (!gpu_temp) return 1;

    ncclRecv(gpu_temp, wire_sz, ncclUint8, peer, comm, stream);

    /* Download received data to CPU */
    void *wire = malloc(wire_sz);
    if (!wire) { cudaFree(gpu_temp); return 1; }
    cudaMemcpyAsync(wire, gpu_temp, wire_sz, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(gpu_temp);

    /* Decompress directly into user's CPU buffer */
    int rc = unpack_decompress(wire, wire_sz, buf, data_bytes);
    free(wire);
    return rc;
}

/*===========================================================================*
 * nccl_exchange_compressed — bi-directional compressed exchange via NCCL     *
 *===========================================================================*/
int nccl_exchange_compressed(const void *sendbuf, size_t send_bytes,
                              void *recvbuf,  size_t recv_bytes,
                              int dst, int src, ncclComm_t comm,
                              cudaStream_t stream)
{
    void *gpu_send = NULL;
    void *gpu_recv = NULL;
    void *wire_send = NULL;
    void *wire_recv = NULL;
    int ret = 0;

    /* ---- Prepare send side: compress CPU sendbuf → pack → upload ---- */
    size_t send_wire = (send_bytes > 0) ? nccl_max_wire_size(send_bytes) : 0;
    if (send_bytes > 0) {
        wire_send = pack_compressed(sendbuf, send_bytes, &send_wire);
        if (!wire_send) { ret = 1; goto out; }
        cudaMalloc(&gpu_send, send_wire);
        if (!gpu_send) { ret = 1; goto out; }
        cudaMemcpyAsync(gpu_send, wire_send, send_wire,
                        cudaMemcpyHostToDevice, stream);
    }

    /* ---- Prepare recv side ---- */
    size_t recv_wire = (recv_bytes > 0) ? nccl_max_wire_size(recv_bytes) : 0;
    if (recv_bytes > 0) {
        cudaMalloc(&gpu_recv, recv_wire);
        if (!gpu_recv) { ret = 1; goto out; }
    }

    cudaStreamSynchronize(stream);

    /* Batch both p2p operations so NCCL's proxy sees both peer's
     * send+recv before scheduling — avoids the recv-then-send deadlock
     * that occurs when both sides post recv() first on the stream. */
    ncclGroupStart();
    if (recv_bytes > 0)
        ncclRecv(gpu_recv, recv_wire, ncclUint8, src, comm, stream);
    if (send_bytes > 0)
        ncclSend(gpu_send, send_wire, ncclUint8, dst, comm, stream);
    ncclGroupEnd();

    /* ---- Process received data ---- */
    if (recv_bytes > 0) {
        wire_recv = malloc(recv_wire);
        if (!wire_recv) { ret = 1; goto out; }
        cudaMemcpyAsync(wire_recv, gpu_recv, recv_wire,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        ret = unpack_decompress(wire_recv, recv_wire, recvbuf, recv_bytes);
    }

out:
    free(wire_send);
    free(wire_recv);
    if (gpu_send) { cudaStreamSynchronize(stream); cudaFree(gpu_send); }
    if (gpu_recv) { cudaStreamSynchronize(stream); cudaFree(gpu_recv); }
    return ret;
}

/*===========================================================================*
 * nccl_reduce_cpu — CPU-side reduction for use inside collectives            *
 *===========================================================================*/
int nccl_reduce_cpu(void *dst, const void *src, size_t count,
                     ncclDataType_t dtype, ncclRedOp_t op)
{
    if (count == 0) return 0;
    if (op == ncclAvg) return 1;

    switch (dtype) {
    case ncclFloat32: {
        float *d = (float *)dst;
        const float *s = (const float *)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: return 1;
        }
        break;
    }
    case ncclFloat64: {
        double *d = (double *)dst;
        const double *s = (const double *)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: return 1;
        }
        break;
    }
    case ncclInt32: {
        int *d = (int *)dst;
        const int *s = (const int *)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: return 1;
        }
        break;
    }
    case ncclInt64: {
        long long *d = (long long *)dst;
        const long long *s = (const long long *)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: return 1;
        }
        break;
    }
    default:
        return 1;
    }
    return 0;
}
