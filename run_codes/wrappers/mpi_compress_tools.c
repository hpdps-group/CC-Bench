#include "mpi_compress_tools.h"
#include <stdlib.h>
#include <string.h>

/*
 * Weak default: compression is always enabled.
 * plain_mpi_compress overrides this with an env-var-aware version.
 */
int __attribute__((weak)) compression_enabled(void) { return 1; }

/*===========================================================================*
 * Pending-send management                                                   *
 *===========================================================================*/
#define MAX_PENDING 256

static struct {
    void       *buf;
    MPI_Request req;
} pending_bufs[MAX_PENDING];

static int npending = 0;

void flush_pending(void)
{
    if (npending == 0) return;

    MPI_Request *reqs = malloc((size_t)npending * sizeof(MPI_Request));
    if (!reqs) { npending = 0; return; }

    for (int i = 0; i < npending; i++)
        reqs[i] = pending_bufs[i].req;

    MPI_Waitall(npending, reqs, MPI_STATUSES_IGNORE);

    for (int i = 0; i < npending; i++)
        free(pending_bufs[i].buf);

    free(reqs);
    npending = 0;
}

int add_pending_send(void *buf, MPI_Request req)
{
    if (npending >= MAX_PENDING) return -1;
    pending_bufs[npending].buf = buf;
    pending_bufs[npending].req = req;
    npending++;
    return 0;
}

void ensure_pending_capacity(int needed)
{
    if (npending + needed > MAX_PENDING)
        flush_pending();
}

/*===========================================================================*
 * Compressed point-to-point primitives                                      *
 *===========================================================================*/

/* Pack raw data into wire format: [int32_t size][data] */
static void *pack_raw(const void *data, size_t data_size, int *out_len)
{
    int sz = (int)data_size;
    int total = (int)sizeof(int) + sz;
    void *msg = malloc((size_t)total);
    if (!msg) { *out_len = 0; return NULL; }
    memcpy(msg, &sz, sizeof(int));
    memcpy((char*)msg + sizeof(int), data, (size_t)sz);
    *out_len = total;
    return msg;
}

/* Compress data then pack into wire format */
static void *pack_compressed(const void *data, size_t data_size,
                             int *out_len)
{
    size_t comp_cap = data_size + 64 + data_size / 16;
    void *comp = malloc(comp_cap);
    if (!comp) { *out_len = 0; return NULL; }

    size_t comp_size = comp_cap;
    int ret = mpi_compress((void*)data, data_size, comp, &comp_size);
    if (ret != 0) {
        free(comp);
        *out_len = 0;
        return NULL;
    }

    int sz = (int)comp_size;
    int total = (int)sizeof(int) + sz;
    void *msg = malloc((size_t)total);
    if (!msg) { free(comp); *out_len = 0; return NULL; }
    memcpy(msg, &sz, sizeof(int));
    memcpy((char*)msg + sizeof(int), comp, (size_t)sz);
    free(comp);
    *out_len = total;
    return msg;
}

int send_compressed(const void *buf, size_t data_size,
                    int dest, int tag, MPI_Comm comm)
{
    if (data_size == 0) return MPI_SUCCESS;
    ensure_pending_capacity(1);

    void *msg;
    int msg_len;

    if (!compression_enabled()) {
        msg = pack_raw(buf, data_size, &msg_len);
    } else {
        msg = pack_compressed(buf, data_size, &msg_len);
    }

    if (!msg) return MPI_ERR_NO_MEM;

    MPI_Request req;
    int rc = MPI_Isend(msg, msg_len, MPI_BYTE, dest, tag, comm, &req);
    if (rc != MPI_SUCCESS) {
        free(msg);
        return rc;
    }
    if (add_pending_send(msg, req) != 0) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        free(msg);
        return MPI_ERR_NO_MEM;
    }
    return MPI_SUCCESS;
}

int recv_decompress(void *buf, size_t data_size,
                    int source, int tag, MPI_Comm comm)
{
    if (data_size == 0) return MPI_SUCCESS;

    MPI_Status status;
    int msg_len;

    MPI_Probe(source, tag, comm, &status);
    MPI_Get_count(&status, MPI_BYTE, &msg_len);

    if (msg_len < (int)sizeof(int)) return MPI_ERR_TRUNCATE;

    void *msg = malloc((size_t)msg_len);
    if (!msg) return MPI_ERR_NO_MEM;

    MPI_Recv(msg, msg_len, MPI_BYTE, status.MPI_SOURCE, tag, comm,
             MPI_STATUS_IGNORE);

    int comp_size;
    memcpy(&comp_size, msg, sizeof(int));

    if (!compression_enabled()) {
        memcpy(buf, (char*)msg + sizeof(int), data_size);
    } else {
        mpi_decompress((char*)msg + sizeof(int), (size_t)comp_size,
                       buf, data_size);
    }

    free(msg);
    return MPI_SUCCESS;
}

int exchange_compressed(const void *sendbuf, size_t send_dsize,
                        void *recvbuf,  size_t recv_dsize,
                        int dst, int src, int tag, MPI_Comm comm)
{
    size_t comp_cap = send_dsize + 64 + send_dsize / 16;
    void *comp = NULL;
    int my_size = 0;

    if (send_dsize > 0) {
        if (!compression_enabled()) {
            my_size = (int)send_dsize;
            comp = malloc(send_dsize ? send_dsize : 1);
            if (!comp) return MPI_ERR_NO_MEM;
            memcpy(comp, sendbuf, send_dsize);
        } else {
            comp = malloc(comp_cap);
            if (!comp) return MPI_ERR_NO_MEM;
            size_t out = comp_cap;
            int ret = mpi_compress((void*)sendbuf, send_dsize, comp, &out);
            if (ret != 0) { free(comp); return MPI_ERR_INTERN; }
            my_size = (int)out;
        }
    }

    int peer_size = 0;

    MPI_Sendrecv(&my_size,  1, MPI_INT, dst, tag,
                 &peer_size, 1, MPI_INT, src, tag,
                 comm, MPI_STATUS_IGNORE);

    int rc = MPI_SUCCESS;
    void *peer_comp = NULL;
    if (peer_size > 0) {
        peer_comp = malloc((size_t)peer_size);
        if (!peer_comp) { rc = MPI_ERR_NO_MEM; goto out; }
    }

    int my_send = (my_size > 0) ? my_size : 1;
    int peer_recv = (peer_size > 0) ? peer_size : 1;
    void *my_sendbuf = comp ? comp : (void*)sendbuf;

    MPI_Sendrecv(my_sendbuf, my_send, MPI_BYTE, dst, tag + 1,
                 peer_comp,  peer_recv, MPI_BYTE, src, tag + 1,
                 comm, MPI_STATUS_IGNORE);

    if (peer_size > 0 && recv_dsize > 0) {
        if (!compression_enabled()) {
            memcpy(recvbuf, peer_comp, min_sz((size_t)peer_size, recv_dsize));
        } else {
            mpi_decompress(peer_comp, (size_t)peer_size, recvbuf, recv_dsize);
        }
    }

out:
    free(comp);
    free(peer_comp);
    return rc;
}
