/*
 * plain_mpi_compress.c — Plain MPI collectives with compression wrappers
 *
 * Implements core MPI collectives using MPI_Send/MPI_Recv with
 * compress-before-send / decompress-after-receive at each step.
 * Designed for benchmarking pure compressor impact on MPI communication.
 *
 * Collective implementations:
 *   MPI_Bcast              — binomial tree
 *   MPI_Reduce             — binomial tree
 *   MPI_Allreduce          — reduce-scatter + allgather (ring)
 *   MPI_Gather             — linear (all → root)
 *   MPI_Scatter            — linear (root → all)
 *   MPI_Allgather          — ring
 *   MPI_Alltoall           — pairwise exchange
 *   MPI_Reduce_scatter_block — recursive halving
 *   MPI_Reduce_scatter     — wrapper around recursive-halving core
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "plain_mpi_compress.h"

/*===========================================================================*
 * Compression control — env-var bypass (MPI_COMPRESS_DISABLE)               *
 *===========================================================================*/
int compression_enabled(void)
{
    static int checked = 0;
    static int enabled = 1;
    if (!checked) {
        char *e = getenv("MPI_COMPRESS_DISABLE");
        if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y'))
            enabled = 0;
        checked = 1;
    }
    return enabled;
}

/*===========================================================================*
 * Compression hooks — weak pass-through defaults                            *
 *===========================================================================*/
int __attribute__((weak))
mpi_compress(void *input, size_t input_size,
             void *output, size_t *output_size)
{
    if (output != input)
        memcpy(output, input, input_size);
    *output_size = input_size;
    return 0;
}

int __attribute__((weak))
mpi_decompress(void *input, size_t input_size,
               void *output, size_t output_size)
{
    (void)input_size;
    if (output != input)
        memcpy(output, input, output_size);
    return 0;
}

/* ---- MPI_Bcast (binomial tree) ---- */

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    if (count == 0 || size == 1) return MPI_SUCCESS;

    size_t dsz = (size_t)count * type_size(datatype);
    int tr = rank ^ root;   /* tree-space rank: root becomes 0 */

    /* ---- receive from parent ---- */
    int mask;
    for (mask = 1; mask < size; mask <<= 1) {
        if (tr & mask) {
            recv_decompress(buffer, dsz, (tr ^ mask) ^ root,
                            TAG_BCAST, comm);
            break;
        }
    }

    /* ---- send to children (non-blocking, flush before return) ---- */
    for (mask >>= 1; mask > 0; mask >>= 1) {
        int child = (tr | mask) ^ root;
        if (child < size)
            send_compressed(buffer, dsz, child, TAG_BCAST, comm);
    }

    flush_pending();
    return MPI_SUCCESS;
}

/* ---- MPI_Reduce (binomial tree) ---- */

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    if (count == 0 || size == 1) {
        if (rank == root && sendbuf != MPI_IN_PLACE && sendbuf != recvbuf)
            memcpy(recvbuf, sendbuf, (size_t)count * type_size(datatype));
        return MPI_SUCCESS;
    }

    size_t dsz = (size_t)count * type_size(datatype);
    int tr = rank ^ root;

    /* Local accumulation buffer */
    void *buf = malloc(dsz);
    if (!buf) return MPI_ERR_NO_MEM;

    if (rank == root && sendbuf == MPI_IN_PLACE)
        memcpy(buf, recvbuf, dsz);
    else
        memcpy(buf, sendbuf, dsz);

    /* ---- receive from children and reduce ---- */
    for (int m = 1; m < size; m <<= 1) {
        if (!(tr & m)) {
            int child = (tr | m) ^ root;
            if (child < size) {
                void *tmp = malloc(dsz);
                if (!tmp) { free(buf); return MPI_ERR_NO_MEM; }
                recv_decompress(tmp, dsz, child, TAG_REDUCE, comm);
                MPI_Reduce_local(tmp, buf, count, datatype, op);
                free(tmp);
            }
        }
    }

    /* ---- send reduced result to parent ---- */
    for (int m = 1; m < size; m <<= 1) {
        if (tr & m) {
            int parent = (tr ^ m) ^ root;
            send_compressed(buf, dsz, parent, TAG_REDUCE, comm);
            break;
        }
    }

    flush_pending();

    if (rank == root)
        memcpy(recvbuf, buf, dsz);

    free(buf);
    return MPI_SUCCESS;
}

/* ---- MPI_Allreduce (ring: reduce-scatter + allgather) ---- */

static int ring_allreduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t base = (size_t)count / (size_t)size;
    int rem   = count % size;

    /* per-chunk sizes in bytes */
    size_t chunk_sz  = base * type_size(datatype);
    size_t rem_extra = type_size(datatype);
    size_t dsz_total = (size_t)count * type_size(datatype);

    /* Allocate work buffer */
    void *buf = malloc(dsz_total);
    if (!buf) return MPI_ERR_NO_MEM;

    if (sendbuf == MPI_IN_PLACE)
        memcpy(buf, recvbuf, dsz_total);
    else
        memcpy(buf, sendbuf, dsz_total);

    int tag = TAG_ALLREDUCE;

    /* ---- Phase 1: reduce-scatter ---- */
    for (int step = 0; step < size - 1; step++) {
        int send_rank = (rank - step + size) % size;
        int recv_rank = (rank - step - 1 + size) % size;

        /* offsets and sizes */
        size_t send_off = send_rank * chunk_sz + (size_t)min_sz(send_rank, rem) * rem_extra;
        size_t send_sz  = (send_rank < rem) ? chunk_sz + rem_extra : chunk_sz;
        size_t recv_off = recv_rank * chunk_sz + (size_t)min_sz(recv_rank, rem) * rem_extra;
        size_t recv_sz  = (recv_rank < rem) ? chunk_sz + rem_extra : chunk_sz;

        int dst = (rank + 1) % size;
        int src = (rank - 1 + size) % size;

        /* Compress, exchange, decompress, reduce */
        size_t comp_cap = send_sz + 64 + send_sz / 16;
        void *comp = malloc(comp_cap);
        if (!comp) { free(buf); return MPI_ERR_NO_MEM; }
        size_t comp_out = comp_cap;
        int my_sz, peer_sz;

        if (!compression_enabled()) {
            my_sz = (int)send_sz;
            memcpy(comp, (char*)buf + send_off, send_sz);
        } else {
            int ret = mpi_compress((char*)buf + send_off, send_sz,
                                   comp, &comp_out);
            if (ret) { free(comp); free(buf); return MPI_ERR_INTERN; }
            my_sz = (int)comp_out;
        }

        /* Exchange sizes */
        MPI_Sendrecv(&my_sz, 1, MPI_INT, dst, tag,
                     &peer_sz, 1, MPI_INT, src, tag,
                     comm, MPI_STATUS_IGNORE);

        /* Exchange data */
        void *peer_comp = malloc((size_t)peer_sz);
        if (!peer_comp) { free(comp); free(buf); return MPI_ERR_NO_MEM; }
        int s_sz = (my_sz > 0) ? my_sz : 1;
        int r_sz = (peer_sz > 0) ? peer_sz : 1;
        MPI_Sendrecv(comp, s_sz, MPI_BYTE, dst, tag + 1,
                     peer_comp, r_sz, MPI_BYTE, src, tag + 1,
                     comm, MPI_STATUS_IGNORE);

        if (peer_sz > 0 && recv_sz > 0) {
            void *tmp = malloc(recv_sz);
            if (!tmp) { free(comp); free(peer_comp); free(buf); return MPI_ERR_NO_MEM; }
            if (!compression_enabled()) {
                memcpy(tmp, peer_comp, min_sz((size_t)peer_sz, recv_sz));
            } else {
                mpi_decompress(peer_comp, (size_t)peer_sz, tmp, recv_sz);
            }
            MPI_Reduce_local(tmp, (char*)buf + recv_off,
                             (recv_rank < rem) ? (int)base + 1 : (int)base,
                             datatype, op);
            free(tmp);
        }

        free(comp);
        free(peer_comp);
    }

    /* ---- Phase 2: allgather ---- */
    for (int step = 0; step < size - 1; step++) {
        int send_rank = (rank - step + 1 + size) % size;
        int recv_rank = (rank - step + size) % size;

        size_t send_off = send_rank * chunk_sz + (size_t)min_sz(send_rank, rem) * rem_extra;
        size_t send_sz  = (send_rank < rem) ? chunk_sz + rem_extra : chunk_sz;
        size_t recv_off = recv_rank * chunk_sz + (size_t)min_sz(recv_rank, rem) * rem_extra;
        size_t recv_sz  = (recv_rank < rem) ? chunk_sz + rem_extra : chunk_sz;

        int dst = (rank + 1) % size;
        int src = (rank - 1 + size) % size;

        exchange_compressed((char*)buf + send_off, send_sz,
                            (char*)buf + recv_off, recv_sz,
                            dst, src, tag + 2, comm);
    }

    memcpy(recvbuf, buf, dsz_total);
    free(buf);
    return MPI_SUCCESS;
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int size;
    PMPI_Comm_size(comm, &size);

    if (count == 0) return MPI_SUCCESS;
    if (size == 1) {
        if (sendbuf != MPI_IN_PLACE && sendbuf != recvbuf)
            memcpy(recvbuf, sendbuf, (size_t)count * type_size(datatype));
        return MPI_SUCCESS;
    }

    return ring_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

/* ---- MPI_Gather (linear: all → root) ---- */

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t snd_sz = (size_t)sendcount * type_size(sendtype);
    size_t rcv_sz = (size_t)recvcount * type_size(recvtype);

    if (rank == root) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy((char*)recvbuf + root * rcv_sz, sendbuf, snd_sz);

        for (int i = 0; i < size; i++) {
            if (i == root) continue;
            recv_decompress((char*)recvbuf + i * rcv_sz, rcv_sz,
                            i, TAG_GATHER, comm);
        }
    } else {
        send_compressed(sendbuf, snd_sz, root, TAG_GATHER, comm);
    }

    flush_pending();
    return MPI_SUCCESS;
}

/* ---- MPI_Scatter (linear: root → all) ---- */

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t snd_sz = (size_t)sendcount * type_size(sendtype);
    size_t rcv_sz = (size_t)recvcount * type_size(recvtype);

    if (rank == root) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, (const char*)sendbuf + root * snd_sz, snd_sz);

        for (int i = 0; i < size; i++) {
            if (i == root) continue;
            send_compressed((const char*)sendbuf + i * snd_sz, snd_sz,
                            i, TAG_SCATTER, comm);
        }
    } else {
        recv_decompress(recvbuf, rcv_sz, root, TAG_SCATTER, comm);
    }

    flush_pending();
    return MPI_SUCCESS;
}

/* ---- MPI_Allgather (ring) ---- */

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t snd_sz = (size_t)sendcount * type_size(sendtype);
    size_t rcv_sz = (size_t)recvcount * type_size(recvtype);

    if (size == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, min_sz(snd_sz, rcv_sz));
        return MPI_SUCCESS;
    }

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char*)recvbuf + rank * rcv_sz, sendbuf, min_sz(snd_sz, rcv_sz));
    }

    int tag = TAG_ALLGATHER;

    for (int step = 0; step < size - 1; step++) {
        int send_rank = (rank - step + size) % size;
        int recv_rank = (rank - step - 1 + size) % size;
        int dst = (rank + 1) % size;
        int src = (rank - 1 + size) % size;

        exchange_compressed((char*)recvbuf + send_rank * rcv_sz, rcv_sz,
                            (char*)recvbuf + recv_rank * rcv_sz, rcv_sz,
                            dst, src, tag, comm);
    }

    return MPI_SUCCESS;
}

/* ---- MPI_Alltoall (pairwise exchange) ---- */

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t snd_sz = (size_t)sendcount * type_size(sendtype);
    size_t rcv_sz = (size_t)recvcount * type_size(recvtype);

    /* local copy */
    memcpy((char*)recvbuf + rank * rcv_sz,
           (const char*)sendbuf + rank * snd_sz,
            min_sz(snd_sz, rcv_sz));

    int tag = TAG_ALLTOALL;

    /* Fire off all sends (non-blocking) */
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        send_compressed((const char*)sendbuf + i * snd_sz, snd_sz,
                        i, tag, comm);
    }

    /* Receive from everyone (blocking) */
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        recv_decompress((char*)recvbuf + i * rcv_sz, rcv_sz,
                        i, tag, comm);
    }

    flush_pending();
    return MPI_SUCCESS;
}

/* ---- MPI_Reduce_scatter_block (recursive halving) ---- */

int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf,
                             int recvcount, MPI_Datatype datatype,
                             MPI_Op op, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    int total_count = recvcount * size;
    size_t dsz = (size_t)total_count * type_size(datatype);
    size_t chunk = (size_t)recvcount * type_size(datatype);

    if (size == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, chunk);
        return MPI_SUCCESS;
    }

    size_t np2  = (size_t)next_pow2(size);
    size_t buf_count = np2 * (size_t)recvcount;
    void *buf = malloc(buf_count * type_size(datatype));
    if (!buf) return MPI_ERR_NO_MEM;

    if (sendbuf == MPI_IN_PLACE)
        memcpy(buf, recvbuf, dsz);
    else
        memcpy(buf, sendbuf, dsz);

    if ((size_t)size < np2)
        memset((char*)buf + dsz, 0, (buf_count - (size_t)total_count) * type_size(datatype));

    int tag = TAG_REDUCE_SCAT;
    int remain = size;
    int mask = 1;

    while (mask < size) {
        int new_remain = (remain + 1) / 2;

        if (rank < remain) {
            int peer = rank ^ mask;
            if (peer < remain) {
                size_t half_sz = chunk * (size_t)(remain / 2);

                if (rank < peer) {
                    void *tmp = malloc(half_sz);
                    if (!tmp) { free(buf); return MPI_ERR_NO_MEM; }
                    exchange_compressed(
                        (char*)buf + half_sz, half_sz,
                        tmp, half_sz,
                        peer, peer, tag, comm);
                    MPI_Reduce_local(tmp, buf,
                                     (int)(half_sz / type_size(datatype)),
                                     datatype, op);
                    free(tmp);
                } else {
                    void *tmp = malloc(half_sz);
                    if (!tmp) { free(buf); return MPI_ERR_NO_MEM; }
                    exchange_compressed(
                        buf, half_sz,
                        tmp, half_sz,
                        peer, peer, tag, comm);
                    MPI_Reduce_local(tmp, (char*)buf + half_sz,
                                     (int)(half_sz / type_size(datatype)),
                                     datatype, op);
                    memmove(buf, (char*)buf + half_sz, half_sz);
                    free(tmp);
                }
            } else if (rank >= new_remain) {
                memmove((char*)buf + chunk * (size_t)(rank - new_remain),
                        (char*)buf + chunk * (size_t)rank,
                        chunk);
            }
        }

        remain = new_remain;
        mask <<= 1;
    }

    memcpy(recvbuf, buf, chunk);
    free(buf);
    return MPI_SUCCESS;
}

/* ---- MPI_Reduce_scatter ---- */

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf,
                       const int recvcounts[], MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    int first = recvcounts[0];
    int uniform = 1;
    for (int i = 1; i < size; i++) {
        if (recvcounts[i] != first) { uniform = 0; break; }
    }

    if (uniform)
        return MPI_Reduce_scatter_block(sendbuf, recvbuf, first,
                                        datatype, op, comm);

    int total_count = 0;
    for (int i = 0; i < size; i++) total_count += recvcounts[i];

    size_t dsz = (size_t)total_count * type_size(datatype);
    size_t my_chunk = (size_t)recvcounts[rank] * type_size(datatype);

    if (size == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, my_chunk);
        return MPI_SUCCESS;
    }

    void *full = NULL;
    if (rank == 0) {
        full = malloc(dsz);
        if (!full) return MPI_ERR_NO_MEM;
    }

    int tr = rank ^ 0;
    void *buf = malloc(dsz);
    if (!buf) { free(full); return MPI_ERR_NO_MEM; }
    if (sendbuf == MPI_IN_PLACE)
        memcpy(buf, recvbuf, dsz);
    else
        memcpy(buf, sendbuf, dsz);

    for (int m = 1; m < size; m <<= 1) {
        if (!(tr & m)) {
            int child = tr | m;
            if (child < size) {
                void *tmp = malloc(dsz);
                if (!tmp) { free(buf); free(full); return MPI_ERR_NO_MEM; }
                recv_decompress(tmp, dsz, child, TAG_REDUCE_SCAT, comm);
                MPI_Reduce_local(tmp, buf, total_count, datatype, op);
                free(tmp);
            }
        }
    }
    for (int m = 1; m < size; m <<= 1) {
        if (tr & m) {
            int parent = tr ^ m;
            send_compressed(buf, dsz, parent, TAG_REDUCE_SCAT, comm);
            break;
        }
    }
    flush_pending();

    if (rank == 0) memcpy(full, buf, dsz);
    free(buf);

    if (rank == 0) {
        size_t offset = 0;
        for (int i = 0; i < size; i++) {
            size_t sz = (size_t)recvcounts[i] * type_size(datatype);
            if (i == 0) {
                memcpy(recvbuf, full, sz);
            } else {
                send_compressed((const char*)full + offset, sz,
                                i, TAG_REDUCE_SCAT + 1, comm);
            }
            offset += sz;
        }
    } else {
        recv_decompress(recvbuf, my_chunk, 0, TAG_REDUCE_SCAT + 1, comm);
    }

    flush_pending();
    free(full);
    return MPI_SUCCESS;
}

/* ---- MPI_Allgatherv (sequential broadcast, variable sizes) ---- */

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int *recvcounts, const int *displs,
                   MPI_Datatype recvtype, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t snd_sz = (size_t)sendcount * type_size(sendtype);
    size_t rcv_esz = type_size(recvtype);

    if (size == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, (size_t)sendcount * rcv_esz);
        return MPI_SUCCESS;
    }

    /* Place local data */
    memcpy((char*)recvbuf + (size_t)displs[rank] * rcv_esz,
           sendbuf, snd_sz);

    /* Sequential broadcast of each rank's chunk */
    for (int root = 0; root < size; root++) {
        size_t root_sz = (size_t)recvcounts[root] * rcv_esz;
        if (root_sz == 0) continue;

        if (rank == root) {
            for (int p = 0; p < size; p++) {
                if (p == root) continue;
                send_compressed((const char*)recvbuf + (size_t)displs[root] * rcv_esz,
                                root_sz, p, TAG_ALLGATHERV, comm);
            }
        } else {
            recv_decompress((char*)recvbuf + (size_t)displs[root] * rcv_esz,
                            root_sz, root, TAG_ALLGATHERV, comm);
        }
    }

    flush_pending();
    return MPI_SUCCESS;
}

/* ---- MPI_Scatterv (linear root -> all, variable sizes) ---- */

int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t rcv_sz = (size_t)recvcount * type_size(recvtype);
    size_t snd_esz = type_size(sendtype);

    if (size == 1) {
        if (rank == root && sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, rcv_sz);
        return MPI_SUCCESS;
    }

    if (rank == root) {
        /* Root's own chunk */
        memcpy(recvbuf, (const char*)sendbuf + (size_t)displs[root] * snd_esz,
               rcv_sz);

        /* Send each non-root rank its chunk */
        for (int i = 0; i < size; i++) {
            if (i == root) continue;
            size_t peer_sz = (size_t)sendcounts[i] * snd_esz;
            if (peer_sz == 0) continue;
            send_compressed((const char*)sendbuf + (size_t)displs[i] * snd_esz,
                            peer_sz, i, TAG_SCATTERV, comm);
        }
    } else {
        recv_decompress(recvbuf, rcv_sz, root, TAG_SCATTERV, comm);
    }

    flush_pending();
    return MPI_SUCCESS;
}
