#ifndef MPI_COMPRESS_TOOLS_H
#define MPI_COMPRESS_TOOLS_H

#include <mpi.h>
#include <stddef.h>

/*===========================================================================*
 * Tag constants — each collective uses a distinct base tag                  *
 *===========================================================================*/
#define TAG_BCAST       0x2000
#define TAG_REDUCE      0x2100
#define TAG_ALLREDUCE   0x2200
#define TAG_GATHER      0x2300
#define TAG_SCATTER     0x2400
#define TAG_ALLGATHER   0x2500
#define TAG_ALLTOALL    0x2600
#define TAG_REDUCE_SCAT 0x2700

/*===========================================================================*
 * Utility — MPI-independent helpers                                         *
 *===========================================================================*/
static inline size_t min_sz(size_t a, size_t b) { return a < b ? a : b; }

static inline int next_pow2(int x)
{
    int v = 1;
    while (v < x) v <<= 1;
    return v;
}

/* Size of an MPI_Datatype in bytes */
static inline size_t type_size(MPI_Datatype dt)
{
    int sz;
    MPI_Type_size(dt, &sz);
    return (size_t)sz;
}

/*===========================================================================*
 * Compression hooks — override at link time                                  *
 *===========================================================================*/
int mpi_compress(void *input, size_t input_size,
                 void *output, size_t *output_size);
int mpi_decompress(void *input, size_t input_size,
                   void *output, size_t output_size);

/*===========================================================================*
 * Pending-send management — for non-blocking compressed sends               *
 *===========================================================================*/
int  add_pending_send(void *buf, MPI_Request req);
void flush_pending(void);
void ensure_pending_capacity(int needed);

/*===========================================================================*
 * Compressed point-to-point primitives                                      *
 *===========================================================================*/
int send_compressed(const void *buf, size_t data_size,
                    int dest, int tag, MPI_Comm comm);
int recv_decompress(void *buf, size_t data_size,
                    int source, int tag, MPI_Comm comm);
int exchange_compressed(const void *sendbuf, size_t send_dsize,
                        void *recvbuf,  size_t recv_dsize,
                        int dst, int src, int tag, MPI_Comm comm);

#endif /* MPI_COMPRESS_TOOLS_H */
