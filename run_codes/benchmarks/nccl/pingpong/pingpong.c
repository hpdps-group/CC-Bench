/**
 * NCCL PingPong latency benchmark.
 *
 * Rank 0 sequentially ping-pongs with partners 1..n-1 for each message size.
 * Measures one-way latency = round-trip / 2.
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "utils.h"
#include "validation.h"
#include "base_impl.h"
#include "nccl/nccl_base.h"
#include "nccl/nccl_utils.h"

#define NCCL_ID_FILE "/tmp/nccl_bench_pingpong_id"

/* ── NCCL bootstrap ──────────────────────────────────────────────────────── */
static ncclComm_t bootstrap_nccl(int rank, int size)
{
    ncclUniqueId id;
    if (size == 1) {
        ncclGetUniqueId(&id);
    } else if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
        FILE *f = fopen(NCCL_ID_FILE, "wb");
        fwrite(&id, sizeof(id), 1, f); fclose(f);
    }
    if (size > 1 && rank != 0) {
        struct stat st;
        int w = 0;
        while (stat(NCCL_ID_FILE, &st) != 0) { usleep(10000); w++; if (w > 3000) exit(1); }
        FILE *f = fopen(NCCL_ID_FILE, "rb");
        fread(&id, sizeof(id), 1, f); fclose(f);
    }
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    return comm;
}

int main(int argc, char **argv)
{
    int rank = 0, size = 1;
    const char *r = getenv("OMPI_COMM_WORLD_RANK"); if (r) rank = atoi(r);
    const char *s = getenv("OMPI_COMM_WORLD_SIZE"); if (s) size = atoi(s);
    if (!r) { r = getenv("PMI_RANK"); if (r) rank = atoi(r); s = getenv("PMI_SIZE"); if (s) size = atoi(s); }
    if (!r) { r = getenv("SLURM_PROCID"); if (r) rank = atoi(r); s = getenv("SLURM_NPROCS"); if (s) size = atoi(s); }

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Error: Need at least 2 processes\n");
        return 1;
    }

    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    cudaSetDevice(rank % (ndev > 0 ? ndev : 1));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ncclComm_t comm = bootstrap_nccl(rank, size);

    test_config_t config = parse_arguments(argc, argv);
    ncclDataType_t dtype = data_type_to_nccl(config.data_type);
    size_t esz = nccl_dtype_size(dtype);

    size_t max_sz = config.max_message_size;
    void *d_buf = NULL;
    cudaMalloc(&d_buf, max_sz);
    cudaMemset(d_buf, 0, max_sz);

    /* ── Banner ──────────────────────────────────────────────────── */
    if (rank == 0) {
        printf("=== NCCL PingPong Test ===\n");
        printf("Processes: %d\n", size);
        printf("Message sizes: ");
        size_iter_t banner_it;
        size_iter_init(&banner_it, &config);
        size_t banner_sz;
        while (size_iter_next(&banner_it, &banner_sz)) printf("%zu ", banner_sz);
        printf("\nIterations: %d (warmup: %d)\n\n",
               config.iterations, config.warmup_iterations);
        printf("%-10s %-6s  %-14s %-12s  %-8s %-8s %-8s %-8s\n",
               "Size(B)", "Partner", "Lat(us)", "BW(GB/s)",
               "send0", "recv0", "recvP", "sendP");
    }

    /* ── Main loop ────────────────────────────────────────────────── */
    size_iter_t sz_it;

    for (int partner = 1; partner < size; partner++) {
        size_iter_init(&sz_it, &config);
        size_t sz;

        while (size_iter_next(&sz_it, &sz)) {
            int count = (int)(sz / esz);
            if (count < 1) count = 1;

            if (rank == 0 || rank == partner) {
                /* Warmup */
                cudaStreamSynchronize(stream);
                for (int w = 0; w < config.warmup_iterations; w++) {
                    if (rank == 0) {
                        ncclSend(d_buf, count, dtype, partner, comm, stream);
                        ncclRecv(d_buf, count, dtype, partner, comm, stream);
                    } else {
                        ncclRecv(d_buf, count, dtype, 0, comm, stream);
                        ncclSend(d_buf, count, dtype, 0, comm, stream);
                    }
                    cudaStreamSynchronize(stream);
                }

                /* Measured iterations */
                double round_total = 0.0;
                double send0_total = 0.0, recv0_total = 0.0;
                double recvP_total = 0.0, sendP_total = 0.0;

                for (int i = 0; i < config.iterations; i++) {
                    cudaEvent_t ev0, ev1;
                    cudaEventCreate(&ev0);
                    cudaEventCreate(&ev1);

                    if (rank == 0) {
                        cudaEventRecord(ev0, stream);
                        ncclSend(d_buf, count, dtype, partner, comm, stream);
                        cudaEventRecord(ev1, stream);
                        cudaEventSynchronize(ev1);
                        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
                        send0_total += (double)ms / 1000.0;

                        cudaEventRecord(ev0, stream);
                        ncclRecv(d_buf, count, dtype, partner, comm, stream);
                        cudaEventRecord(ev1, stream);
                        cudaEventSynchronize(ev1);
                        cudaEventElapsedTime(&ms, ev0, ev1);
                        recv0_total += (double)ms / 1000.0;

                        round_total += send0_total + recv0_total;
                    } else {
                        cudaEventRecord(ev0, stream);
                        ncclRecv(d_buf, count, dtype, 0, comm, stream);
                        cudaEventRecord(ev1, stream);
                        cudaEventSynchronize(ev1);
                        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
                        recvP_total += (double)ms / 1000.0;

                        cudaEventRecord(ev0, stream);
                        ncclSend(d_buf, count, dtype, 0, comm, stream);
                        cudaEventRecord(ev1, stream);
                        cudaEventSynchronize(ev1);
                        cudaEventElapsedTime(&ms, ev0, ev1);
                        sendP_total += (double)ms / 1000.0;
                    }

                    cudaEventDestroy(ev0);
                    cudaEventDestroy(ev1);
                }

                /* Send partner times to rank 0 via GPU mem + ncclSend/Recv */
                double partner_times[2] = {recvP_total, sendP_total};
                double host_times[2] = {0, 0};

                if (rank == partner) {
                    void *d_pt;
                    cudaMalloc(&d_pt, 2 * sizeof(double));
                    cudaMemcpy(d_pt, partner_times, 2 * sizeof(double), cudaMemcpyHostToDevice);
                    ncclSend(d_pt, 2, ncclFloat64, 0, comm, stream);
                    cudaStreamSynchronize(stream);
                    cudaFree(d_pt);
                }
                if (rank == 0) {
                    void *d_pt;
                    cudaMalloc(&d_pt, 2 * sizeof(double));
                    ncclRecv(d_pt, 2, ncclFloat64, partner, comm, stream);
                    cudaStreamSynchronize(stream);
                    cudaMemcpy(host_times, d_pt, 2 * sizeof(double), cudaMemcpyDeviceToHost);
                    cudaFree(d_pt);
                }

                /* Rank 0 reports */
                if (rank == 0) {
                    double avg_rt    = round_total / config.iterations;
                    double lat_us    = avg_rt * 1e6 / 2.0;
                    double bw        = (double)sz / avg_rt;

                    double avg_send0 = send0_total / config.iterations * 1e6;
                    double avg_recv0 = recv0_total / config.iterations * 1e6;
                    double avg_recvP = host_times[0] / config.iterations * 1e6;
                    double avg_sendP = host_times[1] / config.iterations * 1e6;

                    printf("%-10zu %-6d  lat=%-8.2f us  bw=%-8.2f GB/s  "
                           "send0=%.2f recv0=%.2f recvP=%.2f sendP=%.2f us\n",
                           sz, partner, lat_us, bw / 1e9,
                           avg_send0, avg_recv0, avg_recvP, avg_sendP);
                }
            }
        }
    }

    cudaFree(d_buf);
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
    if (rank == 0) remove(NCCL_ID_FILE);

    if (rank == 0)
        printf("\n=== Test Complete ===\n");

    return 0;
}
