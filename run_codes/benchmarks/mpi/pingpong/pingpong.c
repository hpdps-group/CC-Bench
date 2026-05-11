/**
 * MPI PingPong latency benchmark.
 *
 * Rank 0 sequentially ping-pongs with partners 1..n-1 for each message size.
 * Measures one-way latency = round-trip / 2.
 * Automatically detects intra-node vs inter-node via MPI_Get_processor_name.
 *
 * Supports both -m MIN:MAX:INCR and -L X,Y,Z message size specifications.
 * CSV output via -c flag.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "mpi/mpi_utils.h"

#define TAG 0

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Error: Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    test_config_t config = parse_arguments(argc, argv);
    MPI_Datatype dtype = data_type_to_mpi(config.data_type);

    int dtype_size;
    MPI_Type_size(dtype, &dtype_size);

    /* Default CSV path */
    if (config.save_csv && config.csv_path[0] == '\0')
        snprintf(config.csv_path, sizeof(config.csv_path), "results/pingpong.csv");

    /* ── Gather processor names for intra/inter detection ──────── */
    char my_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(my_name, &name_len);

    char *all_names = malloc((size_t)size * MPI_MAX_PROCESSOR_NAME);
    MPI_Allgather(my_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  all_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

    /* ── Allocate max-size buffer ──────────────────────────────── */
    size_t max_sz = config.use_size_list
                        ? config.size_list[config.num_sizes - 1]
                        : config.max_message_size;
    void *buf = allocate_aligned_buffer(max_sz, 64);
    if (!buf) {
        if (rank == 0) fprintf(stderr, "Error: buffer allocation failed\n");
        free(all_names);
        MPI_Finalize();
        return 1;
    }
    memset(buf, 0, max_sz);

    /* ── CSV header ────────────────────────────────────────────── */
    if (rank == 0 && config.save_csv)
        mpi_csv_write(config.csv_path,
                      "msg_size,partner,comm_type,latency_us,bandwidth_gbps,"
                      "send0_us,recv0_us,recvP_us,sendP_us", NULL);

    /* ── Print banner ──────────────────────────────────────────── */
    if (rank == 0) {
        printf("=== MPI PingPong Test ===\n");
        printf("Processes: %d\n", size);
        printf("Message sizes: ");
        size_iter_t banner_it;
        size_iter_init(&banner_it, &config);
        size_t banner_sz;
        while (size_iter_next(&banner_it, &banner_sz))
            printf("%zu ", banner_sz);
        printf("\nIterations: %d (warmup: %d)\n\n",
               config.iterations, config.warmup_iterations);

        printf("Intra-node ranks: ");
        for (int i = 1; i < size; i++)
            if (strncmp(all_names + (size_t)i * MPI_MAX_PROCESSOR_NAME,
                        my_name, MPI_MAX_PROCESSOR_NAME) == 0)
                printf("%d ", i);
        printf("\nInter-node ranks: ");
        for (int i = 1; i < size; i++)
            if (strncmp(all_names + (size_t)i * MPI_MAX_PROCESSOR_NAME,
                        my_name, MPI_MAX_PROCESSOR_NAME) != 0)
                printf("%d ", i);
        printf("\n\n");

        printf("%-10s %-6s %-6s  %-14s %-12s  %-8s %-8s %-8s %-8s\n",
               "Size(B)", "Partner", "Type", "Lat(us)", "BW(GB/s)",
               "send0", "recv0", "recvP", "sendP");
    }

    /* ── Main loop: rank 0 × partner 1..n-1 ───────────────────── */
    size_iter_t sz_it;
    MPI_Status status;

    for (int partner = 1; partner < size; partner++) {
        size_iter_init(&sz_it, &config);
        size_t sz;

        while (size_iter_next(&sz_it, &sz)) {
            int count = (int)(sz / (size_t)dtype_size);
            if (count < 1) count = 1;

            /* Only rank 0 and current partner participate */
            if (rank == 0 || rank == partner) {
                /* Warmup */
                for (int w = 0; w < config.warmup_iterations; w++) {
                    if (rank == 0) {
                        MPI_Send(buf, count, dtype, partner, TAG, MPI_COMM_WORLD);
                        MPI_Recv(buf, count, dtype, partner, TAG, MPI_COMM_WORLD, &status);
                    } else {
                        MPI_Recv(buf, count, dtype, 0, TAG, MPI_COMM_WORLD, &status);
                        MPI_Send(buf, count, dtype, 0, TAG, MPI_COMM_WORLD);
                    }
                }

                /* Measured iterations with per-call timing */
                double send0_total = 0.0, recv0_total = 0.0;  /* rank 0 */
                double recvP_total = 0.0, sendP_total = 0.0;  /* partner */
                double round_total = 0.0;  /* round-trip from rank 0 */
                for (int i = 0; i < config.iterations; i++) {
                    if (rank == 0) {
                        double t0 = MPI_Wtime();
                        double t = MPI_Wtime();
                        MPI_Send(buf, count, dtype, partner, TAG, MPI_COMM_WORLD);
                        send0_total += MPI_Wtime() - t;
                        t = MPI_Wtime();
                        MPI_Recv(buf, count, dtype, partner, TAG, MPI_COMM_WORLD, &status);
                        recv0_total += MPI_Wtime() - t;
                        round_total += MPI_Wtime() - t0;
                    } else {
                        double t = MPI_Wtime();
                        MPI_Recv(buf, count, dtype, 0, TAG, MPI_COMM_WORLD, &status);
                        recvP_total += MPI_Wtime() - t;
                        t = MPI_Wtime();
                        MPI_Send(buf, count, dtype, 0, TAG, MPI_COMM_WORLD);
                        sendP_total += MPI_Wtime() - t;
                    }
                }

                /* Partner sends accumulated times to rank 0 */
                double partner_times[2] = {0.0, 0.0};
                if (rank == partner) {
                    partner_times[0] = recvP_total;
                    partner_times[1] = sendP_total;
                    MPI_Send(partner_times, 2, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
                }
                if (rank == 0) {
                    MPI_Recv(partner_times, 2, MPI_DOUBLE, partner, TAG, MPI_COMM_WORLD, &status);
                }

                /* Rank 0 reports */
                if (rank == 0) {
                    double avg_rt    = round_total / config.iterations;
                    double lat_us    = avg_rt * 1e6 / 2.0;  /* one-way */
                    double bw        = (double)sz / avg_rt; /* bytes/sec */

                    double avg_send0 = send0_total / config.iterations * 1e6;
                    double avg_recv0 = recv0_total / config.iterations * 1e6;
                    double avg_recvP = partner_times[0] / config.iterations * 1e6;
                    double avg_sendP = partner_times[1] / config.iterations * 1e6;

                    int is_intra = (strncmp(all_names + (size_t)partner * MPI_MAX_PROCESSOR_NAME,
                                            my_name, MPI_MAX_PROCESSOR_NAME) == 0);
                    const char *ct = is_intra ? "intra" : "inter";

                    printf("%-10zu %-6d %-6s  lat=%-8.2f us  bw=%-8.2f GB/s  "
                           "send0=%.2f recv0=%.2f recvP=%.2f sendP=%.2f us\n",
                           sz, partner, ct, lat_us, bw / 1e9,
                           avg_send0, avg_recv0, avg_recvP, avg_sendP);

                    if (config.save_csv)
                        mpi_csv_write(config.csv_path, NULL,
                                      "%zu,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                                      sz, partner, ct, lat_us, bw / 1e9,
                                      avg_send0, avg_recv0, avg_recvP, avg_sendP);
                }
            }
        }
    }

    /* ── Cleanup ───────────────────────────────────────────────── */
    free(all_names);
    free_aligned_buffer(buf);

    if (rank == 0)
        printf("\n=== Test Complete ===\n");

    MPI_Finalize();
    return 0;
}
