/**
 * MPI_Overlap — measurement of compute/communication mutual interference.
 *
 * Uses dual threads: one for MPI, one for lightweight compute.
 * Measures how much each side slows down the other.
 *
 * Processes paired as n (even, "initiator") and n+1 (odd, "partner").
 * Requires even number of processes.
 *
 * Three phases per message size:
 *   1. Pure compute   — N_WU work units, no MPI (baseline throughput)
 *   2. Pure comm      — MPI_Send / MPI_Recv (baseline latency)
 *   3. Parallel       — dual threads: comm + compute simultaneously
 *                       Thread A: MPI_Send (init) / MPI_Recv (partner) → signal stop
 *                       Thread B: work_unit loop until signaled
 *
 * Per-phase metrics:
 *   Comp(us)    — pure compute time for N_WU work units
 *   Send(us)    — pure MPI_Send time
 *   Recv(us)    — pure MPI_Recv time
 *   PSend(us)   — MPI_Send time with concurrent compute on another core
 *   PRecv(us)   — MPI_Recv time with concurrent compute on another core
 *   CSlow%      — compute throughput drop during parallel phase
 *   SSlow%      — send slowdown % (parallel send / pure send - 1)
 *   RSlow%      — recv slowdown % (parallel recv / pure recv - 1)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include "utils.h"
#include "mpi/mpi_utils.h"

/* PMPI declarations */
extern int PMPI_Init_thread(int *argc, char ***argv, int required, int *provided);
extern int PMPI_Finalize(void);
extern int PMPI_Barrier(MPI_Comm comm);
extern int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, int root,
                       MPI_Comm comm);
extern int PMPI_Comm_rank(MPI_Comm comm, int *rank);
extern int PMPI_Comm_size(MPI_Comm comm, int *size);
extern int PMPI_Type_size(MPI_Datatype datatype, int *size);

/* ── Work unit ─────────────────────────────────────────────────────────── */

#define N_WU      20000000UL  /* 20M iterations for pure compute baseline */
#define SCR_LEN   512         /* scratch buffer length (fits L1 cache) */

/* Lightweight memory-touching workload — multiply-add chain on scratch buffer.
 * Simulates real-world inner loop (e.g., compression) with L1 cache footprint. */
static inline void work_unit(int *scr, int i) {
    int idx = i % SCR_LEN;
    scr[idx] = scr[idx] * 3 + scr[(idx + 1) % SCR_LEN];
}

/* ── Parallel phase: shared state ──────────────────────────────────────── */

static pthread_barrier_t   s_barrier;
static atomic_int          s_comm_done;  /* 1 = comm thread finished */

/* Thread argument: communication */
typedef struct {
    void       *buf;
    int         count;
    MPI_Datatype dtype;
    int         partner;
    int         tag;
    int         is_sender;  /* 1 = MPI_Send, 0 = MPI_Recv */
    double     *t_comm;     /* output: comm duration */
} comm_arg_t;

/* Thread argument: compute */
typedef struct {
    unsigned long *count;    /* output: work units completed */
    int           *scratch;  /* scratch buffer */
} comp_arg_t;

static void *comm_thread_fn(void *arg) {
    comm_arg_t *a = (comm_arg_t *)arg;

    pthread_barrier_wait(&s_barrier);

    double t0 = MPI_Wtime();
    if (a->is_sender)
        MPI_Send(a->buf, a->count, a->dtype, a->partner, a->tag, MPI_COMM_WORLD);
    else
        MPI_Recv(a->buf, a->count, a->dtype, a->partner, a->tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double t1 = MPI_Wtime();

    *a->t_comm = t1 - t0;

    atomic_store(&s_comm_done, 1);
    return NULL;
}

static void *comp_thread_fn(void *arg) {
    comp_arg_t *a = (comp_arg_t *)arg;
    unsigned long cnt = 0;

    pthread_barrier_wait(&s_barrier);

    while (!atomic_load(&s_comm_done)) {
        work_unit(a->scratch, cnt);
        cnt++;
    }

    *a->count = cnt;
    return NULL;
}

/* ── Per-size test ─────────────────────────────────────────────────────── */

static void run_test_size(int rank, int size,
                           size_t msg_size, MPI_Datatype datatype,
                           int is_initiator, int partner,
                           void *sendbuf, void *recvbuf,
                           const test_config_t *config) {
    int dtype_size;
    PMPI_Type_size(datatype, &dtype_size);
    int count = (int)(msg_size / dtype_size);
    if (count <= 0) count = 1;
    int scratch[SCR_LEN] = {0};

    const int n_init = size / 2;
    const int tag = 100;

    /* Accumulators (summed across iterations) */
    double t_comp        = 0.0;
    double t_send_pure   = 0.0;
    double t_recv_pure   = 0.0;
    double t_send_para   = 0.0;
    double t_recv_para   = 0.0;
    double cnt_send_para = 0.0;  /* work units done during parallel send */
    double cnt_recv_para = 0.0;  /* work units done during parallel recv */

    for (int iter = 0; iter < config->iterations; iter++) {
        /* ═════════════════════════════════════════════════════════════════
         * Phase 1: Pure compute — N_WU work units (both ranks)
         * ═════════════════════════════════════════════════════════════════ */
        PMPI_Barrier(MPI_COMM_WORLD);
        {
            atomic_int dummy = 0;
            unsigned long cnt = 0;
            double t0 = MPI_Wtime();
            while (!atomic_load(&dummy)) {
                work_unit(scratch, cnt);
                cnt++;
                if (cnt >= N_WU) break;
            }
            double t1 = MPI_Wtime();
            t_comp += t1 - t0;
            /* prevent optimization of scratch buffer */
            volatile int sink = scratch[0];
            (void)sink;
        }

        /* ═════════════════════════════════════════════════════════════════
         * Phase 2: Pure communication — MPI_Send / MPI_Recv
         * ═════════════════════════════════════════════════════════════════ */
        PMPI_Barrier(MPI_COMM_WORLD);
        if (is_initiator) {
            double t0 = MPI_Wtime();
            MPI_Send(sendbuf, count, datatype, partner, tag, MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            t_send_pure += t1 - t0;
        } else {
            double t0 = MPI_Wtime();
            MPI_Recv(recvbuf, count, datatype, partner, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double t1 = MPI_Wtime();
            t_recv_pure += t1 - t0;
        }

        /* ═════════════════════════════════════════════════════════════════
         * Phase 3: Parallel — dual threads (comm + compute simultaneously)
         * ═════════════════════════════════════════════════════════════════ */
        PMPI_Barrier(MPI_COMM_WORLD);
        {
            double t_comm_para = 0.0;
            unsigned long comp_cnt = 0;
            pthread_t th_comm, th_comp;

            atomic_store(&s_comm_done, 0);

            /* Thread 1 does MPI_Send (initiator) or MPI_Recv (partner) */
            comm_arg_t ca = {
                .buf       = is_initiator ? sendbuf : recvbuf,
                .count     = count,
                .dtype     = datatype,
                .partner   = partner,
                .tag       = tag,
                .is_sender = is_initiator,
                .t_comm    = &t_comm_para,
            };
            /* Thread 2 does work units until signaled */
            comp_arg_t pa = {
                .count    = &comp_cnt,
                .scratch  = scratch,
            };

            pthread_create(&th_comm, NULL, comm_thread_fn, &ca);
            pthread_create(&th_comp, NULL, comp_thread_fn, &pa);

            pthread_join(th_comm, NULL);
            pthread_join(th_comp, NULL);

            if (is_initiator) {
                t_send_para   += t_comm_para;
                cnt_send_para += (double)comp_cnt;
            } else {
                t_recv_para   += t_comm_para;
                cnt_recv_para += (double)comp_cnt;
            }
            /* prevent optimization of scratch buffer */
            volatile int sink = scratch[0];
            (void)sink;
        }
    }

    /* ── Average ───────────────────────────────────────────────────────── */
    int n = config->iterations;
    t_comp    /= n;
    t_send_pure /= n;
    t_recv_pure /= n;
    t_send_para   /= n;
    t_recv_para   /= n;
    cnt_send_para /= n;
    cnt_recv_para /= n;

    /* ── Reduce to rank 0 (sum across same-role ranks) ─────────────────── */
    /* Layout: [comp, send_pure, recv_pure, send_para, recv_para,
                cnt_send_para, cnt_recv_para] */
    double local[7] = {t_comp, t_send_pure, t_recv_pure,
                       t_send_para, t_recv_para,
                       cnt_send_para, cnt_recv_para};
    double global[7] = {0};

    /* For pure compute, ALL ranks participate.
     * For send metrics, ONLY initiators contribute.
     * For recv metrics, ONLY partners contribute.
     *
     * We sum everything across all ranks with MPI_SUM.
     * On rank 0: compute / size, send metrics / n_init, recv metrics / n_init. */
    PMPI_Reduce(local, global, 7, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ── Report from rank 0 ────────────────────────────────────────────── */
    if (rank == 0) {
        double g_comp         = global[0] / size;
        double g_send_pure    = global[1] / n_init;
        double g_recv_pure    = global[2] / n_init;
        double g_send_para    = global[3] / n_init;
        double g_recv_para    = global[4] / n_init;
        double g_cnt_send     = global[5] / n_init;
        double g_cnt_recv     = global[6] / n_init;

        /* compute throughput: work units per second */
        double comp_thru      = (double)N_WU / g_comp;
        double send_thru      = g_cnt_send / g_send_para;
        double recv_thru      = g_cnt_recv / g_recv_para;

        double comp_slow_send = 0.0;
        double comp_slow_recv = 0.0;
        double send_slow      = 0.0;
        double recv_slow      = 0.0;

        if (comp_thru > 0.0) {
            comp_slow_send = (send_thru / comp_thru - 1.0) * 100.0;
            comp_slow_recv = (recv_thru / comp_thru - 1.0) * 100.0;
        }
        if (g_send_pure > 0.0)
            send_slow = (g_send_para / g_send_pure - 1.0) * 100.0;
        if (g_recv_pure > 0.0)
            recv_slow = (g_recv_para / g_recv_pure - 1.0) * 100.0;

        printf("%-10zu %-10.2f %-10.2f %-10.2f "
               "%-10.2f %-10.2f %-10.0f %-10.0f "
               "%-9.1f %-9.1f %-9.1f %-9.1f\n",
               msg_size,
               g_comp * 1e6, g_send_pure * 1e6, g_recv_pure * 1e6,
               g_send_para * 1e6, g_recv_para * 1e6,
               g_cnt_send, g_cnt_recv,
               comp_slow_send, comp_slow_recv, send_slow, recv_slow);
        fflush(stdout);
    }
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int provided;
    PMPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int rank, size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_SERIALIZED) {
        if (rank == 0)
            fprintf(stderr, "Error: MPI does not support MPI_THREAD_SERIALIZED\n");
        PMPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (size < 2 || size % 2 != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: Need even number of processes (got %d)\n", size);
        PMPI_Finalize();
        return 1;
    }

    test_config_t config = parse_arguments(argc, argv);

    int is_initiator = (rank % 2 == 0);
    int partner      = is_initiator ? rank + 1 : rank - 1;

    MPI_Datatype datatype = data_type_to_mpi(config.data_type);
    size_t max_size = config.max_message_size;

    void *sendbuf = allocate_aligned_buffer(max_size, 64);
    void *recvbuf = allocate_aligned_buffer(max_size, 64);
    if (!sendbuf || !recvbuf) {
        if (rank == 0)
            fprintf(stderr, "Error: Memory allocation failed\n");
        free_aligned_buffer(sendbuf);
        free_aligned_buffer(recvbuf);
        PMPI_Finalize();
        return 1;
    }

    /* Initialize barrier once */
    pthread_barrier_init(&s_barrier, NULL, 2);

    /* Header */
    if (rank == 0) {
        printf("=== MPI Overlap (dual-thread) ===\n");
        printf("Processes: %d\n", size);
        printf("Work units: %lu\n\n", (unsigned long)N_WU);
        printf("%-10s %-10s %-10s %-10s "
               "%-10s %-10s %-10s %-10s "
               "%-9s %-9s %-9s %-9s\n",
               "# Size", "Comp", "Send", "Recv",
               "PSend", "PRecv", "CntSnd", "CntRcv",
               "CSnd%", "CRcv%", "SSlow%", "RSlow%");
        printf("%-10s %-10s %-10s %-10s "
               "%-10s %-10s %-10s %-10s "
               "%-9s %-9s %-9s %-9s\n",
               "(bytes)", "(us)", "(us)", "(us)",
               "(us)", "(us)", "", "",
               "", "", "", "");
    }

    size_iter_t msg_iter;
    size_iter_init(&msg_iter, &config);
    size_t msg_size;
    while (size_iter_next(&msg_iter, &msg_size)) {
        /* Fill buffer with minimal data */
        memset(sendbuf, 0, msg_size);
        memset(recvbuf, 0, msg_size);

        run_test_size(rank, size, msg_size, datatype,
                      is_initiator, partner,
                      sendbuf, recvbuf, &config);
    }

    /* ── Legend ──────────────────────────────────────────────── */
    if (rank == 0) {
        printf("\n");
        printf("--- Column definitions ---\n");
        printf("Comp      Pure-compute time for %lu work units (us)\n", (unsigned long)N_WU);
        printf("Send      Pure MPI_Send time (us)\n");
        printf("Recv      Pure MPI_Recv time (us)\n");
        printf("PSend     MPI_Send time with concurrent compute (us)\n");
        printf("PRecv     MPI_Recv time with concurrent compute (us)\n");
        printf("CntSnd    Work units completed during parallel Send\n");
        printf("CntRcv    Work units completed during parallel Recv\n");
        printf("CSnd%%     Compute throughput change during Send: (CntSnd/PSend)/(N_WU/Comp)-1\n");
        printf("          Negative = compute slowed, -100%% = no compute done\n");
        printf("CRcv%%     Same as CSnd%% but during Recv\n");
        printf("SSlow%%    Send slowdown: PSend/Send-1. Positive = slower, negative = faster\n");
        printf("RSlow%%    Recv slowdown: PRecv/Recv-1\n");
        printf("\n");
        printf("Note: CntSnd/CntRcv = 0 when the message is so small that\n");
        printf("      MPI_Send/Recv finishes before the compute thread\n");
        printf("      completes even one work unit.\n");
        fflush(stdout);
    }

    pthread_barrier_destroy(&s_barrier);
    free_aligned_buffer(sendbuf);
    free_aligned_buffer(recvbuf);
    PMPI_Finalize();
    return 0;
}
