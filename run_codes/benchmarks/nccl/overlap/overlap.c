/**
 * NCCL Overlap — measurement of compute/communication mutual interference.
 *
 * Uses dual threads: one for NCCL, one for lightweight compute.
 * Measures how much each side slows down the other.
 *
 * Processes paired as n (even, "initiator") and n+1 (odd, "partner").
 * Requires even number of processes.
 *
 * Three phases per message size:
 *   1. Pure compute   — N_WU work units, no NCCL (baseline throughput)
 *   2. Pure comm      — ncclSend / ncclRecv (baseline latency)
 *   3. Parallel       — dual threads: comm + compute simultaneously
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <sys/stat.h>
#include "utils.h"

/* ── Work unit ─────────────────────────────────────────────────────────── */

#define N_WU      20000000UL  /* 20M iterations for pure compute baseline */
#define SCR_LEN   512         /* scratch buffer length */

static inline void work_unit(int *scr, unsigned long i) {
    int idx = (int)(i % SCR_LEN);
    scr[idx] = scr[idx] * 3 + scr[(idx + 1) % SCR_LEN];
}

/* ── Parallel phase: shared state ──────────────────────────────────────── */

static pthread_barrier_t  s_barrier;
static atomic_int         s_comm_done;

typedef struct {
    void       *d_buf;
    size_t      count;
    ncclDataType_t dtype;
    int         partner;
    int         is_sender;
    ncclComm_t  comm;
    cudaStream_t stream;
    double     *t_comm;     /* seconds */
} comm_arg_t;

typedef struct {
    unsigned long *count;    /* work units completed */
    int           *scratch;
} comp_arg_t;

static void *comm_thread_fn(void *arg) {
    comm_arg_t *a = (comm_arg_t *)arg;

    pthread_barrier_wait(&s_barrier);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, a->stream);
    if (a->is_sender) {
        ncclResult_t _ret = ncclSend(a->d_buf, a->count, a->dtype, a->partner,
                                     a->comm, a->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[pid=%d] ncclSend FAILED at count=%zu error=%d — aborting\n",
                    getpid(), a->count, (int)_ret);
            exit(1);
        }
    } else {
        ncclResult_t _ret = ncclRecv(a->d_buf, a->count, a->dtype, a->partner,
                                     a->comm, a->stream);
        if (_ret != ncclSuccess) {
            fprintf(stderr, "[pid=%d] ncclRecv FAILED at count=%zu error=%d — aborting\n",
                    getpid(), a->count, (int)_ret);
            exit(1);
        }
    }
    cudaEventRecord(stop, a->stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    *a->t_comm = (double)ms / 1000.0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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

/* ── Helpers (no MPI dependency) ────────────────────────────────────────── */

static int get_rank(void) {
    const char *r;
    r = getenv("OMPI_COMM_WORLD_RANK"); if (r) return atoi(r);
    r = getenv("PMI_RANK");             if (r) return atoi(r);
    r = getenv("SLURM_PROCID");         if (r) return atoi(r);
    return 0;
}
static int get_size(void) {
    const char *s;
    s = getenv("OMPI_COMM_WORLD_SIZE"); if (s) return atoi(s);
    s = getenv("PMI_SIZE");             if (s) return atoi(s);
    s = getenv("SLURM_NPROCS");         if (s) return atoi(s);
    return 1;
}

static size_t nccl_dt_size(ncclDataType_t dt) {
    switch (dt) {
    case ncclInt8: case ncclUint8:   return 1;
    case ncclFloat16:                return 2;
    case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
    case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
    default: return 4;
    }
}

/* ── Per-size test ─────────────────────────────────────────────────────── */

static void run_test_size(int rank, int size,
                           size_t msg_size, ncclDataType_t dtype,
                           int is_initiator, int partner,
                           ncclComm_t comm, cudaStream_t stream,
                           void *d_buf, const test_config_t *config)
{
    size_t esz = nccl_dt_size(dtype);
    int count = (int)(msg_size / esz);
    if (count <= 0) count = 1;
    int scratch[SCR_LEN] = {0};
    const int n_init = size / 2;

    double t_comp        = 0.0;
    double t_send_pure   = 0.0;
    double t_recv_pure   = 0.0;
    double t_send_para   = 0.0;
    double t_recv_para   = 0.0;
    double cnt_send_para = 0.0;
    double cnt_recv_para = 0.0;

    for (int iter = 0; iter < config->iterations; iter++) {
        /* ══════════════════════════════════════════════════════════
         * Phase 1: Pure compute
         * ══════════════════════════════════════════════════════════ */
        cudaStreamSynchronize(stream);
        {
            atomic_int dummy = 0;
            unsigned long cnt = 0;
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            while (!atomic_load(&dummy)) {
                work_unit(scratch, cnt);
                cnt++;
                if (cnt >= N_WU) break;
            }
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            double sec = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;
            t_comp += sec;
            volatile int sink = scratch[0]; (void)sink;
        }

        /* ══════════════════════════════════════════════════════════
         * Phase 2: Pure communication
         * ══════════════════════════════════════════════════════════ */
        cudaStreamSynchronize(stream);
        {
            cudaEvent_t ev_s, ev_e;
            cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
            cudaEventRecord(ev_s, stream);
            if (is_initiator) {
                ncclResult_t _ret = ncclSend(d_buf, count, dtype, partner, comm, stream);
                if (_ret != ncclSuccess) {
                    fprintf(stderr, "[rank=%d] ncclSend FAILED at count=%d iter=%d "
                                    "error=%d — aborting\n",
                            rank, count, iter, (int)_ret);
                    exit(1);
                }
            } else {
                ncclResult_t _ret = ncclRecv(d_buf, count, dtype, partner, comm, stream);
                if (_ret != ncclSuccess) {
                    fprintf(stderr, "[rank=%d] ncclRecv FAILED at count=%d iter=%d "
                                    "error=%d — aborting\n",
                            rank, count, iter, (int)_ret);
                    exit(1);
                }
            }
            cudaEventRecord(ev_e, stream);
            cudaEventSynchronize(ev_e);
            float ms;
            cudaEventElapsedTime(&ms, ev_s, ev_e);
            if (is_initiator) t_send_pure += (double)ms / 1000.0;
            else              t_recv_pure += (double)ms / 1000.0;
            cudaEventDestroy(ev_s); cudaEventDestroy(ev_e);
        }

        /* ══════════════════════════════════════════════════════════
         * Phase 3: Parallel
         * ══════════════════════════════════════════════════════════ */
        cudaStreamSynchronize(stream);
        {
            double t_comm_para = 0.0;
            unsigned long comp_cnt = 0;
            pthread_t th_comm, th_comp;

            atomic_store(&s_comm_done, 0);

            comm_arg_t ca = {
                .d_buf     = d_buf,
                .count     = (size_t)count,
                .dtype     = dtype,
                .partner   = partner,
                .is_sender = is_initiator,
                .comm      = comm,
                .stream    = stream,
                .t_comm    = &t_comm_para,
            };
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
            volatile int sink = scratch[0]; (void)sink;
        }
    }

    /* ── Average ───────────────────────────────────────────────── */
    int n = config->iterations;
    t_comp    /= n;
    t_send_pure /= n;
    t_recv_pure /= n;
    t_send_para   /= n;
    t_recv_para   /= n;
    cnt_send_para /= n;
    cnt_recv_para /= n;

    /* ── Report (rank 0 prints all, others skip) ────────────────── */
    if (rank == 0) {
        double g_comp         = t_comp;          /* already avg */
        double g_send_pure    = t_send_pure;
        double g_recv_pure    = t_recv_pure;
        double g_send_para    = t_send_para;
        double g_recv_para    = t_recv_para;
        double g_cnt_send     = cnt_send_para;
        double g_cnt_recv     = cnt_recv_para;

        double comp_thru      = (double)N_WU / g_comp;
        double send_thru      = g_cnt_send / (g_send_para > 0 ? g_send_para : 1e-12);
        double recv_thru      = g_cnt_recv / (g_recv_para > 0 ? g_recv_para : 1e-12);

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

/* ── NCCL bootstrap (no MPI) ────────────────────────────────────────────── */

#define NCCL_ID_FILE "nccl_id_file/nccl_bench_overlap_id"

static ncclComm_t init_nccl(int rank, int size) {
    ncclUniqueId id;
    if (size == 1) {
        ncclGetUniqueId(&id);
    } else if (rank == 0) {
        ncclGetUniqueId(&id);
        FILE *f = fopen(NCCL_ID_FILE, "wb");
        fwrite(&id, sizeof(id), 1, f);
        fclose(f);
    }
    if (size > 1 && rank != 0) {
        struct stat st;
        int waited = 0;
        while (stat(NCCL_ID_FILE, &st) != 0) { usleep(10000); waited++; if (waited > 3000) exit(1); }
        FILE *f = fopen(NCCL_ID_FILE, "rb");
        fread(&id, sizeof(id), 1, f);
        fclose(f);
    }
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);
    return comm;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    int rank = get_rank();
    int size = get_size();

    if (size < 2 || size % 2 != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: Need even number of processes (got %d)\n", size);
        return 1;
    }

    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    cudaSetDevice(rank % (ndev > 0 ? ndev : 1));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ncclComm_t comm = init_nccl(rank, size);

    test_config_t config = parse_arguments(argc, argv);
    ncclDataType_t dtype = data_type_to_nccl(config.data_type);
    size_t esz = nccl_dt_size(dtype);

    size_t max_sz = config.max_message_size;
    void *d_buf = NULL;
    cudaMalloc(&d_buf, max_sz);
    cudaMemset(d_buf, 0, max_sz);

    int is_initiator = (rank % 2 == 0);
    int partner      = is_initiator ? rank + 1 : rank - 1;

    pthread_barrier_init(&s_barrier, NULL, 2);

    /* Header */
    if (rank == 0) {
        printf("=== NCCL Overlap (dual-thread) ===\n");
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

    size_iter_t it;
    size_iter_init(&it, &config);
    size_t sz;
    while (size_iter_next(&it, &sz)) {
        run_test_size(rank, size, sz, dtype,
                      is_initiator, partner,
                      comm, stream, d_buf, &config);
    }

    /* Legend */
    if (rank == 0) {
        printf("\n");
        printf("--- Column definitions ---\n");
        printf("Comp      Pure-compute time for %lu work units (us)\n", (unsigned long)N_WU);
        printf("Send      Pure ncclSend time (us)\n");
        printf("Recv      Pure ncclRecv time (us)\n");
        printf("PSend     ncclSend time with concurrent compute (us)\n");
        printf("PRecv     ncclRecv time with concurrent compute (us)\n");
        printf("CntSnd    Work units completed during parallel Send\n");
        printf("CntRcv    Work units completed during parallel Recv\n");
        printf("CSnd%%     Compute throughput change during Send\n");
        printf("CRcv%%     Same as CSnd%% but during Recv\n");
        printf("SSlow%%    Send slowdown: PSend/Send-1\n");
        printf("RSlow%%    Recv slowdown: PRecv/Recv-1\n");
        fflush(stdout);
    }

    pthread_barrier_destroy(&s_barrier);
    cudaFree(d_buf);
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
    if (rank == 0) remove(NCCL_ID_FILE);

    return 0;
}
