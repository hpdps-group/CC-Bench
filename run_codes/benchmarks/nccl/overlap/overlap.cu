/**
 * NCCL Overlap with GPU Compute — measurement of GPU compute / NCCL mutual interference.
 *
 * Uses two CUDA streams: one for GPU compute kernel, one for NCCL communication.
 * Measures how much each side slows down the other on the same GPU.
 *
 * Processes paired as n (even, "initiator") and n+1 (odd, "partner").
 * Requires even number of processes.
 *
 * Three phases per message size:
 *   1. Pure GPU compute   — GPU kernel on stream_comp (baseline)
 *   2. Pure NCCL comm     — ncclSend / ncclRecv on stream_comm (baseline)
 *   3. Parallel           — GPU compute + NCCL simultaneously on two streams
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "nccl/nccl_utils.h"

#ifdef __cplusplus
}
#endif

/* ── GPU compute configuration ──────────────────────────────────────────── */

/** Total iterations across all threads (adjust for GPU capability). */
#define N_GPU_ITER      20000000000ULL

/**
 * GPU work buffer size (must be power of two).  64M × 8B = 512 MB.
 * Larger than GPU L2 cache (~50 MB on H100), so every access goes to HBM.
 */
#define GPU_BUF_LOG2    26
#define GPU_BUF_ELEMS   (1ULL << GPU_BUF_LOG2)

#define BLOCK_DIM       256
#define GRID_DIM        128

/**
 * GPU compute kernel — memory-bandwidth-heavy read-modify-write chain.
 *
 * Each thread walks the buffer by a stride, performing dependent
 * loads/stores that cannot be optimized away.  The buffer is larger
 * than L2 cache, every access hits HBM and competes with NCCL traffic.
 */
__global__ void gpu_work_kernel(volatile unsigned long long *buf,
                                unsigned long long n_iter)
{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long mask = GPU_BUF_ELEMS - 1;

    for (unsigned long long i = tid; i < n_iter; i += stride) {
        int idx = (int)(i & mask);
        buf[idx] = buf[idx] * 3 + 1;
    }
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

/* ── NCCL bootstrap ─────────────────────────────────────────────────────── */

#define NCCL_ID_FILE "nccl_id_file/nccl_bench_overlap_id"

static ncclComm_t init_nccl(int rank, int size) {
    ncclUniqueId id;
    if (size == 1) {
        ncclGetUniqueId(&id);
    } else if (rank == 0) {
        remove(NCCL_ID_FILE);
        ncclGetUniqueId(&id);
        FILE *f = fopen(NCCL_ID_FILE, "wb");
        fwrite(&id, sizeof(id), 1, f);
        fclose(f);
    }
    if (size > 1 && rank != 0) {
        struct stat st;
        int waited = 0;
        while (stat(NCCL_ID_FILE, &st) != 0) {
            usleep(10000);
            waited++;
            if (waited > 3000) exit(1);
        }
        FILE *f = fopen(NCCL_ID_FILE, "rb");
        fread(&id, sizeof(id), 1, f);
        fclose(f);
    }
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);
    return comm;
}

/* ── Per-size test ─────────────────────────────────────────────────────── */

static void run_test_size(int rank, int size,
                          size_t msg_size, ncclDataType_t dtype,
                          int is_initiator, int partner,
                          ncclComm_t comm,
                          cudaStream_t stream_comp, cudaStream_t stream_comm,
                          volatile unsigned long long *d_comp_buf,
                          void *d_buf, const test_config_t *config)
{
    size_t esz = nccl_dt_size(dtype);
    int count = (int)(msg_size / esz);
    if (count <= 0) count = 1;

    double t_comp      = 0.0;    /* pure GPU compute (seconds) */
    double t_comm_pure = 0.0;    /* pure NCCL send/recv (seconds) */
    double t_comp_para = 0.0;    /* GPU compute during concurrent comm */
    double t_comm_para = 0.0;    /* NCCL comm during concurrent compute */

    for (int iter = 0; iter < config->iterations; iter++) {
        /* ═════════════════════════════════════════════════════════════════
         * Phase 1: Pure GPU compute (baseline kernel throughput)
         * ═════════════════════════════════════════════════════════════════ */
        cudaStreamSynchronize(stream_comp);
        {
            cudaEvent_t ev_s, ev_e;
            cudaEventCreate(&ev_s);
            cudaEventCreate(&ev_e);

            cudaEventRecord(ev_s, stream_comp);
            gpu_work_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream_comp>>>(
                d_comp_buf, N_GPU_ITER);
            cudaEventRecord(ev_e, stream_comp);
            cudaEventSynchronize(ev_e);

            float ms;
            cudaEventElapsedTime(&ms, ev_s, ev_e);
            t_comp += (double)ms / 1000.0;

            cudaEventDestroy(ev_s);
            cudaEventDestroy(ev_e);
        }

        /* ═════════════════════════════════════════════════════════════════
         * Phase 2: Pure NCCL communication
         * ═════════════════════════════════════════════════════════════════ */
        cudaStreamSynchronize(stream_comm);
        {
            cudaEvent_t ev_s, ev_e;
            cudaEventCreate(&ev_s);
            cudaEventCreate(&ev_e);

            cudaEventRecord(ev_s, stream_comm);
            if (is_initiator) {
                ncclResult_t _ret = ncclSend(d_buf, count, dtype, partner,
                                             comm, stream_comm);
                if (_ret != ncclSuccess) {
                    fprintf(stderr, "[rank=%d] ncclSend FAILED count=%d "
                                    "iter=%d error=%d\n",
                            rank, count, iter, (int)_ret);
                    exit(1);
                }
            } else {
                ncclResult_t _ret = ncclRecv(d_buf, count, dtype, partner,
                                             comm, stream_comm);
                if (_ret != ncclSuccess) {
                    fprintf(stderr, "[rank=%d] ncclRecv FAILED count=%d "
                                    "iter=%d error=%d\n",
                            rank, count, iter, (int)_ret);
                    exit(1);
                }
            }
            cudaEventRecord(ev_e, stream_comm);
            cudaEventSynchronize(ev_e);

            float ms;
            cudaEventElapsedTime(&ms, ev_s, ev_e);
            t_comm_pure += (double)ms / 1000.0;

            cudaEventDestroy(ev_s);
            cudaEventDestroy(ev_e);
        }

        /* ═════════════════════════════════════════════════════════════════
         * Phase 3: Parallel — GPU compute + NCCL simultaneously
         * ═════════════════════════════════════════════════════════════════ */
        cudaStreamSynchronize(stream_comp);
        cudaStreamSynchronize(stream_comm);
        {
            cudaEvent_t comp_s, comp_e, comm_s, comm_e;
            cudaEventCreate(&comp_s); cudaEventCreate(&comp_e);
            cudaEventCreate(&comm_s); cudaEventCreate(&comm_e);

            /* Launch GPU compute on stream_comp */
            cudaEventRecord(comp_s, stream_comp);
            gpu_work_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream_comp>>>(
                d_comp_buf, N_GPU_ITER);
            cudaEventRecord(comp_e, stream_comp);

            /* Launch NCCL comm on stream_comm */
            cudaEventRecord(comm_s, stream_comm);
            if (is_initiator) {
                ncclResult_t _ret = ncclSend(d_buf, count, dtype, partner,
                                             comm, stream_comm);
                if (_ret != ncclSuccess) {
                    fprintf(stderr, "[rank=%d] ncclSend FAILED (para) "
                                    "count=%d iter=%d error=%d\n",
                            rank, count, iter, (int)_ret);
                    exit(1);
                }
            } else {
                ncclResult_t _ret = ncclRecv(d_buf, count, dtype, partner,
                                             comm, stream_comm);
                if (_ret != ncclSuccess) {
                    fprintf(stderr, "[rank=%d] ncclRecv FAILED (para) "
                                    "count=%d iter=%d error=%d\n",
                            rank, count, iter, (int)_ret);
                    exit(1);
                }
            }
            cudaEventRecord(comm_e, stream_comm);

            /* Wait for both to finish */
            cudaEventSynchronize(comm_e);
            cudaEventSynchronize(comp_e);

            float comp_ms, comm_ms;
            cudaEventElapsedTime(&comp_ms, comp_s, comp_e);
            cudaEventElapsedTime(&comm_ms, comm_s, comm_e);

            t_comp_para += (double)comp_ms / 1000.0;
            t_comm_para += (double)comm_ms / 1000.0;

            cudaEventDestroy(comp_s); cudaEventDestroy(comp_e);
            cudaEventDestroy(comm_s); cudaEventDestroy(comm_e);
        }
    }

    /* ── Average ───────────────────────────────────────────────────────── */
    int n = config->iterations;
    t_comp      /= n;
    t_comm_pure /= n;
    t_comp_para /= n;
    t_comm_para /= n;

    /* ── Report (rank 0 prints, others skip) ───────────────────────────── */
    if (rank == 0) {
        double comp_slow = 0.0;
        double comm_slow = 0.0;
        if (t_comp > 0.0)
            comp_slow = (t_comp_para / t_comp - 1.0) * 100.0;
        if (t_comm_pure > 0.0)
            comm_slow = (t_comm_para / t_comm_pure - 1.0) * 100.0;

        printf("%-10zu %-10.2f %-10.2f "
               "%-10.2f %-10.2f "
               "%-9.1f %-9.1f\n",
               msg_size,
               t_comp * 1e6, t_comm_pure * 1e6,
               t_comp_para * 1e6, t_comm_para * 1e6,
               comp_slow, comm_slow);
        fflush(stdout);
    }
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

    /* Two CUDA streams: one for GPU compute, one for NCCL comm */
    cudaStream_t stream_comp, stream_comm;
    cudaStreamCreate(&stream_comp);
    cudaStreamCreate(&stream_comm);

    ncclComm_t comm = init_nccl(rank, size);

    test_config_t config = parse_arguments(argc, argv);
    ncclDataType_t dtype = data_type_to_nccl(config.data_type);
    size_t esz = nccl_dt_size(dtype);

    size_t max_sz = config.max_message_size;

    /* NCCL communication buffer */
    void *d_buf = NULL;
    cudaMalloc(&d_buf, max_sz);
    cudaMemset(d_buf, 0, max_sz);

    /* GPU compute work buffer (allocated once, reused across phases) */
    volatile unsigned long long *d_comp_buf = NULL;
    size_t comp_buf_bytes = GPU_BUF_ELEMS * sizeof(unsigned long long);
    if (cudaMalloc(&d_comp_buf, comp_buf_bytes) != cudaSuccess) {
        fprintf(stderr, "[rank=%d] Failed to allocate GPU comp buffer "
                        "(%zu bytes)\n", rank, comp_buf_bytes);
        return 1;
    }
    cudaMemset((void*)d_comp_buf, 0, comp_buf_bytes);

    int is_initiator = (rank % 2 == 0);
    int partner      = is_initiator ? rank + 1 : rank - 1;

    /* Header */
    if (rank == 0) {
        printf("=== NCCL Overlap (GPU compute) ===\n");
        printf("Processes: %d\n", size);
        printf("GPU iter:  %llu\n", (unsigned long long)N_GPU_ITER);
        printf("Comp buf:  %zu MB\n\n", comp_buf_bytes / (1024 * 1024));
        printf("%-10s %-10s %-10s "
               "%-10s %-10s "
               "%-9s %-9s\n",
               "# Size", "Comp", "Comm",
               "PComp", "PComm",
               "CSlow%", "SSlow%");
        printf("%-10s %-10s %-10s "
               "%-10s %-10s "
               "%-9s %-9s\n",
               "(bytes)", "(us)", "(us)",
               "(us)", "(us)",
               "", "");
    }

    size_iter_t it;
    size_iter_init(&it, &config);
    size_t sz;
    while (size_iter_next(&it, &sz)) {
        run_test_size(rank, size, sz, dtype,
                      is_initiator, partner,
                      comm,
                      stream_comp, stream_comm,
                      d_comp_buf, d_buf, &config);
    }

    /* Legend */
    if (rank == 0) {
        printf("\n");
        printf("--- Column definitions ---\n");
        printf("Comp      Pure GPU compute kernel time (us)\n");
        printf("Comm      Pure NCCL send/recv time (us)\n");
        printf("PComp     GPU compute kernel time with concurrent NCCL (us)\n");
        printf("PComm     NCCL time with concurrent GPU compute (us)\n");
        printf("CSlow%%    Compute slowdown: PComp/Comp - 1 (%%)\n");
        printf("SSlow%%    Comm slowdown: PComm/Comm - 1 (%%)\n");
        printf("\n");
        printf("Positive CSlow%%/SSlow%% means the operation was slowed down\n");
        printf("by concurrent GPU work.  0%% = no interference.\n");
        fflush(stdout);
    }

    cudaFree((void*)d_comp_buf);
    cudaFree(d_buf);
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream_comp);
    cudaStreamDestroy(stream_comm);
    if (rank == 0) remove(NCCL_ID_FILE);

    return 0;
}
