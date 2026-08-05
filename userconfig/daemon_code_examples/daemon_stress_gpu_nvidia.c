/*
 * daemon_stress_gpu_nvidia.c — GPU stress daemon (NVIDIA CUDA)
 *
 * Spawns one worker thread per GPU; each thread repeatedly launches SHORT
 * FINITE burn kernels. Every cycle, with probability STRESS_GPU_PERCENT/100 a
 * burn kernel is launched and runs to completion (releasing all SM slots and
 * HBM traffic); with probability (100-pct)/100 the cycle is skipped and the
 * GPU is genuinely free. The GPU is therefore busy ~pct% of the time.
 *
 * Why re-launching finite kernels instead of one persistent kernel? A
 * persistent kernel holds its blocks FOREVER — even with per-thread activity
 * toggling, the SM slots stay occupied and the allreduce victim is always
 * squeezed into the same fixed slice (~80 GB/s from 2% to 95%, flat). Finite
 * kernels RELEASE the GPU between launches: at low pct the victim mostly runs
 * on a free GPU and only contends ~pct% of the time, giving a GRADED,
 * monotonic slowdown. The per-launch duration is ~100us (far below a multi-ms
 * collective), so every iteration averages over many busy/free windows.
 *
 * Environment:
 *   STRESS_GPU_PERCENT      — launch probability per cycle 0-100 (default 50)
 *   STRESS_GPU_ITERS        — RMW iterations per burn launch (default 100000,
 *                             ~100us at the full grid)
 *   STRESS_GPU_CYCLE_GAP_US — sleep (us) on a skipped cycle (default 150)
 *
 * Requires:
 *   CUDA runtime headers + libcudart / libcuda at build time.
 *   Compiled together with daemon_gpu_burn_kernel.cu (nvcc).
 *   Only the NVIDIA driver is required at runtime.
 */

#include "daemon_helper.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

/* External kernel launcher — compiled from run_codes/daemons/src/daemon_gpu_burn_kernel.cu */
#include "daemon_gpu_burn_kernel.h"

/* ── Tunables ──────────────────────────────────────────────────── */
#define THREADS     256     /* threads per block                   */
#define STRESS_BUF_BYTES (256 * 1024 * 1024)  /* 256 MB per GPU — exceeds L2 cache */

/* ── Shared stop flag ──────────────────────────────────────────── */
static volatile sig_atomic_t g_worker_stop = 0;

/* Cheap per-worker LCG for the stochastic launch decisions. */
static unsigned int lcg_next(unsigned int *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

/* ── Per-GPU worker context ────────────────────────────────────── */
typedef struct {
    int             gpu_idx;
    cudaStream_t    stream;
    double         *d_out;
    int             gpu_pct; /* launch probability (STRESS_GPU_PERCENT) */
    size_t          d_out_size;
} gpu_worker_t;

/* ── Clamp utility ─────────────────────────────────────────────── */
static int clamp(int val, int lo, int hi)
{
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/* ── Worker thread — one per GPU ───────────────────────────────── */
static void *worker_routine(void *arg)
{
    gpu_worker_t *w = (gpu_worker_t *)arg;

    /* Attach to the GPU */
    cudaError_t err = cudaSetDevice(w->gpu_idx);
    if (err != cudaSuccess) {
        fprintf(stderr, "[stress_gpu %d] cudaSetDevice failed: %s\n",
                w->gpu_idx, cudaGetErrorString(err));
        return NULL;
    }

    /* Create stream */
    err = cudaStreamCreate(&w->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[stress_gpu %d] cudaStreamCreate failed: %s\n",
                w->gpu_idx, cudaGetErrorString(err));
        return NULL;
    }

    /* Allocate device output buffer (256 MB — exceeds L2 cache) */
    w->d_out_size = (size_t)STRESS_BUF_BYTES;
    err = cudaMalloc(&w->d_out, w->d_out_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[stress_gpu %d] cudaMalloc failed: %s\n",
                w->gpu_idx, cudaGetErrorString(err));
        return NULL;
    }

    /* Retrieve finite-kernel length (per-launch RMW iterations). At the full
     * grid this is ~100us for the default — short enough that a multi-ms
     * collective averages over many launch cycles. */
    int iterations = 100000;
    const char *env_i = getenv("STRESS_GPU_ITERS");
    if (env_i) iterations = clamp(atoi(env_i), 1000, 10000000);

    /* Gap (us) to sleep on a SKIPPED cycle, i.e. when no burn kernel runs.
     * Keep it comparable to the burn-kernel duration so the GPU is busy
     * roughly pct% of the time. */
    int cycle_gap_us = 150;
    const char *env_g = getenv("STRESS_GPU_CYCLE_GAP_US");
    if (env_g) cycle_gap_us = clamp(atoi(env_g), 1, 1000000);

    /* ── Derive grid from the ACTUAL device (device-agnostic) ───── */
    int sm_count = 0;
    err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, w->gpu_idx);
    if (err != cudaSuccess || sm_count < 1) {
        fprintf(stderr, "[stress_gpu %d] cannot query SM count: %s\n",
                w->gpu_idx, cudaGetErrorString(err));
        return NULL;
    }
    int blocks_per_sm = gpu_burn_query_blocks_per_sm(THREADS);
    if (blocks_per_sm < 1) blocks_per_sm = 1;

    long long max_blocks = (long long)sm_count * blocks_per_sm;
    /* Fixed grid: all co-resident slots minus one per SM, so while a burn
     * kernel IS resident the victim's kernels still have slots to run on. */
    int blocks = (int)(max_blocks - sm_count);
    if (blocks < 1) blocks = 1;

    fprintf(stderr, "[stress_gpu %d] SM=%d blocks/SM=%d max_blocks=%lld pct=%d (launch prob) -> blocks=%d iters=%d\n",
            w->gpu_idx, sm_count, blocks_per_sm, max_blocks, w->gpu_pct, blocks, iterations);

    /* ── Optional delayed kernel start (STRESS_GPU_START_DELAY, seconds) ── */
    int start_delay = 0;
    const char *env_d = getenv("STRESS_GPU_START_DELAY");
    if (env_d) start_delay = clamp(atoi(env_d), 0, 3600);
    if (start_delay > 0) {
        fprintf(stderr, "[stress_gpu %d] delaying kernel start by %d s\n",
                w->gpu_idx, start_delay);
        for (int s = 0; s < start_delay && !g_worker_stop && !daemon_should_stop(); s++)
            daemon_interruptible_sleep(1.0);
    }

    /* ── Re-launch loop ────────────────────────────────────────────
     * Each cycle: with probability pct/100 launch a finite burn kernel and
     * wait for it to COMPLETE (releasing all SM slots and HBM traffic); with
     * probability (100-pct)/100 skip and sleep the gap (GPU genuinely free).
     * A multi-ms collective therefore averages over busy/free windows, giving
     * a GRADED slowdown ~proportional to pct — unlike a persistent kernel,
     * which holds every SM slot forever and leaves the victim a fixed slice. */
    unsigned int seed = (unsigned int)w->gpu_idx * 2654435761u + 0x9E3779B9u;
    while (!g_worker_stop && !daemon_should_stop()) {
        if ((int)(lcg_next(&seed) % 100) < w->gpu_pct) {
            launch_gpu_burn(w->d_out, iterations, w->stream, blocks, THREADS);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[stress_gpu %d] burn launch failed: %s\n",
                        w->gpu_idx, cudaGetErrorString(err));
                break;
            }
            cudaStreamSynchronize(w->stream);   /* kernel completes -> slots free */
        } else {
            daemon_interruptible_sleep((double)cycle_gap_us / 1e6);
        }
    }

    /* Cleanup */
    cudaFree(w->d_out);
    cudaStreamDestroy(w->stream);
    w->d_out   = NULL;
    w->stream  = NULL;
    return NULL;
}

/* ── Main ──────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    /* Read launch probability % from environment (0 = true baseline, no stress) */
    int gpu_pct = 50;
    const char *env = getenv("STRESS_GPU_PERCENT");
    if (env) gpu_pct = clamp(atoi(env), 0, 100);

    /* Initialise CUDA runtime */
    int ngpu = 0;
    cudaError_t err = cudaGetDeviceCount(&ngpu);
    if (err != cudaSuccess || ngpu < 1) {
        fprintf(stderr, "[daemon_stress_gpu_nvidia] no CUDA-capable GPU found\n");
        return 1;
    }

    /* Signals */
    daemon_setup_signal_handler();
    daemon_set_signal_file("daemon_signals/daemon_stress_gpu_nvidia.signal");

    fprintf(stderr, "[daemon_stress_gpu_nvidia] starting, %d GPU(s) @ %d%% launch "
            "probability, %d threads/block (re-launching finite burn kernels)\n",
            ngpu, gpu_pct, THREADS);

    /* Create per-GPU workers */
    gpu_worker_t *workers = calloc((size_t)ngpu, sizeof(gpu_worker_t));
    pthread_t    *threads = malloc((size_t)ngpu * sizeof(pthread_t));
    if (!workers || !threads) {
        fprintf(stderr, "[daemon_stress_gpu_nvidia] malloc failed\n");
        free(workers);
        free(threads);
        return 1;
    }

    for (int i = 0; i < ngpu; i++) {
        workers[i].gpu_idx = i;
        workers[i].gpu_pct = gpu_pct;
        pthread_create(&threads[i], NULL, worker_routine, &workers[i]);
    }

    /* Main thread monitors signal file */
    while (!daemon_should_stop()) {
        int sig = daemon_check_signal_file();
        if (sig == 2) {          /* EXIT / FLUSH_EXIT */
            g_worker_stop = 1;
            break;
        }
        if (sig == 1) {          /* PAUSE */
            daemon_interruptible_sleep(0.100);
            continue;
        }
        daemon_interruptible_sleep(0.100);
    }

    /* Join all workers */
    for (int i = 0; i < ngpu; i++) {
        pthread_join(threads[i], NULL);
    }

    free(workers);
    free(threads);
    fprintf(stderr, "[daemon_stress_gpu_nvidia] stopped\n");
    return 0;
}
