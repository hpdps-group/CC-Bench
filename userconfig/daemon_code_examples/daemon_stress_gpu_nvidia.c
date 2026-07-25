/*
 * daemon_stress_gpu_nvidia.c — GPU stress daemon (NVIDIA CUDA)
 *
 * Spawns one worker thread per GPU; each thread repeatedly launches
 * compute kernels to keep the device busy at a configurable duty cycle.
 * This mirrors daemon_stress_cpu.c but for GPU compute units.
 *
 * Environment:
 *   STRESS_GPU_PERCENT   — per-GPU target utilisation 1-100 (default 50)
 *   STRESS_GPU_ITERS     — inner-loop iterations per kernel launch
 *                          (default 10000, increase for longer-running kernels)
 *
 * Requires:
 *   CUDA runtime headers + libcudart / libcuda at build time.
 *   Compiled together with daemon_stress_gpu_nvidia_kernel.cu (nvcc).
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
#define CYCLE_SEC   0.100   /* 100 ms duty-cycle granularity       */
#define BLOCKS      256     /* thread-blocks per kernel launch     */
#define THREADS     256     /* threads per block                   */
#define STRESS_BUF_BYTES (256 * 1024 * 1024)  /* 256 MB per GPU — exceeds L2 cache */

/* ── Shared stop flag ──────────────────────────────────────────── */
static volatile sig_atomic_t g_worker_stop = 0;

/* ── Per-GPU worker context ────────────────────────────────────── */
typedef struct {
    int          gpu_idx;
    cudaStream_t stream;
    double      *d_out;
    double       work_sec;
    double       sleep_sec;
    size_t       d_out_size;
} gpu_worker_t;

/* ── Launch a burst of kernels to fill work_sec ────────────────── */
static void burn_gpu(gpu_worker_t *w, int iterations)
{
    double t0 = daemon_get_time();
    while (daemon_get_time() - t0 < w->work_sec) {
        launch_gpu_burn(w->d_out, iterations, w->stream, BLOCKS, THREADS);
        cudaStreamSynchronize(w->stream);

        /* If signalled during a burn burst, exit early */
        if (g_worker_stop || daemon_should_stop())
            break;
    }
}

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

    /* Retrieve iterations */
    int iterations = 10000;
    const char *env_i = getenv("STRESS_GPU_ITERS");
    if (env_i) iterations = clamp(atoi(env_i), 1000, 1000000);

    /* Stress loop */
    while (!g_worker_stop && !daemon_should_stop()) {
        burn_gpu(w, iterations);
        if (w->sleep_sec > 0.0)
            daemon_interruptible_sleep(w->sleep_sec);
    }

    /* Cleanup */
    cudaStreamSynchronize(w->stream);
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

    /* Read target GPU% from environment */
    int gpu_pct = 50;
    const char *env = getenv("STRESS_GPU_PERCENT");
    if (env) gpu_pct = clamp(atoi(env), 1, 100);

    double work_sec  = CYCLE_SEC * gpu_pct / 100.0;
    double sleep_sec = CYCLE_SEC - work_sec;

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

    fprintf(stderr, "[daemon_stress_gpu_nvidia] starting, %d GPU(s) @ %d%% duty, "
            "%dx%d threads/launch\n", ngpu, gpu_pct, BLOCKS, THREADS);

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
        workers[i].gpu_idx   = i;
        workers[i].work_sec  = work_sec;
        workers[i].sleep_sec = sleep_sec;
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
            daemon_interruptible_sleep(CYCLE_SEC);
            continue;
        }
        daemon_interruptible_sleep(CYCLE_SEC);
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
