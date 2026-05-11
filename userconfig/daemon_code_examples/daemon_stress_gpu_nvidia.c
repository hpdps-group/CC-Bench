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
 *   CUDA toolkit headers (cuda.h) and libcuda.so at build time.
 *   Only the NVIDIA driver is required at runtime.
 *
 * Compile (manual — register_daemons.sh currently passes -lm -lpthread only):
 *   gcc -O2 -Wall -I./run_codes/daemons/include \
 *       run_codes/daemons/src/daemon_helper.c \
 *       userconfig/daemon_code_examples/daemon_stress_gpu_nvidia.c \
 *       -o bin/daemons/daemon_stress_gpu_nvidia \
 *       -lm -lpthread -lcuda
 *
 * Or add -lcuda to the CC line in register_daemons.sh for this file.
 */

#include "daemon_helper.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

/* ── Tunables ──────────────────────────────────────────────────── */
#define CYCLE_SEC   0.100   /* 100 ms duty-cycle granularity       */
#define BLOCKS      256     /* thread-blocks per kernel launch     */
#define THREADS     256     /* threads per block                   */

/* ── PTX for the compute kernel ────────────────────────────────── *
 *
 * Equivalent CUDA source:
 *   __global__ void gpu_burn(double *out, int iterations) {
 *       int tid = threadIdx.x + blockIdx.x * blockDim.x;
 *       double x = 1.0;
 *       for (int i = 0; i < iterations; i++) {
 *           x = x * 1.0000001 + 0.0000001;
 *           x = x / 1.0000001 - 0.0000001;
 *       }
 *       out[tid] = x;   // prevent dead-code elimination
 *   }
 * ──────────────────────────────────────────────────────────────── */
static const char gpu_burn_ptx[] =
    ".version 7.0\n"
    ".target sm_60\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry gpu_burn(\n"
    "  .param .u64 out,\n"
    "  .param .u32 iterations\n"
    ")\n"
    "{\n"
    "  .reg .u32  tid, bid, ntid, iters, i;\n"
    "  .reg .u64  ptr, off;\n"
    "  .reg .f64  x;\n"
    "  .reg .pred p;\n"
    "\n"
    "  mov.u32        tid,  %tid.x;\n"
    "  mov.u32        bid,  %ctaid.x;\n"
    "  mov.u32        ntid, %ntid.x;\n"
    "  mad.lo.u32     tid,  bid, ntid, tid;\n"
    "\n"
    "  ld.param.u32   iters, [iterations];\n"
    "  ld.param.u64   ptr,   [out];\n"
    "\n"
    "  mov.f64        x, 1.0;\n"
    "  mov.u32        i, 0;\n"
    "\n"
    "loop:\n"
    "  mul.f64        x, x, 1.0000001;\n"
    "  add.f64        x, x, 0.0000001;\n"
    "  div.f64        x, x, 1.0000001;\n"
    "  sub.f64        x, x, 0.0000001;\n"
    "  add.u32        i, i, 1;\n"
    "  setp.lt.u32    p, i, iters;\n"
    "  @p bra         loop;\n"
    "\n"
    "  mul.wide.u32   off, tid, 8;\n"
    "  add.u64        ptr, ptr, off;\n"
    "  cvta.to.global.u64  ptr, ptr;\n"
    "  st.global.f64  [ptr], x;\n"
    "  ret;\n"
    "}\n";

/* ── Shared stop flag ──────────────────────────────────────────── */
static volatile sig_atomic_t g_worker_stop = 0;

/* ── Per-GPU worker context ────────────────────────────────────── */
typedef struct {
    int         gpu_idx;
    CUcontext   ctx;
    CUmodule    mod;
    CUfunction  func;
    CUstream    stream;
    CUdeviceptr d_out;
    double      work_sec;
    double      sleep_sec;
    size_t      d_out_size;
} gpu_worker_t;

/* ── Launch a burst of kernels to fill work_sec ────────────────── */
static void burn_gpu(gpu_worker_t *w, int iterations)
{
    void *args[2];
    args[0] = &w->d_out;
    args[1] = &iterations;

    double t0 = daemon_get_time();
    while (daemon_get_time() - t0 < w->work_sec) {
        CUresult res = cuLaunchKernel(
            w->func,
            BLOCKS, 1, 1,          /* grid dim  */
            THREADS, 1, 1,         /* block dim */
            0,                     /* shared mem */
            w->stream,             /* stream    */
            args, NULL);           /* kernel args */
        if (res != CUDA_SUCCESS)
            break;
        cuStreamSynchronize(w->stream);

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
    CUresult res;
    res = cuCtxSetCurrent(w->ctx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[stress_gpu %d] cuCtxSetCurrent failed\n", w->gpu_idx);
        return NULL;
    }

    /* Load module and get kernel function */
    res = cuModuleLoadData(&w->mod, gpu_burn_ptx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[stress_gpu %d] cuModuleLoadData failed\n", w->gpu_idx);
        return NULL;
    }
    res = cuModuleGetFunction(&w->func, w->mod, "gpu_burn");
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[stress_gpu %d] cuModuleGetFunction failed\n", w->gpu_idx);
        return NULL;
    }

    /* Create stream */
    res = cuStreamCreate(&w->stream, CU_STREAM_NON_BLOCKING);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[stress_gpu %d] cuStreamCreate failed\n", w->gpu_idx);
        return NULL;
    }

    /* Allocate device output buffer (one double per thread) */
    w->d_out_size = (size_t)BLOCKS * THREADS * sizeof(double);
    res = cuMemAlloc(&w->d_out, w->d_out_size);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[stress_gpu %d] cuMemAlloc failed\n", w->gpu_idx);
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
    cuStreamSynchronize(w->stream);
    cuMemFree(w->d_out);
    cuStreamDestroy(w->stream);
    cuModuleUnload(w->mod);
    w->d_out   = 0;
    w->mod     = NULL;
    w->func    = NULL;
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

    /* Initialise CUDA driver API */
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[daemon_stress_gpu_nvidia] cuInit failed "
                "(NVIDIA driver loaded?)\n");
        return 1;
    }

    int ngpu = 0;
    res = cuDeviceGetCount(&ngpu);
    if (res != CUDA_SUCCESS || ngpu < 1) {
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

        CUdevice dev;
        cuDeviceGet(&dev, i);
        cuCtxCreate(&workers[i].ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev);

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
        cuCtxDestroy(workers[i].ctx);
        workers[i].ctx = NULL;
    }

    free(workers);
    free(threads);
    fprintf(stderr, "[daemon_stress_gpu_nvidia] stopped\n");
    return 0;
}
