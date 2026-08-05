/*
 * daemon_gpu_burn_kernel.cu — Finite GPU burn kernel (re-launch based)
 *
 * The daemon repeatedly launches SHORT FINITE burn kernels. Each launch runs
 * a strided HBM read-modify-write over the whole buffer for `iterations` and
 * then COMPLETES — releasing every SM slot and all HBM traffic. The daemon
 * launches with probability STRESS_GPU_PERCENT/100 each cycle, so the GPU is
 * busy ~pct% of the time and genuinely free the rest.
 *
 * Why finite re-launches instead of one persistent kernel:
 *
 *   A persistent kernel holds its blocks FOREVER. Even with per-thread
 *   activity toggling, every SM slot stays occupied, so the victim is always
 *   squeezed into the same fixed slice of the GPU and the allreduce stayed
 *   flat at ~80 GB/s from 2% to 95%. Finite kernels RELEASE the GPU between
 *   launches — at low pct the victim mostly runs on a free GPU and only
 *   contends ~pct% of the time, giving a GRADED, monotonic slowdown.
 *
 * Shutdown: kernels are short (~100us), so once the daemon breaks its launch
 * loop the GPU frees up immediately; no stop flag is needed.
 *
 * The launch grid is fixed by the daemon at max_blocks - sm_count (all slots
 * minus one per SM) so the victim's own kernels can still be scheduled while
 * a burn kernel is resident.
 */
#include <cuda_runtime.h>

/* Per-GPU buffer size — must be >> L2 cache (~40 MB on A800). */
#define BUF_ELEMS (256UL * 1024UL * 1024UL / sizeof(double))  /* ~33.6M */

__global__ void gpu_burn_finite_kernel(double *buf, int n_elems, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    /* Per-thread strided RMW cursor. Additive pointer wrap instead of a
     * 64-bit `%` (a runtime modulo would make the kernel division-bound and
     * cap its HBM traffic). */
    double *p = buf + tid;
    const double *buf_end = buf + n_elems;

    for (int i = 0; i < iterations; i++) {
        /* Strided HBM RMW, coalesced within a warp. */
        *p = *p * 1.0000001 + 0.0000001;
        p += stride;
        if (p >= buf_end) p -= n_elems;   /* stride <= n_elems, safe */
    }
}

/*
 * launch_gpu_burn() — launch ONE finite burn kernel with `blocks` co-resident
 * blocks, running `iterations` RMW steps per thread. Returns after the kernel
 * is ENQUEUED; the caller syncs the stream to wait for completion (at which
 * point all SM slots and HBM traffic are released).
 */
extern "C" void launch_gpu_burn(double *d_out, int iterations,
                                cudaStream_t stream, int blocks, int threads)
{
    gpu_burn_finite_kernel<<<blocks, threads, 0, stream>>>(
        d_out, (int)BUF_ELEMS, iterations);
}

/*
 * gpu_burn_query_blocks_per_sm() — how many of the burn kernel's blocks can
 * run co-resident on one SM for the given block size.
 *
 * Uses the CUDA occupancy API, so the result is correct on ANY NVIDIA GPU.
 */
extern "C" int gpu_burn_query_blocks_per_sm(int threads)
{
    int num_blocks = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, gpu_burn_finite_kernel, threads, 0);
    if (err != cudaSuccess || num_blocks < 1)
        return 1;
    return num_blocks;
}
