/*
 * daemon_gpu_burn_kernel.cu — Memory-bandwidth-heavy GPU stress kernel
 *
 * Each thread does a strided read-modify-write across a buffer much larger
 * than GPU L2 cache (~40 MB on A100/A800), so every iteration hits HBM.
 * This competes with NCCL DMA and compress/decompress kernels for HBM
 * bandwidth, faithfully simulating training-step compute load.
 *
 * Pre-compiled with nvcc so no runtime PTX JIT is needed.
 * Provides launch_gpu_burn() callable from C daemon code.
 *
 * The caller (daemon_stress_gpu_nvidia) must pre-allocate the buffer
 * via cudaMalloc with at least BUF_BYTES bytes before calling this.
 */
#include <cuda_runtime.h>

/* Per-GPU buffer size — must be >> L2 cache (~40 MB on A800). */
#define BUF_ELEMS (256UL * 1024UL * 1024UL / sizeof(double))  /* ~33.6M */

__global__ void gpu_burn_kernel(double *buf, int n_elems, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = 0; i < iterations; i++) {
        /* Strided walk over the entire buffer; every access hits HBM.
         * Accesses within a warp are coalesced for maximal throughput. */
        int idx = (tid + (long long)i * stride) % n_elems;
        buf[idx] = buf[idx] * 1.0000001 + 0.0000001;
    }
}

extern "C" void launch_gpu_burn(double *d_out, int iterations,
                                cudaStream_t stream,
                                int blocks, int threads)
{
    gpu_burn_kernel<<<blocks, threads, 0, stream>>>(d_out, (int)BUF_ELEMS, iterations);
}
