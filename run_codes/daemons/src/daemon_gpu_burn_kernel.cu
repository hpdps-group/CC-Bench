/*
 * daemon_gpu_burn_kernel.cu — GPU burn kernel for stress daemons
 *
 * Pre-compiled with nvcc so no runtime PTX JIT is needed.
 * Provides launch_gpu_burn() callable from C daemon code.
 */
#include <cuda_runtime.h>

__global__ void gpu_burn_kernel(double *out, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double x = 1.0;
    for (int i = 0; i < iterations; i++) {
        x = x * 1.0000001 + 0.0000001;
        x = x / 1.0000001 - 0.0000001;
    }
    out[tid] = x;
}

extern "C" void launch_gpu_burn(double *d_out, int iterations,
                                cudaStream_t stream,
                                int blocks, int threads)
{
    gpu_burn_kernel<<<blocks, threads, 0, stream>>>(d_out, iterations);
}
