/* daemon_gpu_burn_kernel.h — C-linkage declaration for launch_gpu_burn() */
#ifndef DAEMON_GPU_BURN_KERNEL_H
#define DAEMON_GPU_BURN_KERNEL_H

#include <cuda_runtime.h>

void launch_gpu_burn(double *d_out, int iterations,
                     cudaStream_t stream,
                     int blocks, int threads);

#endif
