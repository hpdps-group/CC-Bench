/* daemon_gpu_burn_kernel.h — C-linkage declaration for launch_gpu_burn() */
#ifndef DAEMON_GPU_BURN_KERNEL_H
#define DAEMON_GPU_BURN_KERNEL_H

#include <cuda_runtime.h>

/* Launch ONE finite burn kernel with `blocks` co-resident blocks, each thread
 * doing `iterations` strided HBM RMWs, then the kernel COMPLETES (releasing
 * all SM slots and HBM traffic). The daemon re-launches these with probability
 * STRESS_GPU_PERCENT/100 per cycle — see daemon_stress_gpu_nvidia.c.
 */
void launch_gpu_burn(double *d_out, int iterations,
                     cudaStream_t stream, int blocks, int threads);

/* Max co-resident blocks of the burn kernel per SM for the given block size
 * (via cudaOccupancyMaxActiveBlocksPerMultiprocessor). Device-agnostic. */
int gpu_burn_query_blocks_per_sm(int threads);

#endif
