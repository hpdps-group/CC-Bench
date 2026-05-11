/**
 * mapper_gpu.c — example: record the GPU visible to the calling process.
 *
 * Uses cudaGetDevice() to discover which GPU this rank is bound to.
 *
 * Convention:
 *   - file name = mapper_gpu.c  →  function name = mapper_gpu()
 *   - signature: void mapper_<name>(device_map_ctx_t *ctx)
 *
 * NOTE: This file requires the CUDA toolkit (cuda_runtime.h).
 * It is only compiled when CUDA is available.
 *
 * Compile: ./scripts/register_job_device_mapper.sh
 */

#include "job_device_mapper.h"
#include <cuda_runtime.h>

void mapper_gpu(device_map_ctx_t *ctx)
{
    int current_dev = -1;
    cudaError_t err = cudaGetDevice(&current_dev);
    if (err != cudaSuccess || current_dev < 0)
        return;

    device_map_add_entry(ctx, "gpu", current_dev);

#if 0
    /* Alternative: enumerate ALL visible GPUs */
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) == cudaSuccess) {
        for (int i = 0; i < dev_count; i++)
            if (device_map_add_entry(ctx, "gpu", i) != 0) break;
    }
#endif
}
