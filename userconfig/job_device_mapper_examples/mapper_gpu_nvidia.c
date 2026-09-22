/**
 * mapper_gpu_nvidia.c — example: record the GPU visible to the calling process.
 *
 * Uses cudaGetDevice() to discover which GPU this rank is bound to.
 *
 * Convention:
 *   - file name = mapper_gpu_nvidia.c  →  function name = mapper_gpu_nvidia()
 *   - signature: void mapper_<name>(device_map_ctx_t *ctx)
 *
 * The function name MUST match the file name (minus the "mapper_" prefix):
 * register_job_device_mapper.sh derives the generated forward declaration and
 * the dispatcher call from the file name, so a mismatch compiles into an
 * undefined symbol that only surfaces when the .so is dlopen'd.
 *
 * NOTE: This file requires the CUDA toolkit (cuda_runtime.h + libcudart).
 * register_job_device_mapper.sh adds the toolkit include/lib paths and links
 * -lcudart exactly when a mapper needs them, and fails the build if no toolkit
 * is found — rather than emitting a .so with an undefined cudaGetDevice.
 *
 * Compile: ./scripts/register_job_device_mapper.sh
 */

#include "job_device_mapper.h"
#include <cuda_runtime.h>

void mapper_gpu_nvidia(device_map_ctx_t *ctx)
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
