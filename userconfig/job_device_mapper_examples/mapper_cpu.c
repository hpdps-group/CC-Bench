/**
 * mapper_cpu.c — example: record CPU affinity for the calling process.
 *
 * Uses sched_getaffinity() to discover which logical CPUs this process
 * is allowed to run on, then adds one entry per CPU.
 *
 * Convention:
 *   - file name = mapper_cpu.c  →  function name = mapper_cpu()
 *   - signature: void mapper_<name>(device_map_ctx_t *ctx)
 *
 * Compile: ./scripts/register_job_device_mapper.sh
 *
 * NOTE: sched_getaffinity is a GNU extension.  If you compile with strict
 * C standards (-std=c99 etc.), define _GNU_SOURCE before including <sched.h>.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "job_device_mapper.h"
#include <sched.h>
#include <unistd.h>

void mapper_cpu(device_map_ctx_t *ctx)
{
    cpu_set_t *cpuset = NULL;
    int nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc <= 0) nproc = 1024;

    size_t cpuset_size = CPU_ALLOC_SIZE(nproc);
    cpuset = CPU_ALLOC(nproc);
    if (!cpuset) return;

    CPU_ZERO_S(cpuset_size, cpuset);

    if (sched_getaffinity(0, cpuset_size, cpuset) != 0) {
        CPU_FREE(cpuset);
        return;
    }

    for (int i = 0; i < nproc; i++) {
        if (CPU_ISSET_S(i, cpuset_size, cpuset)) {
            if (device_map_add_entry(ctx, "cpu", i) != 0)
                break;
        }
    }

    CPU_FREE(cpuset);
}
