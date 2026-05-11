/**
 * job_device_mapper.h — record which CPUs/GPUs each MPI rank is using.
 *
 * Each MPI rank calls map_job_to_device() at init time, populating
 * device_map_ctx_t with {rank, node, device_type, device_number} entries.
 * Rank 0 then gathers everything and writes perf_files/map.csv.
 *
 * Users write custom mapper .c files in:
 *   userconfig/job_device_mapper_examples/   (reference examples)
 *   userconfig/your_job_device_mapper/       (user's own mappers)
 *
 * Run scripts/register_job_device_mapper.sh to regenerate
 * the mapper_generated.c and compile bin/libs/libjob_device_mapper.so.
 *
 * The benchmark (mpi_utils.c) uses dlopen to optionally load the .so
 * at runtime — safe to skip if not present.
 */

#ifndef JOB_DEVICE_MAPPER_H
#define JOB_DEVICE_MAPPER_H

#include <stddef.h>

#define DEVICE_MAP_MAX_ENTRIES  4096
#define DEVICE_MAP_NODE_NAME_MAX 64
#define DEVICE_MAP_TYPE_MAX      16

/* ── one resource binding ───────────────────────────── */
typedef struct {
    int  rank;                                /* MPI rank           */
    char node[DEVICE_MAP_NODE_NAME_MAX];       /* hostname           */
    char device_type[DEVICE_MAP_TYPE_MAX];     /* "cpu" or "gpu"     */
    int  device_number;                        /* CPU ID / GPU index */
} device_map_entry_t;

/* ── per-rank context, populated by mapper functions ── */
typedef struct {
    int  rank;
    int  size;
    device_map_entry_t entries[DEVICE_MAP_MAX_ENTRIES];
    int  num_entries;
} device_map_ctx_t;

/* ── Framework API ──────────────────────────────────── */

/**
 * Initialise a device-map context (zero entries, record rank/size).
 */
void device_map_init(device_map_ctx_t *ctx, int rank, int size);

/**
 * Append one entry for the calling rank.
 * The node field is filled automatically via gethostname().
 * Returns 0 on success, -1 if the ring is full.
 */
int  device_map_add_entry(device_map_ctx_t *ctx,
                          const char *device_type, int device_number);

/**
 * MPI_Gatherv all entries to rank 0 and write perf_files/map.csv.
 * Must be called by every rank.
 */
void device_map_flush_to_file(device_map_ctx_t *ctx);

/**
 * No-op (kept for API symmetry).
 */
void device_map_destroy(device_map_ctx_t *ctx);

/* ── User mapper entry point ──────────────────────────
 *
 * A WEAK no-op default lives in job_device_mapper.c so the symbol
 * is always available.  The script register_job_device_mapper.sh
 * compiles the real overrides into mapper_generated.c → .so.
 * mpi_utils.c loads the .so via dlopen at runtime.
 */
void map_job_to_device(device_map_ctx_t *ctx);

#endif /* JOB_DEVICE_MAPPER_H */
