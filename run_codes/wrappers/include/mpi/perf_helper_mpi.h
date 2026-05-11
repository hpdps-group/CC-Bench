#ifndef PERF_HELPER_MPI_H
#define PERF_HELPER_MPI_H

#include <mpi.h>
#include "perf_helper.h"

/**
 * perf_helper_mpi — MPI-specific TLS perf state management.
 *
 * - Each thread gets its own perf_state_t (TLS).
 * - First call to perf_mpi_get_tls() lazily initializes.
 * - perf_mpi_flush() writes data to perf_mpi_<rank>.txt and resets.
 * - A destructor auto-flushes at exit (MPI must still be alive).
 */

/**
 * Return pointer to this thread's perf state, initializing if needed.
 */
perf_state_t *perf_mpi_get_tls(void);

/**
 * Flush this thread's perf data to a per-rank file ("perf_mpi_<rank>.txt")
 * and destroy the state (free allocations, reset counters).
 *
 * Safe to call multiple times — second call is a no-op.
 */
void perf_mpi_flush(void);

/**
 * Node map for intra / inter-node communication detection.
 *
 * Lazy one-time Allgather of all processor names across MPI_COMM_WORLD.
 * Safe to call before MPI_Init (becomes no-op until MPI is ready).
 */
void perf_mpi_init_node_map(void);

/**
 * Return the local rank in MPI_COMM_WORLD (-1 if not yet mapped).
 */
int  perf_mpi_world_rank(void);

/**
 * Return non-zero when @p rank (in MPI_COMM_WORLD) is on the same node.
 * Returns 0 if the rank is out of range or the node map is unavailable.
 */
int  perf_mpi_is_intra(int rank);

/**
 * Return the message size in bytes for an MPI operation.
 * count * sizeof(MPI_Datatype element).
 */
double perf_mpi_msg_size(int count, MPI_Datatype datatype);

#endif /* PERF_HELPER_MPI_H */