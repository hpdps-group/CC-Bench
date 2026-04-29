#ifndef PERF_HELPER_MPI_H
#define PERF_HELPER_MPI_H

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

#endif /* PERF_HELPER_MPI_H */