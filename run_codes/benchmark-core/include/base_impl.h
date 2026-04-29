/**
 * Base implementation loading interface.
 *
 * Each backend (mpi/nccl/rccl) provides a load_base_impl() that
 * prepares the reference ("definitely correct") implementation:
 *   - MPI: built-in PMPI (no dlopen needed)
 *   - NCCL/RCCL: dlopen the vendor library discovered by findso
 */

#ifndef BASE_IMPL_H
#define BASE_IMPL_H

/* Default path for base-implementation state file */
#define BASE_SO_FILE "bin/libs/base_so_name"

/**
 * Load / verify the base implementation.
 * The state file must have been written by the standalone findso binary
 * (./bin/<backend>/findso) before calling this.
 *
 * For MPI: confirms the state file and prints a message.
 * For NCCL/RCCL: dlopen's the recorded .so and resolves function pointers.
 *
 * Returns 0 on success, -1 on failure.
 */
int load_base_impl(const char *state_path);

#endif /* BASE_IMPL_H */