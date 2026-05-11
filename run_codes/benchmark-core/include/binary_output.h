/**
 * binary_output.h — write reference / user result buffers to binary files.
 *
 * Two modes:
 *   single-writer — one rank (root / last) writes the full buffer.
 *   multi-writer  — every rank writes a chunk at its own offset into
 *                   the same file via pwrite (zero-copy, no MPI I/O).
 *
 * The output path is derived as:  <base>_<suffix>.bin
 *   e.g.  results/allreduce_reference.bin
 *         results/allreduce_user.bin
 */

#ifndef BINARY_OUTPUT_H
#define BINARY_OUTPUT_H

#include <stddef.h>
#include "utils.h"

/**
 * Single-writer: only @p writer_rank opens, writes all @p bytes, and closes.
 * All other ranks are no-ops.
 */
void write_binary_single(const char *base_path, const char *suffix,
                         const void *buf, size_t bytes,
                         int rank, int writer_rank);

/**
 * Multi-writer: rank 0 creates the file, then every rank writes
 * @p chunk_size bytes at offset @p rank * @p chunk_size.
 *
 * Collective: all ranks MUST call this together (rank 0 must create
 * the file before others open it).  On MPI backends a barrier/fsync
 * is expected before this call.
 */
void write_binary_multi(const char *base_path, const char *suffix,
                        const void *buf, size_t chunk_size,
                        int rank, int nranks);

/**
 * Accumulate @p buf element-wise into @p accum (in-place).
 * Both buffers have @p count elements of type @p dtype.
 * Caller must ensure @p accum is initialized to zero before first call.
 */
void binary_accumulate(void *accum, const void *buf, int count, data_type_t dtype);

/**
 * Divide each element of @p accum by @p divisor (in-place).
 * @p count elements of type @p dtype.
 */
void binary_average(void *accum, int count, data_type_t dtype, int divisor);

#endif /* BINARY_OUTPUT_H */
