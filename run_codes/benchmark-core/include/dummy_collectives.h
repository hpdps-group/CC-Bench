/**
 * dummy_collectives.h — CPU-based simulation of NCCL collectives
 * for reference generation.
 *
 * These functions compute the expected result of a collective operation
 * entirely on CPU, using only the data source (file or pattern) and
 * the number of simulated ranks.  No NCCL, CUDA, or multi-process
 * coordination is needed.
 *
 * Each process (rank) independently computes the reference for its own
 * position, making this safe under LD_PRELOAD interception.
 */

#ifndef DUMMY_COLLECTIVES_H
#define DUMMY_COLLECTIVES_H

#include <stddef.h>
#include "utils.h"

/**
 * Load one simulated rank's input data from file or pattern.
 *
 * @param input_file   Path to binary data file, or NULL for pattern
 * @param datatype     Element data type (TYPE_FLOAT, TYPE_DOUBLE, etc.)
 * @param pattern_type Buffer fill pattern (ignored when input_file is set)
 * @param buf          Output buffer (must hold at least `bytes`)
 * @param bytes        Number of bytes per rank
 * @param sim_rank     Which simulated rank to load (0 .. nranks-1)
 * @param nranks       Total number of simulated ranks
 */
void dummy_load_rank_data(const char *input_file,
                          data_type_t datatype,
                          pattern_type_t pattern_type,
                          void *buf, size_t bytes,
                          int sim_rank, int nranks);

/**
 * CPU allreduce: element-wise sum across all nranks.
 *
 * Loads each simulated rank's data incrementally (one at a time) and
 * accumulates into `ref`.  Memory usage: O(bytes) per iteration, not
 * O(bytes * nranks).
 *
 * @param ref          Output reference buffer (allreduce result, same for all ranks)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count        Number of elements per rank
 * @param nranks       Total number of ranks
 */
void dummy_allreduce(void *ref,
                     const char *input_file,
                     data_type_t datatype,
                     pattern_type_t pattern_type,
                     int count, int nranks);

/**
 * CPU alltoall: rearrange data as expected.
 *
 * For each source rank s, loads that rank's full send buffer and
 * extracts the chunk destined for the calling rank.
 *
 * @param ref          Output reference buffer (per-rank recv data)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count_per    Elements per peer (sendcount / recvcount)
 * @param nranks       Total number of ranks
 * @param rank         Calling rank (0 .. nranks-1)
 */
void dummy_alltoall(void *ref,
                    const char *input_file,
                    data_type_t datatype,
                    pattern_type_t pattern_type,
                    int count_per, int nranks, int rank);

/**
 * CPU allgather: concatenate all ranks' data.
 *
 * @param ref          Output buffer (nranks * count elements)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count        Number of elements per rank
 * @param nranks       Total number of ranks
 * @param rank         Calling rank (ignored for allgather, result is same for all)
 */
void dummy_allgather(void *ref,
                     const char *input_file,
                     data_type_t datatype,
                     pattern_type_t pattern_type,
                     int count, int nranks);

/**
 * CPU reduce: element-wise sum to root rank.
 *
 * @param ref          Output reference buffer (root gets the result, others unused)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count        Number of elements per rank
 * @param nranks       Total number of ranks
 * @param root         Root rank that receives the result
 */
void dummy_reduce(void *ref,
                  const char *input_file,
                  data_type_t datatype,
                  pattern_type_t pattern_type,
                  int count, int nranks, int root);

/**
 * CPU scatter: root sends a chunk of data to each rank.
 *
 * @param ref          Output reference buffer (per-rank chunk)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count        Number of elements each rank receives
 * @param nranks       Total number of ranks
 * @param root         Root rank
 * @param rank         Calling rank
 */
void dummy_scatter(void *ref,
                   const char *input_file,
                   data_type_t datatype,
                   pattern_type_t pattern_type,
                   int count, int nranks, int root, int rank);

/**
 * CPU gather: all ranks send data to root.
 *
 * @param ref          Output reference buffer (root gets nranks*count elements)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count        Number of elements per rank
 * @param nranks       Total number of ranks
 * @param root         Root rank
 */
void dummy_gather(void *ref,
                  const char *input_file,
                  data_type_t datatype,
                  pattern_type_t pattern_type,
                  int count, int nranks, int root);

/**
 * CPU reduce_scatter: element-wise sum then scatter.
 *
 * @param ref          Output reference buffer (per-rank chunk of the result)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count        Number of elements the calling rank receives
 * @param nranks       Total number of ranks
 * @param rank         Calling rank
 */
void dummy_reduce_scatter(void *ref,
                          const char *input_file,
                          data_type_t datatype,
                          pattern_type_t pattern_type,
                          int count, int nranks, int rank);

/**
 * CPU broadcast: root sends data to all ranks.
 *
 * @param ref          Output reference buffer (same for all ranks)
 * @param input_file   Data source file (NULL = pattern)
 * @param datatype     Element type
 * @param pattern_type Pattern (used when input_file is NULL)
 * @param count        Number of elements
 * @param nranks       Total number of ranks
 * @param root         Root rank
 */
void dummy_bcast(void *ref,
                 const char *input_file,
                 data_type_t datatype,
                 pattern_type_t pattern_type,
                 int count, int nranks, int root);

#endif /* DUMMY_COLLECTIVES_H */
