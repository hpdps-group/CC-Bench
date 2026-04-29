#ifndef PLAIN_MPI_COMPRESS_H
#define PLAIN_MPI_COMPRESS_H

/*
 * plain_mpi_compress — MPI-compression benchmark layer.
 *
 * Provides the env-var-based compression bypass (MPI_COMPRESS_DISABLE)
 * and weak pass-through defaults for mpi_compress / mpi_decompress.
 *
 * User wrappers that provide their own strong mpi_compress / mpi_decompress
 * do NOT need this header; include only mpi_compress_tools.h for the
 * function declarations and point-to-point primitives.
 */

#include "mpi_compress_tools.h"

int compression_enabled(void);

#endif /* PLAIN_MPI_COMPRESS_H */