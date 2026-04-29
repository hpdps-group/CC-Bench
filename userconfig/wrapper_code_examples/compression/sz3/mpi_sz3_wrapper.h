#ifndef MPI_SZ3_WRAPPER_H
#define MPI_SZ3_WRAPPER_H

#include <stddef.h>

/*
 * SZ3-backed MPI compression wrapper (dynamic-library linking approach).
 *
 * At runtime the wrapper dlopens libSZ3.so (or libSZ3c.so) and resolves
 * SZ_compress_args / SZ_decompress / free_buf via dlsym.
 *
 * Environment variables:
 *   MPI_SZ3_LIB_PATH  — full path to the SZ3 shared library (optional)
 *   MPI_SZ3_TYPE      — 0 = float, 1 = double  (default: 0)
 *   MPI_SZ3_EB_MODE   — 0 = ABS, 1 = REL       (default: 0)
 *   MPI_SZ3_ABS_EB    — absolute error bound    (default: 1e-6)
 *   MPI_SZ3_REL_EB    — relative error bound    (default: 0.01)
 */

int mpi_compress(void *input, size_t input_size,
                 void *output, size_t *output_size);
int mpi_decompress(void *input, size_t input_size,
                   void *output, size_t output_size);

#endif /* MPI_SZ3_WRAPPER_H */