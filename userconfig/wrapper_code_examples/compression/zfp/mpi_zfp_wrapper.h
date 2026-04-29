#ifndef MPI_ZFP_WRAPPER_H
#define MPI_ZFP_WRAPPER_H

#include <stddef.h>

/*
 * ZFP-backed MPI compression wrapper (co-compilation approach).
 *
 * The wrapper directly includes <zfp.h> and calls zfp_compress /
 * zfp_decompress.  Build with:
 *   -I /path/to/zfp/include
 * and either link against libzfp or compile selected zfp source files
 * together with this translation unit.
 *
 * Environment variables:
 *   MPI_ZFP_TYPE   — 1=int32  2=int64  3=float  4=double  (default: 3)
 *   MPI_ZFP_MODE   — 1=expert  2=rate  3=precision  4=accuracy  5=reversible
 *                    (default: 2 = fixed-rate)
 *   MPI_ZFP_RATE   — bits per value in fixed-rate mode    (default: 16)
 *   MPI_ZFP_PREC   — precision in fixed-precision mode     (default: 10)
 *   MPI_ZFP_TOL    — tolerance in fixed-accuracy mode      (default: 1e-3)
 */

int mpi_compress(void *input, size_t input_size,
                 void *output, size_t *output_size);
int mpi_decompress(void *input, size_t input_size,
                   void *output, size_t output_size);

#endif /* MPI_ZFP_WRAPPER_H */