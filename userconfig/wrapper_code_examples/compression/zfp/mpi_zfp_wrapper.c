/*
 * ZFP MPI compression wrapper — co-compilation approach.
 *
 * Approach
 * --------
 * The wrapper directly includes <zfp.h> and calls the ZFP high-level
 * compression / decompression API.  At build time the compiler needs
 * ZFP's include path (and optionally its source files or library).
 *
 * Build (co-compilation with ZFP sources — no pre-built library needed)
 * --------------------------------------------------------------------
 *   gcc -fPIC -shared -O2                                              \
 *       -I/path/to/userconfig/wrapper_code_examples/communication/include                 \
 *       -I/path/to/plugin_projects/compression/zfp/include             \
 *       -o libmpi_zfp_wrapper.so                                       \
 *       mpi_zfp_wrapper.c                                              \
 *       /path/to/zfp/src/bitstream.c                                    \
 *       /path/to/zfp/src/zfp.c                                          \
 *       /path/to/zfp/src/encode*.c                                      \
 *       /path/to/zfp/src/decode*.c                                      \
 *       /path/to/zfp/src/template/*.c                                   \
 *       /path/to/zfp/src/share/parallel.c
 *
 * Alternative build (link against pre-built libzfp):
 *   gcc -fPIC -shared -O2                                              \
 *       -I/path/to/userconfig/wrapper_code_examples/communication/include                 \
 *       -I/path/to/plugin_projects/compression/zfp/include             \
 *       -o libmpi_zfp_wrapper.so                                       \
 *       mpi_zfp_wrapper.c                                              \
 *       -L/path/to/zfp/build/lib -lzfp
 *
 * Run
 * ---
 *   LD_PRELOAD=./libmpi_zfp_wrapper.so mpirun -np 4 ./your_benchmark
 *
 * Environment variables (all optional)
 * ------------------------------------
 *   MPI_ZFP_TYPE   — 1=int32  2=int64  3=float  4=double  (default: 3)
 *   MPI_ZFP_MODE   — 1=expert  2=rate  3=precision  4=accuracy  5=reversible
 *                    (default: 2 = fixed-rate)
 *   MPI_ZFP_RATE   — bits per value in fixed-rate mode    (default: 16)
 *   MPI_ZFP_PREC   — precision in fixed-precision mode     (default: 10)
 *   MPI_ZFP_TOL    — tolerance in fixed-accuracy mode      (default: 1e-3)
 *
 * For expert mode (MPI_ZFP_MODE=1), set all four via:
 *   MPI_ZFP_MINBITS MPI_ZFP_MAXBITS MPI_ZFP_MAXPREC MPI_ZFP_MINEXP
 */

#include "mpi_compress_tools.h"

#include <zfp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*===========================================================================*
 * Helper — read ZFP configuration from environment                           *
 *===========================================================================*/
static zfp_type   get_zfp_type(void)
{
    const char *env = getenv("MPI_ZFP_TYPE");
    if (!env) return zfp_type_float;
    int t = atoi(env);
    if (t >= 1 && t <= 4) return (zfp_type)t;
    return zfp_type_float;
}

static int configure_stream(zfp_stream *zfp, zfp_type type)
{
    const char *env;
    int mode = 2; /* fixed-rate by default */

    env = getenv("MPI_ZFP_MODE");
    if (env) mode = atoi(env);

    switch (mode) {
    case 1: { /* expert */
        uint minbits = 0, maxbits = 0, maxprec = 0;
        int  minexp  = 0;
        env = getenv("MPI_ZFP_MINBITS"); if (env) minbits = (uint)atoi(env);
        env = getenv("MPI_ZFP_MAXBITS"); if (env) maxbits = (uint)atoi(env);
        env = getenv("MPI_ZFP_MAXPREC"); if (env) maxprec = (uint)atoi(env);
        env = getenv("MPI_ZFP_MINEXP");  if (env) minexp  = atoi(env);
        return zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp)
               ? 0 : -1;
    }
    case 2: { /* fixed-rate */
        double rate = 16.0;
        env = getenv("MPI_ZFP_RATE"); if (env) rate = atof(env);
        zfp_stream_set_rate(zfp, rate, type, 1, 0);
        return 0;
    }
    case 3: { /* fixed-precision */
        uint prec = 10;
        env = getenv("MPI_ZFP_PREC"); if (env) prec = (uint)atoi(env);
        zfp_stream_set_precision(zfp, prec);
        return 0;
    }
    case 4: { /* fixed-accuracy */
        double tol = 1e-3;
        env = getenv("MPI_ZFP_TOL"); if (env) tol = atof(env);
        zfp_stream_set_accuracy(zfp, tol);
        return 0;
    }
    case 5: /* reversible (lossless) */
        zfp_stream_set_reversible(zfp);
        return 0;
    default:
        return -1;
    }
}

/*===========================================================================*
 * mpi_compress                                                              *
 *===========================================================================*/
int mpi_compress(void *input, size_t input_size,
                 void *output, size_t *output_size)
{
    if (!input || !output || !output_size)
        return -1;
    if (input_size == 0) {
        *output_size = 0;
        return 0;
    }

    zfp_type type = get_zfp_type();
    size_t elem_size = zfp_type_size(type);
    size_t n = input_size / elem_size;
    if (n == 0) return -1;

    /* Allocate a buffer large enough for the worst-case compressed output */
    size_t buf_cap = input_size + 1024;    /* generous margin */
    void *buf = malloc(buf_cap);
    if (!buf) return -1;

    /* Set up ZFP stream with the raw buffer as backing */
    bitstream *bs   = stream_open(buf, buf_cap);
    zfp_stream *zfp = zfp_stream_open(bs);
    if (!zfp) { stream_close(bs); free(buf); return -1; }

    if (configure_stream(zfp, type) != 0) {
        zfp_stream_close(zfp);
        stream_close(bs);
        free(buf);
        return -1;
    }

    /* Describe the 1-D input field */
    zfp_field *field = zfp_field_1d(input, type, n);
    if (!field) {
        zfp_stream_close(zfp);
        stream_close(bs);
        free(buf);
        return -1;
    }

    /* Compress */
    size_t compressed_size = zfp_compress(zfp, field);
    if (compressed_size == 0) {
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(bs);
        free(buf);
        return -1;
    }

    /* Copy compressed data into the caller's output buffer */
    if (compressed_size > *output_size) {
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(bs);
        free(buf);
        return -1;
    }

    memcpy(output, buf, compressed_size);
    *output_size = compressed_size;

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    free(buf);
    return 0;
}

/*===========================================================================*
 * mpi_decompress                                                            *
 *===========================================================================*/
int mpi_decompress(void *input, size_t input_size,
                   void *output, size_t output_size)
{
    if (!input || !output)
        return -1;
    if (input_size == 0 || output_size == 0)
        return 0;

    zfp_type type = get_zfp_type();
    size_t elem_size = zfp_type_size(type);
    size_t n = output_size / elem_size;
    if (n == 0) return -1;

    /*
     * Create a bitstream that reads from the compressed input,
     * and a zfp_stream configured the same way as during compression.
     */
    bitstream *bs   = stream_open(input, input_size);
    zfp_stream *zfp = zfp_stream_open(bs);
    if (!zfp) { stream_close(bs); return -1; }

    if (configure_stream(zfp, type) != 0) {
        zfp_stream_close(zfp);
        stream_close(bs);
        return -1;
    }

    /* The output field describes the shape of the decompressed data.
     * zfp_decompress will write directly into the caller's output buffer. */
    zfp_field *field = zfp_field_1d(output, type, n);
    if (!field) {
        zfp_stream_close(zfp);
        stream_close(bs);
        return -1;
    }

    size_t result = zfp_decompress(zfp, field);
    if (result == 0) {
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(bs);
        return -1;
    }

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
}