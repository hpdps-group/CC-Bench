/*
 * mpi_sz3_wrapper.c  —  SZ3 compression wrapper (dlopen mode)
 *
 * What this demonstrates (compression_kernel_selection mode 2)
 * --------------------
 * You have a prebuilt libSZ3.so.  Your wrapper dlopen(3)s it at
 * runtime and calls its functions.  No compile-time dependency on
 * SZ3 headers or libraries — only -ldl is needed.
 *
 * Build
 * -----
 *   mode: 2
 *   source_files: [".../mpi_sz3_wrapper.c"]
 *   libraries:    ["-ldl"]
 *   output_name:  "libtest_sz3.so"
 *
 * Run
 * ---
 *   LD_PRELOAD=libtest_sz3.so
 *   LD_LIBRARY_PATH=/path/to/SZ3/build/lib:$LD_LIBRARY_PATH
 *   mpirun -np 4 ./your_benchmark
 *
 * Environment variables (read at compression time)
 * ------------------------------------------------------------
 *   MPI_SZ3_TYPE       — 0 = float, 1 = double  (default: 0)
 *   MPI_SZ3_EB_MODE    — 0 = ABS, 1 = REL       (default: 0)
 *   MPI_SZ3_ABS_EB     — absolute error bound   (default: 1e-6)
 *   MPI_SZ3_REL_EB     — relative error bound   (default: 0.01)
 */

#include "mpi_compress_tools.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*===========================================================================*
 * SZ3 function pointer typedefs (resolved via dlsym)                        *
 *===========================================================================*/
typedef unsigned char *(*sz_compress_args_fn)(
    int dataType, void *data, size_t *outSize,
    int errBoundMode,
    double absErrBound, double relBoundRatio, double pwrBoundRatio,
    size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

typedef void *(*sz_decompress_fn)(
    int dataType, unsigned char *bytes, size_t byteLength,
    size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

static sz_compress_args_fn sz_compress_args  = NULL;
static sz_decompress_fn    sz_decompress     = NULL;

/*===========================================================================*
 * Dynamic library loader — called once on first use                         *
 *===========================================================================*/
static int load_sz3_lib(void)
{
    static int loaded = 0;
    static int failed = 0;
    if (loaded) return 0;
    if (failed) return -1;

    void *handle = NULL;

    /* Try common names; LD_LIBRARY_PATH tells the loader where to look. */
    static const char *candidates[] = {
        "libSZ3c.so",
        "libSZ3.so",
        "libsz3c.so",
        "libsz3.so",
    };
    for (int i = 0; i < (int)(sizeof candidates / sizeof candidates[0]); i++) {
        handle = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            fprintf(stderr, "[SZ3 wrapper] loaded %s\n", candidates[i]);
            break;
        }
    }

    if (!handle) {
        fprintf(stderr, "[SZ3 wrapper] failed to load SZ3 library."
                        " Set LD_LIBRARY_PATH to the SZ3 lib directory.\n");
        failed = 1;
        return -1;
    }

    /* 3. Resolve symbols */
    sz_compress_args = (sz_compress_args_fn)dlsym(handle, "SZ_compress_args");
    sz_decompress    = (sz_decompress_fn)dlsym(handle, "SZ_decompress");

    if (!sz_compress_args || !sz_decompress) {
        fprintf(stderr, "[SZ3 wrapper] dlsym failed: %s\n", dlerror());
        dlclose(handle);
        failed = 1;
        return -1;
    }

    loaded = 1;
    return 0;
}

/*===========================================================================*
 * Helper — read SZ3 config from environment (cached after first call)       *
 *===========================================================================*/
static int get_config(int *data_type, int *eb_mode,
                      double *abs_eb, double *rel_eb)
{
    static int init = 0;
    static int s_data_type = 0;
    static int s_eb_mode   = 0;
    static double s_abs_eb = 1e-6;
    static double s_rel_eb = 0.01;

    if (!init) {
        const char *env;
        env = getenv("MPI_SZ3_TYPE");
        if (env) s_data_type = atoi(env);
        env = getenv("MPI_SZ3_EB_MODE");
        if (env) s_eb_mode   = atoi(env);
        env = getenv("MPI_SZ3_ABS_EB");
        if (env) s_abs_eb    = atof(env);
        env = getenv("MPI_SZ3_REL_EB");
        if (env) s_rel_eb    = atof(env);
        init = 1;
    }

    *data_type = s_data_type;
    *eb_mode   = s_eb_mode;
    *abs_eb    = s_abs_eb;
    *rel_eb    = s_rel_eb;
    return 0;
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

    if (load_sz3_lib() != 0)
        return -1;

    int data_type, eb_mode;
    double abs_eb, rel_eb;
    get_config(&data_type, &eb_mode, &abs_eb, &rel_eb);

    size_t elem_size = (data_type == 1) ? sizeof(double) : sizeof(float);
    size_t n = input_size / elem_size;
    if (n == 0) return -1;

    size_t out_capacity = *output_size;
    *output_size = n;

    unsigned char *compressed = sz_compress_args(
        data_type, input, output_size,
        eb_mode, abs_eb, rel_eb, 0.0,
        0, 0, 0, 0, n
    );

    if (!compressed) {
        *output_size = 0;
        return -1;
    }

    if (*output_size > out_capacity) {
        free(compressed);
        return -1;
    }

    memcpy(output, compressed, *output_size);
    free(compressed);
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

    if (load_sz3_lib() != 0)
        return -1;

    int data_type, eb_mode;
    double abs_eb, rel_eb;
    get_config(&data_type, &eb_mode, &abs_eb, &rel_eb);
    (void)eb_mode;
    (void)abs_eb;
    (void)rel_eb;

    size_t elem_size = (data_type == 1) ? sizeof(double) : sizeof(float);
    size_t n = output_size / elem_size;

    void *decompressed = sz_decompress(
        data_type, (unsigned char *)input, input_size,
        0, 0, 0, 0, n
    );

    if (!decompressed)
        return -1;

    memcpy(output, decompressed, output_size);
    free(decompressed);
    return 0;
}