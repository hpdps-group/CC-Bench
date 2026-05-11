/*
 * sz3_for_zccl.c —  LD_PRELOAD wrapper: replace ZCCL's internal
 *                   compression with SZ3.
 *
 * Intercepts all ZCCL float compress/decompress functions that go
 * through PLT (verified: libZCCL.so uses @plt for every call).
 *
 * Compressed layout:
 *   [0..7]    size_t  sz3_payload_size    (SZ3 output, NOT counting this header)
 *   [8..]     SZ3 compressed data         (produced by SZ_compress_args)
 *
 * Build:
 *   gcc -fPIC -shared -O2 -o libsz3_for_zccl.so \
 *       sz3_for_zccl.c -ldl
 *
 * Run:
 *   LD_PRELOAD=./libsz3_for_zccl.so:...existing preloads... \
 *   LD_LIBRARY_PATH=/path/to/SZ3/lib:$LD_LIBRARY_PATH \
 *   mpirun -np 4 ./benchmark
 *
 * Environment variables:
 *   SZ3_EB_MODE  — 0 = ABS (default), 1 = REL
 *   SZ3_ABS_EB   — absolute error bound (default: use ZCCL's absErrBound)
 *   SZ3_REL_EB   — relative error bound when SZ3_EB_MODE=1  (default: 0.01)
 *   SZ3_VERBOSE  — if set, print debug info to stderr
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 *  SZ3 function pointer types (C API)                                 *
 * ------------------------------------------------------------------ */
typedef unsigned char *(*sz_compress_args_t)(
    int dataType, void *data, size_t *outSize,
    int errBoundMode,
    double absErrBound, double relBoundRatio, double pwrBoundRatio,
    size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

typedef void *(*sz_decompress_t)(
    int dataType, unsigned char *bytes, size_t byteLength,
    size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

typedef void (*sz_free_buf_t)(void *p);

/* ------------------------------------------------------------------ *
 *  Resolved SZ3 symbols (dlopen once)                                 *
 * ------------------------------------------------------------------ */
static sz_compress_args_t real_sz_compress = NULL;
static sz_decompress_t    real_sz_decompress = NULL;
static sz_free_buf_t      real_sz_free = NULL;

/* Cached config from environment */
static int    g_eb_mode   = 0;       /* ABS */
static double g_abs_eb    = 0.0;     /* 0 = use ZCCL's absErrBound   */
static double g_rel_eb    = 0.01;
static int    g_verbose   = 0;
static int    g_loaded    = 0;

/* ------------------------------------------------------------------ *
 *  Load SZ3 shared library (one shot)                                 *
 * ------------------------------------------------------------------ */
static int load_sz3(void)
{
    if (g_loaded) return 0;
    if (g_loaded < 0) return -1;      /* previously failed */

    const char *libs[] = {"libSZ3c.so", "libSZ3.so", NULL};
    void *h = NULL;
    for (int i = 0; libs[i]; i++) {
        h = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            if (g_verbose)
                fprintf(stderr, "[sz3_for_zccl] loaded %s\n", libs[i]);
            break;
        }
    }
    if (!h) {
        fprintf(stderr, "[sz3_for_zccl] FATAL: no SZ3 library found (tried libSZ3c.so, libSZ3.so)\n");
        g_loaded = -1;
        return -1;
    }

    real_sz_compress = (sz_compress_args_t)dlsym(h, "SZ_compress_args");
    real_sz_decompress = (sz_decompress_t)dlsym(h, "SZ_decompress");
    real_sz_free = (sz_free_buf_t)dlsym(h, "free_buf");

    if (!real_sz_compress || !real_sz_decompress || !real_sz_free) {
        fprintf(stderr, "[sz3_for_zccl] FATAL: dlsym SZ3 symbols failed\n");
        g_loaded = -1;
        return -1;
    }

    const char *env;
    env = getenv("SZ3_EB_MODE");
    if (env) g_eb_mode = atoi(env);
    env = getenv("SZ3_ABS_EB");
    if (env) g_abs_eb = atof(env);
    env = getenv("SZ3_REL_EB");
    if (env) g_rel_eb = atof(env);
    if (getenv("SZ3_VERBOSE")) g_verbose = 1;

    g_loaded = 1;
    return 0;
}

/* ------------------------------------------------------------------ *
 *  Internal helpers — the actual SZ3 compress / decompress            *
 * ------------------------------------------------------------------ */

/* Compress float data via SZ3 and write [8-byte size][payload] to out.
 * Returns the total bytes written to out (payload + 8), or 0 on error. */
static size_t internal_compress(unsigned char *out, const float *data,
                                size_t nb_ele, float abs_err)
{
    /* Use env-override when set, otherwise ZCCL's absErrBound */
    double eb = g_abs_eb > 0.0 ? g_abs_eb : (double)abs_err;
    size_t sz3_sz = 0;
    unsigned char *cmp = real_sz_compress(
        0, (void *)data, &sz3_sz,
        g_eb_mode, eb, g_rel_eb, 0.0,
        0, 0, 0, 0, nb_ele
    );

    if (!cmp || sz3_sz == 0) {
        if (g_verbose)
            fprintf(stderr, "[sz3_for_zccl] SZ_compress_args returned empty\n");
        return 0;
    }

    /* [8-byte header][SZ3 payload] */
    memcpy(out, &sz3_sz, sizeof(size_t));
    memcpy(out + sizeof(size_t), cmp, sz3_sz);
    size_t total = sz3_sz + sizeof(size_t);

    real_sz_free(cmp);
    return total;
}

/* Decompress data produced by internal_compress().
 * Writes nb_ele floats to out.  Returns 0 on success, -1 on error. */
static int internal_decompress(float *out, size_t nb_ele,
                               const unsigned char *in)
{
    size_t cmp_sz;
    memcpy(&cmp_sz, in, sizeof(size_t));

    float *dec = (float *)real_sz_decompress(
        0, (unsigned char *)in + sizeof(size_t), cmp_sz,
        0, 0, 0, 0, nb_ele
    );

    if (!dec) {
        if (g_verbose)
            fprintf(stderr, "[sz3_for_zccl] SZ_decompress failed\n");
        return -1;
    }

    memcpy(out, dec, nb_ele * sizeof(float));
    real_sz_free(dec);
    return 0;
}

/* ------------------------------------------------------------------ *
 *  Intercepted functions                                              *
 * ------------------------------------------------------------------ */

/* ---------- compress: pre-allocated output buffer ---------- */
void ZCCL_float_openmp_threadblock_arg(unsigned char *outputBytes,
                                       float *oriData,
                                       size_t *outSize,
                                       float absErrBound,
                                       size_t nbEle,
                                       int blockSize)
{
    (void)blockSize;
    if (load_sz3() != 0) { *outSize = 0; return; }
    if (nbEle == 0)       { *outSize = 0; return; }
    *outSize = internal_compress(outputBytes, oriData, nbEle, absErrBound);
}

void ZCCL_float_single_thread_arg(unsigned char *outputBytes,
                                  float *oriData,
                                  size_t *outSize,
                                  float absErrBound,
                                  size_t nbEle,
                                  int blockSize)
{
    (void)blockSize;
    if (load_sz3() != 0) { *outSize = 0; return; }
    if (nbEle == 0)       { *outSize = 0; return; }
    *outSize = internal_compress(outputBytes, oriData, nbEle, absErrBound);
}

/* ---------- compress: self-allocated output buffer ---------- */
unsigned char *ZCCL_float_openmp_threadblock(float *oriData,
                                              size_t *outSize,
                                              float absErrBound,
                                              size_t nbEle,
                                              int blockSize)
{
    (void)blockSize;
    if (load_sz3() != 0) { *outSize = 0; return NULL; }
    if (nbEle == 0)       { *outSize = 0; return NULL; }

    /* Allocate maximally sized buffer (same as ZCCL does internally) */
    size_t cap = nbEle * sizeof(float);
    unsigned char *buf = (unsigned char *)malloc(cap);
    if (!buf) { *outSize = 0; return NULL; }

    *outSize = internal_compress(buf, oriData, nbEle, absErrBound);
    if (*outSize == 0) { free(buf); return NULL; }
    return buf;
}

unsigned char *ZCCL_float_openmp_threadblock_randomaccess(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
    /* Same as above — the "randomaccess" property is irrelevant for SZ3 */
    return ZCCL_float_openmp_threadblock(oriData, outSize, absErrBound, nbEle, blockSize);
}

/* ---------- compress: split-record (extra params ignored) ---------- */
void ZCCL_float_single_thread_arg_split_record(unsigned char *outputBytes,
                                                float *oriData,
                                                size_t *outSize,
                                                float absErrBound,
                                                size_t nbEle,
                                                int blockSize,
                                                unsigned char *chunk_arr,
                                                size_t chunk_iter)
{
    (void)blockSize; (void)chunk_arr; (void)chunk_iter;
    if (load_sz3() != 0) { *outSize = 0; return; }
    if (nbEle == 0)       { *outSize = 0; return; }
    *outSize = internal_compress(outputBytes, oriData, nbEle, absErrBound);
}

/* ---------- decompress: pre-allocated output buffer ---------- */
void ZCCL_float_decompress_openmp_threadblock_arg(float *newData,
                                                   size_t nbEle,
                                                   float absErrBound,
                                                   int blockSize,
                                                   unsigned char *cmpBytes)
{
    (void)absErrBound; (void)blockSize;
    if (load_sz3() != 0) return;
    if (nbEle == 0)      return;
    internal_decompress(newData, nbEle, cmpBytes);
}

void ZCCL_float_decompress_single_thread_arg(float *newData,
                                              size_t nbEle,
                                              float absErrBound,
                                              int blockSize,
                                              unsigned char *cmpBytes)
{
    (void)absErrBound; (void)blockSize;
    if (load_sz3() != 0) return;
    if (nbEle == 0)      return;
    internal_decompress(newData, nbEle, cmpBytes);
}

/* ---------- decompress: self-allocated output buffer ---------- */
void ZCCL_float_decompress_openmp_threadblock(float **newData,
                                               size_t nbEle,
                                               float absErrBound,
                                               int blockSize,
                                               unsigned char *cmpBytes)
{
    (void)absErrBound; (void)blockSize;
    if (load_sz3() != 0) { *newData = NULL; return; }
    if (nbEle == 0)       { *newData = NULL; return; }

    *newData = (float *)malloc(nbEle * sizeof(float));
    if (!*newData) return;
    internal_decompress(*newData, nbEle, cmpBytes);
}

/* ---------- homomorphic add (NOT supported with SZ3) ---------- */
void ZCCL_float_homomophic_add_openmp_threadblock(
    unsigned char *final_cmpBytes, size_t *final_cmpSize,
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, unsigned char *cmpBytes2)
{
    (void)final_cmpBytes; (void)final_cmpSize;
    (void)nbEle; (void)absErrBound; (void)blockSize;
    (void)cmpBytes; (void)cmpBytes2;
    fprintf(stderr, "[sz3_for_zccl] ERROR: homomorphic add not supported "
                    "with SZ3 compressor.  Do not use ZCCL_MODE with HO.\n");
    abort();
}

void ZCCL_float_homomophic_add_single_thread(
    unsigned char *final_cmpBytes, size_t *final_cmpSize,
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, unsigned char *cmpBytes2)
{
    /* Same fatal error */
    ZCCL_float_homomophic_add_openmp_threadblock(
        final_cmpBytes, final_cmpSize, nbEle, absErrBound, blockSize,
        cmpBytes, cmpBytes2);
}
