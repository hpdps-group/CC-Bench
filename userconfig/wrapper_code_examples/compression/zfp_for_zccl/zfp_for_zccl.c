/*
 * zfp_for_zccl.c —  LD_PRELOAD wrapper: replace ZCCL's internal
 *                   compression with ZFP.
 *
 * Intercepts all ZCCL float compress/decompress functions that go
 * through PLT (verified: libZCCL.so uses @plt for every call).
 *
 * The wrapper embeds a ZFP header (via zfp_write_header) so that
 * decompression is self-describing — no extra environment needed
 * on the receive side.
 *
 * Compressed layout:
 *   [0..7]    size_t  total_zfp_data_size  (ZFP header + blocks; excludes this header)
 *   [8..]     ZFP data                     (written by zfp_write_header + zfp_compress)
 *
 * Build:
 *   gcc -fPIC -shared -O2 -o libzfp_for_zccl.so           \
 *       zfp_for_zccl.c                                    \
 *       -I /path/to/zfp/include                           \
 *       -L /path/to/zfp/lib -lzfp
 *
 * Or co-compile with ZFP sources (no prebuilt lib needed):
 *   gcc -fPIC -shared -O2 -o libzfp_for_zccl.so           \
 *       zfp_for_zccl.c                                    \
 *       -I /path/to/zfp/include                           \
 *       /path/to/zfp/src/bitstream.c                      \
 *       /path/to/zfp/src/zfp.c                            \
 *       /path/to/zfp/src/encode*.c                        \
 *       /path/to/zfp/src/decode*.c                        \
 *       /path/to/zfp/src/template/*.c                     \
 *       /path/to/zfp/src/share/parallel.c
 *
 * Run:
 *   LD_PRELOAD=./libzfp_for_zccl.so:...existing preloads... \
 *   mpirun -np 4 ./benchmark
 *
 * Environment variables (only read on the compress side; header carries config):
 *   ZFP_MODE   - 2=rate 3=precision 4=accuracy 5=reversible  (default: 2)
 *   ZFP_RATE   - bits/value for fixed-rate mode  (default: 16)
 *   ZFP_PREC   - precision for fixed-precision   (default: 10)
 *   ZFP_TOL    - tolerance for fixed-accuracy    (default: 1e-3)
 *   ZFP_VERBOSE - if set, print debug info
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zfp.h>

/* ------------------------------------------------------------------ *
 *  Cached config from environment (one-shot)                          *
 * ------------------------------------------------------------------ */
static int  g_mode      = 2;      /* fixed-rate */
static double g_rate    = 16.0;
static uint  g_prec     = 10;
static double g_tol     = 1e-3;
static int  g_verbose   = 0;
static int  g_loaded    = 0;

static void load_config(void)
{
    if (g_loaded) return;
    const char *env;

    env = getenv("ZFP_MODE");
    if (env) g_mode = atoi(env);
    env = getenv("ZFP_RATE");
    if (env) g_rate = atof(env);
    env = getenv("ZFP_PREC");
    if (env) g_prec = (uint)atoi(env);
    env = getenv("ZFP_TOL");
    if (env) g_tol = atof(env);
    if (getenv("ZFP_VERBOSE")) g_verbose = 1;

    g_loaded = 1;
}

/* ------------------------------------------------------------------ *
 *  Configure a zfp_stream from the cached environment settings        *
 * ------------------------------------------------------------------ */
static void configure_stream(zfp_stream *stream, zfp_type type)
{
    switch (g_mode) {
    case 1: /* expert — use defaults */
        break;
    case 3: /* fixed-precision */
        zfp_stream_set_precision(stream, g_prec);
        break;
    case 4: /* fixed-accuracy */
        zfp_stream_set_accuracy(stream, g_tol);
        break;
    case 5: /* reversible */
        zfp_stream_set_reversible(stream);
        break;
    default: /* 2 — fixed-rate */
        zfp_stream_set_rate(stream, g_rate, type, 1, 0);
        break;
    }
}

/* ------------------------------------------------------------------ *
 *  Internal helpers — the actual ZFP compress / decompress            *
 * ------------------------------------------------------------------ */

/* Compress via ZFP and write [8-byte size][ZFP data] to out.
 * Returns total bytes written (payload + 8), or 0 on error. */
static size_t internal_compress(unsigned char *out, const float *data,
                                size_t nb_ele, float abs_err)
{
    (void)abs_err;  /* ZFP uses its own tolerance from env vars */

    load_config();

    /* Temporary buffer for ZFP output (worst case) */
    size_t cap = nb_ele * sizeof(float) + 4096;
    void *tmp = malloc(cap);
    if (!tmp) return 0;

    bitstream  *bs   = stream_open(tmp, cap);
    zfp_stream *zfp  = zfp_stream_open(bs);
    if (!zfp) { stream_close(bs); free(tmp); return 0; }

    configure_stream(zfp, zfp_type_float);

    zfp_field *field = zfp_field_1d((void *)data, zfp_type_float, nb_ele);
    if (!field) { zfp_stream_close(zfp); stream_close(bs); free(tmp); return 0; }

    if (g_verbose)
        fprintf(stderr, "[zfp_for_zccl] compress nb_ele=%zu, mode=%d\n", nb_ele, g_mode);

    /* Write ZFP self-describing header so decompress can auto-configure */
    zfp_write_header(zfp, field, ZFP_HEADER_FULL);

    /* Compress */
    size_t ret = zfp_compress(zfp, field);
    if (ret == 0) {
        if (g_verbose)
            fprintf(stderr, "[zfp_for_zccl] zfp_compress failed (ret=0)\n");
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(bs);
        free(tmp);
        return 0;
    }

    /* Total bytes in bitstream = header + compressed blocks */
    size_t total = (size_t)stream_rtell(bs);

    if (g_verbose)
        fprintf(stderr, "[zfp_for_zccl] compressed %zu -> %zu bytes (ratio %.1f)\n",
                nb_ele * sizeof(float), total,
                (double)(nb_ele * sizeof(float)) / (double)total);

    /* [8-byte size][ZFP payload] */
    memcpy(out, &total, sizeof(size_t));
    memcpy(out + sizeof(size_t), tmp, total);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    free(tmp);

    return total + sizeof(size_t);
}

/* Decompress data produced by internal_compress().
 * Writes nb_ele floats to out.  Returns 0 on success, -1 on error. */
static int internal_decompress(float *out, size_t nb_ele,
                               const unsigned char *in)
{
    load_config();

    size_t total;
    memcpy(&total, in, sizeof(size_t));

    bitstream  *bs   = stream_open((void *)(in + sizeof(size_t)), total);
    zfp_stream *zfp  = zfp_stream_open(bs);
    if (!zfp) return -1;

    /* Create field pointing to our output buffer; zfp_read_header will
     * populate dimensions/type from the header without touching data ptr. */
    zfp_field *field = zfp_field_1d(out, zfp_type_float, nb_ele);
    if (!field) { zfp_stream_close(zfp); stream_close(bs); return -1; }

    /* Auto-configure stream & field from embedded header */
    uint header_read = zfp_read_header(zfp, field, ZFP_HEADER_FULL);
    if (!header_read) {
        if (g_verbose)
            fprintf(stderr, "[zfp_for_zccl] zfp_read_header failed\n");
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(bs);
        return -1;
    }

    /* Decompress directly into out (field->data == out) */
    size_t ret = zfp_decompress(zfp, field);
    if (ret == 0) {
        if (g_verbose)
            fprintf(stderr, "[zfp_for_zccl] zfp_decompress failed\n");
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
    if (nbEle == 0) { *outSize = 0; return; }
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
    if (nbEle == 0) { *outSize = 0; return; }
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
    if (nbEle == 0) { *outSize = 0; return NULL; }

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
    return ZCCL_float_openmp_threadblock(oriData, outSize, absErrBound, nbEle, blockSize);
}

/* ---------- compress: split-record ---------- */
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
    if (nbEle == 0) { *outSize = 0; return; }
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
    if (nbEle == 0) return;
    internal_decompress(newData, nbEle, cmpBytes);
}

void ZCCL_float_decompress_single_thread_arg(float *newData,
                                              size_t nbEle,
                                              float absErrBound,
                                              int blockSize,
                                              unsigned char *cmpBytes)
{
    (void)absErrBound; (void)blockSize;
    if (nbEle == 0) return;
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
    if (nbEle == 0) { *newData = NULL; return; }

    *newData = (float *)malloc(nbEle * sizeof(float));
    if (!*newData) return;
    internal_decompress(*newData, nbEle, cmpBytes);
}

/* ---------- homomorphic add (NOT supported with ZFP) ---------- */
void ZCCL_float_homomophic_add_openmp_threadblock(
    unsigned char *final_cmpBytes, size_t *final_cmpSize,
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, unsigned char *cmpBytes2)
{
    (void)final_cmpBytes; (void)final_cmpSize;
    (void)nbEle; (void)absErrBound; (void)blockSize;
    (void)cmpBytes; (void)cmpBytes2;
    fprintf(stderr, "[zfp_for_zccl] ERROR: homomorphic add not supported "
                    "with ZFP compressor.  Do not use ZCCL_MODE with HO.\n");
    abort();
}

void ZCCL_float_homomophic_add_single_thread(
    unsigned char *final_cmpBytes, size_t *final_cmpSize,
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, unsigned char *cmpBytes2)
{
    ZCCL_float_homomophic_add_openmp_threadblock(
        final_cmpBytes, final_cmpSize, nbEle, absErrBound, blockSize,
        cmpBytes, cmpBytes2);
}
