/*
 * perfcoccl.c  —  LD_PRELOAD wrapper that profiles COCCL
 *                  (NCCL + compression extensions) operations.
 *
 * Interception groups:
 *   1. ncclSend / ncclRecv                — point-to-point
 *   2. ncclCompress / ncclDecompress      — lossy compression / decompression
 *   3. ncclDecompressReduce               — decompress + reduce fused
 *   4. ncclDecompReduceComp               — decompress + reduce + compress fused
 *   5. ncclAllReduce / ncclBroadcast /
 *      ncclAllGather / ncclReduceScatter  — collective operations
 *
 * Relies on dlsym(RTLD_NEXT) to chain to the real implementation
 * (in the COCCL-modified libnccl.so).
 *
 * Build
 * -----
 *   cc -O2 -fPIC -shared -o libperfcoccl.so                                     \
 *         -I/path/to/run_codes/wrappers/include                                \
 *         -I/path/to/coccl/src/include                                         \
 *         -I/path/to/nccl/include                                              \
 *         perfcoccl.c                                                          \
 *         /path/to/run_codes/wrappers/src/perf_helper.c                        \
 *         /path/to/run_codes/wrappers/src/nccl/perf_helper_nccl.c              \
 *         -ldl -lm
 *
 * Run
 * ---
 *   LD_PRELOAD=libperfcoccl.so ./app
 *
 * Output
 * ------
 *   perf_nccl_<rank>.csv  — one per rank, auto-flushed at exit.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "perf_helper.h"
#include "nccl/perf_helper_nccl.h"

/* COCCL comm-operation enum (from compress.h) */
typedef enum {
  AlltoAll       = 0,
  AlltoAll_Inter = 1,
  AllReduce      = 2,
  AllReduce_Inter = 3,
  AllGather      = 4,
  AllGather_Inter = 5,
  ReduceScatter  = 6,
  ReduceScatter_Inter = 7,
  SendRecv       = 8,
  SendRecv_BWD   = 9,
} ncclCommOp_t;

/* ── helpers ────────────────────────────────────────────────────── */
static perf_state_t *get_state(void)  { return perf_nccl_get_tls(); }

/* perf_nccl_get_real() / PERF_NCCL_REAL provided by perf_helper_nccl.h */

/* ═══════════════════════════════════════════════════════════════════
 * Group 1 — NCCL point-to-point: send / receive
 * ═══════════════════════════════════════════════════════════════════ */

ncclResult_t ncclSend(const void *sendbuff, size_t count,
                       ncclDataType_t datatype, int peer,
                       ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*real)(const void *, size_t, ncclDataType_t,
                               int, ncclComm_t, cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclSend");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(sendbuff, count, datatype, peer, comm, stream);
  double t1 = perf_get_time();

  perf_notedown(get_state(), "ncclSend", t0, 4,
      (perf_attr_t[]){
          {"duration",  t1 - t0},
          {"msg_bytes", (double)count * nccl_type_size(datatype)},
          {"count",     (double)count},
          {"intra",     perf_nccl_is_intra(peer) ? 1.0 : 0.0},
      });
  return ret;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count,
                       ncclDataType_t datatype, int peer,
                       ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*real)(void *, size_t, ncclDataType_t,
                               int, ncclComm_t, cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclRecv");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(recvbuff, count, datatype, peer, comm, stream);
  double t1 = perf_get_time();

  int src = peer;
  perf_notedown(get_state(), "ncclRecv", t0, 4,
      (perf_attr_t[]){
          {"duration",  t1 - t0},
          {"msg_bytes", (double)count * nccl_type_size(datatype)},
          {"count",     (double)count},
          {"intra",     (src >= 0 && perf_nccl_is_intra(src)) ? 1.0 : 0.0},
      });
  return ret;
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 2 — COCCL compression / decompression
 * ═══════════════════════════════════════════════════════════════════ */

ncclResult_t ncclCompress(const void *orgbuff, void **compbuff,
                           const size_t orgChunkCount,
                           ncclDataType_t orgDatatype,
                           size_t *compChunkCount,
                           ncclDataType_t *compDatatype,
                           const size_t numChunks, const int rank,
                           ncclCommOp_t commOp, cudaStream_t stream) {
  static ncclResult_t (*real)(const void *, void **,
                               const size_t, ncclDataType_t,
                               size_t *, ncclDataType_t *,
                               const size_t, const int,
                               ncclCommOp_t, cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclCompress");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(orgbuff, compbuff, orgChunkCount, orgDatatype,
                           compChunkCount, compDatatype,
                           numChunks, rank, commOp, stream);
  double t1 = perf_get_time();

  double in_bytes  = (double)orgChunkCount * nccl_type_size(orgDatatype)
                     * (double)numChunks;
  double out_bytes = 0;
  if (compChunkCount && compDatatype)
    out_bytes = (double)(*compChunkCount)
                * nccl_type_size(*compDatatype) * (double)numChunks;

  perf_notedown(get_state(), "ncclCompress", t0, 5,
      (perf_attr_t[]){
          {"duration",          t1 - t0},
          {"input_bytes",       in_bytes},
          {"output_bytes",      out_bytes},
          {"compression_ratio", in_bytes > 0 ? out_bytes / in_bytes : 1.0},
          {"numChunks",         (double)numChunks},
      });
  return ret;
}

ncclResult_t ncclDecompress(void *decompbuff, const void *compbuff,
                             const size_t decompChunkCount,
                             ncclDataType_t decompDatatype,
                             const size_t compChunkCount,
                             ncclDataType_t compDatatype,
                             const size_t numChunks,
                             ncclCommOp_t commOp, cudaStream_t stream) {
  static ncclResult_t (*real)(void *, const void *,
                               const size_t, ncclDataType_t,
                               const size_t, ncclDataType_t,
                               const size_t, ncclCommOp_t,
                               cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclDecompress");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(decompbuff, compbuff, decompChunkCount,
                           decompDatatype, compChunkCount, compDatatype,
                           numChunks, commOp, stream);
  double t1 = perf_get_time();

  double out_bytes = (double)decompChunkCount
                     * nccl_type_size(decompDatatype) * (double)numChunks;

  perf_notedown(get_state(), "ncclDecompress", t0, 3,
      (perf_attr_t[]){
          {"duration",     t1 - t0},
          {"output_bytes", out_bytes},
          {"numChunks",    (double)numChunks},
      });
  return ret;
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 3 — COCCL decompress + reduce fused
 * ═══════════════════════════════════════════════════════════════════ */

ncclResult_t ncclDecompressReduce(void *reducebuff, const void *compbuff,
                                   const size_t compChunkCount,
                                   ncclDataType_t compDatatype,
                                   const size_t reduceChunkCount,
                                   ncclDataType_t reduceDataType,
                                   const size_t numChunks,
                                   ncclCommOp_t commOp,
                                   cudaStream_t stream) {
  static ncclResult_t (*real)(void *, const void *,
                               const size_t, ncclDataType_t,
                               const size_t, ncclDataType_t,
                               const size_t, ncclCommOp_t,
                               cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclDecompressReduce");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(reducebuff, compbuff, compChunkCount, compDatatype,
                           reduceChunkCount, reduceDataType,
                           numChunks, commOp, stream);
  double t1 = perf_get_time();

  perf_notedown(get_state(), "ncclDecompressReduce", t0, 3,
      (perf_attr_t[]){
          {"duration",       t1 - t0},
          {"reduceChunkCnt", (double)reduceChunkCount},
          {"numChunks",      (double)numChunks},
      });
  return ret;
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 4 — COCCL decompress + reduce + compress fused
 * ═══════════════════════════════════════════════════════════════════ */

ncclResult_t ncclDecompReduceComp(const void *compbuff, void **recompbuff,
                                   const size_t orgChunkCount,
                                   ncclDataType_t orgDatatype,
                                   const size_t compChunkCount,
                                   ncclDataType_t compDatatype,
                                   size_t *reCompChunkCount,
                                   ncclDataType_t *reCompDatatype,
                                   const size_t numChunks,
                                   ncclCommOp_t commOp,
                                   cudaStream_t stream) {
  static ncclResult_t (*real)(const void *, void **,
                               const size_t, ncclDataType_t,
                               const size_t, ncclDataType_t,
                               size_t *, ncclDataType_t *,
                               const size_t, ncclCommOp_t,
                               cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclDecompReduceComp");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(compbuff, recompbuff, orgChunkCount, orgDatatype,
                           compChunkCount, compDatatype,
                           reCompChunkCount, reCompDatatype,
                           numChunks, commOp, stream);
  double t1 = perf_get_time();

  perf_notedown(get_state(), "ncclDecompReduceComp", t0, 3,
      (perf_attr_t[]){
          {"duration",        t1 - t0},
          {"orgChunkCnt",     (double)orgChunkCount},
          {"numChunks",       (double)numChunks},
      });
  return ret;
}

/* ═══════════════════════════════════════════════════════════════════
 * Group 5 — NCCL collectives
 * ═══════════════════════════════════════════════════════════════════ */

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff,
                            size_t count, ncclDataType_t datatype,
                            ncclRedOp_t op, ncclComm_t comm,
                            cudaStream_t stream) {

  static ncclResult_t (*real)(const void *, void *, size_t,
                               ncclDataType_t, ncclRedOp_t,
                               ncclComm_t, cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclAllReduce");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(sendbuff, recvbuff, count, datatype, op, comm, stream);
  double t1 = perf_get_time();

  perf_notedown(get_state(), "ncclAllReduce", t0, 2,
      (perf_attr_t[]){
          {"duration",   t1 - t0},
          {"data_bytes", (double)count * nccl_type_size(datatype)},
      });
  return ret;
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff,
                            size_t count, ncclDataType_t datatype,
                            int root, ncclComm_t comm,
                            cudaStream_t stream) {

  static ncclResult_t (*real)(const void *, void *, size_t,
                               ncclDataType_t, int,
                               ncclComm_t, cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclBroadcast");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(sendbuff, recvbuff, count, datatype, root, comm, stream);
  double t1 = perf_get_time();

  perf_notedown(get_state(), "ncclBroadcast", t0, 2,
      (perf_attr_t[]){
          {"duration",   t1 - t0},
          {"data_bytes", (double)count * nccl_type_size(datatype)},
      });
  return ret;
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                            size_t sendcount, ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream) {

  static ncclResult_t (*real)(const void *, void *, size_t,
                               ncclDataType_t, ncclComm_t,
                               cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclAllGather");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  double t1 = perf_get_time();

  perf_notedown(get_state(), "ncclAllGather", t0, 2,
      (perf_attr_t[]){
          {"duration",     t1 - t0},
          {"sendcount",    (double)sendcount},
      });
  return ret;
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                                size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream) {

  static ncclResult_t (*real)(const void *, void *, size_t,
                               ncclDataType_t, ncclRedOp_t,
                               ncclComm_t, cudaStream_t) = NULL;
  if (!real) real = perf_nccl_get_real("ncclReduceScatter");
  if (!real) return ncclInternalError;

  double t0 = perf_get_time();
  ncclResult_t ret = real(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  double t1 = perf_get_time();

  perf_notedown(get_state(), "ncclReduceScatter", t0, 2,
      (perf_attr_t[]){
          {"duration",   t1 - t0},
          {"data_bytes", (double)recvcount * nccl_type_size(datatype)},
      });
  return ret;
}
