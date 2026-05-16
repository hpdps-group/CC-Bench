/*
 * perfuccl.c  —  LD_PRELOAD wrapper that profiles NCCL-level
 *                 operations inside UCCL's libp2p.so / libuccl.so.
 *
 * UCCL exports ncclSend / ncclRecv (C-linkage, via nccl_dl.cc) and
 * the application's underlying libnccl exports standard NCCL collectives.
 * This wrapper intercepts those C ABI functions to measure latency and
 * bandwidth of NCCL operations flowing through UCCL.
 *
 * LIMITATION
 * ----------
 * UCCL's C++ API (uccl_engine_send, uccl_engine_recv in uccl_engine.h)
 * uses C++ name-mangling and complex types (std::vector, etc.), making
 * LD_PRELOAD interposition infeasible for those entry points.  To profile
 * the full C++ send/recv path including UCCL's internal dispatch
 * (RDMA / IPC / SHM), add direct instrumentation inside UCCL source.
 *
 * This wrapper gives you the next best thing: per-call timing of the
 * NCCL operations that UCCL ultimately invokes on the backend.
 *
 * Build
 * -----
 *   cc -O2 -fPIC -shared -o libperfuccl.so                                     \
 *         -I/path/to/run_codes/wrappers/include                                \
 *         -I/path/to/nccl/include                                              \
 *         perfuccl.c                                                           \
 *         /path/to/run_codes/wrappers/src/perf_helper.c                        \
 *         /path/to/run_codes/wrappers/src/nccl/perf_helper_nccl.c              \
 *         -ldl -lm
 *
 * Run
 * ---
 *   LD_PRELOAD=libperfuccl.so ./app
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

#include "perf_helper.h"
#include "nccl/perf_helper_nccl.h"

/* ── helpers ────────────────────────────────────────────────────── */
static perf_state_t *get_state(void)  { return perf_nccl_get_tls(); }

/* ═══════════════════════════════════════════════════════════════════
 * Group 0 — NCCL init: intercept ncclCommInitRank to initialise the
 *            GPU node map BEFORE any collective / P2P operation.
 * ═══════════════════════════════════════════════════════════════════ */

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks,
                               ncclUniqueId commId, int myrank) {
  static ncclResult_t (*real)(ncclComm_t *, int, ncclUniqueId, int) = NULL;
  if (!real) real = perf_nccl_get_real("ncclCommInitRank");
  if (!real) return ncclInternalError;

  ncclResult_t ret = real(comm, nranks, commId, myrank);
  if (ret == ncclSuccess)
    perf_nccl_init_node_map(*comm);  /* safe: comm is ready, no outer collective */
  return ret;
}

/* ═══════════════════════════════════════════════════════════════════
 * NCCL collectives intercepted at the C ABI layer
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
 * Group 2 — NCCL collectives (from the real libnccl.so on the system)
 *           Intercepted at the C ABI layer — works with any NCCL,
 *           including when called through UCCL's NCCL endpoints.
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
