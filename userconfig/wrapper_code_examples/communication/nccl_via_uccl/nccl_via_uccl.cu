/**
 * nccl_via_uccl.cu — NCCL API over UCCL P2P RDMA with DietGPU compression.
 *
 * LD_PRELOAD this shared library to intercept NCCL calls and route them
 * through UCCL P2P's RDMA transport.  DietGPU float compression is enabled
 * via the UCCL_P2P_COMPRESS_STRATEGY=encode environment variable (set before
 * launching the process).
 *
 * Bootstrap uses file-based metadata exchange (no MPI/torch.distributed
 * dependency).  Each rank writes its UCCL endpoint metadata to a rank-specific
 * file under NCCL_COMM_ID (default nccl_id_file/nccl_bench_id/).
 *
 * Implemented: ncclGetUniqueId, ncclCommInitRank, ncclCommDestroy,
 *              ncclSend, ncclRecv, ncclAllReduce, ncclAllToAll,
 *              ncclBroadcast, ncclReduce, ncclGetErrorString.
 *
 * Build (via mybench build system — communication_lib_selection.jsonc mode 3):
 *   source_dirs: ["userconfig/wrapper_code_examples/communication/nccl_via_uccl"]
 *   include_dirs: ["plugin_projects/ccl/uccl/p2p"]
 *   libraries: ["-labspath/uccl/p2p/libuccl_p2p.so", "-lcudart", "-lcuda", "-lpthread", "-ldl"]
 *   output_name: "libnccl_via_uccl.so"
 *
 *   nvcc picks up .cu from source_dirs, links against libuccl_p2p.
 *   Set UCCL_P2P_COMPRESS_STRATEGY=encode at runtime to enable DietGPU.
 */

#include <cuda_runtime.h>

/* NCCL types supplied by UCCL's nccl_types.h (included transitively via engine.h).
 * Do NOT also include <nccl.h> — it would conflict. */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/* UCCL C++ API — pulls in uccl::FloatType, nccl_types.h, Endpoint */
#include "engine.h"

/* ncclRedOp_t is not in UCCL's nccl_types.h — define it here */
typedef enum {
  ncclSum    = 0,
  ncclProd   = 1,
  ncclMax    = 2,
  ncclMin    = 3,
  ncclAvg    = 4,
  ncclNumOps = 5,
} ncclRedOp_t;

/* ========================================================================= *
 * Constants
 * ========================================================================= */
#define NCCL_ID_DIR "nccl_id_file"
#define NCCL_ID_FILE NCCL_ID_DIR "/nccl_bench_id"

/* ========================================================================= *
 * Debug logging (NCCL_DEBUG=INFO / WARN)
 * ========================================================================= */
static int nccl_debug_enabled() {
    static int checked = 0;
    static int val = 0;
    if (!checked) {
        char *e = getenv("NCCL_DEBUG");
        if (e && (strcmp(e, "INFO") == 0 || strcmp(e, "WARN") == 0))
            val = 1;
        __sync_synchronize();
        checked = 1;
    }
    return val;
}

/* Check if UCCL_DEBUG is set to INFO or WARN */
static int uccl_debug_enabled() {
    static int checked = 0;
    static int val = 0;
    if (!checked) {
        char *e = getenv("UCCL_DEBUG");
        if (e && (strcmp(e, "INFO") == 0 || strcmp(e, "WARN") == 0))
            val = 1;
        __sync_synchronize();
        checked = 1;
    }
    return val;
}

/* Poll timeout in loop iterations.
 *
 * Base value: UCCL_P2P_POLL_TIMEOUT env var (default 30000000 ≈ 30 s).
 * When NCCL_DEBUG=INFO/WARN or UCCL_DEBUG=INFO/WARN, automatically
 * multiplied by 10× to 300000000 to avoid spurious timeouts during
 * debug logging which slows down proxy thread processing.
 */
// Wall-clock timeout in milliseconds.  The env var is in ms too.
// Default: 60000 (60 seconds) — enough for large message prepost resize.
static int64_t poll_timeout_ms() {
    static int64_t val = 0;
    if (val != 0) return val;
    int64_t base = 60000;
    char *e = getenv("UCCL_P2P_POLL_TIMEOUT_MS");
    if (e) {
        char *end = nullptr;
        int64_t v = strtoll(e, &end, 10);
        if (end != e && v > 0) base = v;
    }
    val = base;
    return val;
}

/* ========================================================================= *
 * NCCL communicator — our internal structure
 * ========================================================================= */
struct nccl_comm {
    int rank;
    int nranks;
    int local_gpu_idx;           /* CUDA device ordinal */

    /* UCCL endpoint (per-process) */
    Endpoint* endpoint;

    /* UCCL connect→send_channel_groups_, accept→recv_channel_groups_ (asymmetric).
     * Two-phase connection establishes both directions per peer. */
    std::vector<uint64_t> send_conn_ids;  /* for sending to peer */
    std::vector<uint64_t> recv_conn_ids;  /* for receiving from peer */

    /* Temporary ring-allgather buffer — per-call alloc/free.
     * Non-persistent to reproduce the original rkey-stale race. */
};

/* Global state — only one active communicator at a time (fine for benchmarks) */
static std::mutex g_mutex;
static nccl_comm* g_active_comm = nullptr;

/* Best-effort rank for logging (0 = unknown) */
static int log_rank() {
    nccl_comm* c = g_active_comm;
    return c ? c->rank : 0;
}

#define LOG(...) do { \
    if (nccl_debug_enabled()) { \
        fprintf(stderr, "[nccl_via_uccl][rank=%d] ", log_rank()); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } \
} while (0)

/* ========================================================================= *
 * Forward declarations — extern "C" NCCL API replacements
 * ========================================================================= */
/* UCCL Endpoint API provides C-linkage NCCL entry points through
 * UCCL's nccl_types.h (basic ops) and our definitions below (collectives). */


/* ========================================================================= *
 * Helpers
 * ========================================================================= */

/* Element size of NCCL data type */
static size_t dtype_size(ncclDataType_t dtype) {
    switch (dtype) {
    case ncclInt8:    case ncclUint8:   return 1;
    case ncclFloat16: return 2;
    case ncclFloat32:                   return 4;
    case ncclInt32:   case ncclUint32:  return 4;
    case ncclInt64:   case ncclUint64:  return 8;
    case ncclFloat64:                   return 8;
    default:                            return 4;
    }
}

/* Map NCCL dtype → UCCL FloatType for DietGPU compression */
static uccl::FloatType to_float_type(ncclDataType_t dtype) {
    switch (dtype) {
    case ncclFloat16: return uccl::FloatType::kFloat16;
    case ncclFloat32: return uccl::FloatType::kFloat32;
    case ncclFloat64: return uccl::FloatType::kFloat32; /* DietGPU splits */
    default:          return uccl::FloatType::kUndefined;
    }
}

/* File path for rank's bootstrap metadata */
static std::string meta_path(int rank) {
    const char* base = getenv("NCCL_COMM_ID");
    if (!base) base = NCCL_ID_DIR "/nccl_bench_id";
    return std::string(base) + ".meta." + std::to_string(rank);
}

static void ensure_dir(const char* path) {
    std::string s(path);
    auto pos = s.rfind('/');
    if (pos != std::string::npos) {
        std::string dir = s.substr(0, pos);
        mkdir(dir.c_str(), 0755);
    }
}

static void write_meta(const std::string& path, const std::vector<uint8_t>& data) {
    ensure_dir(path.c_str());
    /* Atomic write: write to a unique temp file, fsync, then rename().
     * POSIX guarantees rename() is atomic even over NFS — readers see
     * either the old file (before rename) or the new file (after rename). */
    std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE* fp = fopen(tmp.c_str(), "wb");
    if (!fp) { LOG("ERROR: cannot write %s", tmp.c_str()); return; }
    uint32_t sz = data.size();
    fwrite(&sz, sizeof(sz), 1, fp);
    fwrite(data.data(), 1, sz, fp);
    fflush(fp);
    fsync(fileno(fp));
    fclose(fp);
    rename(tmp.c_str(), path.c_str());
}

static std::vector<uint8_t> read_meta(const std::string& path) {
    for (int attempt = 0; attempt < 30; attempt++) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        uint32_t sz = 0;
        f.read((char*)&sz, sizeof(sz));
        if (sz == 0 || sz > 4096) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        std::vector<uint8_t> data(sz);
        f.read((char*)data.data(), sz);
        if (f.gcount() != (std::streamsize)sz) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        /* Validate: IP (bytes 0-3) must be non-zero; port (bytes 4-5) must be non-zero */
        if (sz >= 7) {
            uint32_t ip;
            memcpy(&ip, data.data(), 4);
            uint16_t port = ntohs(*(uint16_t*)(data.data() + 4));
            if (ip == 0 || port == 0) {
                // LOG("read_meta: stale data (ip=%x port=%d), retry %d", ip, port, attempt);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
        }
        return data;
    }
    LOG("ERROR: read_meta failed for %s after retries", path.c_str());
    return {};
}

static bool wait_file(const std::string& path, int timeout_sec = 30) {
    struct stat st;
    for (int i = 0; i < timeout_sec * 10; i++) {
        if (stat(path.c_str(), &st) == 0) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

/* Register a GPU buffer with UCCL (no caching — always re-register to avoid
 * stale MR issues when cudaFree + cudaMalloc reuses the same VA).
 * Returns UINT64_MAX on failure (since mr_id=0 is valid for the first MR). */
static uint64_t reg_buffer(nccl_comm* comm, uintptr_t addr,
                            size_t size, ncclDataType_t dtype) {
    uint64_t mr_id = 0;
    uccl::FloatType ft = to_float_type(dtype);
    bool ok = comm->endpoint->reg((void*)addr, size, mr_id, ft);
    if (!ok) {
        LOG("ERROR: reg(%zu bytes @ %lx) failed", size, addr);
        return UINT64_MAX;
    }
    // LOG("registered %zu bytes @ %lx as mr_id=%lu (ft=%d)", size, addr, mr_id, (int)ft);
    return mr_id;
}

/* Blocking UCCL send (async + poll). */
static bool uccl_send_wait(nccl_comm* comm, uint64_t conn_id,
                            void* data, size_t bytes, uint64_t mr_id) {
    uint64_t tid;
    if (!comm->endpoint->send_async(conn_id, mr_id, data, bytes, &tid))
        return false;
    bool done = false;
    while (!done) {
        if (!comm->endpoint->poll_async(tid, &done))
            return false;
    }
    return true;
}

/* Blocking UCCL recv (async + poll). */
static bool uccl_recv_wait(nccl_comm* comm, uint64_t conn_id,
                            void* data, size_t bytes, uint64_t mr_id) {
    uint64_t tid;
    if (!comm->endpoint->recv_async(conn_id, mr_id, data, bytes, &tid))
        return false;
    bool done = false;
    while (!done) {
        if (!comm->endpoint->poll_async(tid, &done))
            return false;
    }
    return true;
}

/* ========================================================================= *
 * CPU reduction helper (element-wise, for GPU data via cudaMemcpy)
 * ========================================================================= */
static void cpu_reduce(void* dst, const void* src, size_t count,
                        ncclDataType_t dtype, ncclRedOp_t op) {
    switch (dtype) {
    case ncclFloat32: {
        float* d = (float*)dst;
        const float* s = (const float*)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: break;
        }
        break;
    }
    case ncclFloat64: {
        double* d = (double*)dst;
        const double* s = (const double*)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: break;
        }
        break;
    }
    case ncclInt32: {
        int* d = (int*)dst;
        const int* s = (const int*)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: break;
        }
        break;
    }
    case ncclInt64: {
        long long* d = (long long*)dst;
        const long long* s = (const long long*)src;
        switch (op) {
        case ncclSum:  for (size_t i = 0; i < count; i++) d[i] += s[i]; break;
        case ncclProd: for (size_t i = 0; i < count; i++) d[i] *= s[i]; break;
        case ncclMax:  for (size_t i = 0; i < count; i++) if (s[i] > d[i]) d[i] = s[i]; break;
        case ncclMin:  for (size_t i = 0; i < count; i++) if (s[i] < d[i]) d[i] = s[i]; break;
        default: break;
        }
        break;
    }
    default:
        break;
    }
}

/* ========================================================================= *
 * ncclGetUniqueId — token for file-path coordination
 * ========================================================================= */
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    memset(uniqueId, 0, sizeof(*uniqueId));
    uniqueId->internal[0] = 'U';
    uniqueId->internal[1] = 'C';
    uniqueId->internal[2] = 'C';
    uniqueId->internal[3] = 'L';
    return ncclSuccess;
}

/* ========================================================================= *
 * ncclCommInitRank — bootstrap + UCCL engine + full-mesh connections
 * ========================================================================= */
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks,
                               ncclUniqueId commId, int myrank) {
    // fprintf(stderr, "[p-rank%d] ncclCommInitRank nranks=%d\n", myrank, nranks);

    if (!newcomm || nranks < 1 || myrank < 0 || myrank >= nranks)
        return ncclInvalidArgument;

    (void)commId;

    nccl_comm* comm = new nccl_comm;
    comm->rank = myrank;
    comm->nranks = nranks;
    comm->send_conn_ids.resize(nranks, UINT64_MAX);
    comm->recv_conn_ids.resize(nranks, UINT64_MAX);

    /* Determine local GPU index */
    int dev;
    cudaGetDevice(&dev);
    comm->local_gpu_idx = dev;

    /* ── Step 1: Create UCCL endpoint ── */
    try {
        comm->endpoint = new Endpoint(comm->local_gpu_idx);
    } catch (const std::exception& e) {
        LOG("ERROR: failed to create UCCL Endpoint: %s", e.what());
        delete comm;
        return ncclInternalError;
    }

    /* Tell the RDMA endpoint our rank so process_meta uses actual rank IDs
     * instead of auto-incrementing.  This makes conn->uccl_conn_id_.flow_id
     * equal the sender's rank on the accept side. */
    comm->endpoint->set_rank(myrank);

    /* ── Step 2: Write endpoint metadata to shared file ── */
    std::vector<uint8_t> my_meta = comm->endpoint->get_metadata();
    std::string my_path = meta_path(myrank);
    write_meta(my_path, my_meta);
    // LOG("wrote metadata to %s", my_path.c_str());

//     /* ── Step 3: Ensure all peers' metadata exist ── */
    for (int r = 0; r < nranks; r++) {
        if (r == myrank) continue;
        std::string r_path = meta_path(r);
        if (!wait_file(r_path, 60)) {
            LOG("ERROR: timeout waiting for rank %d metadata", r);
            delete comm->endpoint;
            delete comm;
            return ncclInternalError;
        }
    }

    /* ── Step 4: Read all peers' metadata ── */
    std::vector<std::vector<uint8_t>> all_meta(nranks);
    all_meta[myrank] = my_meta;
    for (int r = 0; r < nranks; r++) {
        if (r == myrank) continue;
        all_meta[r] = read_meta(meta_path(r));
        if (all_meta[r].empty()) {
            LOG("ERROR: failed to read rank %d metadata", r);
            delete comm->endpoint;
            delete comm;
            return ncclInternalError;
        }
    }

    /* ── Step 5: Establish full-mesh connections (two-phase) ──
     *
     * Phase 1 (original): accept from lower peers (→recv), connect to higher (→send).
     * Phase 2 (reverse):  connect to lower peers (→send), accept from higher (→recv).
     * After both phases each peer has both send_conn_ids and recv_conn_ids. */
    // LOG("establishing connections (rank %d of %d)...", myrank, nranks);

    {
//         // Phase-1 accept: accept from all lower peers, then match each to the
//         // correct rank.  accept() returns connections in non-deterministic
//         // order (depends on OOB message arrival), so we CANNOT use the loop
//         // counter as the peer rank — recv_conn_ids[peer] would point at the
//         // wrong peer if rank 1's metadata arrives before rank 0's.
//         //
//         // Instead we collect all incoming connections, then match by
//         // peer IP + remote GPU index against all_meta[].
        struct AcceptResult {
            uint64_t conn_id;
            std::string ip;
            int gpu;
        };
        std::vector<AcceptResult> pending;
        for (int count = 0; count < myrank; count++) {
            // LOG("  phase-1 waiting for accept (%d/%d)...", count, myrank);
            std::string ip_buf;
            int remote_gpu = -1;
            uint64_t conn_id;
            bool ok = comm->endpoint->accept(ip_buf, remote_gpu, conn_id);
            if (!ok) {
                LOG("ERROR: phase-1 accept failed (count=%d/%d)",
                    count, myrank);
                delete comm->endpoint;
                delete comm;
                return ncclInternalError;
            }
            pending.push_back({conn_id, ip_buf, remote_gpu});
            // LOG("  phase-1 got connection ip=%s gpu=%d (pending=%zu)",
            //     ip_buf.c_str(), remote_gpu, pending.size());
        }
        // Match each pending accept to the correct peer using flow_id.
        // With set_rank() called above, conn->uccl_conn_id_.flow_id equals
        // the sender's actual rank — no need for IP/GPU heuristics.
        for (auto const& ar : pending) {
            Conn* c = comm->endpoint->get_conn(ar.conn_id);
            if (!c) {
                LOG("ERROR: phase-1 accept conn %lu not found", ar.conn_id);
                delete comm->endpoint;
                delete comm;
                return ncclInternalError;
            }
            int peer_rank = static_cast<int>(c->uccl_conn_id_.flow_id);
            if (peer_rank < 0 || peer_rank >= myrank) {
                LOG("ERROR: phase-1 accept invalid rank %d from %s gpu=%d",
                    peer_rank, ar.ip.c_str(), ar.gpu);
                delete comm->endpoint;
                delete comm;
                return ncclInternalError;
            }
            comm->recv_conn_ids[peer_rank] = ar.conn_id;
        }
    }

    for (int peer = myrank + 1; peer < nranks; peer++) {
        auto [peer_ip, peer_port, peer_gpu_bdf] =
            Endpoint::parse_metadata(all_meta[peer]);

        // LOG("  connecting to rank %d at %s:%d...", peer, peer_ip.c_str(), peer_port);

        uint64_t conn_id;
//         /* remote_gpu_idx=0 is safe for cross-node RDMA: each process drives one GPU */
        bool ok = comm->endpoint->connect(peer_ip, 0, (int)peer_port, conn_id);
        if (!ok) {
            LOG("ERROR: connect to rank %d failed", peer);
            delete comm->endpoint;
            delete comm;
            return ncclInternalError;
        }
        comm->send_conn_ids[peer] = conn_id;
        // LOG("  connected to rank %d (send_conn_id=%lu)", peer, conn_id);
    }

//     /* ── Phase 2: reverse direction (connect to lower, accept from higher) ──
//      * Phase-1 connect gave us send-only channels to higher peers.
//      * Phase-1 accept gave us recv-only channels from lower peers.
//      * This phase establishes the missing directions so every peer has both
//      * send_conn_ids and recv_conn_ids for all-to-all bidirectional usage. */
    // LOG("  phase 2: establishing reverse-direction connections...");

    for (int peer = 0; peer < myrank; peer++) {
        auto [peer_ip, peer_port, peer_gpu_bdf] =
            Endpoint::parse_metadata(all_meta[peer]);

        // LOG("  connecting to rank %d (phase 2)...", peer);

        uint64_t conn_id;
        bool ok = comm->endpoint->connect(peer_ip, 0, (int)peer_port, conn_id);
        if (!ok) {
            LOG("ERROR: phase-2 connect to rank %d failed", peer);
            delete comm->endpoint;
            delete comm;
            return ncclInternalError;
        }
        comm->send_conn_ids[peer] = conn_id;
        // LOG("  phase-2 connected to rank %d (send_conn_id=%lu)", peer, conn_id);
    }

    {
//         // Phase-2 accept: accept from all higher peers, then match.
//         // Same ordering issue as Phase 1 — accept order is
//         // non-deterministic.
        struct AcceptResult2 {
            uint64_t conn_id;
            std::string ip;
            int gpu;
        };
        std::vector<AcceptResult2> pending2;
        int n_higher = nranks - myrank - 1;
        for (int count = 0; count < n_higher; count++) {
            // LOG("  phase-2 waiting for accept (%d/%d)...", count, n_higher);
            std::string ip_buf;
            int remote_gpu = -1;
            uint64_t conn_id;
            bool ok = comm->endpoint->accept(ip_buf, remote_gpu, conn_id);
            if (!ok) {
                LOG("ERROR: phase-2 accept failed (count=%d/%d)",
                    count, n_higher);
                delete comm->endpoint;
                delete comm;
                return ncclInternalError;
            }
            pending2.push_back({conn_id, ip_buf, remote_gpu});
            // LOG("  phase-2 got connection ip=%s gpu=%d (pending=%zu)",
            //     ip_buf.c_str(), remote_gpu, pending2.size());
        }
        // Match each pending accept to the correct higher peer using flow_id.
        for (auto const& ar : pending2) {
            Conn* c = comm->endpoint->get_conn(ar.conn_id);
            if (!c) {
                LOG("ERROR: phase-2 accept conn %lu not found", ar.conn_id);
                delete comm->endpoint;
                delete comm;
                return ncclInternalError;
            }
            int peer_rank = static_cast<int>(c->uccl_conn_id_.flow_id);
            if (peer_rank <= myrank || peer_rank >= nranks) {
                LOG("ERROR: phase-2 accept invalid rank %d from %s gpu=%d",
                    peer_rank, ar.ip.c_str(), ar.gpu);
                delete comm->endpoint;
                delete comm;
                return ncclInternalError;
            }
            comm->recv_conn_ids[peer_rank] = ar.conn_id;
        }
    }

    // fprintf(stderr, "[p-rank%d] connections done, entering barrier\n", myrank);
    // LOG("all connections established for rank %d", myrank);

    /* ── Step 7: Global ready barrier — wait for all peers to finish setup ──
//      *
//      * Without this barrier a fast rank may start ncclAllReduce (and send RDMA
//      * WRITE WITH IMMEDIATE) before a slower peer has finished establishing its
//      * RDMA connections.  The IMM then targets a QP / CQ that doesn't exist yet
//      * on the slow peer, the completion is silently lost, and the ring allgather
//      * deadlocks until a 30 s timeout.
//      *
//      * Two-phase file barrier to eliminate TOCTOU race:
//      *   Phase 1 — write .ready, wait for all peers' .ready.
//      *   Phase 2 — write .ready2, wait for all peers' .ready2.
//      * Every rank enters phase 2 only after ALL peers have passed phase 1,
//      * so no rank deletes .ready before another rank has confirmed it exists.
//      * Metadata directory is shared via NCCL_COMM_ID. */
    {
//         // fprintf(stderr, "[dbg-rank%d] entering barrier phase 1\n", myrank);

        /* Phase 1 */
        std::string r1 = meta_path(myrank) + ".ready";
        FILE* fp = fopen(r1.c_str(), "w");
        if (fp) { fputc('1', fp); fflush(fp); fsync(fileno(fp)); fclose(fp); }
        // fprintf(stderr, "[dbg-rank%d] wrote .ready, waiting for peers\n", myrank);

        for (int r = 0; r < nranks; r++) {
            if (r == myrank) continue;
            std::string pr = meta_path(r) + ".ready";
            // fprintf(stderr, "[dbg-rank%d] phase-1 waiting for peer %d\n", myrank, r);
            struct stat st;
            int polled = 0;
            while (stat(pr.c_str(), &st) != 0) {
                if (++polled % 600 == 0) /* 60 s */;
                    // fprintf(stderr, "[dbg-rank%d] phase-1 still waiting for peer %d (600 polls)\n", myrank, r);
                usleep(100000);
            }
            // fprintf(stderr, "[dbg-rank%d] phase-1 peer %d ready\n", myrank, r);
        }

        // fprintf(stderr, "[dbg-rank%d] phase-1 done, entering phase 2\n", myrank);

        /* Phase 2 — only after every peer's .ready was confirmed */
        std::string r2 = meta_path(myrank) + ".ready2";
        fp = fopen(r2.c_str(), "w");
        if (fp) { fputc('1', fp); fflush(fp); fsync(fileno(fp)); fclose(fp); }

        for (int r = 0; r < nranks; r++) {
            if (r == myrank) continue;
            std::string pr2 = meta_path(r) + ".ready2";
            // fprintf(stderr, "[dbg-rank%d] phase-2 waiting for peer %d\n", myrank, r);
            struct stat st2 = {};
            int polled2 = 0;
            while (stat(pr2.c_str(), &st2) != 0) {
                if (++polled2 % 600 == 0) /* 60 s */;
                    // fprintf(stderr, "[dbg-rank%d] phase-2 still waiting for peer %d\n", myrank, r);
                usleep(100000);
            }
            // fprintf(stderr, "[dbg-rank%d] phase-2 peer %d ready\n", myrank, r);
        }

        // fprintf(stderr, "[p-rank%d] barrier done\n", myrank);

        /* NOTE: .ready / .ready2 files NOT removed here — another rank may still be
         * waiting for this rank's .ready2 in its phase-2 stat() loop.  Removing would
         * cause a TOCTOU hang.  The benchmark script cleans *.meta.* at startup. */
    }

    remove(my_path.c_str());

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_active_comm = comm;
    }

    // fprintf(stderr, "[p-rank%d] ncclCommInitRank done\n", myrank);
    *newcomm = reinterpret_cast<ncclComm_t>(comm);
    return ncclSuccess;
}

/* ========================================================================= *
 * ncclCommDestroy — file-barrier then graceful teardown
 * ========================================================================= *
 * Before tearing down RDMA resources, we synchronise all peers via a
 * file-based barrier (same directory as metadata).  This prevents a
 * use-after-free race where a fast rank destroys its Endpoint while
 * slower peers are still driving RDMA traffic or establishing
 * connections to it.
 * ========================================================================= */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c) return ncclInvalidArgument;

    int rank = c->rank;
    // fprintf(stderr, "[p-rank%d] ncclCommDestroy\n", rank);
    int nranks = c->nranks;
    // LOG("ncclCommDestroy rank=%d/%d", rank, nranks);

//     /* ── Step 1: Signal "I'm done" via a .done file ── */
    std::string done = meta_path(rank) + ".done";
    {
        FILE* fp = fopen(done.c_str(), "w");
        if (fp) { fputc('1', fp); fflush(fp); fsync(fileno(fp)); fclose(fp); }
    }

    /* ── Step 2: Wait for all peers' .done files (barrier, 60 s timeout) ── */
    for (int r = 0; r < nranks; r++) {
        if (r == rank) continue;
        std::string peer_done = meta_path(r) + ".done";
        int waited = 0;
        for (; waited < 600; waited++) {
            struct stat st;
            if (stat(peer_done.c_str(), &st) == 0) break;
            usleep(100000); /* 100 ms */
        }
            if (waited >= 600);
            // LOG("WARN: timeout waiting for rank %d done file", r);
    }

//     /* ── Step 3: Tear down RDMA resources (safe now) ── */
    if (c->endpoint) {
        delete c->endpoint;
        c->endpoint = nullptr;
    }

//     /* ── Step 4: Clean up done file ── */
    remove(done.c_str());

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_active_comm == c) g_active_comm = nullptr;
    }

    delete c;
    return ncclSuccess;
}

/* ========================================================================= *
 * ncclSend — GPU buffer → UCCL RDMA to peer
 * ========================================================================= */
ncclResult_t ncclSend(const void* sendbuff, size_t count,
                       ncclDataType_t datatype, int peer,
                       ncclComm_t comm, cudaStream_t stream) {
    (void)stream;
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c || peer < 0 || peer >= c->nranks) return ncclInvalidArgument;
    if (c->send_conn_ids[peer] == UINT64_MAX) return ncclInvalidArgument;

    size_t bytes = count * dtype_size(datatype);
    if (bytes == 0) return ncclSuccess;

    uint64_t mr_id = reg_buffer(c, (uintptr_t)sendbuff, bytes, datatype);
    if (mr_id == UINT64_MAX) return ncclInternalError;

    return uccl_send_wait(c, c->send_conn_ids[peer],
                          const_cast<void*>(sendbuff), bytes, mr_id)
           ? ncclSuccess : ncclInternalError;
}

/* ========================================================================= *
 * ncclRecv — UCCL RDMA from peer → GPU buffer
 * ========================================================================= */
ncclResult_t ncclRecv(void* recvbuff, size_t count,
                       ncclDataType_t datatype, int peer,
                       ncclComm_t comm, cudaStream_t stream) {
    (void)stream;
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c || peer < 0 || peer >= c->nranks) return ncclInvalidArgument;
    if (c->recv_conn_ids[peer] == UINT64_MAX) return ncclInvalidArgument;

    size_t bytes = count * dtype_size(datatype);
    if (bytes == 0) return ncclSuccess;

    uint64_t mr_id = reg_buffer(c, (uintptr_t)recvbuff, bytes, datatype);
    if (mr_id == UINT64_MAX) return ncclInternalError;

    return uccl_recv_wait(c, c->recv_conn_ids[peer],
                          recvbuff, bytes, mr_id)
           ? ncclSuccess : ncclInternalError;
}

/* ========================================================================= *
 * ncclAllReduce — allgather + CPU reduce
 *
 * Approach (simple, correct):
 *   1. Allgather: each rank contributes its N-th of the total data.
 *      Ring allgather via UCCL RDMA (GPU↔GPU).
 *   2. Download full gathered buffer to CPU.
 *   3. Element-wise reduce on CPU.
 *   4. Upload result back to recvbuff.
 * ========================================================================= */
extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                            size_t count, ncclDataType_t datatype,
                            ncclRedOp_t op, ncclComm_t comm,
                            cudaStream_t stream) {
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    int rank = c->rank;
    // fprintf(stderr, "[p-rank%d] ncclAllReduce entry count=%zu\n", rank, count);
    int nranks = c->nranks;
    size_t elt_size = dtype_size(datatype);
    size_t bytes = count * elt_size;
    static std::atomic<uint64_t> ar_seq{0};
    uint64_t my_seq = ar_seq.fetch_add(1, std::memory_order_relaxed);

    // fprintf(stderr, "[dbg-%d] AR ENTRY rank=%d nranks=%d bytes=%zu seq=%lu\n",
            //     getpid(), rank, nranks, bytes, my_seq);
    // LOG("AR ENTRY count=%zu dtype=%d nranks=%d bytes=%zu",
        //     count, (int)datatype, nranks, bytes);

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                            cudaMemcpyDeviceToDevice, stream);
        return ncclSuccess;
    }

    /* ── Allocate/persist GPU temp for allgather ── */
    size_t total_bytes = (size_t)nranks * bytes;

    /* Per-call allocation — register, use, then dereg/free.
     * This is the original pattern that reproduces the VA-reuse /
     * stale-rkey race (cudaFree + cudaMalloc may hand back the same
     * virtual address, and the new ibv_reg_mr produces a different
     * rkey while peers may still hold the old one). */
    void* gpu_all;
    cudaMalloc(&gpu_all, total_bytes);
    uint64_t mr_all = reg_buffer(c, (uintptr_t)gpu_all, total_bytes, datatype);
    if (mr_all == UINT64_MAX) {
        cudaFree(gpu_all);
        return ncclInternalError;
    }

    /* Copy local data into our slot */
    cudaMemcpyAsync((uint8_t*)gpu_all + (size_t)rank * bytes, sendbuff, bytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    // LOG("  registered gpu_all (%zu bytes @ %p) as mr_id=%lu",
        //     total_bytes, gpu_all, mr_all);

    int pred = (rank == 0) ? nranks - 1 : rank - 1;
    int succ = (rank + 1) % nranks;

    for (int step = 0; step < nranks - 1; step++) {
        int send_idx = (rank - step + nranks) % nranks;
        int recv_idx = (rank - step - 1 + nranks) % nranks;

        void* send_ptr = (uint8_t*)gpu_all + (size_t)send_idx * bytes;
        void* recv_ptr = (uint8_t*)gpu_all + (size_t)recv_idx * bytes;

        uint64_t send_tid, recv_tid;
        bool send_ok = c->endpoint->send_async(c->send_conn_ids[succ], mr_all,
                                                send_ptr, bytes, &send_tid);
        bool recv_ok = c->endpoint->recv_async(c->recv_conn_ids[pred], mr_all,
                                                recv_ptr, bytes, &recv_tid);
        // fprintf(stderr, "[dbg-%d] AR step=%d/2 bytes=%zu send_ok=%d recv_ok=%d send_tid=0x%lx recv_tid=0x%lx seq=%lu\n",
        //         getpid(), step, bytes, (int)send_ok, (int)recv_ok,
        //         send_tid, recv_tid, my_seq);
        // LOG("  step %d: send_ok=%d recv_ok=%d", step, (int)send_ok, (int)recv_ok);
        if (!send_ok || !recv_ok) {
            fprintf(stderr, "[p-rank%d] ERROR: send_async(succ=%d)=%d recv_async(pred=%d)=%d at step %d\n",
                    rank, succ, (int)send_ok, pred, (int)recv_ok, step);
//             /* Wait for any operation that WAS posted before cleanup,
//              * otherwise the proxy thread may crash accessing a
//              * stale MR after we dereg/free it. */
            if (send_ok) {
                bool done = false;
                while (!done) c->endpoint->poll_async(send_tid, &done);
            }
            if (recv_ok) {
                bool done = false;
                while (!done) c->endpoint->poll_async(recv_tid, &done);
            }
            // fprintf(stderr, "[dbg-%d] AR EXIT_ERR rank=%d bytes=%zu seq=%lu (send_ok=%d recv_ok=%d)\n",
        //             getpid(), rank, bytes, my_seq, (int)send_ok, (int)recv_ok);
            return ncclInternalError;
        }

        // fprintf(stderr, "[dbg-%d] AR step=%d/2 ENTER_POLL_RECV bytes=%zu seq=%lu\n",
        //             getpid(), step, bytes, my_seq);
        /* Busy-wait for recv completion (with wall-clock timeout) */
        bool done = false;
        int64_t poll_loops = 0;
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(poll_timeout_ms());
        while (!done) {
            c->endpoint->poll_async(recv_tid, &done);
            if (++poll_loops && std::chrono::steady_clock::now() > deadline) {
                fprintf(stderr, "[p-rank%d] ERROR: timeout waiting for recv_tid from pred=%d on step %d (poll_loops=%" PRId64 ")\n", rank, pred, step, poll_loops);
                /* Try to drain the pending send before cleanup. If the send
                 * drain also times out, the proxy thread is still using our
                 * MR/mhandle — we MUST NOT dereg/free or we get SIGSEGV in
                 * the proxy thread (use-after-free of P2PMhandle). */
                done = false; poll_loops = 0;
                deadline = std::chrono::steady_clock::now()
                           + std::chrono::milliseconds(poll_timeout_ms());
                while (!done) {
                    c->endpoint->poll_async(send_tid, &done);
                    if (++poll_loops && std::chrono::steady_clock::now() > deadline) {
                        fprintf(stderr, "[p-rank%d] ERROR: send drain timed out on step %d (succ=%d, poll_loops=%" PRId64 ") — leaking resources to avoid proxy thread crash\n", rank, step, succ, poll_loops);
                        return ncclInternalError;
                    }
                }
                // LOG("  step %d: send drain OK after recv timeout", step);
                return ncclInternalError;
            }
        }
        // LOG("  step %d: recv done", step);
//         // fprintf(stderr, "[dbg-%d] AR step=%d/2 RECV_DONE bytes=%zu poll_loops=%d seq=%lu\n",
        //             getpid(), step, bytes, poll_loops, my_seq);

        /* Busy-wait for send completion (with wall-clock timeout) */
        // fprintf(stderr, "[dbg-%d] AR step=%d/2 ENTER_POLL_SEND bytes=%zu seq=%lu\n",
        //             getpid(), step, bytes, my_seq);
        done = false;
        poll_loops = 0;
        deadline = std::chrono::steady_clock::now()
                   + std::chrono::milliseconds(poll_timeout_ms());
        while (!done) {
            c->endpoint->poll_async(send_tid, &done);
            if (++poll_loops && std::chrono::steady_clock::now() > deadline) {
                fprintf(stderr, "[p-rank%d] ERROR: timeout waiting for send_tid on step %d (succ=%d, poll_loops=%" PRId64 ", recv already done) — leaking resources to avoid UAF\n", rank, step, succ, poll_loops);
                /* Recv already done, but send proxy thread still holds the
                 * mhandle. Skip dereg/free to avoid a use-after-free. */
                return ncclInternalError;
            }
        }
        // LOG("  step %d: send done", step);
//         // fprintf(stderr, "[dbg-%d] AR step=%d/2 SEND_DONE bytes=%zu poll_loops=%d seq=%lu\n",
        //             getpid(), step, bytes, poll_loops, my_seq);
    }

    // fprintf(stderr, "[dbg-%d] AR EXIT_OK rank=%d bytes=%zu seq=%lu\n",
        //             getpid(), rank, bytes, my_seq);

    /* ── Ensure all proxy-thread cudaMemcpy H2D ops are visible on GPU ──
     *
     * The recv proxy thread issues cudaMemcpy(H2D, →gpu_all) on its own
     * per-thread default CUDA stream.  Without an explicit global sync, the
     * main thread's subsequent cudaMemcpy(D2H, from gpu_all) may execute on
     * the GPU before the proxy thread's H2D data has landed, reading stale
     * zeros instead of the RDMA payload (a cross-thread stream ordering bug).
     */
    cudaDeviceSynchronize();

    /* ── Download full gathered data to CPU (keep gpu_all alive) ── */
    std::vector<uint8_t> cpu_all(total_bytes);
    cudaMemcpy(cpu_all.data(), gpu_all, total_bytes, cudaMemcpyDeviceToHost);

    /* ── Diagnostic: per-slot checksum (disabled) ── */
    /* Re-enable by uncommenting the block below.
    {
      static std::atomic<int> _diag_seq{0};
      int _ds = _diag_seq.fetch_add(1, std::memory_order_relaxed);
      fprintf(stderr, "[SLOT_DIAG seq=%d rank=%d]", _ds, rank);
      for (int r = 0; r < nranks; r++) {
        float const* _slot = reinterpret_cast<float const*>(
            cpu_all.data() + (size_t)r * bytes);
        size_t _n = count > 1024 ? 1024 : count;
        double _sum = 0.0;
        for (size_t i = 0; i < _n; i++) _sum += (double)_slot[i];
        uint64_t _bits;
        memcpy(&_bits, &_sum, sizeof(_bits));
        fprintf(stderr, " %d:%016lx", r, (unsigned long)_bits);
      }
      fprintf(stderr, "\n");
    }
    */

    /* ── CPU reduce: use first slot (rank 0's data) as accumulator ── */
    /* Data is arranged as [rank0, rank1, ..., rank_{N-1}] in cpu_all */
    void* acc = cpu_all.data();
    size_t acc_count = count;
    for (int r = 1; r < nranks; r++) {
        void* src = cpu_all.data() + (size_t)r * bytes;
        cpu_reduce(acc, src, acc_count, datatype, op);
    }

    /* ── Upload result ── */
    cudaMemcpyAsync(recvbuff, acc, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    /* ── Cleanup per-call buffer ── */
    c->endpoint->dereg(mr_all);
    cudaFree(gpu_all);

    // fprintf(stderr, "[p-rank%d] ncclAllReduce done\n", rank);
    return ncclSuccess;
}

/* ========================================================================= *
 * ncclAllToAll — batched all-recv-then-all-send over RDMA
 *
 * Recv channel groups are a limited UCCL resource (~6 per endpoint).  To
 * avoid "Recv channel group not found" errors, we process peers in batches
 * of 4: post recvs for the batch, post sends for the batch, poll, repeat.
 * ========================================================================= */
extern "C" int ncclAllToAll(const void* sendbuf, size_t sendcount, ncclDataType_t sendtype,
                            void* recvbuf, size_t recvcount, ncclDataType_t recvtype,
                            ncclComm_t comm, cudaStream_t stream) {
    (void)recvtype;
    (void)recvcount;
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c) return 1;

    int nranks = c->nranks;
    int rank = c->rank;
    size_t chunk = sendcount * dtype_size(sendtype);

    /* Build peer list (all ranks except self) */
    int peers[256];
    int n_peers = 0;
    for (int i = 0; i < nranks; i++) {
        if (i != rank) peers[n_peers++] = i;
    }

    const int BATCH = 4;
    for (int b = 0; b < n_peers; b += BATCH) {
        int end = (b + BATCH < n_peers) ? b + BATCH : n_peers;

        /* Post recvs for this batch */
        uint64_t recv_tids[256];
        for (int j = b; j < end; j++) {
            int p = peers[j];
            void* r = (uint8_t*)recvbuf + (size_t)p * chunk;
            uint64_t mr = reg_buffer(c, (uintptr_t)r, chunk, sendtype);
            if (mr == UINT64_MAX) return 1;
            if (!c->endpoint->recv_async(c->recv_conn_ids[p], mr, r, chunk, &recv_tids[j]))
                return 1;
        }

        /* Post sends for this batch */
        uint64_t send_tids[256];
        for (int j = b; j < end; j++) {
            int p = peers[j];
            const void* s = (const uint8_t*)sendbuf + (size_t)p * chunk;
            uint64_t mr = reg_buffer(c, (uintptr_t)s, chunk, sendtype);
            if (mr == UINT64_MAX) return 1;
            if (!c->endpoint->send_async(c->send_conn_ids[p], mr, s, chunk, &send_tids[j]))
                return 1;
        }

        /* Poll sends then recvs for this batch */
        for (int j = b; j < end; j++) {
            bool done = false;
            while (!done) { c->endpoint->poll_async(send_tids[j], &done); }
        }
        for (int j = b; j < end; j++) {
            bool done = false;
            while (!done) { c->endpoint->poll_async(recv_tids[j], &done); }
        }
    }

    /* Own chunk */
    if (sendbuf != recvbuf && chunk > 0)
        cudaMemcpyAsync((char*)recvbuf + (size_t)rank * chunk,
                        (const char*)sendbuf + (size_t)rank * chunk,
                        chunk, cudaMemcpyDeviceToDevice, stream);

    return 0;
}

/* ========================================================================= *
 * ncclBroadcast — binomial tree
 * ========================================================================= */
extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
                            size_t count, ncclDataType_t datatype,
                            int root, ncclComm_t comm,
                            cudaStream_t stream) {
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    int nranks = c->nranks;
    int rank = c->rank;
    size_t bytes = count * dtype_size(datatype);

    void* buf = recvbuff;
    if (rank == root && sendbuff != recvbuff)
        cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                        cudaMemcpyDeviceToDevice, stream);

    int tr = rank ^ root; /* tree-space: root becomes 0 */

    /* Receive from parent */
    for (int mask = 1; mask < nranks; mask <<= 1) {
        if (tr & mask) {
            int parent = (tr ^ mask) ^ root;
            uint64_t mr = reg_buffer(c, (uintptr_t)buf, bytes, datatype);
            if (mr == UINT64_MAX) return ncclInternalError;
            if (!uccl_recv_wait(c, c->recv_conn_ids[parent], buf, bytes, mr))
                return ncclInternalError;
            break;
        }
    }

    /* Send to children */
    for (int mask = 1; mask < nranks; mask <<= 1) {
        int child = (tr | mask) ^ root;
        if (child < nranks) {
            uint64_t mr = reg_buffer(c, (uintptr_t)buf, bytes, datatype);
            if (mr == UINT64_MAX) return ncclInternalError;
            if (!uccl_send_wait(c, c->send_conn_ids[child], buf, bytes, mr))
                return ncclInternalError;
        }
    }

    return ncclSuccess;
}

/* ========================================================================= *
 * ncclReduce — binomial tree with CPU reduction
 *
 * Each rank that receives from a child downloads the data to CPU, reduces
 * with its local contribution, and keeps the result in a CPU buffer.
 * Non-root ranks upload their final CPU buffer for send to parent.
 * Root uploads the final result to recvbuff.
 * ========================================================================= */
extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff,
                         size_t count, ncclDataType_t datatype,
                         ncclRedOp_t op, int root,
                         ncclComm_t comm, cudaStream_t stream) {
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess;

    int rank = c->rank;
    int nranks = c->nranks;
    size_t elt_size = dtype_size(datatype);
    size_t bytes = count * elt_size;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                            cudaMemcpyDeviceToDevice, stream);
        return ncclSuccess;
    }

    /* ── Download local data to CPU buffer as initial accumulator ── */
    std::vector<uint8_t> cpu_acc(bytes);
    cudaMemcpy(cpu_acc.data(), sendbuff, bytes, cudaMemcpyDeviceToHost);

    int tr = rank ^ root;
    size_t acc_count = count;

    /* Temp GPU buffer for receiving from children */
    void* gpu_tmp = nullptr;
    cudaMalloc(&gpu_tmp, bytes);
    std::vector<uint8_t> cpu_tmp(bytes);

    /* ── Binomial tree: receive from children, then send to parent ── */
    for (int mask = 1; mask < nranks; mask <<= 1) {
        if (tr & mask) {
            /* I'm a child: send accumulated result to parent */
            int parent = (rank ^ mask);
            cudaMemcpy(gpu_tmp, cpu_acc.data(), bytes, cudaMemcpyHostToDevice);

            uint64_t mr = reg_buffer(c, (uintptr_t)gpu_tmp, bytes, datatype);
            if (mr == UINT64_MAX) { cudaFree(gpu_tmp); return ncclInternalError; }

            if (!uccl_send_wait(c, c->send_conn_ids[parent], gpu_tmp, bytes, mr)) {
                cudaFree(gpu_tmp);
                return ncclInternalError;
            }
            /* Done for non-root */
            cudaFree(gpu_tmp);
            return ncclSuccess;
        } else {
            /* I may be parent of child = rank | mask */
            int child = (tr | mask) ^ root;
            if (child < nranks) {
                uint64_t mr = reg_buffer(c, (uintptr_t)gpu_tmp, bytes, datatype);
                if (mr == UINT64_MAX) { cudaFree(gpu_tmp); return ncclInternalError; }

                if (!uccl_recv_wait(c, c->recv_conn_ids[child], gpu_tmp, bytes, mr)) {
                    cudaFree(gpu_tmp);
                    return ncclInternalError;
                }

                cudaMemcpy(cpu_tmp.data(), gpu_tmp, bytes, cudaMemcpyDeviceToHost);
                cpu_reduce(cpu_acc.data(), cpu_tmp.data(), acc_count, datatype, op);
            }
        }
    }

    cudaFree(gpu_tmp);

    /* ── Root uploads final result ── */
    if (rank == root) {
        cudaMemcpyAsync(recvbuff, cpu_acc.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return ncclSuccess;
}

/* ========================================================================= *
 * ncclCommUserRank — extract rank from our comm struct
 *
 * Called by nccl_extensions.c (libnccl_extensions.so) to get rank/nranks
 * from the communicator.  Without these, the call falls through to real
 * NCCL which reads garbage from our fake comm pointer.
 * ========================================================================= */
extern "C" ncclResult_t ncclCommUserRank(ncclComm_t comm, int* rank) {
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c || !rank) return ncclInvalidArgument;
    *rank = c->rank;
    return ncclSuccess;
}

extern "C" ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    nccl_comm* c = reinterpret_cast<nccl_comm*>(comm);
    if (!c || !count) return ncclInvalidArgument;
    *count = c->nranks;
    return ncclSuccess;
}

/* ========================================================================= *
 * ncclGroupStart / ncclGroupEnd — batch no-op for sync wrappers
 *
 * Real NCCL batches operations inside group_start/end to optimise proxy
 * dispatch.  Our UCCL send/recv are synchronous (block until DMA completes),
 * so there is nothing to batch.  These exist because nccl_extensions.c wraps
 * its send/recv loops in ncclGroupStart/End.
 * ========================================================================= */
extern "C" ncclResult_t ncclGroupStart() {
    return ncclSuccess;
}

extern "C" ncclResult_t ncclGroupEnd() {
    return ncclSuccess;
}

/* ========================================================================= *
 * ncclGetErrorString
 * ========================================================================= */
const char* ncclGetErrorString(ncclResult_t error) {
    switch (error) {
    case ncclSuccess:             return "ncclSuccess";
    case ncclUnhandledCudaError:  return "ncclUnhandledCudaError";
    case ncclSystemError:         return "ncclSystemError";
    case ncclInternalError:       return "ncclInternalError";
    case ncclInvalidArgument:     return "ncclInvalidArgument";
    case ncclInvalidUsage:        return "ncclInvalidUsage";
    case ncclNumResults:          return "ncclNumResults";
    default:                      return "ncclUnknownError";
    }
}
