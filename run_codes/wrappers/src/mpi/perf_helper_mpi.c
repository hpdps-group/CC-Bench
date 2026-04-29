#include "mpi/perf_helper_mpi.h"
#include <mpi.h>
#include <stdio.h>

/* ── TLS state ──────────────────────────────────────────────────── */
static __thread perf_state_t tls_state;
static __thread int          tls_active = 0;

perf_state_t *perf_mpi_get_tls(void)
{
    if (!tls_active) {
        perf_init(&tls_state);
        tls_active = 1;
    }
    return &tls_state;
}

/* ── flush ───────────────────────────────────────────────────────── */
void perf_mpi_flush(void)
{
    if (!tls_active || tls_state.total_records == 0) return;

    int rank = 0;
    int flag = 0;
    PMPI_Initialized(&flag);
    if (flag)
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* MPI may already be shutting down — best-effort write */
    perf_flush_to_rank_file(&tls_state, rank, "perf_mpi");
    perf_destroy(&tls_state);
    tls_active = 0;
}

/* ── auto-flush at exit ──────────────────────────────────────────── */
__attribute__((destructor))
static void perf_mpi_on_exit(void)
{
    perf_mpi_flush();
}