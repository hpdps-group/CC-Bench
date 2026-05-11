#include "mpi/perf_helper_mpi.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Global (per-process) state ────────────────────────────────── */
static perf_state_t   g_state;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void do_init(void)
{
    perf_init(&g_state);
}

perf_state_t *perf_mpi_get_tls(void)
{
    pthread_once(&g_once, do_init);
    return &g_state;
}

/* ── Node map for intra/inter detection ──────────────────────────── */
#define PERF_MPI_NAME_LEN MPI_MAX_PROCESSOR_NAME

static char           s_my_name[PERF_MPI_NAME_LEN];
static char          *s_all_names = NULL;
static int            s_world_rank = -1;
static int            s_world_size = 0;
static int            s_node_map_ready = 0;
static pthread_once_t g_node_map_once = PTHREAD_ONCE_INIT;

static void do_init_node_map(void)
{
    int flag = 0;
    PMPI_Initialized(&flag);
    if (!flag) return;

    int name_len;
    PMPI_Get_processor_name(s_my_name, &name_len);
    PMPI_Comm_rank(MPI_COMM_WORLD, &s_world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &s_world_size);

    if (s_world_size > 0) {
        s_all_names = malloc((size_t)s_world_size * PERF_MPI_NAME_LEN);
        if (s_all_names) {
            PMPI_Allgather(s_my_name, PERF_MPI_NAME_LEN, MPI_CHAR,
                           s_all_names, PERF_MPI_NAME_LEN, MPI_CHAR,
                           MPI_COMM_WORLD);
            s_node_map_ready = 1;
        }
    }
}

void perf_mpi_init_node_map(void)
{
    pthread_once(&g_node_map_once, do_init_node_map);
}

int perf_mpi_world_rank(void)
{
    return s_world_rank;
}

int perf_mpi_is_intra(int rank)
{
    if (!s_node_map_ready || rank < 0 || rank >= s_world_size)
        return 0;
    return strncmp(s_all_names + (size_t)rank * PERF_MPI_NAME_LEN,
                   s_my_name, PERF_MPI_NAME_LEN) == 0;
}

double perf_mpi_msg_size(int count, MPI_Datatype datatype)
{
    int dtype_size = 0;
    PMPI_Type_size(datatype, &dtype_size);
    return (double)count * (double)dtype_size;
}

/* ── flush ───────────────────────────────────────────────────────── */
void perf_mpi_flush(void)
{
    if (g_state.total_records == 0) return;

    int rank = 0;
    int flag = 0;
    PMPI_Initialized(&flag);
    if (flag)
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* MPI may already be shutting down — best-effort write */
    perf_flush_to_rank_file(&g_state, rank, "perf_function");
    perf_destroy(&g_state);
}

/* ── auto-flush at exit ──────────────────────────────────────────── */
__attribute__((destructor))
static void perf_mpi_on_exit(void)
{
    perf_mpi_flush();
    free(s_all_names);
    s_all_names = NULL;
}
