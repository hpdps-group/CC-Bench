/**
 * User wrapper template — implement your own MPI functions here.
 *
 * Fill in the functions below with your custom algorithms.
 * When compiled as a shared library and loaded via LD_PRELOAD, these
 * functions will intercept the corresponding MPI calls from any application.
 *
 * RULES:
 *   1. Use STANDARD MPI function names (MPI_Allreduce, MPI_Bcast, etc.)
 *   2. To call the original MPI implementation (no recursion), use PMPI_*:
 *          PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
 *      PMPI is the MPI Profiling Interface — always the original implementation.
 *   3. If you only implement some functions, the unimplemented ones will
 *      automatically fallback to the original MPI library.
 *   4. Read config from environment variables using getenv() — no need
 *      for global state. Your slurm_config.jsonc can set these via
 *      "environment_variables" or "wrapper_env_vars".
 *
 * Compile: mpicc -shared -fPIC -o libuser_wrapper.so user_wrapper_template.c
 * Run:     LD_PRELOAD=./libuser_wrapper.so mpirun -np 4 ./your_app
 *
 * Environment variables (all optional — define your own):
 *   MPI_TOLERANCE         Algorithm tolerance
 *   MPI_BLOCK_SIZE        Block size
 *   MPI_COMPRESSION_RATIO Compression ratio
 *   MPI_DEBUG             Debug output (0/1)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* PMPI declarations — call the original MPI without recursion */
extern int PMPI_Init(int *argc, char ***argv);
extern int PMPI_Finalize(void);
extern int PMPI_Comm_rank(MPI_Comm comm, int *rank);
extern int PMPI_Comm_size(MPI_Comm comm, int *size);
extern int PMPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
extern int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
extern int PMPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                      int root, MPI_Comm comm);
extern int PMPI_Barrier(MPI_Comm comm);
extern int PMPI_Send(const void *buf, int count, MPI_Datatype datatype,
                     int dest, int tag, MPI_Comm comm);
extern int PMPI_Recv(void *buf, int count, MPI_Datatype datatype,
                     int source, int tag, MPI_Comm comm, MPI_Status *status);
extern int PMPI_Isend(const void *buf, int count, MPI_Datatype datatype,
                      int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern int PMPI_Irecv(void *buf, int count, MPI_Datatype datatype,
                      int source, int tag, MPI_Comm comm, MPI_Request *request);
extern int PMPI_Wait(MPI_Request *request, MPI_Status *status);
extern int PMPI_Waitall(int count, MPI_Request array_of_requests[],
                        MPI_Status array_of_statuses[]);

/*===========================================================================*
 *  INIT / FINALIZE — pass through to original MPI                          *
 *  Delete these if you don't need custom init logic.                       *
 *===========================================================================*/

int MPI_Init(int *argc, char ***argv) {
    return PMPI_Init(argc, argv);
}

int MPI_Finalize(void) {
    return PMPI_Finalize();
}

int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    return PMPI_Comm_rank(comm, rank);
}

int MPI_Comm_size(MPI_Comm comm, int *size) {
    return PMPI_Comm_size(comm, size);
}

/*===========================================================================*
 *  MPI_Allreduce — EXAMPLE                                                 *
 *                                                                          *
 *  This shows the pattern:                                                 *
 *    1. Read config from environment variables                             *
 *    2. Call your custom function with those values                        *
 *    3. On failure, fallback to PMPI_Allreduce (original MPI)              *
 *                                                                          *
 *  For example, if you have a custom function like:                        *
 *    int MPI_Allreduce_ZCCL_RI2_mt_oa_record(                              *
 *        const void *sendbuf, void *recvbuf,                               *
 *        float compressionRatio, float tolerance, int blockSize,           *
 *        MPI_Aint count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)  *
 *                                                                          *
 *  You would read compressionRatio, tolerance, blockSize from env vars     *
 *  and call it here.  See the commented example below.                     *
 *===========================================================================*/

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

    /* ---- YOUR CODE HERE ---- */

    /*
     * Example: read config from environment variables and dispatch
     * to your custom function.  These vars are set in slurm_config.jsonc
     * under "environment_variables" or "wrapper_env_vars".
     *
     *   // 1. Read parameters from environment
     *   float tolerance = 1e-5;
     *   int blockSize = 1024;
     *   float compressionRatio = 0.5;
     *   char *env;
     *   if ((env = getenv("MPI_TOLERANCE")) != NULL)
     *       tolerance = atof(env);
     *   if ((env = getenv("MPI_BLOCK_SIZE")) != NULL)
     *       blockSize = atoi(env);
     *   if ((env = getenv("MPI_COMPRESSION_RATIO")) != NULL)
     *       compressionRatio = atof(env);
     *
     *   // 2. Debug output
     *   if ((env = getenv("MPI_DEBUG")) != NULL && atoi(env)) {
     *       int rank;
     *       PMPI_Comm_rank(comm, &rank);
     *       fprintf(stderr, "[User] rank=%d Allreduce count=%d "
     *               "tol=%e bs=%d comp=%f\n",
     *               rank, count, tolerance, blockSize, compressionRatio);
     *   }
     *
     *   // 3. Call your custom function : ZCCL as an example
     *   int ret = MPI_Allreduce_ZCCL_RI2_mt_oa_record(
     *       sendbuf, recvbuf,
     *       compressionRatio, tolerance, blockSize,
     *       (MPI_Aint)count, datatype, op, comm);
     *
     *   // 4. Fallback to original MPI on failure
     *   if (ret != MPI_SUCCESS)
     *       return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
     *   return ret;
     */

    /* Default: call original MPI (safe fallback, no recursion) */
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

/*===========================================================================*
 *  MPI_Reduce — implement or delete to fallback to original                *
 *===========================================================================*/

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {

    /* ---- YOUR CODE HERE ---- */

    return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

/*===========================================================================*
 *  MPI_Bcast — implement or delete to fallback to original                 *
 *===========================================================================*/

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {

    /* ---- YOUR CODE HERE ---- */

    return PMPI_Bcast(buffer, count, datatype, root, comm);
}

/*===========================================================================*
 *  MPI_Barrier — implement or delete to fallback to original               *
 *===========================================================================*/

int MPI_Barrier(MPI_Comm comm) {

    /* ---- YOUR CODE HERE ---- */

    return PMPI_Barrier(comm);
}
