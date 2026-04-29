/**
 * MPI: find base (reference) implementation.
 *
 * Standalone executable. Verifies that PMPI is available
 * (every compliant MPI implementation exports PMPI_* symbols)
 * and records "builtin:pmpi" to the state file.
 *
 * Usage: ./bin/mpi/findso
 * Returns 0 on success, 1 on failure.
 */

#define _GNU_SOURCE
#include "base_impl.h"
#include <dlfcn.h>
#include <stdio.h>

static int find_base_so(const char *output_path)
{
    void *sym = dlsym(RTLD_DEFAULT, "PMPI_Allreduce");
    if (!sym) {
        fprintf(stderr, "[findso] ERROR: PMPI_Allreduce not found — "
                "MPI implementation may not support PMPI\n");
        return -1;
    }

    FILE *f = fopen(output_path, "w");
    if (!f) {
        perror("[findso] fopen");
        return -1;
    }
    fprintf(f, "builtin:pmpi\n");
    fclose(f);

    printf("[findso] PMPI verified, reference: %p\n", sym);
    return 0;
}

int main(void)
{
    return find_base_so(BASE_SO_FILE) != 0;
}