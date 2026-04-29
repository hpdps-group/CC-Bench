/**
 * MPI: load base (reference) implementation.
 *
 * PMPI is linked at build-time and requires no runtime dlopen.
 * This module just verifies the state file written by findso.c
 * and prints a confirmation.
 */

#include "base_impl.h"
#include <stdio.h>
#include <string.h>

int load_base_impl(const char *state_path)
{
    FILE *f = fopen(state_path, "r");
    if (!f) {
        perror("[base_impl] fopen");
        return -1;
    }

    char buf[128];
    if (!fgets(buf, sizeof(buf), f)) {
        fprintf(stderr, "[base_impl] state file is empty\n");
        fclose(f);
        return -1;
    }
    fclose(f);

    /* Strip trailing newline */
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

    /* suppress per-rank log; enabled via -v if needed */
    return 0;
}