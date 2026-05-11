/**
 * RCCL: find base (reference) implementation.
 *
 * Standalone executable. Discovers the real librccl.so (or libnccl.so)
 * path by trying dlsym(RTLD_DEFAULT) + dladdr first, then falling back
 * to common sonames. Writes the .so path to the state file.
 *
 * Usage: ./bin/rccl/findso
 * Returns 0 on success, 1 on failure.
 */

#define _GNU_SOURCE
#include "base_impl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static int find_base_so(const char *output_path)
{
    const char *so_path = NULL;

    /* 1. Check if ncclAllReduce is already loaded */
    void *sym = dlsym(RTLD_DEFAULT, "ncclAllReduce");
    if (sym) {
        Dl_info info;
        if (dladdr(sym, &info) && info.dli_fname)
            so_path = info.dli_fname;
    }

    /* 2. Try common RCCL/NCCL sonames */
    if (!so_path) {
        const char *candidates[] = {
            "librccl.so.1",
            "librccl.so",
            "libnccl.so.2",
            "libnccl.so",
        };
        for (int i = 0; i < 4; i++) {
            void *h = dlopen(candidates[i], RTLD_LAZY | RTLD_NOLOAD);
            if (!h)
                h = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
            if (h) {
                void *s = dlsym(h, "ncclAllReduce");
                if (s) {
                    Dl_info info;
                    if (dladdr(s, &info) && info.dli_fname)
                        so_path = info.dli_fname;
                    /* Keep h open — so_path points into this library's memory */
                    break;
                }
                dlclose(h);
            }
        }
    }

    if (!so_path) {
        fprintf(stderr, "[findso] ERROR: cannot locate librccl.so "
                "(ncclAllReduce not found)\n");
        return -1;
    }

    FILE *f = fopen(output_path, "w");
    if (!f) {
        perror("[findso] fopen");
        return -1;
    }
    fprintf(f, "%s\n", so_path);
    fclose(f);

    printf("[findso] RCCL base implementation: %s\n", so_path);
    return 0;
}

int main(void)
{
    return find_base_so(BASE_SO_FILE) != 0;
}