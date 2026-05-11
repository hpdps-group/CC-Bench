/**
 * NCCL: find base (reference) implementation.
 *
 * Standalone executable. Discovers the real libnccl.so path by
 * trying dlsym(RTLD_DEFAULT) + dladdr first, then falling back
 * to common sonames.  Writes the .so path to the state file.
 *
 * Custom NCCL implementations (COCCL, ZCCL, etc.) that share the
 * same soname "libnccl.so" are excluded so the real NCCL is used
 * as the reference.
 *
 * Usage: ./bin/nccl/findso
 * Returns 0 on success, 1 on failure.
 */

#define _GNU_SOURCE
#include "base_impl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * Substrings that identify custom NCCL implementations.
 * Any library whose path contains one of these is NOT considered
 * the "real" NCCL and will be skipped when locating the base impl.
 */
static const char *excluded_patterns[] = {
    "coccl",
    "zccl",
    NULL  /* sentinel */
};

/* Returns 1 if path contains an excluded pattern */
static int is_excluded(const char *path)
{
    if (!path) return 0;
    for (int i = 0; excluded_patterns[i]; i++) {
        if (strstr(path, excluded_patterns[i]))
            return 1;
    }
    return 0;
}

static int find_base_so(const char *output_path)
{
    const char *so_path = NULL;
    Dl_info info;

    /* ── 1. Check if ncclAllReduce is already loaded ────────────── */
    void *sym = dlsym(RTLD_DEFAULT, "ncclAllReduce");
    if (sym && dladdr(sym, &info) && info.dli_fname) {
        if (!is_excluded(info.dli_fname))
            so_path = info.dli_fname;
    }

    /* ── 2. Try common sonames, skip excluded paths ─────────────── */
    if (!so_path) {
        const char *candidates[] = {
            "libnccl.so.2",
            "libnccl.so",
            "libnccl.so.1",
            NULL
        };
        for (int i = 0; candidates[i]; i++) {
            void *h = dlopen(candidates[i], RTLD_LAZY | RTLD_NOLOAD);
            if (!h)
                h = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
            if (!h) continue;

            void *s = dlsym(h, "ncclAllReduce");
            if (s && dladdr(s, &info) && info.dli_fname) {
                if (!is_excluded(info.dli_fname)) {
                    so_path = info.dli_fname;
                    /* Keep h open — so_path points into this library's memory */
                    break;
                }
                /* Excluded: keep looking */
            }
            dlclose(h);
        }
    }

    /* ── 3. Scan each directory in LD_LIBRARY_PATH manually ──────── */
    if (!so_path) {
        const char *llp = getenv("LD_LIBRARY_PATH");
        if (llp) {
            char *copy = strdup(llp);
            if (copy) {
                const char *candidates[] = {
                    "libnccl.so.2", "libnccl.so", "libnccl.so.1", NULL
                };
                char *dir = copy, *next;
                do {
                    next = strchr(dir, ':');
                    if (next) *next = '\0';

                    /* Skip excluded directories */
                    if (!is_excluded(dir)) {
                        for (int i = 0; candidates[i]; i++) {
                            char full[4096];
                            int n = snprintf(full, sizeof(full), "%s/%s",
                                             dir, candidates[i]);
                            if (n < 0 || (size_t)n >= sizeof(full)) continue;

                            void *h = dlopen(full, RTLD_LAZY | RTLD_LOCAL);
                            if (!h) continue;

                            void *s = dlsym(h, "ncclAllReduce");
                            Dl_info info2;
                            if (s && dladdr(s, &info2) && info2.dli_fname) {
                                if (!is_excluded(info2.dli_fname)) {
                                    so_path = info2.dli_fname;
                                    /* Keep h open — so_path points into this library's memory */
                                    break;
                                }
                            }
                            dlclose(h);
                        }
                        if (so_path) break;
                    }

                    dir = next ? next + 1 : NULL;
                } while (dir);
                free(copy);
            }
        }
    }

    if (!so_path) {
        fprintf(stderr, "[findso] ERROR: cannot locate libnccl.so "
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

    printf("[findso] NCCL base implementation: %s\n", so_path);
    return 0;
}

int main(void)
{
    return find_base_so(BASE_SO_FILE) != 0;
}
