/**
 * Utility functions — generic, no architecture dependency.
 * Uses data_type_t for type information instead of MPI/NCCL types.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

/* Generic data type enum (architecture-independent) */
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_DOUBLE,
    TYPE_CHAR
} data_type_t;

/* Buffer data pattern types */
typedef enum {
    PATTERN_PLAIN         = 0,  /* all 1 (uniform) */
    PATTERN_RANK_LINEAR   = 1,  /* rank + index */
    PATTERN_RANK_PRODUCT  = 2,  /* (rank+1) * (index+1) */
    PATTERN_SEQUENTIAL    = 3   /* rank * count + index */
} pattern_type_t;

/* File reading */
typedef enum {
    FILE_FORMAT_BINARY,
    FILE_FORMAT_TEXT
} file_format_t;

/**
 * Read data from file into buffer.
 */
int read_from_file(
    const char *filename,
    void *buffer,
    size_t buffer_size,
    file_format_t format,
    data_type_t datatype
);

/**
 * Write buffer to file.
 */
int write_to_file(
    const char *filename,
    const void *buffer,
    size_t buffer_size,
    file_format_t format,
    data_type_t datatype
);

/**
 * Allocate aligned memory for communication buffers.
 */
void *allocate_aligned_buffer(size_t size, size_t alignment);

/**
 * Free aligned buffer.
 */
void free_aligned_buffer(void *ptr);

/**
 * Initialize buffer with pattern based on rank.
 */
/**
 * Parse pattern type from string ("plain", "rank_linear", etc.).
 * Aborts on unknown string.
 */
pattern_type_t parse_pattern_type(const char *name);

void init_buffer_pattern(
    void *buffer,
    int count,
    data_type_t datatype,
    pattern_type_t pattern_type,
    int rank
);

/**
 * Print buffer contents (for debugging).
 */
void print_buffer(
    const void *buffer,
    int count,
    data_type_t datatype,
    int max_to_print
);

/**
 * Message size iterator — abstracts over -m and -L modes.
 *
 * Usage:
 *   size_iter_t it;
 *   size_iter_init(&it, &config);
 *   size_t sz;
 *   while (size_iter_next(&it, &sz)) { ... }
 */
typedef struct {
    int use_list;
    int index;
    int num_sizes;
    size_t size_list[256];
    size_t min_size, max_size;
    int incr;
    int incr_mode;  /* 0=multiply, 1=add */
} size_iter_t;

/**
 * Parse command-line arguments for test configuration.
 */
typedef struct {
    /* Message size range (-m MIN:MAX:INCR) */
    size_t min_message_size;
    size_t max_message_size;
    int message_size_incr;
    int message_size_incr_mode;  /* 0=multiply (default), 1=add (-A) */

    /* Explicit size list (-L X,Y,Z), overrides min/max/incr */
    int use_size_list;
    int num_sizes;
    size_t size_list[256];

    /* Iterations */
    int iterations;
    int warmup_iterations;

    /* CSV output (-c, -o) */
    int save_csv;
    char csv_path[512];

    /* Binary output (-b, -B) — write reference + user data for correctness */
    int save_binary;
    char bin_path[512];

    /* File input */
    char *input_file;
    file_format_t file_format;

    /* Validation */
    double tolerance;
    int validate;
    int compute_metrics;
    uint32_t metrics_mask;

    /* Data */
    pattern_type_t pattern_type;
    data_type_t data_type;

    /* Reference save/load (-S DIR, -R DIR) */
    char save_ref_path[512];   /* non-empty = save-ref mode: run once, save to dir */
    char ref_dir[512];         /* non-empty = load reference from dir instead of computing */
} test_config_t;

test_config_t parse_arguments(int argc, char **argv);

void size_iter_init(size_iter_t *it, const test_config_t *config);
int  size_iter_next(size_iter_t *it, size_t *size);

#endif /* UTILS_H */