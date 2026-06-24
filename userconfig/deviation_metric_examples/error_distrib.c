static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

double compute_error_distrib(const void *buf1, const void *buf2,
                             int count, data_type_t dtype)
{
    if (count <= 0) return 0.0;

    double max_err = 0.0;
    for (int i = 0; i < count; i++) {
        double d = fabs(to_double(buf1, dtype, i) - to_double(buf2, dtype, i));
        if (d > max_err) max_err = d;
    }

    const char *out = getenv("METRIC_ERROR_DISTRIB_OUTPUT");
    if (!out) out = "error_distrib.txt";

    float *err = (float *)malloc((size_t)count * sizeof(float));
    if (!err) {
        FILE *f = fopen(out, "a");
        if (f) { fprintf(f, "Max=%.6e  msg_bytes=%zu\n", max_err, (size_t)count * get_element_size(dtype)); fclose(f); }
        return -1.0;
    }

    for (int i = 0; i < count; i++)
        err[i] = (float)fabs(to_double(buf1, dtype, i) - to_double(buf2, dtype, i));

    qsort(err, (size_t)count, sizeof(float), cmp_float);

    int c = count - 1;
    double p50 = err[(int)(0.50 * c)], p95  = err[(int)(0.95 * c)];
    double p99 = err[(int)(0.99 * c)], p999 = err[(int)(0.999 * c)];

    FILE *f = fopen(out, "a");
    if (f) {
        fprintf(f, "P50=%.6e  P95=%.6e  P99=%.6e  P99.9=%.6e  Max=%.6e  msg_bytes=%zu\n",
                p50, p95, p99, p999, max_err, (size_t)count * get_element_size(dtype));
        fclose(f);
    }

    free(err);
    return p50;
}
