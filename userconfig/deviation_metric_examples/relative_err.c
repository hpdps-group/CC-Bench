/* Element-wise Relative Error: |ref - comp| / max(|ref|, eps).
 * buf1 = user (compressed), buf2 = reference (original).
 * Extra env vars:
 *   METRIC_RELATIVE_ERR_EPS — denominator floor to avoid division by zero (default: 1e-8)
 */
double compute_relative_err(const void *buf1, const void *buf2, int count, data_type_t dtype) {
    if (count <= 0) return 0.0;

    double eps = 1e-8;
    const char *e = getenv("METRIC_RELATIVE_ERR_EPS");
    if (e) eps = atof(e);

    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        double a = to_double(buf1, dtype, i);
        double b = to_double(buf2, dtype, i);
        double denom = fabs(b);
        if (denom < eps) denom = eps;
        sum += fabs(a - b) / denom;
    }
    return sum / count;
}
