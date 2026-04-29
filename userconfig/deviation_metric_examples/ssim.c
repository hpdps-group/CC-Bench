/* Structural Similarity (1D simplified).
 * Extra env vars:
 *   METRIC_SSIM_K1 — stability constant factor (default: 0.01)
 *   METRIC_SSIM_K2 — stability constant factor (default: 0.03)
 */
double compute_ssim(const void *buf1, const void *buf2, int count, data_type_t dtype) {
    if (count <= 0) return 0.0;

    double k1 = 0.01, k2 = 0.03;
    const char *ek1 = getenv("METRIC_SSIM_K1");
    const char *ek2 = getenv("METRIC_SSIM_K2");
    if (ek1) k1 = atof(ek1);
    if (ek2) k2 = atof(ek2);

    double mean1 = 0.0, mean2 = 0.0;
    for (int i = 0; i < count; i++) {
        mean1 += to_double(buf1, dtype, i);
        mean2 += to_double(buf2, dtype, i);
    }
    mean1 /= count;
    mean2 /= count;

    double var1 = 0.0, var2 = 0.0, cov = 0.0;
    for (int i = 0; i < count; i++) {
        double v1 = to_double(buf1, dtype, i);
        double v2 = to_double(buf2, dtype, i);
        var1 += (v1 - mean1) * (v1 - mean1);
        var2 += (v2 - mean2) * (v2 - mean2);
        cov  += (v1 - mean1) * (v2 - mean2);
    }
    var1 /= count;
    var2 /= count;
    cov  /= count;

    double C1 = k1 * k1;
    double C2 = k2 * k2;
    double num = (2.0 * mean1 * mean2 + C1) * (2.0 * cov + C2);
    double den = (mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2);
    return (den == 0.0) ? 1.0 : num / den;
}