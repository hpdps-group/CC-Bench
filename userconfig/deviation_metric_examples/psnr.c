/* Peak Signal-to-Noise Ratio.
 * Extra env vars:
 *   METRIC_PSNR_MAX_VAL — max possible value (default: auto-detect from dtype)
 */
double compute_psnr(const void *buf1, const void *buf2, int count, data_type_t dtype) {
    if (count <= 0) return INFINITY;

    /* Compute MSE inline (no dependency on other metrics) */
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = to_double(buf1, dtype, i) - to_double(buf2, dtype, i);
        sum += diff * diff;
    }
    double mse = sum / count;
    if (mse <= 0.0) return INFINITY;

    double max_val = 0.0;
    const char *env = getenv("METRIC_PSNR_MAX_VAL");
    if (env) {
        max_val = atof(env);
    } else {
        switch (dtype) {
            case TYPE_CHAR:   max_val = 255.0; break;
            case TYPE_INT:    max_val = 65535.0; break;
            case TYPE_FLOAT:  max_val = 1.0; break;
            case TYPE_DOUBLE: max_val = 1.0; break;
            default:          max_val = 255.0; break;
        }
    }
    return 20.0 * log10(max_val) - 10.0 * log10(mse);
}