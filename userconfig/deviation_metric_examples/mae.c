/* Mean Absolute Error */
double compute_mae(const void *buf1, const void *buf2, int count, data_type_t dtype) {
    if (count <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += fabs(to_double(buf1, dtype, i) - to_double(buf2, dtype, i));
    }
    return sum / count;
}