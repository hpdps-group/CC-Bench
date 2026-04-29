/* Mean Squared Error */
double compute_mse(const void *buf1, const void *buf2, int count, data_type_t dtype) {
    if (count <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = to_double(buf1, dtype, i) - to_double(buf2, dtype, i);
        sum += diff * diff;
    }
    return sum / count;
}