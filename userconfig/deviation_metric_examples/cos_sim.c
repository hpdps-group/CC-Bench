/* Cosine Similarity: dot(buf1, buf2) / (||buf1|| * ||buf2||).
 * Returns 1.0 when both norms are zero (identical zero vectors).
 * Returns 0.0 when exactly one norm is zero (no similarity).
 */
double compute_cos_sim(const void *buf1, const void *buf2, int count, data_type_t dtype) {
    if (count <= 0) return 0.0;

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < count; i++) {
        double a = to_double(buf1, dtype, i);
        double b = to_double(buf2, dtype, i);
        dot  += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }

    if (norm1 == 0.0 && norm2 == 0.0) return 1.0;
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    return dot / (sqrt(norm1) * sqrt(norm2));
}
