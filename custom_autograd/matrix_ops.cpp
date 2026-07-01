#include "matrix_ops.hpp"

// Default constructor
RawMatrix::RawMatrix() : rows(0), cols(0) {}

// Zero-initialized constructor
RawMatrix::RawMatrix(size_t r, size_t c) : data(r * c, 0.0f), rows(r), cols(c) {}

// Value-initialized constructor with size checking
RawMatrix::RawMatrix(size_t r, size_t c, const std::vector<float>& d) : data(d), rows(r), cols(c) {
    if (d.size() != r * c) {
        throw std::invalid_argument("RawMatrix initialization error: data vector size does not match rows * cols");
    }
}

/**
 * Transposes matrix A (R x C) into an output matrix (C x R).
 */
RawMatrix transpose(const RawMatrix& A) {
    RawMatrix out(A.cols, A.rows);
    for (size_t r = 0; r < A.rows; ++r) {
        size_t r_idx = r * A.cols;
        for (size_t c = 0; c < A.cols; ++c) {
            out.data[c * A.rows + r] = A.data[r_idx + c];
        }
    }
    return out;
}

/**
 * Multiplies matrices A (M x K) and B (K x N).
 * Optimized for cache locality: pre-transposes matrix B so the inner loop
 * performs a dot-product between two contiguous rows in memory.
 */
RawMatrix matmul(const RawMatrix& A, const RawMatrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix multiplication error: Dimension mismatch (A.cols != B.rows)");
    }
    
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    RawMatrix out(M, N);
    
    // Cache locality optimization: Transpose matrix B to align columns contiguously in memory
    RawMatrix BT = transpose(B);
    
    for (size_t i = 0; i < M; ++i) {
        size_t i_K = i * K;
        size_t i_N = i * N;
        for (size_t j = 0; j < N; ++j) {
            size_t j_K = j * K;
            float sum = 0.0f;
            // Both arrays accessed linearly (cache friendly stride 1)
            for (size_t k = 0; k < K; ++k) {
                sum += A.data[i_K + k] * BT.data[j_K + k];
            }
            out.data[i_N + j] = sum;
        }
    }
    return out;
}

/**
 * Adds two matrices element-wise. Requires matching dimensions.
 */
RawMatrix elementwise_add(const RawMatrix& A, const RawMatrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Elementwise addition error: Dimension mismatch (matrices must have matching shapes)");
    }
    
    RawMatrix out(A.rows, A.cols);
    size_t total_elements = A.rows * A.cols;
    
    for (size_t i = 0; i < total_elements; ++i) {
        out.data[i] = A.data[i] + B.data[i];
    }
    return out;
}
