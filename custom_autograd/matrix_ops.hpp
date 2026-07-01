#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>

/**
 * Lightweight structure holding flat row-major matrix data and dimensions.
 */
struct RawMatrix {
    std::vector<float> data;
    size_t rows;
    size_t cols;

    // Constructors
    RawMatrix();
    RawMatrix(size_t r, size_t c);
    RawMatrix(size_t r, size_t c, const std::vector<float>& d);
};

// core operations declarations
RawMatrix matmul(const RawMatrix& A, const RawMatrix& B);
RawMatrix transpose(const RawMatrix& A);
RawMatrix elementwise_add(const RawMatrix& A, const RawMatrix& B);

#endif // MATRIX_OPS_HPP
