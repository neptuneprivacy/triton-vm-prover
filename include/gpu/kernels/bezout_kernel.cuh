/**
 * GPU Bézout Coefficient Computation Kernel
 * 
 * Computes the Bézout coefficient polynomials for RAM table using GPU-accelerated
 * NTT-based polynomial multiplication and parallel prefix/suffix products.
 * 
 * For n unique RAM pointers, this replaces O(n² log n) CPU computation with
 * O(n log² n) GPU-parallel computation.
 */

#pragma once

#ifdef TRITON_CUDA_ENABLED

#include "types/b_field_element.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Compute Bézout coefficient polynomials on GPU
 * 
 * Given unique_ramps (the distinct RAM pointer values), computes:
 * - rp(x) = Π (x - r) for r in unique_ramps
 * - fd(x) = rp'(x) (derivative)
 * - b(x) = interpolation polynomial where b(r_i) = 1/fd(r_i)
 * - a(x) = (1 - fd*b) / rp
 * 
 * Returns (a_coeffs, b_coeffs) as vectors of BFieldElement
 * 
 * @param unique_ramps Vector of unique RAM pointer values
 * @param stream CUDA stream to use
 * @return Pair of coefficient vectors (a, b)
 */
std::pair<std::vector<BFieldElement>, std::vector<BFieldElement>>
gpu_compute_bezout_coefficients(
    const std::vector<BFieldElement>& unique_ramps,
    cudaStream_t stream = nullptr
);

/**
 * GPU polynomial multiplication using NTT
 * 
 * @param d_a Device pointer to polynomial a coefficients
 * @param a_size Size of polynomial a
 * @param d_b Device pointer to polynomial b coefficients
 * @param b_size Size of polynomial b
 * @param d_result Device pointer for result (must be allocated with size a_size + b_size - 1)
 * @param stream CUDA stream
 */
void gpu_poly_mul(
    const uint64_t* d_a, size_t a_size,
    const uint64_t* d_b, size_t b_size,
    uint64_t* d_result,
    cudaStream_t stream = nullptr
);

/**
 * Compute product tree on GPU: given n linear factors (x - r_i),
 * compute their product as a polynomial.
 * Uses divide-and-conquer with parallel NTT multiplication at each level.
 * 
 * @param d_roots Device pointer to root values r_0, r_1, ..., r_{n-1}
 * @param n Number of roots
 * @param d_result Device pointer for result polynomial (size n+1)
 * @param stream CUDA stream
 */
void gpu_product_tree(
    const uint64_t* d_roots,
    size_t n,
    uint64_t* d_result,
    cudaStream_t stream = nullptr
);

/**
 * Parallel polynomial evaluation at multiple points
 * 
 * @param d_coeffs Device pointer to polynomial coefficients
 * @param degree Polynomial degree
 * @param d_points Device pointer to evaluation points
 * @param n Number of points
 * @param d_results Device pointer for results (size n)
 * @param stream CUDA stream
 */
void gpu_poly_eval_batch(
    const uint64_t* d_coeffs, size_t degree,
    const uint64_t* d_points, size_t n,
    uint64_t* d_results,
    cudaStream_t stream = nullptr
);

/**
 * Batch field element inversion on GPU
 * 
 * @param d_values Device pointer to values to invert
 * @param n Number of values
 * @param d_results Device pointer for results
 * @param stream CUDA stream
 */
void gpu_batch_inversion(
    const uint64_t* d_values,
    size_t n,
    uint64_t* d_results,
    cudaStream_t stream = nullptr
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

