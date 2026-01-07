#pragma once

/**
 * Quotient Computation CUDA Kernel Declarations
 * 
 * GPU-accelerated operations for quotient polynomial computation.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Compute batch of (x - offset)^{-1} for each domain point
 * Used for initial zerofier: 1/(x - 1)
 */
void zerofier_inv_gpu(
    const uint64_t* d_domain_points,
    uint64_t offset,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Compute (x^power - 1)^{-1} for each domain point
 * Used for consistency/transition zerofiers
 */
void power_zerofier_inv_gpu(
    const uint64_t* d_domain_points,
    size_t power,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Compute weighted sum of XFieldElements
 * result = sum(values[i] * weights[i])
 * 
 * @param d_values Input XFieldElements (3 * num_elements)
 * @param d_weights Weights (3 * num_elements for XFE, or num_elements for BFE)
 * @param d_output Output XFieldElement (3 elements)
 * @param num_elements Number of elements to sum
 * @param weights_are_xfe True if weights are XFieldElements
 */
void xfe_weighted_sum_gpu(
    const uint64_t* d_values,
    const uint64_t* d_weights,
    uint64_t* d_output,
    size_t num_elements,
    bool weights_are_xfe,
    cudaStream_t stream = 0
);

/**
 * Batch weighted sum: compute weighted sums for multiple rows
 * Each row has num_elements values with shared weights
 * 
 * @param d_values [num_rows * num_elements * 3]
 * @param d_weights [num_elements * 3]
 * @param d_output [num_rows * 3]
 */
void xfe_weighted_sum_batch_gpu(
    const uint64_t* d_values,
    const uint64_t* d_weights,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_elements,
    cudaStream_t stream = 0
);

/**
 * Element-wise XFE multiply: output[i] = a[i] * b[i]
 */
void xfe_elementwise_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Scale XFE by BFE inverse: output[i] = xfe[i] * bfe_inv[i]
 */
void xfe_scale_by_inv_gpu(
    const uint64_t* d_xfe,
    const uint64_t* d_bfe_inv,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

