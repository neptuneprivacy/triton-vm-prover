/**
 * Quotient Computation CUDA Kernel Implementation
 * 
 * GPU-accelerated operations for quotient polynomial computation:
 * - Weighted sum of XFieldElements
 * - Zerofier inverse batch computation
 * - Quotient combination
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Zerofier Inverse Kernels
// ============================================================================

/**
 * Compute batch of (x - g^i)^{-1} for each row
 * Used for initial zerofier: 1/(x - 1)
 * @param d_domain_points Evaluation points x_i
 * @param d_offset The value to subtract (e.g., 1 for initial zerofier)
 * @param d_output Output inverses
 * @param n Number of points
 */
__global__ void zerofier_inv_kernel(
    const uint64_t* d_domain_points,
    uint64_t offset,
    uint64_t* d_output,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute (x - offset)^{-1}
    uint64_t x = d_domain_points[idx];
    uint64_t diff = bfield_sub_impl(x, offset);
    d_output[idx] = bfield_inv_impl(diff);
}

/**
 * Compute (x^n - 1)^{-1} for each evaluation point
 * Used for consistency/transition zerofier
 */
__global__ void power_zerofier_inv_kernel(
    const uint64_t* d_domain_points,
    size_t power,
    uint64_t* d_output,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t x = d_domain_points[idx];
    uint64_t x_pow = bfield_pow_impl(x, power);
    uint64_t diff = bfield_sub_impl(x_pow, 1);
    d_output[idx] = bfield_inv_impl(diff);
}

// ============================================================================
// XFieldElement Weighted Sum Kernel
// ============================================================================

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
__global__ void xfe_weighted_sum_kernel(
    const uint64_t* d_values,
    const uint64_t* d_weights,
    uint64_t* d_output,
    size_t num_elements,
    bool weights_are_xfe
) {
    // Single-threaded accumulation (could be parallelized with reduction)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
    
    for (size_t i = 0; i < num_elements; ++i) {
        uint64_t v0 = d_values[i * 3 + 0];
        uint64_t v1 = d_values[i * 3 + 1];
        uint64_t v2 = d_values[i * 3 + 2];
        uint64_t t0, t1, t2;
        
        if (weights_are_xfe) {
            uint64_t w0 = d_weights[i * 3 + 0];
            uint64_t w1 = d_weights[i * 3 + 1];
            uint64_t w2 = d_weights[i * 3 + 2];
            xfield_mul_impl(v0, v1, v2, w0, w1, w2, t0, t1, t2);
        } else {
            uint64_t w = d_weights[i];
            t0 = bfield_mul_impl(v0, w);
            t1 = bfield_mul_impl(v1, w);
            t2 = bfield_mul_impl(v2, w);
        }
        
        sum0 = bfield_add_impl(sum0, t0);
        sum1 = bfield_add_impl(sum1, t1);
        sum2 = bfield_add_impl(sum2, t2);
    }
    
    d_output[0] = sum0;
    d_output[1] = sum1;
    d_output[2] = sum2;
}

/**
 * Batch weighted sum: compute weighted sums for multiple rows
 * Each row has num_elements values with shared weights
 */
__global__ void xfe_weighted_sum_batch_kernel(
    const uint64_t* d_values,     // [num_rows * num_elements * 3]
    const uint64_t* d_weights,    // [num_elements * 3]
    uint64_t* d_output,           // [num_rows * 3]
    size_t num_rows,
    size_t num_elements
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
    
    for (size_t i = 0; i < num_elements; ++i) {
        size_t v_offset = (row * num_elements + i) * 3;
        uint64_t v0 = d_values[v_offset + 0];
        uint64_t v1 = d_values[v_offset + 1];
        uint64_t v2 = d_values[v_offset + 2];
        
        size_t w_offset = i * 3;
        uint64_t w0 = d_weights[w_offset + 0];
        uint64_t w1 = d_weights[w_offset + 1];
        uint64_t w2 = d_weights[w_offset + 2];
        
        uint64_t t0, t1, t2;
        xfield_mul_impl(v0, v1, v2, w0, w1, w2, t0, t1, t2);
        
        sum0 = bfield_add_impl(sum0, t0);
        sum1 = bfield_add_impl(sum1, t1);
        sum2 = bfield_add_impl(sum2, t2);
    }
    
    d_output[row * 3 + 0] = sum0;
    d_output[row * 3 + 1] = sum1;
    d_output[row * 3 + 2] = sum2;
}

/**
 * Element-wise XFE multiply: output[i] = a[i] * b[i]
 */
__global__ void xfe_elementwise_mul_kernel(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_output,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t offset = idx * 3;
    uint64_t a0 = d_a[offset + 0];
    uint64_t a1 = d_a[offset + 1];
    uint64_t a2 = d_a[offset + 2];
    uint64_t b0 = d_b[offset + 0];
    uint64_t b1 = d_b[offset + 1];
    uint64_t b2 = d_b[offset + 2];
    
    uint64_t r0, r1, r2;
    xfield_mul_impl(a0, a1, a2, b0, b1, b2, r0, r1, r2);
    
    d_output[offset + 0] = r0;
    d_output[offset + 1] = r1;
    d_output[offset + 2] = r2;
}

/**
 * Multiply XFE by BFE inverse: output[i] = xfe[i] * bfe_inv[i]
 */
__global__ void xfe_scale_by_inv_kernel(
    const uint64_t* d_xfe,
    const uint64_t* d_bfe_inv,
    uint64_t* d_output,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const uint64_t* xfe = d_xfe + idx * 3;
    uint64_t inv = d_bfe_inv[idx];
    
    d_output[idx * 3 + 0] = bfield_mul_impl(xfe[0], inv);
    d_output[idx * 3 + 1] = bfield_mul_impl(xfe[1], inv);
    d_output[idx * 3 + 2] = bfield_mul_impl(xfe[2], inv);
}

// ============================================================================
// Host Interface
// ============================================================================

void zerofier_inv_gpu(
    const uint64_t* d_domain_points,
    uint64_t offset,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    zerofier_inv_kernel<<<grid_size, block_size, 0, stream>>>(
        d_domain_points, offset, d_output, n
    );
}

void power_zerofier_inv_gpu(
    const uint64_t* d_domain_points,
    size_t power,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    power_zerofier_inv_kernel<<<grid_size, block_size, 0, stream>>>(
        d_domain_points, power, d_output, n
    );
}

void xfe_weighted_sum_gpu(
    const uint64_t* d_values,
    const uint64_t* d_weights,
    uint64_t* d_output,
    size_t num_elements,
    bool weights_are_xfe,
    cudaStream_t stream
) {
    xfe_weighted_sum_kernel<<<1, 1, 0, stream>>>(
        d_values, d_weights, d_output, num_elements, weights_are_xfe
    );
}

void xfe_weighted_sum_batch_gpu(
    const uint64_t* d_values,
    const uint64_t* d_weights,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    xfe_weighted_sum_batch_kernel<<<grid_size, block_size, 0, stream>>>(
        d_values, d_weights, d_output, num_rows, num_elements
    );
}

void xfe_elementwise_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfe_elementwise_mul_kernel<<<grid_size, block_size, 0, stream>>>(
        d_a, d_b, d_output, n
    );
}

void xfe_scale_by_inv_gpu(
    const uint64_t* d_xfe,
    const uint64_t* d_bfe_inv,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfe_scale_by_inv_kernel<<<grid_size, block_size, 0, stream>>>(
        d_xfe, d_bfe_inv, d_output, n
    );
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
