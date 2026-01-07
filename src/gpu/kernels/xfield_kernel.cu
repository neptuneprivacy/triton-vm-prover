/**
 * XFieldElement CUDA Kernel Implementation
 * 
 * Extension field F_p³ = F_p[x] / (x³ - x - 1)
 * Elements are (c0, c1, c2) representing c0 + c1*X + c2*X²
 * where X³ = X - 1 (from the polynomial x³ - x - 1 = 0)
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/xfield_kernel.cuh"

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * XField addition kernel
 * Each XFieldElement is 3 consecutive uint64_t values
 */
__global__ void xfield_add_kernel(
    const uint64_t* a,  // Input: 3*n elements
    const uint64_t* b,  // Input: 3*n elements
    uint64_t* c,        // Output: 3*n elements
    size_t n            // Number of XFieldElements
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t offset = idx * 3;
    xfield_add_impl(
        a[offset], a[offset + 1], a[offset + 2],
        b[offset], b[offset + 1], b[offset + 2],
        c[offset], c[offset + 1], c[offset + 2]
    );
}

/**
 * XField subtraction kernel
 */
__global__ void xfield_sub_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t offset = idx * 3;
    xfield_sub_impl(
        a[offset], a[offset + 1], a[offset + 2],
        b[offset], b[offset + 1], b[offset + 2],
        c[offset], c[offset + 1], c[offset + 2]
    );
}

/**
 * XField multiplication kernel
 */
__global__ void xfield_mul_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t offset = idx * 3;
    xfield_mul_impl(
        a[offset], a[offset + 1], a[offset + 2],
        b[offset], b[offset + 1], b[offset + 2],
        c[offset], c[offset + 1], c[offset + 2]
    );
}

/**
 * XField negation kernel
 */
__global__ void xfield_neg_kernel(
    const uint64_t* a,
    uint64_t* c,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t offset = idx * 3;
    xfield_neg_impl(
        a[offset], a[offset + 1], a[offset + 2],
        c[offset], c[offset + 1], c[offset + 2]
    );
}

/**
 * XField scalar multiplication kernel
 */
__global__ void xfield_scalar_mul_kernel(
    const uint64_t* a,
    const uint64_t* scalars,  // BFieldElement scalars
    uint64_t* c,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t offset = idx * 3;
    xfield_scalar_mul_impl(
        a[offset], a[offset + 1], a[offset + 2],
        scalars[idx],
        c[offset], c[offset + 1], c[offset + 2]
    );
}

/**
 * XField inversion kernel
 */
__global__ void xfield_inv_kernel(
    const uint64_t* a,
    uint64_t* c,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t offset = idx * 3;
    xfield_inv_impl(
        a[offset], a[offset + 1], a[offset + 2],
        c[offset], c[offset + 1], c[offset + 2]
    );
}

// ============================================================================
// Host Interface Functions
// ============================================================================

void xfield_add_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfield_add_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);
}

void xfield_sub_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfield_sub_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);
}

void xfield_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfield_mul_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);
}

void xfield_neg_gpu(
    const uint64_t* d_a,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfield_neg_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_c, n);
}

void xfield_scalar_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_scalars,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfield_scalar_mul_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_scalars, d_c, n);
}

void xfield_inv_gpu(
    const uint64_t* d_a,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    xfield_inv_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_c, n);
}

// ============================================================================
// Batch Inversion Kernels (Montgomery's Trick)
// ============================================================================

/**
 * Forward pass: compute running products
 * products[i] = products[i-1] * values[i]
 * Single-threaded for correctness (dependencies between iterations)
 */
__global__ void xfield_batch_inv_forward_kernel(
    const uint64_t* values,   // Input: n XFE values (3*n uint64)
    uint64_t* products,       // Output: n running products (3*n uint64)
    size_t n
) {
    // Single thread computes all products
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Initialize product[0] = values[0]
    products[0] = values[0];
    products[1] = values[1];
    products[2] = values[2];
    
    // Forward pass: products[i] = products[i-1] * values[i]
    for (size_t i = 1; i < n; i++) {
        size_t prev = (i - 1) * 3;
        size_t curr = i * 3;
        xfield_mul_impl(
            products[prev], products[prev + 1], products[prev + 2],
            values[curr], values[curr + 1], values[curr + 2],
            products[curr], products[curr + 1], products[curr + 2]
        );
    }
}

/**
 * Backward pass: compute inverses from products
 * Uses the total product inverse and propagates back
 * Single-threaded for correctness
 */
__global__ void xfield_batch_inv_backward_kernel(
    uint64_t* values,         // Input/Output: n XFE values, will contain inverses
    const uint64_t* products, // Input: n running products
    size_t n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Compute inverse of total product (products[n-1])
    size_t last = (n - 1) * 3;
    uint64_t total_inv0, total_inv1, total_inv2;
    xfield_inv_impl(
        products[last], products[last + 1], products[last + 2],
        total_inv0, total_inv1, total_inv2
    );
    
    // Backward pass
    // inv[n-1] = total_inv * products[n-2]
    // total_inv = total_inv * values[n-1]  (for next iteration)
    for (size_t i = n - 1; i > 0; i--) {
        size_t curr = i * 3;
        size_t prev = (i - 1) * 3;
        
        // Save current value before overwriting
        uint64_t val0 = values[curr];
        uint64_t val1 = values[curr + 1];
        uint64_t val2 = values[curr + 2];
        
        // inv[i] = total_inv * products[i-1]
        xfield_mul_impl(
            total_inv0, total_inv1, total_inv2,
            products[prev], products[prev + 1], products[prev + 2],
            values[curr], values[curr + 1], values[curr + 2]
        );
        
        // Update total_inv = total_inv * values[i]
        xfield_mul_impl(
            total_inv0, total_inv1, total_inv2,
            val0, val1, val2,
            total_inv0, total_inv1, total_inv2
        );
    }
    
    // inv[0] = total_inv (since products[-1] would be 1)
    values[0] = total_inv0;
    values[1] = total_inv1;
    values[2] = total_inv2;
}

void xfield_batch_inv_gpu(
    uint64_t* d_values,
    uint64_t* d_scratch,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    if (n == 1) {
        // Single element - just invert directly
        xfield_inv_gpu(d_values, d_values, 1, stream);
        return;
    }
    
    // Forward pass: compute products
    xfield_batch_inv_forward_kernel<<<1, 1, 0, stream>>>(d_values, d_scratch, n);
    
    // Backward pass: compute inverses in-place
    xfield_batch_inv_backward_kernel<<<1, 1, 0, stream>>>(d_values, d_scratch, n);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
