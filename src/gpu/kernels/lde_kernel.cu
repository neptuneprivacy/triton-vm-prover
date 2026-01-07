/**
 * Low Degree Extension (LDE) CUDA Kernel Implementation
 * 
 * Extends trace polynomials to larger evaluation domain.
 * 
 * LDE Steps:
 * 1. Coset-unscale input by trace_offset^{-i}
 * 2. INTT to get polynomial coefficients
 * 3. Pad with zeros to extended length
 * 4. NTT to evaluate on extended domain
 * 5. Coset-scale by extended_offset^i
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include <cuda_runtime.h>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Host-side helper for modular inverse
static uint64_t host_pow_mod(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t prod = static_cast<__uint128_t>(result) * base;
            result = static_cast<uint64_t>(prod % GOLDILOCKS_P);
        }
        __uint128_t sq = static_cast<__uint128_t>(base) * base;
        base = static_cast<uint64_t>(sq % GOLDILOCKS_P);
        exp >>= 1;
    }
    return result;
}

// ============================================================================
// Coset Scaling Kernels
// ============================================================================

/**
 * Coset scaling kernel
 * Multiplies each element by offset^i
 * Used for converting between coset evaluations and standard evaluations
 */
__global__ void coset_scale_kernel(
    uint64_t* data,
    size_t n,
    uint64_t offset  // The base offset (we compute offset^i for each element)
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute offset^idx
    uint64_t scale = bfield_pow_impl(offset, idx);
    data[idx] = bfield_mul_impl(data[idx], scale);
}

/**
 * Zero-padding kernel
 * Copies input to output and pads with zeros
 */
__global__ void pad_zeros_kernel(
    const uint64_t* input,
    size_t input_size,
    uint64_t* output,
    size_t output_size
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    output[idx] = (idx < input_size) ? input[idx] : 0;
}

// ============================================================================
// Host Interface
// ============================================================================

/**
 * Perform LDE on a single column
 * 
 * Matches CPU behavior:
 * 1. INTT (interpolate) to get polynomial coefficients
 * 2. Pad with zeros to extended length
 * 3. Scale by offset^i (coset shift)
 * 4. NTT to evaluate on extended domain
 * 
 * Note: The trace_offset parameter is ignored to match CPU behavior.
 * 
 * @param d_trace Input trace column (trace_len elements)
 * @param trace_len Size of trace (must be power of 2)
 * @param d_extended Output extended column (extended_len elements)
 * @param extended_len Size of extended domain (must be power of 2, >= trace_len)
 * @param trace_offset Coset offset for trace domain (ignored to match CPU)
 * @param extended_offset Coset offset for extended domain
 * @param stream CUDA stream
 */
void lde_column_gpu(
    const uint64_t* d_trace,
    size_t trace_len,
    uint64_t* d_extended,
    size_t extended_len,
    uint64_t trace_offset,
    uint64_t extended_offset,
    cudaStream_t stream
) {
    (void)trace_offset;  // Unused, matching CPU behavior
    
    int block_size = 256;
    int grid_extended = (extended_len + block_size - 1) / block_size;
    
    // Allocate working buffer for coefficients
    uint64_t* d_coeffs;
    cudaMalloc(&d_coeffs, trace_len * sizeof(uint64_t));
    
    // Step 1: Copy trace to working buffer
    cudaMemcpyAsync(d_coeffs, d_trace, trace_len * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Step 2: INTT to get polynomial coefficients
    ntt_inverse_gpu(d_coeffs, trace_len, stream);
    
    // Step 3: Pad coefficients with zeros to extended length
    pad_zeros_kernel<<<grid_extended, block_size, 0, stream>>>(
        d_coeffs, trace_len, d_extended, extended_len
    );
    
    cudaFree(d_coeffs);
    
    // Step 4: Scale by offset^i (coset shift)
    coset_scale_kernel<<<grid_extended, block_size, 0, stream>>>(
        d_extended, extended_len, extended_offset
    );
    
    // Step 5: NTT to evaluate on extended domain
    ntt_forward_gpu(d_extended, extended_len, stream);
}

/**
 * Batch LDE for multiple columns
 * More efficient than individual calls
 */
void lde_batch_gpu(
    const uint64_t* d_traces,      // All trace columns contiguous (num_columns * trace_len)
    size_t num_columns,
    size_t trace_len,
    uint64_t* d_extended,          // All extended columns contiguous (num_columns * extended_len)
    size_t extended_len,
    uint64_t trace_offset,
    uint64_t extended_offset,
    cudaStream_t stream
) {
    // For now, process columns sequentially
    // TODO: Use multiple streams for parallelism
    for (size_t c = 0; c < num_columns; ++c) {
        lde_column_gpu(
            d_traces + c * trace_len,
            trace_len,
            d_extended + c * extended_len,
            extended_len,
            trace_offset,
            extended_offset,
            stream
        );
    }
}

// Backward compatibility wrappers
void lde_column_device(
    const uint64_t* d_trace,
    size_t trace_len,
    uint64_t* d_extended,
    size_t extended_len,
    uint64_t trace_offset,
    uint64_t extended_offset,
    cudaStream_t stream
) {
    lde_column_gpu(d_trace, trace_len, d_extended, extended_len,
                   trace_offset, extended_offset, stream);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
