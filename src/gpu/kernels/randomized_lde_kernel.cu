/**
 * Randomized LDE CUDA Kernel Implementation
 * 
 * GPU implementation of randomized Low-Degree Extension for zero-knowledge proofs.
 * 
 * Algorithm:
 * 1. INTT to get interpolant coefficients
 * 2. Scale by trace_offset^(-i) (coset interpolation)
 * 3. Compute zerofier * randomizer = shift(randomizer, n) - randomizer * offset^n
 * 4. Add: randomized_poly = interpolant + zerofier * randomizer
 * 5. Evaluate on target domain with NTT
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/randomized_lde_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstdio>
#include <string>
#include <stdexcept>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Helper Kernels
// ============================================================================

/**
 * Scale coefficients by offset^(-i) for coset interpolation
 */
__global__ void coset_interpolate_scale_kernel(
    uint64_t* coeffs,
    size_t n,
    uint64_t offset_inv
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // scale = offset_inv^idx
    uint64_t scale = bfield_pow_impl(offset_inv, idx);
    coeffs[idx] = bfield_mul_impl(coeffs[idx], scale);
}

/**
 * Apply zerofier multiplication: z(x) * p(x) where z(x) = x^n - offset^n
 * Result: shift(p, n) - p * offset^n
 * 
 * For randomizer polynomial with k coefficients:
 * - shifted[n..n+k-1] = randomizer[0..k-1]
 * - result[i] = shifted[i] - randomizer[i] * offset^n
 */
__global__ void zerofier_mul_kernel(
    const uint64_t* randomizer_coeffs,
    size_t randomizer_len,
    size_t trace_len,  // n = domain.length
    uint64_t offset_pow_n,
    uint64_t* output,
    size_t output_len
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_len) return;
    
    // shifted[i] = 0 for i < n, randomizer[i-n] for i >= n
    uint64_t shifted = 0;
    if (idx >= trace_len && (idx - trace_len) < randomizer_len) {
        shifted = randomizer_coeffs[idx - trace_len];
    }
    
    // scaled[i] = randomizer[i] * offset^n (or 0 if i >= randomizer_len)
    uint64_t scaled = 0;
    if (idx < randomizer_len) {
        scaled = bfield_mul_impl(randomizer_coeffs[idx], offset_pow_n);
    }
    
    // result = shifted - scaled
    output[idx] = bfield_sub_impl(shifted, scaled);
}

/**
 * Add two polynomial coefficient arrays
 */
__global__ void poly_add_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t a_len,
    size_t b_len,
    size_t result_len
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result_len) return;
    
    uint64_t av = (idx < a_len) ? a[idx] : 0;
    uint64_t bv = (idx < b_len) ? b[idx] : 0;
    result[idx] = bfield_add_impl(av, bv);
}

/**
 * Scale coefficients by offset^i for coset evaluation
 */
__global__ void coset_eval_scale_kernel(
    uint64_t* coeffs,
    size_t n,
    uint64_t offset
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t scale = bfield_pow_impl(offset, idx);
    coeffs[idx] = bfield_mul_impl(coeffs[idx], scale);
}

/**
 * Pad coefficients with zeros to target length
 */
__global__ void pad_coeffs_kernel(
    const uint64_t* input,
    size_t input_len,
    uint64_t* output,
    size_t output_len
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_len) return;
    
    output[idx] = (idx < input_len) ? input[idx] : 0;
}

/**
 * Accumulate chunked evaluation results with scaling
 */
__global__ void accumulate_chunk_kernel(
    const uint64_t* chunk_eval,
    uint64_t* result,
    size_t domain_len,
    uint64_t scaled_offset
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_len) return;
    
    uint64_t term = bfield_mul_impl(chunk_eval[idx], scaled_offset);
    result[idx] = bfield_add_impl(result[idx], term);
}

// ============================================================================
// Main Randomized LDE Function
// ============================================================================

void randomized_lde_column_gpu(
    const uint64_t* d_trace_column,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_output,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    
    // Allocate working buffers
    uint64_t* d_interpolant;
    cudaMalloc(&d_interpolant, trace_len * sizeof(uint64_t));
    cudaMemcpyAsync(d_interpolant, d_trace_column, trace_len * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Step 1: INTT to get interpolant coefficients
    ntt_inverse_gpu(d_interpolant, trace_len, stream);
    
    // Step 2: Scale by offset^(-i) for coset interpolation
    uint64_t offset_inv = bfield_inv_host(trace_offset);
    int grid1 = (trace_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    coset_interpolate_scale_kernel<<<grid1, BLOCK_SIZE, 0, stream>>>(
        d_interpolant, trace_len, offset_inv
    );
    
    // Step 3: Compute zerofier * randomizer
    // zerofier(x) = x^n - offset^n where n = trace_len
    // Result length = trace_len + randomizer_len
    size_t zerofier_rand_len = trace_len + randomizer_len;
    uint64_t offset_pow_n = bfield_pow_host(trace_offset, trace_len);
    
    uint64_t* d_zerofier_times_rand;
    cudaMalloc(&d_zerofier_times_rand, zerofier_rand_len * sizeof(uint64_t));
    
    int grid2 = (zerofier_rand_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zerofier_mul_kernel<<<grid2, BLOCK_SIZE, 0, stream>>>(
        d_randomizer_coeffs,
        randomizer_len,
        trace_len,
        offset_pow_n,
        d_zerofier_times_rand,
        zerofier_rand_len
    );
    
    // Step 4: Add: randomized_poly = interpolant + zerofier * randomizer
    size_t poly_len = std::max(trace_len, zerofier_rand_len);
    uint64_t* d_randomized_poly;
    cudaMalloc(&d_randomized_poly, poly_len * sizeof(uint64_t));
    
    int grid3 = (poly_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poly_add_kernel<<<grid3, BLOCK_SIZE, 0, stream>>>(
        d_interpolant,
        d_zerofier_times_rand,
        d_randomized_poly,
        trace_len,
        zerofier_rand_len,
        poly_len
    );
    
    // Step 5: Evaluate on target domain
    if (poly_len <= target_len) {
        // Simple case: polynomial fits in one chunk
        // Pad to target_len
        int grid4 = (target_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_coeffs_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
            d_randomized_poly, poly_len, d_output, target_len
        );
        
        // Scale by target_offset^i
        coset_eval_scale_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
            d_output, target_len, target_offset
        );
        
        // Forward NTT
        ntt_forward_gpu(d_output, target_len, stream);
    } else {
        // Chunking case: polynomial is larger than domain length
        // Initialize result to zero
        cudaMemsetAsync(d_output, 0, target_len * sizeof(uint64_t), stream);
        
        // Process chunks
        uint64_t* d_chunk;
        cudaMalloc(&d_chunk, target_len * sizeof(uint64_t));
        
        for (size_t chunk_start = 0; chunk_start < poly_len; chunk_start += target_len) {
            size_t chunk_size = std::min(target_len, poly_len - chunk_start);
            
            // Copy chunk with padding
            int grid4 = (target_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMemsetAsync(d_chunk, 0, target_len * sizeof(uint64_t), stream);
            if (chunk_size > 0) {
                cudaMemcpyAsync(d_chunk, d_randomized_poly + chunk_start,
                                chunk_size * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream);
            }
            
            // Scale by target_offset^i
            coset_eval_scale_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
                d_chunk, target_len, target_offset
            );
            
            // Forward NTT
            ntt_forward_gpu(d_chunk, target_len, stream);
            
            // Accumulate with scaling
            size_t chunk_index = chunk_start / target_len;
            uint64_t coeff_index = chunk_index * target_len;
            uint64_t scaled_offset = (chunk_index == 0) ? 1 : bfield_pow_host(target_offset, coeff_index);
            
            if (chunk_index == 0) {
                cudaMemcpyAsync(d_output, d_chunk, target_len * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream);
            } else {
                accumulate_chunk_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
                    d_chunk, d_output, target_len, scaled_offset
                );
            }
        }
        
        cudaFree(d_chunk);
    }
    
    // Cleanup
    cudaFree(d_interpolant);
    cudaFree(d_zerofier_times_rand);
    cudaFree(d_randomized_poly);
}

// ============================================================================
// Batch Randomized LDE (for all columns)
// ============================================================================

// Optimized single-column LDE that uses pre-allocated buffers (no internal malloc)
static void randomized_lde_column_optimized(
    const uint64_t* d_trace_column,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_output,
    // Pre-allocated buffers (per-stream)
    uint64_t* d_interpolant,     // [trace_len]
    uint64_t* d_zerofier_rand,   // [trace_len + randomizer_len]
    uint64_t* d_randomized_poly, // [max(trace_len, trace_len + randomizer_len)]
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    
    // Copy trace to interpolant buffer
    cudaMemcpyAsync(d_interpolant, d_trace_column, trace_len * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Step 1: INTT to get interpolant coefficients
    ntt_inverse_gpu(d_interpolant, trace_len, stream);
    
    // Step 2: Scale by offset^(-i) for coset interpolation
    uint64_t offset_inv = bfield_inv_host(trace_offset);
    int grid1 = (trace_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    coset_interpolate_scale_kernel<<<grid1, BLOCK_SIZE, 0, stream>>>(
        d_interpolant, trace_len, offset_inv
    );
    
    // Step 3: Compute zerofier * randomizer
    size_t zerofier_rand_len = trace_len + randomizer_len;
    uint64_t offset_pow_n = bfield_pow_host(trace_offset, trace_len);
    
    int grid2 = (zerofier_rand_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zerofier_mul_kernel<<<grid2, BLOCK_SIZE, 0, stream>>>(
        d_randomizer_coeffs,
        randomizer_len,
        trace_len,
        offset_pow_n,
        d_zerofier_rand,
        zerofier_rand_len
    );
    
    // Step 4: Add: randomized_poly = interpolant + zerofier * randomizer
    size_t poly_len = std::max(trace_len, zerofier_rand_len);
    
    int grid3 = (poly_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poly_add_kernel<<<grid3, BLOCK_SIZE, 0, stream>>>(
        d_interpolant,
        d_zerofier_rand,
        d_randomized_poly,
        trace_len,
        zerofier_rand_len,
        poly_len
    );
    
    // Step 5: Evaluate on target domain
    if (poly_len <= target_len) {
        // Pad to target_len
        int grid4 = (target_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_coeffs_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
            d_randomized_poly, poly_len, d_output, target_len
        );
        
        // Scale by target_offset^i
        coset_eval_scale_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
            d_output, target_len, target_offset
        );
        
        // Forward NTT
        ntt_forward_gpu(d_output, target_len, stream);
    } else {
        // Chunking case (rare for typical STARK parameters)
        cudaMemsetAsync(d_output, 0, target_len * sizeof(uint64_t), stream);
        
        for (size_t chunk_start = 0; chunk_start < poly_len; chunk_start += target_len) {
            size_t chunk_size = std::min(target_len, poly_len - chunk_start);
            
            int grid4 = (target_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // Use d_zerofier_rand as temp chunk buffer (safe since we're done with it)
            cudaMemsetAsync(d_zerofier_rand, 0, target_len * sizeof(uint64_t), stream);
            if (chunk_size > 0) {
                cudaMemcpyAsync(d_zerofier_rand, d_randomized_poly + chunk_start,
                                chunk_size * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream);
            }
            
            coset_eval_scale_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
                d_zerofier_rand, target_len, target_offset
            );
            
            ntt_forward_gpu(d_zerofier_rand, target_len, stream);
            
            size_t chunk_index = chunk_start / target_len;
            uint64_t coeff_index = chunk_index * target_len;
            uint64_t scaled_offset = (chunk_index == 0) ? 1 : bfield_pow_host(target_offset, coeff_index);
            
            if (chunk_index == 0) {
                cudaMemcpyAsync(d_output, d_zerofier_rand, target_len * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream);
            } else {
                accumulate_chunk_kernel<<<grid4, BLOCK_SIZE, 0, stream>>>(
                    d_zerofier_rand, d_output, target_len, scaled_offset
                );
            }
        }
    }
}

// Batched version of coset_interpolate_scale for all columns
// LEGACY version - computes powers on the fly (slow)
__global__ void batched_coset_interpolate_scale_kernel_legacy(
    uint64_t* d_data,  // [num_cols * n]
    size_t n,
    size_t num_cols,
    uint64_t offset_inv
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * num_cols;
    if (idx >= total) return;
    
    size_t row = idx % n;  // Position within column
    uint64_t scale = bfield_pow_impl(offset_inv, row);
    d_data[idx] = bfield_mul_impl(d_data[idx], scale);
}

// OPTIMIZED version with precomputed powers and ILP (4 elements per thread)
__global__ void batched_coset_interpolate_scale_kernel(
    uint64_t* d_data,  // [num_cols * n]
    size_t n,
    size_t num_cols,
    const uint64_t* d_powers  // [n] precomputed powers of offset_inv
) {
    // Process 4 elements per thread for better ILP
    // IMPORTANT: CUDA's blockIdx.x/blockDim.x/threadIdx.x are 32-bit. Cast before multiplying
    // to avoid overflow when processing >2^32 elements (e.g., target_len=2^24 and many columns).
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = n * num_cols;
    
    // Pre-fetch data and powers
    uint64_t vals[4], scales[4];
    size_t rows[4];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            rows[i] = idx % n;
            vals[i] = d_data[idx];
            scales[i] = d_powers[rows[i]];
        }
    }
    
    // Compute multiplications (can be pipelined)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            d_data[idx] = bfield_mul_impl(vals[i], scales[i]);
        }
    }
}

// Batched zerofier multiplication for all columns
__global__ void batched_zerofier_mul_kernel(
    const uint64_t* d_randomizer_coeffs,  // [num_cols * randomizer_len]
    size_t randomizer_len,
    size_t trace_len,
    size_t num_cols,
    uint64_t offset_pow_n,
    uint64_t* d_output,  // [num_cols * output_len]
    size_t output_len
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = output_len * num_cols;
    if (idx >= total) return;
    
    size_t col = idx / output_len;
    size_t row = idx % output_len;
    
    // shifted[row] = 0 for row < trace_len, randomizer[row-trace_len] for row >= trace_len
    uint64_t shifted = 0;
    if (row >= trace_len && (row - trace_len) < randomizer_len) {
        shifted = d_randomizer_coeffs[col * randomizer_len + (row - trace_len)];
    }
    
    // scaled[row] = randomizer[row] * offset^n (or 0 if row >= randomizer_len)
    uint64_t scaled = 0;
    if (row < randomizer_len) {
        scaled = bfield_mul_impl(d_randomizer_coeffs[col * randomizer_len + row], offset_pow_n);
    }
    
    d_output[idx] = bfield_sub_impl(shifted, scaled);
}

// FUSED kernel: zerofier_mul + poly_add in single pass
// Eliminates intermediate d_zerofier_rands buffer, reduces memory traffic by 50%
__global__ void fused_zerofier_add_kernel(
    const uint64_t* __restrict__ d_interpolants,      // [num_cols * trace_len]
    const uint64_t* __restrict__ d_randomizer_coeffs, // [num_cols * randomizer_len]
    size_t trace_len,
    size_t randomizer_len,
    size_t num_cols,
    uint64_t offset_pow_n,
    uint64_t* __restrict__ d_output,  // [num_cols * output_len]
    size_t output_len
) {
    // Process 4 elements per thread for ILP
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = output_len * num_cols;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx >= total) return;
        
        size_t col = idx / output_len;
        size_t row = idx % output_len;
        
        // Compute zerofier * randomizer inline
        uint64_t shifted = 0;
        if (row >= trace_len && (row - trace_len) < randomizer_len) {
            shifted = d_randomizer_coeffs[col * randomizer_len + (row - trace_len)];
        }
        
        uint64_t scaled = 0;
        if (row < randomizer_len) {
            scaled = bfield_mul_impl(d_randomizer_coeffs[col * randomizer_len + row], offset_pow_n);
        }
        
        uint64_t zerofier_val = bfield_sub_impl(shifted, scaled);
        
        // Add interpolant value
        uint64_t interp_val = (row < trace_len) ? d_interpolants[col * trace_len + row] : 0;
        
        d_output[idx] = bfield_add_impl(interp_val, zerofier_val);
    }
}

// MEGA-FUSED kernel: zerofier_add + pad_scale in single pass
// Eliminates d_randomized_polys intermediate buffer entirely!
// Computes: output[row] = (interp[row] + zerofier[row]) * power[row] for row < poly_len
//           output[row] = 0 for row >= poly_len
__global__ void fused_zerofier_pad_scale_kernel(
    const uint64_t* __restrict__ d_interpolants,      // [num_cols * trace_len]
    const uint64_t* __restrict__ d_randomizer_coeffs, // [num_cols * randomizer_len]
    const uint64_t* __restrict__ d_powers,            // [target_len] precomputed powers
    size_t trace_len,
    size_t randomizer_len,
    size_t poly_len,       // trace_len + randomizer_len
    size_t num_cols,
    size_t target_len,
    uint64_t offset_pow_n, // trace_offset^trace_len
    uint64_t* __restrict__ d_output  // [num_cols * target_len]
) {
    // Process 4 elements per thread for ILP
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = target_len * num_cols;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx >= total) continue;
        
        size_t col = idx / target_len;
        size_t row = idx % target_len;
        
        uint64_t result;
        
        if (row < poly_len) {
            // Compute zerofier * randomizer inline
            uint64_t shifted = 0;
            if (row >= trace_len && (row - trace_len) < randomizer_len) {
                shifted = d_randomizer_coeffs[col * randomizer_len + (row - trace_len)];
            }
            
            uint64_t scaled = 0;
            if (row < randomizer_len) {
                scaled = bfield_mul_impl(d_randomizer_coeffs[col * randomizer_len + row], offset_pow_n);
            }
            
            uint64_t zerofier_val = bfield_sub_impl(shifted, scaled);
            
            // Add interpolant value
            uint64_t interp_val = (row < trace_len) ? d_interpolants[col * trace_len + row] : 0;
            uint64_t poly_val = bfield_add_impl(interp_val, zerofier_val);
            
            // Scale by power
            result = bfield_mul_impl(poly_val, d_powers[row]);
        } else {
            // Zero padding
            result = 0;
        }
        
        d_output[idx] = result;
    }
}

// ============================================================================
// OPTION A: Sparse kernel (only process poly_len rows, output pre-zeroed)
// Key insight: 87% of output is zeros. Memset first, then only process non-zeros.
// ============================================================================
__global__ void fused_zerofier_pad_scale_sparse_kernel(
    const uint64_t* __restrict__ d_interpolants,      // [num_cols * trace_len]
    const uint64_t* __restrict__ d_randomizer_coeffs, // [num_cols * randomizer_len]
    const uint64_t* __restrict__ d_powers,            // [poly_len] (only need poly_len powers!)
    size_t trace_len,
    size_t randomizer_len,
    size_t poly_len,
    size_t num_cols,
    size_t target_len,
    uint64_t offset_pow_n,
    uint64_t* __restrict__ d_output  // [num_cols * target_len] - PRE-ZEROED!
) {
    // Only process poly_len rows per column (the non-zero region)
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = poly_len * num_cols;  // Much smaller: 13% of full size!
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx >= total) continue;
        
        size_t col = idx / poly_len;
        size_t row = idx % poly_len;
        
        // Compute zerofier * randomizer inline
        uint64_t shifted = 0;
        if (row >= trace_len) {
            shifted = d_randomizer_coeffs[col * randomizer_len + (row - trace_len)];
        }
        
        uint64_t scaled = 0;
        if (row < randomizer_len) {
            scaled = bfield_mul_impl(d_randomizer_coeffs[col * randomizer_len + row], offset_pow_n);
        }
        
        uint64_t zerofier_val = bfield_sub_impl(shifted, scaled);
        
        // Add interpolant value
        uint64_t interp_val = (row < trace_len) ? d_interpolants[col * trace_len + row] : 0;
        uint64_t poly_val = bfield_add_impl(interp_val, zerofier_val);
        
        // Scale by power and write to correct position in output
        uint64_t result = bfield_mul_impl(poly_val, d_powers[row]);
        d_output[col * target_len + row] = result;  // Strided write, but only 13% of elements
    }
}

// ============================================================================
// OPTION B: Row-major output kernel (coalesced writes, requires transpose after)
// ============================================================================
__global__ void fused_zerofier_pad_scale_rowmajor_kernel(
    const uint64_t* __restrict__ d_interpolants,      // [num_cols * trace_len] col-major
    const uint64_t* __restrict__ d_randomizer_coeffs, // [num_cols * randomizer_len] col-major
    const uint64_t* __restrict__ d_powers,            // [target_len]
    size_t trace_len,
    size_t randomizer_len,
    size_t poly_len,
    size_t num_cols,
    size_t target_len,
    uint64_t offset_pow_n,
    uint64_t* __restrict__ d_output  // [target_len * num_cols] ROW-MAJOR!
) {
    // Process by row for coalesced writes
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = target_len * num_cols;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx >= total) continue;
        
        // Row-major indexing: consecutive threads write consecutive memory
        size_t row = idx / num_cols;
        size_t col = idx % num_cols;
        
        uint64_t result;
        
        if (row < poly_len) {
            uint64_t shifted = 0;
            if (row >= trace_len) {
                shifted = d_randomizer_coeffs[col * randomizer_len + (row - trace_len)];
            }
            
            uint64_t scaled = 0;
            if (row < randomizer_len) {
                scaled = bfield_mul_impl(d_randomizer_coeffs[col * randomizer_len + row], offset_pow_n);
            }
            
            uint64_t zerofier_val = bfield_sub_impl(shifted, scaled);
            uint64_t interp_val = (row < trace_len) ? d_interpolants[col * trace_len + row] : 0;
            uint64_t poly_val = bfield_add_impl(interp_val, zerofier_val);
            result = bfield_mul_impl(poly_val, d_powers[row]);
        } else {
            result = 0;
        }
        
        d_output[idx] = result;  // Row-major: coalesced write!
    }
}

// Transpose kernel: row-major to column-major
__global__ void transpose_rowmajor_to_colmajor_kernel(
    const uint64_t* __restrict__ d_input,   // [rows * cols] row-major
    uint64_t* __restrict__ d_output,        // [cols * rows] col-major
    size_t rows,
    size_t cols
) {
    // Use shared memory tiling for efficient transpose
    __shared__ uint64_t tile[32][33];  // +1 to avoid bank conflicts
    
    size_t bx = blockIdx.x * 32;
    size_t by = blockIdx.y * 32;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    
    // Load tile from row-major input (coalesced read)
    size_t in_row = by + ty;
    size_t in_col = bx + tx;
    if (in_row < rows && in_col < cols) {
        tile[ty][tx] = d_input[in_row * cols + in_col];
    }
    
    __syncthreads();
    
    // Write tile to column-major output (coalesced write)
    size_t out_col = by + tx;  // Swapped
    size_t out_row = bx + ty;
    if (out_row < cols && out_col < rows) {
        d_output[out_row * rows + out_col] = tile[tx][ty];  // Transposed read from shared
    }
}

// ============================================================================
// OPTION C: Tiled kernel with shared memory (better cache utilization)
// ============================================================================
__global__ void fused_zerofier_pad_scale_tiled_kernel(
    const uint64_t* __restrict__ d_interpolants,
    const uint64_t* __restrict__ d_randomizer_coeffs,
    const uint64_t* __restrict__ d_powers,
    size_t trace_len,
    size_t randomizer_len,
    size_t poly_len,
    size_t num_cols,
    size_t target_len,
    uint64_t offset_pow_n,
    uint64_t* __restrict__ d_output
) {
    // Each block processes a tile of 256 consecutive rows across all columns
    // This allows caching d_powers in shared memory
    __shared__ uint64_t s_powers[256];
    
    size_t row_base = blockIdx.x * 256;
    size_t col = blockIdx.y;
    size_t tid = threadIdx.x;
    
    // Cooperatively load powers into shared memory
    if (row_base + tid < target_len) {
        s_powers[tid] = d_powers[row_base + tid];
    }
    __syncthreads();
    
    // Each thread processes one row
    size_t row = row_base + tid;
    if (row >= target_len || col >= num_cols) return;
    
    uint64_t result;
    
    if (row < poly_len) {
        uint64_t shifted = 0;
        if (row >= trace_len) {
            shifted = d_randomizer_coeffs[col * randomizer_len + (row - trace_len)];
        }
        
        uint64_t scaled = 0;
        if (row < randomizer_len) {
            scaled = bfield_mul_impl(d_randomizer_coeffs[col * randomizer_len + row], offset_pow_n);
        }
        
        uint64_t zerofier_val = bfield_sub_impl(shifted, scaled);
        uint64_t interp_val = (row < trace_len) ? d_interpolants[col * trace_len + row] : 0;
        uint64_t poly_val = bfield_add_impl(interp_val, zerofier_val);
        result = bfield_mul_impl(poly_val, s_powers[tid]);  // Use cached power
    } else {
        result = 0;
    }
    
    d_output[col * target_len + row] = result;
}

// ============================================================================
// OPTION D: Branchless kernel with predication (avoid warp divergence)
// ============================================================================
__global__ void fused_zerofier_pad_scale_branchless_kernel(
    const uint64_t* __restrict__ d_interpolants,
    const uint64_t* __restrict__ d_randomizer_coeffs,
    const uint64_t* __restrict__ d_powers,
    size_t trace_len,
    size_t randomizer_len,
    size_t poly_len,
    size_t num_cols,
    size_t target_len,
    uint64_t offset_pow_n,
    uint64_t* __restrict__ d_output
) {
    // Process 4 elements per thread
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = target_len * num_cols;
    
    // Prefetch all 4 values
    uint64_t results[4];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx >= total) {
            results[i] = 0;
            continue;
        }
        
        size_t col = idx / target_len;
        size_t row = idx % target_len;
        
        // Branchless: always compute, use predicate to select
        // shifted = (row >= trace_len && row < poly_len) ? d_randomizer[row-trace_len] : 0
        bool in_shifted_range = (row >= trace_len) & (row < poly_len);
        size_t shifted_idx = col * randomizer_len + (row - trace_len);
        uint64_t shifted = in_shifted_range ? __ldg(&d_randomizer_coeffs[shifted_idx]) : 0;
        
        // scaled = (row < randomizer_len) ? d_randomizer[row] * offset_pow_n : 0
        bool in_scaled_range = row < randomizer_len;
        size_t scaled_idx = col * randomizer_len + row;
        uint64_t rand_val = in_scaled_range ? __ldg(&d_randomizer_coeffs[scaled_idx]) : 0;
        uint64_t scaled = bfield_mul_impl(rand_val, offset_pow_n);
        
        uint64_t zerofier_val = bfield_sub_impl(shifted, scaled);
        
        // interp = (row < trace_len) ? d_interpolants[row] : 0
        bool in_interp_range = row < trace_len;
        size_t interp_idx = col * trace_len + row;
        uint64_t interp_val = in_interp_range ? __ldg(&d_interpolants[interp_idx]) : 0;
        
        uint64_t poly_val = bfield_add_impl(interp_val, zerofier_val);
        
        // result = (row < poly_len) ? poly_val * power : 0
        bool in_poly_range = row < poly_len;
        uint64_t power = in_poly_range ? __ldg(&d_powers[row]) : 0;
        results[i] = bfield_mul_impl(poly_val, power);
        
        // Zero out if outside poly range (branchless)
        results[i] = in_poly_range ? results[i] : 0;
    }
    
    // Write all 4 results
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            d_output[idx] = results[i];
        }
    }
}

// Batched polynomial addition for all columns - optimized with ILP
__global__ void batched_poly_add_kernel(
    const uint64_t* __restrict__ d_a,   // [num_cols * a_len]
    const uint64_t* __restrict__ d_b,   // [num_cols * b_len]
    uint64_t* __restrict__ d_result,    // [num_cols * result_len]
    size_t a_len,
    size_t b_len,
    size_t result_len,
    size_t num_cols
) {
    // Process 4 elements per thread for better ILP
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = result_len * num_cols;
    
    uint64_t av[4], bv[4];
    size_t cols[4], rows[4];
    
    // Prefetch values
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            cols[i] = idx / result_len;
            rows[i] = idx % result_len;
            av[i] = (rows[i] < a_len) ? d_a[cols[i] * a_len + rows[i]] : 0;
            bv[i] = (rows[i] < b_len) ? d_b[cols[i] * b_len + rows[i]] : 0;
        }
    }
    
    // Compute and write
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            d_result[idx] = bfield_add_impl(av[i], bv[i]);
        }
    }
}

// Batched pad and coset evaluation scaling with precomputed powers
// OPTIMIZED with ILP - processes 4 elements per thread
__global__ void batched_pad_and_scale_kernel(
    const uint64_t* d_input,  // [num_cols * input_len]
    size_t input_len,
    uint64_t* d_output,       // [num_cols * output_len]
    size_t output_len,
    size_t num_cols,
    const uint64_t* d_powers  // [output_len] precomputed powers of offset
) {
    // Process 4 elements per thread for better ILP and reduced division overhead
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * 4;
    size_t total = output_len * num_cols;
    
    // Pre-compute values for all 4 elements
    uint64_t vals[4], scales[4], results[4];
    size_t cols[4], rows[4];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            cols[i] = idx / output_len;
            rows[i] = idx % output_len;
            vals[i] = (rows[i] < input_len) ? d_input[cols[i] * input_len + rows[i]] : 0;
            scales[i] = d_powers[rows[i]];
        }
    }
    
    // Compute multiplications (can be pipelined)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            results[i] = bfield_mul_impl(vals[i], scales[i]);
        }
    }
    
    // Write results (coalesced within warp)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t idx = base_idx + i;
        if (idx < total) {
            d_output[idx] = results[i];
        }
    }
}

// OPTIMIZED version: Only processes actual data, zeros are done with memset
// This is 8x faster when target_len = 8 * input_len (typical STARK case)
__global__ void batched_scale_data_only_kernel(
    const uint64_t* d_input,  // [num_cols * input_len] column-major
    size_t input_len,
    uint64_t* d_output,       // [num_cols * output_len] column-major (pre-zeroed)
    size_t output_len,
    size_t num_cols,
    const uint64_t* d_powers  // [input_len] precomputed powers (only need first input_len)
) {
    // Process 4 elements per thread for better ILP
    constexpr int ELEMS_PER_THREAD = 4;
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * static_cast<size_t>(ELEMS_PER_THREAD);
    size_t total = input_len * num_cols;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        size_t idx = base_idx + i;
    if (idx >= total) return;
    
        size_t col = idx / input_len;
        size_t row = idx % input_len;
        
        uint64_t val = d_input[idx];  // Coalesced read
        uint64_t scale = d_powers[row];
        // Write to output at [col * output_len + row]
        d_output[col * output_len + row] = bfield_mul_impl(val, scale);
    }
}

// Generate powers of offset incrementally in chunks
// Each block handles a contiguous chunk and computes powers sequentially within block
// Much faster than computing each power independently via repeated squaring
__global__ void generate_powers_chunked_kernel(
    uint64_t* d_powers,
    size_t n,
    uint64_t offset,
    size_t chunk_size,
    const uint64_t* d_chunk_bases  // d_chunk_bases[i] = offset^(i*chunk_size), precomputed
) {
    size_t chunk_idx = blockIdx.x;
    size_t num_chunks = (n + chunk_size - 1) / chunk_size;
    if (chunk_idx >= num_chunks) return;
    
    // Only thread 0 per block does the work (sequential within chunk)
    // This is fast enough for 4096-element chunks
    if (threadIdx.x != 0) return;
    
    size_t start = chunk_idx * chunk_size;
    size_t end = min(start + chunk_size, n);
    
    // Start with the base power for this chunk
    uint64_t power = d_chunk_bases[chunk_idx];
    
    for (size_t i = start; i < end; ++i) {
        d_powers[i] = power;
        power = bfield_mul_impl(power, offset);
    }
}

// Generate chunk bases: d_bases[i] = offset^(i*chunk_size)
__global__ void generate_chunk_bases_kernel(
    uint64_t* d_bases,
    size_t num_chunks,
    uint64_t offset,
    size_t chunk_size
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_chunks) return;
    
    // Compute offset^(idx * chunk_size)
    d_bases[idx] = bfield_pow_impl(offset, idx * chunk_size);
}

// Internal implementation that uses provided scratch buffers
static void randomized_lde_batch_impl(
    const uint64_t* d_trace_table,  // Column-major: col * trace_len + row
    size_t num_cols,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,  // num_cols * randomizer_len
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_output,  // Column-major: col * target_len + row
    uint64_t* d_scratch1,  // Pre-allocated: num_cols * trace_len
    uint64_t* d_scratch2,  // Pre-allocated: num_cols * poly_len
    bool needs_alloc,      // True if scratch not provided
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    const bool profile = std::getenv("TRITON_PROFILE_LDE_DETAIL") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto elapsed = [&t0]() {
        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t0).count();
        t0 = now;
        return ms;
    };
    
    // Allocate working buffers for all columns at once
    size_t zerofier_rand_len = trace_len + randomizer_len;
    size_t poly_len = std::max(trace_len, zerofier_rand_len);
    
    uint64_t* d_interpolants = d_scratch1;
    // Note: d_randomized_polys no longer needed - fused into fused_zerofier_pad_scale_kernel
    (void)d_scratch2;  // Unused now
    
    if (needs_alloc) {
        cudaError_t alloc_err = cudaMalloc(&d_interpolants, num_cols * trace_len * sizeof(uint64_t));
        if (alloc_err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate interpolants buffer for LDE: " + std::string(cudaGetErrorString(alloc_err)) + 
                                   " (num_cols=" + std::to_string(num_cols) + ", trace_len=" + std::to_string(trace_len) + ")");
        }
        if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] alloc: %.2f ms\n", elapsed()); }
    } else {
        if (profile) { printf("      [LDE] alloc: 0.00 ms (pre-allocated)\n"); }
    }
    
    // Step 1: Copy trace to interpolant buffer
    cudaMemcpyAsync(d_interpolants, d_trace_table, 
                    num_cols * trace_len * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
    if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] memcpy: %.2f ms\n", elapsed()); }
    
    // Step 2: Batched INTT to get interpolant coefficients for all columns
    ntt_inverse_batched_gpu(d_interpolants, trace_len, num_cols, stream);
    if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] INTT (%zu cols x %zu): %.2f ms\n", num_cols, trace_len, elapsed()); }
    
    // Debug: dump interpolant coefficients for column 378 (last column) after INTT
    if (std::getenv("TVM_DEBUG_COL378_INTERPOLANT")) {
        cudaStreamSynchronize(stream);
        size_t last_col = num_cols - 1;
        std::vector<uint64_t> h_interp(5);
        cudaError_t err = cudaMemcpy(h_interp.data(), d_interpolants + last_col * trace_len, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DBG] Column 378 interpolant coeffs (first 5, after INTT, cuda=%d):\n", (int)err);
        for (size_t i = 0; i < 5; i++) {
            fprintf(stderr, "  [%zu]: %lu\n", i, h_interp[i]);
        }
        fflush(stderr);
    }
    
    // Step 3: Batched coset scaling (scale by offset^(-i)) with precomputed powers
    uint64_t offset_inv = bfield_inv_host(trace_offset);
    
    // Debug: check offset values
    if (std::getenv("TVM_DEBUG_COL378_INTERPOLANT")) {
        fprintf(stderr, "[DBG] Coset scaling params: trace_offset=%lu, offset_inv=%lu\n", trace_offset, offset_inv);
        fflush(stderr);
    }
    
    // Precompute powers of offset_inv for coset interpolation
    uint64_t* d_interp_powers;
    cudaMalloc(&d_interp_powers, trace_len * sizeof(uint64_t));
    
    // Use chunked approach for efficient power generation
    constexpr size_t INTERP_CHUNK_SIZE = 4096;
    size_t interp_num_chunks = (trace_len + INTERP_CHUNK_SIZE - 1) / INTERP_CHUNK_SIZE;
    
    uint64_t* d_interp_chunk_bases;
    cudaMalloc(&d_interp_chunk_bases, interp_num_chunks * sizeof(uint64_t));
    
    int grid_bases = (interp_num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_chunk_bases_kernel<<<grid_bases, BLOCK_SIZE, 0, stream>>>(
        d_interp_chunk_bases, interp_num_chunks, offset_inv, INTERP_CHUNK_SIZE
    );
    generate_powers_chunked_kernel<<<interp_num_chunks, 1, 0, stream>>>(
        d_interp_powers, trace_len, offset_inv, INTERP_CHUNK_SIZE, d_interp_chunk_bases
    );
    cudaFree(d_interp_chunk_bases);
    if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] gen_interp_powers: %.2f ms\n", elapsed()); }
    
    // Debug: check first few powers
    if (std::getenv("TVM_DEBUG_COL378_INTERPOLANT")) {
        cudaStreamSynchronize(stream);
        std::vector<uint64_t> h_powers(5);
        cudaError_t err = cudaMemcpy(h_powers.data(), d_interp_powers, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DBG] Coset scaling powers (first 5, cuda=%d):\n", (int)err);
        for (size_t i = 0; i < 5; i++) {
            fprintf(stderr, "  power[%zu]: %lu\n", i, h_powers[i]);
        }
        fflush(stderr);
    }
    
    // Apply coset scaling with precomputed powers (4 elements per thread)
    constexpr int INTERP_ELEMS_PER_THREAD = 4;
    int grid_interp = (num_cols * trace_len + BLOCK_SIZE * INTERP_ELEMS_PER_THREAD - 1) / (BLOCK_SIZE * INTERP_ELEMS_PER_THREAD);
    batched_coset_interpolate_scale_kernel<<<grid_interp, BLOCK_SIZE, 0, stream>>>(
        d_interpolants, trace_len, num_cols, d_interp_powers
    );
    cudaFree(d_interp_powers);
    if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] coset_interp: %.2f ms\n", elapsed()); }
    
    // Debug: dump interpolant coefficients for column 378 after coset scaling
    if (std::getenv("TVM_DEBUG_COL378_INTERPOLANT")) {
        cudaStreamSynchronize(stream);
        size_t last_col = num_cols - 1;
        std::vector<uint64_t> h_interp_scaled(5);
        cudaError_t err = cudaMemcpy(h_interp_scaled.data(), d_interpolants + last_col * trace_len, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DBG] Column 378 interpolant coeffs (first 5, after coset scaling, cuda=%d):\n", (int)err);
        for (size_t i = 0; i < 5; i++) {
            fprintf(stderr, "  [%zu]: %lu\n", i, h_interp_scaled[i]);
        }
        fflush(stderr);
    }
    
    // Steps 4+5+6 MEGA-FUSED: zerofier_add + pad_scale in single kernel
    // Eliminates d_randomized_polys intermediate buffer entirely!
    uint64_t offset_pow_n = bfield_pow_host(trace_offset, trace_len);
    
    if (poly_len <= target_len) {
        // Select kernel variant based on environment variable
        // TRITON_PAD_SCALE_MODE: 0=original (best), 1=sparse, 2=rowmajor+transpose, 3=tiled, 4=branchless
        // Note: Sparse mode adds memset overhead, so original is often faster
        const char* mode_env = std::getenv("TRITON_PAD_SCALE_MODE");
        int mode = mode_env ? std::atoi(mode_env) : 0;  // Default to original (best performance)
        
        // Generate powers for target_len (needed for all modes except sparse)
        size_t powers_needed = target_len;
        uint64_t* d_powers;
        cudaMalloc(&d_powers, powers_needed * sizeof(uint64_t));
        
        constexpr size_t POWER_CHUNK_SIZE = 4096;
        size_t num_chunks = (powers_needed + POWER_CHUNK_SIZE - 1) / POWER_CHUNK_SIZE;
        
        uint64_t* d_chunk_bases;
        cudaMalloc(&d_chunk_bases, num_chunks * sizeof(uint64_t));
        
        int grid_bases = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        generate_chunk_bases_kernel<<<grid_bases, BLOCK_SIZE, 0, stream>>>(
            d_chunk_bases, num_chunks, target_offset, POWER_CHUNK_SIZE
        );
        
        generate_powers_chunked_kernel<<<num_chunks, 1, 0, stream>>>(
            d_powers, powers_needed, target_offset, POWER_CHUNK_SIZE, d_chunk_bases
        );
        
        cudaFree(d_chunk_bases);
        if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] gen_powers (%zu): %.2f ms\n", powers_needed, elapsed()); }
        
        if (mode == 1) {
            // OPTION 1: Sparse kernel - memset zeros first, then only process poly_len rows
            // This processes only 13% of elements (poly_len/target_len), much faster!
            // Powers are already generated for poly_len (optimized above)
            cudaMemsetAsync(d_output, 0, num_cols * target_len * sizeof(uint64_t), stream);
            if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] memset_zeros: %.2f ms\n", elapsed()); }
            
            constexpr int SPARSE_ELEMS = 4;
            int grid_sparse = (num_cols * poly_len + BLOCK_SIZE * SPARSE_ELEMS - 1) / (BLOCK_SIZE * SPARSE_ELEMS);
            fused_zerofier_pad_scale_sparse_kernel<<<grid_sparse, BLOCK_SIZE, 0, stream>>>(
                d_interpolants, d_randomizer_coeffs, d_powers,
                trace_len, randomizer_len, poly_len, num_cols, target_len,
                offset_pow_n, d_output
            );
            if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] sparse_kernel (%zu/%zu rows): %.2f ms\n", poly_len, target_len, elapsed()); }
        }
        else if (mode == 2) {
            // OPTION B: Row-major output + transpose
            uint64_t* d_rowmajor;
            cudaMalloc(&d_rowmajor, num_cols * target_len * sizeof(uint64_t));
            
            constexpr int RM_ELEMS = 4;
            int grid_rm = (num_cols * target_len + BLOCK_SIZE * RM_ELEMS - 1) / (BLOCK_SIZE * RM_ELEMS);
            fused_zerofier_pad_scale_rowmajor_kernel<<<grid_rm, BLOCK_SIZE, 0, stream>>>(
                d_interpolants, d_randomizer_coeffs, d_powers,
                trace_len, randomizer_len, poly_len, num_cols, target_len,
                offset_pow_n, d_rowmajor
            );
            if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] rowmajor_kernel: %.2f ms\n", elapsed()); }
            
            // Transpose to column-major
            dim3 block_t(32, 32);
            dim3 grid_t((num_cols + 31) / 32, (target_len + 31) / 32);
            transpose_rowmajor_to_colmajor_kernel<<<grid_t, block_t, 0, stream>>>(
                d_rowmajor, d_output, target_len, num_cols
            );
            if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] transpose: %.2f ms\n", elapsed()); }
            
            cudaFree(d_rowmajor);
        }
        else if (mode == 3) {
            // OPTION C: Tiled kernel with shared memory for d_powers
            dim3 block_tiled(256);
            dim3 grid_tiled((target_len + 255) / 256, num_cols);
            fused_zerofier_pad_scale_tiled_kernel<<<grid_tiled, block_tiled, 0, stream>>>(
                d_interpolants, d_randomizer_coeffs, d_powers,
                trace_len, randomizer_len, poly_len, num_cols, target_len,
                offset_pow_n, d_output
            );
            if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] tiled_kernel: %.2f ms\n", elapsed()); }
        }
        else if (mode == 4) {
            // OPTION D: Branchless kernel with predication
            constexpr int BL_ELEMS = 4;
            int grid_bl = (num_cols * target_len + BLOCK_SIZE * BL_ELEMS - 1) / (BLOCK_SIZE * BL_ELEMS);
            fused_zerofier_pad_scale_branchless_kernel<<<grid_bl, BLOCK_SIZE, 0, stream>>>(
                d_interpolants, d_randomizer_coeffs, d_powers,
                trace_len, randomizer_len, poly_len, num_cols, target_len,
                offset_pow_n, d_output
            );
            if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] branchless_kernel: %.2f ms\n", elapsed()); }
        }
        else {
            // OPTION 0: Original mega-fused kernel (default - best performance)
            constexpr int MEGA_ELEMS = 4;
            int grid_mega = (num_cols * target_len + BLOCK_SIZE * MEGA_ELEMS - 1) / (BLOCK_SIZE * MEGA_ELEMS);
            fused_zerofier_pad_scale_kernel<<<grid_mega, BLOCK_SIZE, 0, stream>>>(
                d_interpolants, d_randomizer_coeffs, d_powers,
                trace_len, randomizer_len, poly_len, num_cols, target_len,
                offset_pow_n, d_output
            );
            if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] fused_zerofier_pad_scale: %.2f ms\n", elapsed()); }
        }
        
        cudaFree(d_powers);
        
        // Debug: dump randomized polynomial coefficients for column 378 before NTT
        if (std::getenv("TVM_DEBUG_COL378_INTERPOLANT")) {
            cudaStreamSynchronize(stream);
            size_t last_col = num_cols - 1;
            std::vector<uint64_t> h_poly(10);
            cudaError_t err = cudaMemcpy(h_poly.data(), d_output + last_col * target_len, 10 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DBG] Column 378 randomized poly coeffs (first 10, before NTT, cuda=%d):\n", (int)err);
            for (size_t i = 0; i < 10; i++) {
                fprintf(stderr, "  [%zu]: %lu\n", i, h_poly[i]);
            }
            fflush(stderr);
        }
        
        // Step 7: Batched forward NTT
        ntt_forward_batched_gpu(d_output, target_len, num_cols, stream);
        if (profile) { cudaStreamSynchronize(stream); printf("      [LDE] NTT (%zu cols x %zu): %.2f ms\n", num_cols, target_len, elapsed()); }
        
        // Debug: dump final LDE values for column 378 after NTT
        if (std::getenv("TVM_DEBUG_COL378_INTERPOLANT")) {
            cudaStreamSynchronize(stream);
            size_t last_col = num_cols - 1;
            std::vector<uint64_t> h_lde(5);
            cudaError_t err = cudaMemcpy(h_lde.data(), d_output + last_col * target_len, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DBG] Column 378 LDE values (first 5, after NTT, cuda=%d):\n", (int)err);
            fprintf(stderr, "  [0]: %lu (expected Rust: 10863564274434948420, GPU computed: 4494745678245561732)\n", h_lde[0]);
            for (size_t i = 1; i < 5; i++) {
                fprintf(stderr, "  [%zu]: %lu\n", i, h_lde[i]);
            }
            fflush(stderr);
        }
    } else {
        // Chunking case: fall back to per-column processing for correctness
        // This is rare for typical STARK parameters
        for (size_t col = 0; col < num_cols; ++col) {
            randomized_lde_column_gpu(
                d_trace_table + col * trace_len,
                trace_len,
                d_randomizer_coeffs + col * randomizer_len,
                randomizer_len,
                trace_offset,
                target_offset,
                target_len,
                d_output + col * target_len,
                stream
            );
        }
    }
    
    // Cleanup only if we allocated
    if (needs_alloc) {
    cudaFree(d_interpolants);
        // d_randomized_polys no longer allocated (fused into kernel)
    }
}

// Public interface - allocates internally (backward compatible)
void randomized_lde_batch_gpu(
    const uint64_t* d_trace_table,
    size_t num_cols,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_output,
    cudaStream_t stream
) {
    randomized_lde_batch_impl(
        d_trace_table, num_cols, trace_len,
        d_randomizer_coeffs, randomizer_len,
        trace_offset, target_offset, target_len,
        d_output,
        nullptr, nullptr, true,  // Allocate internally
        stream
    );
}

// Version with pre-allocated scratch buffers (avoids 45ms allocation overhead)
void randomized_lde_batch_gpu_preallocated(
    const uint64_t* d_trace_table,
    size_t num_cols,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_output,
    uint64_t* d_scratch1,  // Must be >= num_cols * trace_len elements
    uint64_t* d_scratch2,  // Must be >= num_cols * (trace_len + randomizer_len) elements
    cudaStream_t stream
) {
    randomized_lde_batch_impl(
        d_trace_table, num_cols, trace_len,
        d_randomizer_coeffs, randomizer_len,
        trace_offset, target_offset, target_len,
        d_output,
        d_scratch1, d_scratch2, false,  // Use provided buffers
        stream
    );
}

// ============================================================================
// Host helper functions - use simple modular arithmetic for correctness
// ============================================================================

static inline uint64_t bfield_mul_host(uint64_t a, uint64_t b) {
    // Simple modular multiplication using 128-bit arithmetic
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)(prod % GOLDILOCKS_P);
}

uint64_t bfield_pow_host(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    base = base % GOLDILOCKS_P;  // Ensure base is reduced
    
    while (exp > 0) {
        if (exp & 1) {
            result = bfield_mul_host(result, base);
        }
        base = bfield_mul_host(base, base);
        exp >>= 1;
    }
    
    return result;
}

uint64_t bfield_inv_host(uint64_t a) {
    // Fermat's little theorem: a^(-1) = a^(p-2) mod p
    return bfield_pow_host(a, GOLDILOCKS_P - 2);
}

// ============================================================================
// XFieldElement LDE (for aux table)
// ============================================================================

/**
 * Randomized LDE for XFieldElement columns (aux table)
 * 
 * Each XFE column is processed as 3 BFE components.
 * The randomizer is BFE, so it scales all 3 components identically.
 * 
 * Input layout: d_xfe_trace_table is [num_cols * trace_len * 3] where
 *   component k of XFE at (col, row) = d_xfe_trace_table[(col * trace_len + row) * 3 + k]
 * 
 * Output layout: d_xfe_output is [num_cols * target_len * 3] with same structure
 */
void randomized_xfe_lde_batch_gpu(
    const uint64_t* d_xfe_trace_table,  // [num_cols * trace_len * 3]
    size_t num_cols,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,  // [num_cols * randomizer_len * 3] if use_xfe_randomizers, else [num_cols * randomizer_len] BFieldElements
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_xfe_output,  // [num_cols * target_len * 3]
    bool use_xfe_randomizers,  // true if randomizers are XFieldElement (3 components), false if BFieldElement (1 component)
    cudaStream_t stream
) {
    // Allocate temporary buffers for component processing
    // We process all 3 components of all columns together
    size_t total_bfe_cols = num_cols * 3;
    
    uint64_t* d_bfe_trace;
    uint64_t* d_bfe_output;
    uint64_t* d_bfe_randomizers;
    
    cudaMalloc(&d_bfe_trace, total_bfe_cols * trace_len * sizeof(uint64_t));
    cudaMalloc(&d_bfe_output, total_bfe_cols * target_len * sizeof(uint64_t));
    cudaMalloc(&d_bfe_randomizers, total_bfe_cols * randomizer_len * sizeof(uint64_t));
    
    // Rearrange XFE data to component-wise BFE layout
    // Input:  (col, row, component) -> [(col * trace_len + row) * 3 + component]
    // Output: (col * 3 + component, row) -> [(col * 3 + component) * trace_len + row]
    std::vector<uint64_t> h_xfe(num_cols * trace_len * 3);
    std::vector<uint64_t> h_bfe(total_bfe_cols * trace_len);
    std::vector<uint64_t> h_rand(use_xfe_randomizers ? num_cols * randomizer_len * 3 : num_cols * randomizer_len);
    std::vector<uint64_t> h_bfe_rand(total_bfe_cols * randomizer_len);

    cudaMemcpy(h_xfe.data(), d_xfe_trace_table, h_xfe.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rand.data(), d_randomizer_coeffs, h_rand.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Rearrange trace data
    for (size_t col = 0; col < num_cols; ++col) {
        for (size_t row = 0; row < trace_len; ++row) {
            for (size_t comp = 0; comp < 3; ++comp) {
                size_t xfe_idx = (col * trace_len + row) * 3 + comp;
                size_t bfe_col = col * 3 + comp;
                size_t bfe_idx = bfe_col * trace_len + row;
                h_bfe[bfe_idx] = h_xfe[xfe_idx];
            }
        }
        if (use_xfe_randomizers) {
            // XFE randomizers: each coefficient is an XFE (3 components)
            for (size_t r = 0; r < randomizer_len; ++r) {
                size_t rand_base = (col * randomizer_len + r) * 3;
                // Component 0: XFE coefficient 0
                h_bfe_rand[(col * 3 + 0) * randomizer_len + r] = h_rand[rand_base + 0];
                // Component 1: XFE coefficient 1
                h_bfe_rand[(col * 3 + 1) * randomizer_len + r] = h_rand[rand_base + 1];
                // Component 2: XFE coefficient 2
                h_bfe_rand[(col * 3 + 2) * randomizer_len + r] = h_rand[rand_base + 2];
            }
        } else {
            // BFE randomizer lifted to XFE: (b, 0, 0)
            // Only component 0 gets the randomizer, components 1 and 2 get zero
            for (size_t r = 0; r < randomizer_len; ++r) {
                // Component 0: actual randomizer
                h_bfe_rand[(col * 3 + 0) * randomizer_len + r] = h_rand[col * randomizer_len + r];
                // Components 1 and 2: zero (no randomizer contribution)
                h_bfe_rand[(col * 3 + 1) * randomizer_len + r] = 0;
                h_bfe_rand[(col * 3 + 2) * randomizer_len + r] = 0;
            }
        }
    }
    
    cudaMemcpy(d_bfe_trace, h_bfe.data(), h_bfe.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bfe_randomizers, h_bfe_rand.data(), h_bfe_rand.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Process all BFE columns using existing batch LDE
    randomized_lde_batch_gpu(
        d_bfe_trace,
        total_bfe_cols,
        trace_len,
        d_bfe_randomizers,
        randomizer_len,
        trace_offset,
        target_offset,
        target_len,
        d_bfe_output,
        stream
    );
    cudaStreamSynchronize(stream);
    
    // Rearrange output back to XFE layout
    std::vector<uint64_t> h_bfe_out(total_bfe_cols * target_len);
    std::vector<uint64_t> h_xfe_out(num_cols * target_len * 3);
    
    cudaMemcpy(h_bfe_out.data(), d_bfe_output, h_bfe_out.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    for (size_t col = 0; col < num_cols; ++col) {
        for (size_t row = 0; row < target_len; ++row) {
            for (size_t comp = 0; comp < 3; ++comp) {
                size_t bfe_col = col * 3 + comp;
                size_t bfe_idx = bfe_col * target_len + row;
                size_t xfe_idx = (col * target_len + row) * 3 + comp;
                h_xfe_out[xfe_idx] = h_bfe_out[bfe_idx];
            }
        }
    }
    
    cudaMemcpy(d_xfe_output, h_xfe_out.data(), h_xfe_out.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Cleanup
    cudaFree(d_bfe_trace);
    cudaFree(d_bfe_output);
    cudaFree(d_bfe_randomizers);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

