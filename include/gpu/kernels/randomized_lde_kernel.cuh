#pragma once

/**
 * Randomized LDE CUDA Kernel Declarations
 * 
 * GPU-accelerated randomized Low-Degree Extension for zero-knowledge proofs.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Use constants from bfield_kernel.cuh (included in implementation)

/**
 * Randomized LDE for a single column
 * 
 * Implements: LDE = interpolant + zerofier * randomizer
 * where zerofier(x) = x^n - trace_offset^n
 * 
 * @param d_trace_column Trace column values (trace_len elements)
 * @param trace_len Length of trace domain (power of 2)
 * @param d_randomizer_coeffs Randomizer polynomial coefficients
 * @param randomizer_len Number of randomizer coefficients
 * @param trace_offset Trace domain offset
 * @param target_offset Target domain offset
 * @param target_len Target domain length (power of 2)
 * @param d_output Output LDE values (target_len elements)
 * @param stream CUDA stream
 */
void randomized_lde_column_gpu(
    const uint64_t* d_trace_column,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_output,
    cudaStream_t stream = 0
);

/**
 * Batch randomized LDE for all columns
 * 
 * @param d_trace_table Column-major trace table (col * trace_len + row)
 * @param num_cols Number of columns
 * @param trace_len Trace domain length
 * @param d_randomizer_coeffs Randomizer coeffs for all columns (col * randomizer_len + i)
 * @param randomizer_len Number of randomizer coefficients per column
 * @param trace_offset Trace domain offset
 * @param target_offset Target domain offset
 * @param target_len Target domain length
 * @param d_output Column-major output (col * target_len + row)
 * @param stream CUDA stream
 */
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
    cudaStream_t stream = 0
);

/**
 * Same as randomized_lde_batch_gpu but uses pre-allocated scratch buffers
 * to avoid ~45ms allocation overhead per call.
 * 
 * @param d_scratch1 Pre-allocated buffer: >= num_cols * trace_len elements
 * @param d_scratch2 Pre-allocated buffer: >= num_cols * (trace_len + randomizer_len) elements
 */
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
    uint64_t* d_scratch1,
    uint64_t* d_scratch2,
    cudaStream_t stream = 0
);

/**
 * Host-side Goldilocks field inverse
 */
uint64_t bfield_inv_host(uint64_t a);

/**
 * Host-side Goldilocks field power
 */
uint64_t bfield_pow_host(uint64_t base, uint64_t exp);

/**
 * Phase 1: Compute polynomial coefficients from trace (INTT + coset scaling)
 * 
 * This is the expensive step that only needs to be done ONCE per trace table,
 * regardless of how many cosets will be evaluated.
 * 
 * Steps performed:
 * 1. Copy trace to coefficients buffer
 * 2. INTT to get polynomial coefficients
 * 3. Apply trace_offset coset scaling
 * 
 * After calling this, use evaluate_coset_from_coefficients_gpu() for each coset.
 * 
 * @param d_trace_table Column-major trace table
 * @param num_cols Number of columns
 * @param trace_len Trace domain length
 * @param trace_offset Trace domain offset (usually 1)
 * @param d_coefficients Output: polynomial coefficients [num_cols * trace_len]
 * @param stream CUDA stream
 */
void compute_trace_coefficients_gpu(
    const uint64_t* d_trace_table,
    size_t num_cols,
    size_t trace_len,
    uint64_t trace_offset,
    uint64_t* d_coefficients,
    cudaStream_t stream = 0
);

/**
 * Phase 2: Evaluate polynomial at a specific coset from pre-computed coefficients
 * 
 * This is the fast step that can be called multiple times with different coset offsets.
 * 
 * @param d_coefficients Pre-computed coefficients from compute_trace_coefficients_gpu()
 * @param num_cols Number of columns
 * @param trace_len Coefficient count per column
 * @param d_randomizer_coeffs Randomizer coefficients [num_cols * randomizer_len]
 * @param randomizer_len Randomizer length per column
 * @param trace_offset Original trace domain offset
 * @param coset_offset Target coset offset
 * @param d_output Output: LDE values at coset [num_cols * trace_len]
 * @param d_tail_scratch Scratch buffer for tail chunk [num_cols * trace_len]
 * @param stream CUDA stream
 */
void evaluate_coset_from_coefficients_gpu(
    const uint64_t* d_coefficients,
    size_t num_cols,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t coset_offset,
    uint64_t* d_output,
    uint64_t* d_tail_scratch,
    cudaStream_t stream = 0
);

/**
 * Batch randomized LDE for XFieldElement columns (aux table)
 *
 * Each XFE column is processed as 3 BFE components.
 * Supports both BFieldElement and XFieldElement randomizers.
 *
 * Input layout: d_xfe_trace_table is [num_cols * trace_len * 3] where
 *   component k of XFE at (col, row) = d_xfe_trace_table[(col * trace_len + row) * 3 + k]
 *
 * Output layout: d_xfe_output is [num_cols * target_len * 3] with same structure
 *
 * @param d_xfe_trace_table XFE trace data
 * @param num_cols Number of XFE columns
 * @param trace_len Trace domain length
 * @param d_randomizer_coeffs Randomizer coefficients: [num_cols * randomizer_len * 3] if use_xfe_randomizers, else [num_cols * randomizer_len] BFieldElements
 * @param randomizer_len Number of randomizer coefficients per column
 * @param trace_offset Trace domain offset
 * @param target_offset Target domain offset
 * @param target_len Target domain length
 * @param d_xfe_output Output XFE LDE data
 * @param use_xfe_randomizers true if randomizers are XFieldElement (3 components), false if BFieldElement (1 component)
 * @param stream CUDA stream
 */
void randomized_xfe_lde_batch_gpu(
    const uint64_t* d_xfe_trace_table,
    size_t num_cols,
    size_t trace_len,
    const uint64_t* d_randomizer_coeffs,
    size_t randomizer_len,
    uint64_t trace_offset,
    uint64_t target_offset,
    size_t target_len,
    uint64_t* d_xfe_output,
    bool use_xfe_randomizers,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

