#pragma once

/**
 * GPU Quotient Post-Processing Kernel
 *
 * Accelerates NTT-heavy operations in quotient computation:
 * - Polynomial interpolation (inverse NTT)
 * - Coset evaluation (forward NTT)
 * 
 * Note: Constraint evaluation (MASTER_AUX_NUM_CONSTRAINTS constraints per row) remains on CPU
 * due to code complexity causing nvcc compilation issues.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Interpolate XFieldElement column on GPU using 3 parallel BField NTTs
 */
void interpolate_xfield_column_gpu(
    const uint64_t* d_values,  // [n * 3] XFieldElements
    uint64_t* d_coeffs,        // [n * 3] output coefficients
    size_t n,
    uint64_t offset_inv,       // inverse of domain offset
    cudaStream_t stream = 0
);

/**
 * Evaluate polynomial on coset (GPU version)
 */
void evaluate_on_coset_gpu(
    const uint64_t* d_coeffs,  // [n * 3] XFieldElement coefficients
    uint64_t* d_evals,         // [n * 3] output evaluations
    size_t n,
    uint64_t offset,           // coset offset
    cudaStream_t stream = 0
);

/**
 * Full GPU post-processing for quotient:
 *   - interpolate quotient codeword (inverse NTT + coset unscale)
 *   - split into segment polynomials (NUM_QUOTIENT_SEGMENTS)
 *   - evaluate each segment polynomial on FRI domain coset (coset scale + forward NTT)
 *
 * Outputs:
 * - d_segment_coeffs_compact: [num_segments * 3 * (n/num_segments)] (coefficients)
 * - d_segment_codewords_colmajor: [num_segments * 3 * n] (evaluations), layout:
 *      (segment*3 + component) * n + row
 */
void quotient_segmentify_and_lde_gpu(
    const uint64_t* d_quotient_values_xfe, // [n * 3] row-major XFE: [row0 c0,c1,c2, row1...]
    size_t quotient_len,
    size_t num_segments,
    uint64_t quotient_offset_inv,          // inverse of quotient domain offset
    uint64_t fri_offset,                   // FRI domain offset (coset)
    size_t fri_len,                        // FRI domain length
    uint64_t* d_segment_coeffs_compact,    // [num_segments * 3 * (quotient_len/num_segments)]
    uint64_t* d_segment_codewords_colmajor,// [num_segments * 3 * fri_len]
    cudaStream_t stream = 0,
    uint64_t* d_scratch_c0 = nullptr,      // Optional scratch buffer for c0 (must be >= quotient_len)
    uint64_t* d_scratch_c1 = nullptr       // Optional scratch buffer for c1 (must be >= quotient_len)
);

/**
 * Multi-coset quotient segmentify (matching Rust's algorithm)
 *
 * Inputs:
 * - d_quotient_multicoset: [working_len * num_cosets * 3] XFE evaluations (row-major)
 *
 * Outputs:
 * - d_seg_coeffs_compact: [num_segments * 3 * segment_len] where segment_len = working_len / num_segments
 * - d_segment_codewords_colmajor: [num_segments * 3 * fri_len] (evaluations)
 */
void quotient_segmentify_multicoset_gpu(
    const uint64_t* d_quotient_multicoset, // [working_len * num_cosets * 3] XFE row-major
    size_t working_len,
    size_t num_cosets,                    // 8
    size_t num_segments,                  // NUM_QUOTIENT_SEGMENTS (typically 4)
    uint64_t psi_inv,                     // fri_offset.inverse()
    uint64_t iota_inv,                    // iota.inverse()
    uint64_t fri_offset,
    size_t fri_len,
    uint64_t* d_seg_coeffs_compact,       // [num_segments * 3 * segment_len]
    uint64_t* d_segment_codewords_colmajor, // [num_segments * 3 * fri_len]
    cudaStream_t stream = 0
);

/**
 * Simplified segmentify for testing
 */
void quotient_segmentify_simple_gpu(
    const uint64_t* d_quotient_multicoset, // [working_len * num_cosets * 3]
    size_t working_len,
    size_t num_cosets,
    size_t num_segments,
    uint64_t* d_seg_coeffs_compact,       // [num_segments * 3 * segment_len]
    uint64_t* d_segment_codewords_colmajor, // [num_segments * 3 * fri_len]
    size_t fri_len,
    uint64_t fri_offset,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
