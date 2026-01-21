/**
 * Streaming LDE Kernel Header for Frugal Mode
 * 
 * Coset-based streaming LDE computation that trades compute for ~50% memory reduction.
 */

#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Constants from Rust implementation
constexpr size_t STREAMING_NUM_QUOTIENT_SEGMENTS = 4;
constexpr size_t STREAMING_RATIO = 2;  // RANDOMIZED_TRACE_LEN_TO_WORKING_DOMAIN_LEN_RATIO
constexpr size_t STREAMING_NUM_COSETS = STREAMING_NUM_QUOTIENT_SEGMENTS * STREAMING_RATIO;  // 8

/**
 * Evaluate main table polynomial coefficients on a coset of the working domain
 * 
 * @param d_main_coeffs Polynomial coefficients [trace_len × main_width] (column-major)
 * @param d_working_main Output evaluations [working_domain_len × main_width] (column-major)
 * @param trace_len Length of trace domain
 * @param working_domain_len Length of working domain (= randomized_trace_len / 2)
 * @param main_width Number of main table columns (379)
 * @param coset_offset The coset offset (ψ × ι^coset_index)
 * @param zerofier_value Trace zerofier at coset: coset_offset^trace_len - 1
 * @param d_randomizer_coeffs Trace randomizer coefficients [main_width × num_randomizers]
 * @param num_randomizers Number of randomizers per column
 * @param stream CUDA stream
 */
void streaming_evaluate_main_on_coset(
    const uint64_t* d_main_coeffs,
    uint64_t* d_working_main,
    size_t trace_len,
    size_t working_domain_len,
    size_t main_width,
    uint64_t coset_offset,
    uint64_t zerofier_value,
    const uint64_t* d_randomizer_coeffs,
    size_t num_randomizers,
    cudaStream_t stream
);

/**
 * Evaluate aux table polynomial coefficients on a coset of the working domain
 * XFieldElement version (3 BFE components per element)
 */
void streaming_evaluate_aux_on_coset(
    const uint64_t* d_aux_coeffs,
    uint64_t* d_working_aux,
    size_t trace_len,
    size_t working_domain_len,
    size_t aux_width,
    uint64_t coset_offset,
    uint64_t zerofier_value,
    const uint64_t* d_randomizer_coeffs,
    size_t num_randomizers,
    cudaStream_t stream
);

/**
 * Accumulate quotient evaluations from a coset into segment codewords
 * Implements the "segmentify" operation from Rust's JIT LDE approach
 * 
 * @param d_coset_quotient Quotient evaluated on current coset [working_domain_len × 3]
 * @param d_segment_codewords Accumulated segment codewords [4 × working_domain_len × 3]
 * @param working_domain_len Length of working domain
 * @param coset_index Current coset index (0..7)
 * @param iota Primitive root of unity of order (working_domain_len × NUM_COSETS)
 * @param stream CUDA stream
 */
void streaming_accumulate_coset_quotient(
    const uint64_t* d_coset_quotient,
    uint64_t* d_segment_codewords,
    size_t working_domain_len,
    size_t coset_index,
    uint64_t iota,
    cudaStream_t stream
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
