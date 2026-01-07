#pragma once

/**
 * Low Degree Extension (LDE) CUDA Kernel Declarations
 * 
 * GPU-accelerated polynomial extension for STARK proof generation.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Perform LDE on a single column
 * 
 * Extends polynomial evaluations from trace domain to extended domain:
 * 1. Coset-unscale by trace_offset^{-i}
 * 2. INTT to get coefficients
 * 3. Pad with zeros
 * 4. NTT on extended domain
 * 5. Coset-scale by extended_offset^i
 * 
 * @param d_trace Input trace column (trace_len elements)
 * @param trace_len Size of trace (must be power of 2)
 * @param d_extended Output extended column (extended_len elements)
 * @param extended_len Size of extended domain (must be power of 2)
 * @param trace_offset Coset offset for trace domain
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
    cudaStream_t stream = 0
);

/**
 * Batch LDE for multiple columns
 * 
 * @param d_traces All trace columns contiguous (num_columns * trace_len)
 * @param num_columns Number of columns
 * @param trace_len Size of each trace column
 * @param d_extended All extended columns contiguous (num_columns * extended_len)
 * @param extended_len Size of each extended column
 * @param trace_offset Coset offset for trace domain
 * @param extended_offset Coset offset for extended domain
 * @param stream CUDA stream
 */
void lde_batch_gpu(
    const uint64_t* d_traces,
    size_t num_columns,
    size_t trace_len,
    uint64_t* d_extended,
    size_t extended_len,
    uint64_t trace_offset,
    uint64_t extended_offset,
    cudaStream_t stream = 0
);

// Backward compatibility
void lde_column_device(
    const uint64_t* d_trace,
    size_t trace_len,
    uint64_t* d_extended,
    size_t extended_len,
    uint64_t trace_offset,
    uint64_t extended_offset,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

