#pragma once

/**
 * NTT (Number Theoretic Transform) CUDA Kernel Declarations
 * 
 * GPU-accelerated forward and inverse NTT for Goldilocks field.
 * Uses Cooley-Tukey decimation-in-time algorithm.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Initialize NTT constants (roots of unity, etc.)
 * Must be called before any NTT operations.
 * Thread-safe and idempotent.
 */
void ntt_init_constants();

/**
 * Forward NTT: polynomial coefficients → evaluations
 * 
 * Transforms coefficients [c₀, c₁, ..., c_{n-1}] to evaluations
 * [f(ω⁰), f(ω¹), ..., f(ω^{n-1})] where ω is the n-th root of unity.
 * 
 * @param d_data Device pointer to data (n elements, modified in-place)
 * @param n Size (must be power of 2)
 * @param stream CUDA stream
 */
void ntt_forward_gpu(
    uint64_t* d_data,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Inverse NTT: evaluations → polynomial coefficients
 * 
 * Transforms evaluations [f(ω⁰), f(ω¹), ..., f(ω^{n-1})] to
 * coefficients [c₀, c₁, ..., c_{n-1}].
 * Includes scaling by n^{-1}.
 * 
 * @param d_data Device pointer to data (n elements, modified in-place)
 * @param n Size (must be power of 2)
 * @param stream CUDA stream
 */
void ntt_inverse_gpu(
    uint64_t* d_data,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Batch NTT - perform NTT on multiple arrays
 * 
 * @param d_data_ptrs Array of device pointers
 * @param n Size of each array (must be power of 2)
 * @param batch_size Number of arrays
 * @param inverse True for inverse NTT
 * @param stream CUDA stream
 */
void ntt_batch_gpu(
    uint64_t** d_data_ptrs,
    size_t n,
    size_t batch_size,
    bool inverse,
    cudaStream_t stream = 0
);

/**
 * Batched forward NTT for contiguous column-major data
 * 
 * Performs NTT on num_cols columns simultaneously, where all columns
 * are stored contiguously in column-major order.
 * 
 * @param d_data Device pointer to data (num_cols * n elements)
 *               Layout: d_data[col * n + row] = element at (col, row)
 * @param n Size of each column (must be power of 2)
 * @param num_cols Number of columns
 * @param stream CUDA stream
 */
void ntt_forward_batched_gpu(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    cudaStream_t stream = 0
);

/**
 * Batched inverse NTT for contiguous column-major data
 * 
 * @param d_data Device pointer to data (num_cols * n elements)
 * @param n Size of each column (must be power of 2)
 * @param num_cols Number of columns
 * @param stream CUDA stream
 */
void ntt_inverse_batched_gpu(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

