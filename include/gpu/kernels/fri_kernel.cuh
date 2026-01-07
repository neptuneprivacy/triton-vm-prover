#pragma once

/**
 * FRI Protocol CUDA Kernel Declarations
 * 
 * GPU-accelerated FRI folding operations.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Perform FRI folding on GPU
 * 
 * Matches CPU FriRound::split_and_fold():
 *   scaled_offset_inv = challenge * domain_inv[i]
 *   left = (1 + scaled_offset_inv) * codeword[i]
 *   right = (1 - scaled_offset_inv) * codeword[half + i]
 *   folded[i] = (left + right) * two_inv
 * 
 * @param d_codeword Input codeword (XFE, 3 * codeword_len uint64_t)
 * @param codeword_len Number of XFieldElements
 * @param d_challenge Folding challenge (3 uint64_t for XFE)
 * @param d_domain_inv Inverses of domain points (half_len uint64_t)
 * @param two_inv Inverse of 2 in the base field
 * @param d_folded Output folded codeword (3 * half_len uint64_t)
 * @param stream CUDA stream
 */
void fri_fold_gpu(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    const uint64_t* d_domain_inv,
    uint64_t two_inv,
    uint64_t* d_folded,
    cudaStream_t stream = 0
);

/**
 * Perform FRI folding on GPU (optimized): compute domain inverses on-the-fly.
 *
 * Domain is coset points: x_i = offset * generator^i
 * We need inv(x_i) = inv_offset * inv_generator^i, where:
 *   inv_offset   = offset^{-1}
 *   inv_generator = generator^{-1}
 *
 * This avoids:
 * - a separate kernel launch for domain inverses
 * - a large temporary global array (half_len BFEs)
 * - per-element inversions
 */
void fri_fold_gpu_fast(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    uint64_t inv_offset,
    uint64_t inv_generator,
    uint64_t two_inv,
    uint64_t* d_folded,
    cudaStream_t stream = 0
);

/**
 * Compute domain point inverses for FRI folding
 * 
 * @param offset Domain offset
 * @param generator Domain generator
 * @param d_domain_inv Output inverses (half_len uint64_t)
 * @param half_len Number of inverses to compute
 * @param stream CUDA stream
 */
void compute_domain_inverses_gpu(
    uint64_t offset,
    uint64_t generator,
    uint64_t* d_domain_inv,
    size_t half_len,
    cudaStream_t stream = 0
);

/**
 * Compute domain inverses on GPU (optimized): inv(x_i) = inv_offset * inv_generator^i.
 *
 * This avoids per-element inversion and amortizes pow() by processing a small contiguous
 * chunk per thread.
 */
void compute_domain_inverses_fast_gpu(
    uint64_t inv_offset,
    uint64_t inv_generator,
    uint64_t* d_domain_inv,
    size_t half_len,
    cudaStream_t stream = 0
);

/**
 * Backward compatibility wrapper (simplified folding)
 */
void fri_fold_device(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    uint64_t* d_folded,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

