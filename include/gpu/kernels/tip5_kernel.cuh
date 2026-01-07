#pragma once

/**
 * Tip5 Hash CUDA Kernel Declarations
 * 
 * GPU-accelerated Tip5 permutation for Merkle tree construction.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Hash pairs of digests for Merkle tree construction
 * 
 * Computes Tip5(left || right) where left and right are 5-element digests.
 * 
 * @param d_left Left digests (5 elements each)
 * @param d_right Right digests (5 elements each)
 * @param d_output Output digests (5 elements each)
 * @param count Number of pairs
 * @param stream CUDA stream
 */
void hash_pairs_gpu(
    const uint64_t* d_left,
    const uint64_t* d_right,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream = 0
);

/**
 * Hash interleaved pairs of digests for Merkle tree construction.
 *
 * Input layout is interleaved digests:
 *   d_level = [d0, d1, d2, d3, ...] (each digest is 5 u64)
 * and computes:
 *   out[i] = Tip5(d_{2i} || d_{2i+1})
 *
 * This avoids materializing contiguous left/right buffers.
 *
 * @param d_level Interleaved digests (2*count digests, 5 elements each)
 * @param d_output Output digests (count digests, 5 elements each)
 * @param count Number of pairs
 * @param stream CUDA stream
 */
void hash_pairs_strided_gpu(
    const uint64_t* d_level,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream = 0
);

/**
 * Batch Tip5 permutation
 * 
 * @param d_states Device pointer to states (16 elements each)
 * @param num_states Number of states
 * @param stream CUDA stream
 */
void tip5_permutation_gpu(
    uint64_t* d_states,
    size_t num_states,
    cudaStream_t stream = 0
);

/**
 * Backward compatibility wrapper for hash_pairs_gpu
 */
void hash_pairs_device(
    const uint64_t* d_left,
    const uint64_t* d_right,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream = 0
);

/**
 * Initialize Tip5 tables in device memory for Fiat-Shamir operations
 * 
 * Allocates and initializes the S-box, MDS matrix, and round constants
 * tables in device memory. The caller is responsible for freeing these.
 * 
 * @param d_sbox_table Output: S-box lookup table (65536 uint16_t)
 * @param d_mds_matrix Output: MDS matrix first row (16 uint64_t)
 * @param d_round_constants Output: Round constants (128 uint64_t for 8 rounds)
 */
void tip5_init_tables(
    uint16_t* d_sbox_table,
    uint64_t* d_mds_matrix,
    uint64_t* d_round_constants
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

