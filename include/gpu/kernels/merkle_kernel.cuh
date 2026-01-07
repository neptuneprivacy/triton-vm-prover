#pragma once

/**
 * Merkle Tree CUDA Kernel Declarations
 * 
 * GPU-accelerated Merkle tree construction.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Build full Merkle tree on GPU
 * 
 * @param d_leaves Device memory with leaf digests (5 elements each)
 * @param d_tree Device memory for full tree (2*num_leaves - 1 digests)
 * @param num_leaves Number of leaves (must be power of 2)
 * @param stream CUDA stream
 */
void merkle_tree_gpu(
    const uint64_t* d_leaves,
    uint64_t* d_tree,
    size_t num_leaves,
    cudaStream_t stream = 0
);

/**
 * Compute just the Merkle root on GPU
 * 
 * @param d_leaves Device memory with leaf digests (5 elements each)
 * @param d_root Device memory for root digest (5 elements)
 * @param num_leaves Number of leaves (must be power of 2)
 * @param stream CUDA stream
 */
void merkle_root_gpu(
    const uint64_t* d_leaves,
    uint64_t* d_root,
    size_t num_leaves,
    cudaStream_t stream = 0
);

// Backward compatibility
void merkle_tree_device(
    const uint64_t* d_leaves,
    uint64_t* d_tree,
    size_t num_leaves,
    cudaStream_t stream = 0
);

void merkle_root_device(
    const uint64_t* d_leaves,
    uint64_t* d_root,
    size_t num_leaves,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

