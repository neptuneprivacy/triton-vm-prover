/**
 * Merkle Tree CUDA Kernel Implementation
 * 
 * GPU-accelerated Merkle tree construction using Tip5 hash.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/tip5_kernel.cuh"
#include <cuda_runtime.h>

namespace triton_vm {
namespace gpu {
namespace kernels {

constexpr size_t DIGEST_LEN = 5;

// ============================================================================
// Host Interface
// ============================================================================

/**
 * Build full Merkle tree on GPU
 * 
 * Tree layout (1-indexed internal representation, 0-indexed storage):
 *   Storage: [leaves | internal nodes | root]
 *   - leaves at indices [0, num_leaves)
 *   - internal nodes at [num_leaves, 2*num_leaves - 2)
 *   - root at index 2*num_leaves - 2
 * 
 * But we'll use a simpler layout: bottom-up computation
 *   - Level 0 (leaves): indices [0, num_leaves)
 *   - Level 1 (first parents): write to temp buffer
 *   - Continue until root
 * 
 * @param d_leaves Device memory with leaf digests
 * @param d_tree Device memory for full tree (2*num_leaves - 1 digests)
 * @param num_leaves Number of leaves (must be power of 2)
 * @param stream CUDA stream
 */
void merkle_tree_gpu(
    const uint64_t* d_leaves,
    uint64_t* d_tree,
    size_t num_leaves,
    cudaStream_t stream
) {
    if (num_leaves == 0) return;
    if (num_leaves == 1) {
        // Single leaf is also root
        cudaMemcpyAsync(
            d_tree,
            d_leaves,
            DIGEST_LEN * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice,
            stream
        );
        return;
    }
    
    // Tree storage layout:
    // [leaves at 0..num_leaves-1][level1][level2]...[root]
    // Total: 2*num_leaves - 1 nodes
    
    // Copy leaves to beginning of tree (unless caller already wrote leaves in-place).
    if (d_tree != d_leaves) {
        cudaMemcpyAsync(
            d_tree,
            d_leaves,
            num_leaves * DIGEST_LEN * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice,
            stream
        );
    }
    
    // Build tree level by level, bottom-up.
    // NOTE: Keep everything on GPU. Hash interleaved pairs directly (no temp left/right buffers).
    size_t current_level_size = num_leaves;
    size_t current_level_offset = 0;
    size_t next_level_offset = num_leaves;
    
    while (current_level_size > 1) {
        size_t pairs = current_level_size / 2;
        
        // Hash adjacent pairs: tree[current_level_offset + 2*i], tree[current_level_offset + 2*i + 1]
        // -> tree[next_level_offset + i]
        
        uint64_t* parents = d_tree + next_level_offset * DIGEST_LEN;

        // Hash interleaved (d0,d1)->p0, (d2,d3)->p1, ...
        hash_pairs_strided_gpu(
            d_tree + current_level_offset * DIGEST_LEN,
            parents,
            pairs,
            stream
        );
        
        // Move to next level
        current_level_offset = next_level_offset;
        next_level_offset += pairs;
        current_level_size = pairs;
    }
}

/**
 * Compute just the Merkle root on GPU (optimized - no temporary allocation)
 * 
 * This version computes the root in-place by building the tree level by level
 * and only keeping the current level in memory.
 */
void merkle_root_gpu(
    const uint64_t* d_leaves,
    uint64_t* d_root,
    size_t num_leaves,
    cudaStream_t stream
) {
    if (num_leaves == 0) return;
    
    if (num_leaves == 1) {
        cudaMemcpyAsync(
            d_root,
            d_leaves,
            DIGEST_LEN * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice,
            stream
        );
        return;
    }
    
    // For small trees, use a scratch buffer that's reused
    // For large trees, we build level by level in-place
    size_t current_level_size = num_leaves;
    const uint64_t* current_level = d_leaves;
    
    // Allocate scratch space for one level (worst case: num_leaves/2 digests)
    size_t max_level_size = (num_leaves + 1) / 2;
    uint64_t* d_level_a;
    uint64_t* d_level_b;
    cudaMalloc(&d_level_a, max_level_size * DIGEST_LEN * sizeof(uint64_t));
    cudaMalloc(&d_level_b, max_level_size * DIGEST_LEN * sizeof(uint64_t));
    
    uint64_t* d_current = d_level_a;
    uint64_t* d_next = d_level_b;
    
    // Copy leaves to first level buffer
    cudaMemcpyAsync(
        d_current,
        d_leaves,
        num_leaves * DIGEST_LEN * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,
        stream
    );
    
    // Build tree level by level
    while (current_level_size > 1) {
        size_t pairs = current_level_size / 2;
        
        // Hash pairs: current_level[2*i], current_level[2*i+1] -> next_level[i]
        hash_pairs_strided_gpu(
            d_current,
            d_next,
            pairs,
            stream
        );
        
        // Swap buffers
        uint64_t* temp = d_current;
        d_current = d_next;
        d_next = temp;
        
        current_level_size = pairs;
    }
    
    // Root is in d_current[0]
    cudaMemcpyAsync(
        d_root,
        d_current,
        DIGEST_LEN * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,
        stream
    );
    
    cudaFree(d_level_a);
    cudaFree(d_level_b);
}

// Backward compatibility
void merkle_tree_device(
    const uint64_t* d_leaves,
    uint64_t* d_tree,
    size_t num_leaves,
    cudaStream_t stream
) {
    merkle_tree_gpu(d_leaves, d_tree, num_leaves, stream);
}

void merkle_root_device(
    const uint64_t* d_leaves,
    uint64_t* d_root,
    size_t num_leaves,
    cudaStream_t stream
) {
    merkle_root_gpu(d_leaves, d_root, num_leaves, stream);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
