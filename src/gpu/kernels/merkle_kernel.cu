/**
 * Merkle Tree CUDA Kernel Implementation
 * 
 * GPU-accelerated Merkle tree construction using Tip5 hash.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/tip5_kernel.cuh"
#include "gpu/cuda_common.cuh"
#include <cuda_runtime.h>

namespace triton_vm {
namespace gpu {
namespace kernels {

constexpr size_t DIGEST_LEN = 5;

// ============================================================================
// Copy Kernel (works with both device and unified memory)
// ============================================================================

__global__ void copy_digests_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    size_t num_digests
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_digests) return;
    
    size_t base = idx * DIGEST_LEN;
    #pragma unroll
    for (size_t i = 0; i < DIGEST_LEN; ++i) {
        dst[base + i] = src[base + i];
    }
}

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
        // Use cudaMemcpyDefault for unified memory compatibility
        cudaMemcpyKind copy_kind = triton_vm::gpu::use_unified_memory() 
            ? cudaMemcpyDefault 
            : cudaMemcpyDeviceToDevice;
        cudaMemcpyAsync(
            d_tree,
            d_leaves,
            num_leaves * DIGEST_LEN * sizeof(uint64_t),
            copy_kind,
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
 * Compute just the Merkle root on GPU (optimized - computes root directly)
 * 
 * This version computes the root by hashing level by level without storing
 * the full tree. This avoids large memory allocations and copy issues.
 */
void merkle_root_gpu(
    const uint64_t* d_leaves,
    uint64_t* d_root,
    size_t num_leaves,
    cudaStream_t stream
) {
    if (num_leaves == 0) return;
    
    if (num_leaves == 1) {
        // Single leaf: root is the leaf itself
        dim3 block(256);
        dim3 grid(1);
        copy_digests_kernel<<<grid, block, 0, stream>>>(d_leaves, d_root, 1);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    
    // Compute root by hashing level by level
    // We only need two buffers: one for current level, one for next level
    size_t current_size = num_leaves;
    const uint64_t* current_level = d_leaves;
    
    // Allocate buffers for two levels (current and next)
    // Next level will be at most current_size/2
    size_t level_a_size = num_leaves;
    size_t level_b_size = (num_leaves + 1) / 2;
    uint64_t* d_level_a = nullptr;
    uint64_t* d_level_b = nullptr;
    
    size_t level_a_bytes = level_a_size * DIGEST_LEN * sizeof(uint64_t);
    size_t level_b_bytes = level_b_size * DIGEST_LEN * sizeof(uint64_t);
    
    if (triton_vm::gpu::use_unified_memory()) {
        CUDA_CHECK(cudaMallocManaged(&d_level_a, level_a_bytes));
        CUDA_CHECK(cudaMallocManaged(&d_level_b, level_b_bytes));
    } else {
        CUDA_CHECK(cudaMalloc(&d_level_a, level_a_bytes));
        CUDA_CHECK(cudaMalloc(&d_level_b, level_b_bytes));
    }
    
    // Copy leaves to first buffer using cudaMemcpyAsync (works better with unified memory)
    cudaMemcpyKind copy_kind = triton_vm::gpu::use_unified_memory() 
        ? cudaMemcpyDefault 
        : cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpyAsync(
        d_level_a,
        d_leaves,
        num_leaves * DIGEST_LEN * sizeof(uint64_t),
        copy_kind,
        stream
    ));
    
    // Synchronize to ensure copy is complete before hashing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    uint64_t* d_current = d_level_a;
    uint64_t* d_next = d_level_b;
    
    // Hash level by level until we get the root
    while (current_size > 1) {
        size_t pairs = current_size / 2;
        if (pairs == 0) break; // Safety check
        
        // Hash pairs: current[2*i], current[2*i+1] -> next[i]
        // Verify we have enough space
        if (pairs > level_b_size && d_next == d_level_b) {
            throw std::runtime_error("Insufficient buffer size for hash pairs");
        }
        hash_pairs_strided_gpu(d_current, d_next, pairs, stream);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap buffers for next iteration
        uint64_t* temp = d_current;
        d_current = d_next;
        d_next = temp;
        
        current_size = pairs;
    }
    
    // Root is in d_current[0]
    dim3 root_block(256);
    dim3 root_grid(1);
    copy_digests_kernel<<<root_grid, root_block, 0, stream>>>(d_current, d_root, 1);
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize before freeing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_level_a));
    CUDA_CHECK(cudaFree(d_level_b));
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
