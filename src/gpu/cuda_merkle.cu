/**
 * CUDA Merkle Tree Backend Implementation
 */

#ifdef TRITON_CUDA_ENABLED

#include "backend/cuda_backend.hpp"
#include "gpu/cuda_memory.hpp"
#include "gpu/cuda_common.cuh"

namespace triton_vm {
namespace gpu {
namespace kernels {
    // Forward declarations
    void merkle_tree_device(const uint64_t* d_leaves, uint64_t* d_tree, 
                           size_t num_leaves, cudaStream_t stream);
    void merkle_root_device(const uint64_t* d_leaves, uint64_t* d_root,
                           size_t num_leaves, cudaStream_t stream);
}
}

Digest CudaBackend::merkle_root(const Digest* leaves, size_t num_leaves) {
    constexpr size_t DIGEST_LEN = 5;
    size_t total_elements = num_leaves * DIGEST_LEN;
    
    // Flatten digests
    std::vector<uint64_t> h_leaves(total_elements);
    for (size_t i = 0; i < num_leaves; ++i) {
        for (size_t j = 0; j < DIGEST_LEN; ++j) {
            h_leaves[i * DIGEST_LEN + j] = leaves[i][j].value();
        }
    }
    
    // Allocate device memory
    gpu::DeviceBuffer<uint64_t> d_leaves(total_elements);
    gpu::DeviceBuffer<uint64_t> d_root(DIGEST_LEN);
    
    // Upload
    d_leaves.upload(h_leaves.data(), total_elements);
    
    // Execute
    gpu::kernels::merkle_root_device(d_leaves.data(), d_root.data(), num_leaves, stream_);
    cudaStreamSynchronize(stream_);
    
    // Download root
    std::vector<uint64_t> h_root(DIGEST_LEN);
    d_root.download(h_root.data(), DIGEST_LEN);
    
    Digest result;
    for (size_t j = 0; j < DIGEST_LEN; ++j) {
        result[j] = BFieldElement(h_root[j]);
    }
    
    return result;
}

void CudaBackend::merkle_tree_full(
    const Digest* leaves,
    size_t num_leaves,
    Digest* tree
) {
    constexpr size_t DIGEST_LEN = 5;
    size_t total_leaves = num_leaves * DIGEST_LEN;
    size_t total_nodes = (2 * num_leaves - 1) * DIGEST_LEN;
    
    // Flatten digests
    std::vector<uint64_t> h_leaves(total_leaves);
    for (size_t i = 0; i < num_leaves; ++i) {
        for (size_t j = 0; j < DIGEST_LEN; ++j) {
            h_leaves[i * DIGEST_LEN + j] = leaves[i][j].value();
        }
    }
    
    // Allocate device memory
    gpu::DeviceBuffer<uint64_t> d_leaves(total_leaves);
    gpu::DeviceBuffer<uint64_t> d_tree(total_nodes);
    
    // Upload
    d_leaves.upload(h_leaves.data(), total_leaves);
    
    // Execute
    gpu::kernels::merkle_tree_device(d_leaves.data(), d_tree.data(), num_leaves, stream_);
    cudaStreamSynchronize(stream_);
    
    // Download tree
    std::vector<uint64_t> h_tree(total_nodes);
    d_tree.download(h_tree.data(), total_nodes);
    
    // Reconstruct digests
    size_t total_digests = 2 * num_leaves - 1;
    for (size_t i = 0; i < total_digests; ++i) {
        for (size_t j = 0; j < DIGEST_LEN; ++j) {
            tree[i][j] = BFieldElement(h_tree[i * DIGEST_LEN + j]);
        }
    }
}

} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

