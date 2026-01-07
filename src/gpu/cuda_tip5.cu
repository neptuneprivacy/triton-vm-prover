/**
 * CUDA Tip5 Backend Implementation
 */

#ifdef TRITON_CUDA_ENABLED

#include "backend/cuda_backend.hpp"
#include "gpu/cuda_memory.hpp"
#include "gpu/cuda_common.cuh"

namespace triton_vm {
namespace gpu {
namespace kernels {
    // Forward declarations
    void tip5_permutation_device(uint64_t* d_states, size_t num_states, cudaStream_t stream);
    void hash_pairs_device(const uint64_t* d_left, const uint64_t* d_right, 
                          uint64_t* d_output, size_t count, cudaStream_t stream);
}
}

void CudaBackend::tip5_permutation_batch(uint64_t* states, size_t num_states) {
    constexpr size_t STATE_SIZE = 16;
    size_t total_elements = num_states * STATE_SIZE;
    
    gpu::DeviceBuffer<uint64_t> d_states(total_elements);
    
    // Upload
    d_states.upload(states, total_elements);
    
    // Execute
    gpu::kernels::tip5_permutation_device(d_states.data(), num_states, stream_);
    cudaStreamSynchronize(stream_);
    
    // Download
    d_states.download(states, total_elements);
}

void CudaBackend::hash_pairs(
    const Digest* left,
    const Digest* right,
    Digest* output,
    size_t count
) {
    constexpr size_t DIGEST_LEN = 5;
    size_t total_elements = count * DIGEST_LEN;
    
    // Flatten digests to raw u64 arrays
    std::vector<uint64_t> h_left(total_elements);
    std::vector<uint64_t> h_right(total_elements);
    
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < DIGEST_LEN; ++j) {
            h_left[i * DIGEST_LEN + j] = left[i][j].value();
            h_right[i * DIGEST_LEN + j] = right[i][j].value();
        }
    }
    
    // Allocate device memory
    gpu::DeviceBuffer<uint64_t> d_left(total_elements);
    gpu::DeviceBuffer<uint64_t> d_right(total_elements);
    gpu::DeviceBuffer<uint64_t> d_output(total_elements);
    
    // Upload
    d_left.upload(h_left.data(), total_elements);
    d_right.upload(h_right.data(), total_elements);
    
    // Execute
    gpu::kernels::hash_pairs_device(
        d_left.data(), d_right.data(), d_output.data(), count, stream_
    );
    cudaStreamSynchronize(stream_);
    
    // Download and reconstruct digests
    std::vector<uint64_t> h_output(total_elements);
    d_output.download(h_output.data(), total_elements);
    
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < DIGEST_LEN; ++j) {
            output[i][j] = BFieldElement(h_output[i * DIGEST_LEN + j]);
        }
    }
}

} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

