/**
 * CUDA FRI Backend Implementation
 */

#ifdef TRITON_CUDA_ENABLED

#include "backend/cuda_backend.hpp"
#include "gpu/cuda_memory.hpp"
#include "gpu/cuda_common.cuh"

namespace triton_vm {
namespace gpu {
namespace kernels {
    // Forward declaration
    void fri_fold_device(const uint64_t* d_codeword, size_t codeword_len,
                        const uint64_t* d_challenge, uint64_t* d_folded,
                        cudaStream_t stream);
}
}

void CudaBackend::fri_fold(
    const XFieldElement* codeword,
    size_t codeword_len,
    const XFieldElement& challenge,
    XFieldElement* folded
) {
    size_t half_len = codeword_len / 2;
    
    // Flatten XFieldElements to u64 arrays (3 per element)
    std::vector<uint64_t> h_codeword(codeword_len * 3);
    for (size_t i = 0; i < codeword_len; ++i) {
        h_codeword[i * 3] = codeword[i].coeff(0).value();
        h_codeword[i * 3 + 1] = codeword[i].coeff(1).value();
        h_codeword[i * 3 + 2] = codeword[i].coeff(2).value();
    }
    
    uint64_t h_challenge[3] = {
        challenge.coeff(0).value(),
        challenge.coeff(1).value(),
        challenge.coeff(2).value()
    };
    
    // Allocate device memory
    gpu::DeviceBuffer<uint64_t> d_codeword(codeword_len * 3);
    gpu::DeviceBuffer<uint64_t> d_challenge(3);
    gpu::DeviceBuffer<uint64_t> d_folded(half_len * 3);
    
    // Upload
    d_codeword.upload(h_codeword.data(), codeword_len * 3);
    d_challenge.upload(h_challenge, 3);
    
    // Execute
    gpu::kernels::fri_fold_device(
        d_codeword.data(), codeword_len,
        d_challenge.data(), d_folded.data(),
        stream_
    );
    cudaStreamSynchronize(stream_);
    
    // Download
    std::vector<uint64_t> h_folded(half_len * 3);
    d_folded.download(h_folded.data(), half_len * 3);
    
    // Reconstruct XFieldElements
    for (size_t i = 0; i < half_len; ++i) {
        folded[i] = XFieldElement(
            BFieldElement(h_folded[i * 3]),
            BFieldElement(h_folded[i * 3 + 1]),
            BFieldElement(h_folded[i * 3 + 2])
        );
    }
}

} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

