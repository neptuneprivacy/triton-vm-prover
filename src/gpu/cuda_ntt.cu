/**
 * CUDA NTT Backend Implementation
 */

#ifdef TRITON_CUDA_ENABLED

#include "backend/cuda_backend.hpp"
#include "gpu/cuda_memory.hpp"
#include "gpu/kernels/ntt_kernel.cuh"

namespace triton_vm {

void CudaBackend::ntt_forward(BFieldElement* data, size_t n) {
    gpu::DeviceBuffer<uint64_t> d_data(n);
    
    // Upload
    std::vector<uint64_t> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = data[i].value();
    }
    d_data.upload(h_data.data(), n);
    
    // Execute
    gpu::kernels::ntt_forward_gpu(d_data.data(), n, stream_);
    cudaStreamSynchronize(stream_);
    
    // Download
    d_data.download(h_data.data(), n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = BFieldElement(h_data[i]);
    }
}

void CudaBackend::ntt_inverse(BFieldElement* data, size_t n) {
    gpu::DeviceBuffer<uint64_t> d_data(n);
    
    // Upload
    std::vector<uint64_t> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = data[i].value();
    }
    d_data.upload(h_data.data(), n);
    
    // Execute
    gpu::kernels::ntt_inverse_gpu(d_data.data(), n, stream_);
    cudaStreamSynchronize(stream_);
    
    // Download
    d_data.download(h_data.data(), n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = BFieldElement(h_data[i]);
    }
}

void CudaBackend::ntt_batch(BFieldElement** data, size_t n, size_t batch_size) {
    gpu::DeviceBuffer<uint64_t> d_data(n * batch_size);
    
    // Upload all arrays
    std::vector<uint64_t> h_data(n * batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < n; ++i) {
            h_data[b * n + i] = data[b][i].value();
        }
    }
    d_data.upload(h_data.data(), n * batch_size);
    
    // Execute NTT on each array
    for (size_t b = 0; b < batch_size; ++b) {
        gpu::kernels::ntt_forward_gpu(d_data.data() + b * n, n, stream_);
    }
    cudaStreamSynchronize(stream_);
    
    // Download all arrays
    d_data.download(h_data.data(), n * batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < n; ++i) {
            data[b][i] = BFieldElement(h_data[b * n + i]);
        }
    }
}

} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
