/**
 * CUDA Backend Core Implementation
 */

#ifdef TRITON_CUDA_ENABLED

#include "backend/cuda_backend.hpp"
#include "gpu/cuda_common.cuh"
#include <iostream>

namespace triton_vm {

struct CudaBackend::Impl {
    // Future: memory pools, precomputed tables, etc.
};

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    // Print device info on first creation
    static bool info_printed = false;
    if (!info_printed) {
        print_device_info();
        info_printed = true;
    }
}

CudaBackend::~CudaBackend() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CudaBackend::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaBackend::check_error(const char* context) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string msg = std::string(context) + ": " + cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

void CudaBackend::print_device_info() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return;
    }
    
    std::cout << "=================================================\n";
    std::cout << "CUDA Device Information:\n";
    std::cout << "=================================================\n";
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Warp Size: " << prop.warpSize << "\n";
        std::cout << "  Shared Memory/Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
    }
    
    std::cout << "=================================================\n";
}

} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

