/**
 * CUDA Memory Management Implementation
 */

#include "gpu/cuda_memory.hpp"

#ifdef TRITON_CUDA_ENABLED

namespace triton_vm {
namespace gpu {

CudaMemoryPool& CudaMemoryPool::instance() {
    static CudaMemoryPool pool;
    return pool;
}

CudaMemoryPool::~CudaMemoryPool() {
    release_all();
}

void* CudaMemoryPool::allocate(size_t bytes) {
    // Check pool for suitable allocation
    for (auto it = device_pool_.begin(); it != device_pool_.end(); ++it) {
        if (it->size >= bytes) {
            void* ptr = it->ptr;
            device_pool_.erase(it);
            total_pooled_ -= bytes;
            return ptr;
        }
    }
    
    // Allocate new memory
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    total_allocated_ += bytes;
    return ptr;
}

void CudaMemoryPool::deallocate(void* ptr, size_t bytes) {
    // Add to pool for reuse
    device_pool_.push_back({ptr, bytes});
    total_pooled_ += bytes;
}

void* CudaMemoryPool::allocate_pinned(size_t bytes) {
    // Check pool for suitable allocation
    for (auto it = pinned_pool_.begin(); it != pinned_pool_.end(); ++it) {
        if (it->size >= bytes) {
            void* ptr = it->ptr;
            pinned_pool_.erase(it);
            return ptr;
        }
    }
    
    // Allocate new pinned memory
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed");
    }
    
    return ptr;
}

void CudaMemoryPool::deallocate_pinned(void* ptr, size_t bytes) {
    pinned_pool_.push_back({ptr, bytes});
}

void CudaMemoryPool::release_all() {
    for (auto& entry : device_pool_) {
        cudaFree(entry.ptr);
    }
    device_pool_.clear();
    
    for (auto& entry : pinned_pool_) {
        cudaFreeHost(entry.ptr);
    }
    pinned_pool_.clear();
    
    total_pooled_ = 0;
}

} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

