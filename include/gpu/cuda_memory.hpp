#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>

#ifdef TRITON_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace triton_vm {
namespace gpu {

#ifdef TRITON_CUDA_ENABLED

/**
 * RAII wrapper for device memory allocation.
 * Automatically frees memory on destruction.
 */
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : d_ptr_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t count) : d_ptr_(nullptr), size_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&d_ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed: " + 
                    std::string(cudaGetErrorString(err)));
            }
        }
    }
    
    ~DeviceBuffer() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
        }
    }
    
    // Move only
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : d_ptr_(other.d_ptr_), size_(other.size_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (d_ptr_) {
                cudaFree(d_ptr_);
            }
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // No copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    /**
     * Upload data from host to device
     */
    void upload(const T* host_data, size_t count) {
        if (count > size_) {
            throw std::runtime_error("Upload count exceeds buffer size");
        }
        cudaError_t err = cudaMemcpy(d_ptr_, host_data, count * sizeof(T),
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy H2D failed");
        }
    }
    
    void upload(const std::vector<T>& host_data) {
        upload(host_data.data(), host_data.size());
    }
    
    /**
     * Upload data asynchronously
     */
    void upload_async(const T* host_data, size_t count, cudaStream_t stream) {
        if (count > size_) {
            throw std::runtime_error("Upload count exceeds buffer size");
        }
        cudaError_t err = cudaMemcpyAsync(d_ptr_, host_data, count * sizeof(T),
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpyAsync H2D failed");
        }
    }
    
    /**
     * Download data from device to host
     */
    void download(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("Download count exceeds buffer size");
        }
        cudaError_t err = cudaMemcpy(host_data, d_ptr_, count * sizeof(T),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D2H failed");
        }
    }
    
    std::vector<T> download() const {
        std::vector<T> result(size_);
        download(result.data(), size_);
        return result;
    }
    
    /**
     * Download data asynchronously
     */
    void download_async(T* host_data, size_t count, cudaStream_t stream) const {
        if (count > size_) {
            throw std::runtime_error("Download count exceeds buffer size");
        }
        cudaError_t err = cudaMemcpyAsync(host_data, d_ptr_, count * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpyAsync D2H failed");
        }
    }
    
    /**
     * Clear buffer to zero
     */
    void zero() {
        if (d_ptr_) {
            cudaMemset(d_ptr_, 0, size_ * sizeof(T));
        }
    }
    
    T* data() { return d_ptr_; }
    const T* data() const { return d_ptr_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    
    T* begin() { return d_ptr_; }
    T* end() { return d_ptr_ + size_; }
    const T* begin() const { return d_ptr_; }
    const T* end() const { return d_ptr_ + size_; }
    
private:
    T* d_ptr_;
    size_t size_;
};

/**
 * RAII wrapper for pinned (page-locked) host memory.
 * Enables faster CPU-GPU transfers.
 */
template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer() : h_ptr_(nullptr), size_(0) {}
    
    explicit PinnedBuffer(size_t count) : h_ptr_(nullptr), size_(count) {
        if (count > 0) {
            cudaError_t err = cudaMallocHost(&h_ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMallocHost failed");
            }
        }
    }
    
    ~PinnedBuffer() {
        if (h_ptr_) {
            cudaFreeHost(h_ptr_);
        }
    }
    
    // Move only
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : h_ptr_(other.h_ptr_), size_(other.size_) {
        other.h_ptr_ = nullptr;
        other.size_ = 0;
    }
    
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (h_ptr_) {
                cudaFreeHost(h_ptr_);
            }
            h_ptr_ = other.h_ptr_;
            size_ = other.size_;
            other.h_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // No copy
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    
    T* data() { return h_ptr_; }
    const T* data() const { return h_ptr_; }
    size_t size() const { return size_; }
    
    T& operator[](size_t i) { return h_ptr_[i]; }
    const T& operator[](size_t i) const { return h_ptr_[i]; }
    
private:
    T* h_ptr_;
    size_t size_;
};

/**
 * Memory pool for efficient GPU memory reuse.
 * Reduces allocation overhead for repeated operations.
 */
class CudaMemoryPool {
public:
    static CudaMemoryPool& instance();
    
    /**
     * Allocate device memory (may reuse from pool)
     */
    void* allocate(size_t bytes);
    
    /**
     * Return memory to pool
     */
    void deallocate(void* ptr, size_t bytes);
    
    /**
     * Allocate pinned host memory
     */
    void* allocate_pinned(size_t bytes);
    
    /**
     * Return pinned memory to pool
     */
    void deallocate_pinned(void* ptr, size_t bytes);
    
    /**
     * Release all pooled memory
     */
    void release_all();
    
    /**
     * Get statistics
     */
    size_t total_allocated() const { return total_allocated_; }
    size_t total_pooled() const { return total_pooled_; }
    
private:
    CudaMemoryPool() = default;
    ~CudaMemoryPool();
    
    struct PoolEntry {
        void* ptr;
        size_t size;
    };
    
    std::vector<PoolEntry> device_pool_;
    std::vector<PoolEntry> pinned_pool_;
    size_t total_allocated_ = 0;
    size_t total_pooled_ = 0;
};

#else // !TRITON_CUDA_ENABLED

// Stub implementations when CUDA is not available
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t) {
        throw std::runtime_error("CUDA not available");
    }
    void upload(const T*, size_t) { throw std::runtime_error("CUDA not available"); }
    void download(T*, size_t) const { throw std::runtime_error("CUDA not available"); }
    T* data() { return nullptr; }
    size_t size() const { return 0; }
};

template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t) {
        throw std::runtime_error("CUDA not available");
    }
    T* data() { return nullptr; }
    size_t size() const { return 0; }
};

#endif // TRITON_CUDA_ENABLED

} // namespace gpu
} // namespace triton_vm

