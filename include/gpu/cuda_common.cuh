#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace triton_vm {
namespace gpu {

// ============================================================================
// Error Handling
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

// ============================================================================
// Field Constants (Goldilocks)
// ============================================================================

// Goldilocks prime: p = 2^64 - 2^32 + 1
__constant__ uint64_t GOLDILOCKS_PRIME = 0xFFFFFFFF00000001ULL;

// Generator for multiplicative group
__constant__ uint64_t GOLDILOCKS_GENERATOR = 7ULL;

// Primitive 2^32-th root of unity
__constant__ uint64_t GOLDILOCKS_TWO_ADIC_ROOT = 0x185629DCDA58878CULL;

// ============================================================================
// Device Field Arithmetic
// ============================================================================

/**
 * Modular addition: (a + b) mod p
 */
__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    // If overflow or >= p, subtract p
    if (sum < a || sum >= GOLDILOCKS_PRIME) {
        sum -= GOLDILOCKS_PRIME;
    }
    return sum;
}

/**
 * Modular subtraction: (a - b) mod p
 */
__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b) {
    uint64_t diff = a - b;
    // If underflow, add p
    if (a < b) {
        diff += GOLDILOCKS_PRIME;
    }
    return diff;
}

/**
 * Modular multiplication: (a * b) mod p using 128-bit arithmetic
 */
__device__ __forceinline__ uint64_t mul_mod(uint64_t a, uint64_t b) {
    // Use PTX for 64x64 -> 128 multiplication
    uint64_t hi, lo;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
    
    // Barrett-like reduction for Goldilocks
    // p = 2^64 - 2^32 + 1, so 2^64 ≡ 2^32 - 1 (mod p)
    uint64_t result = lo;
    
    // hi * 2^64 ≡ hi * (2^32 - 1) = hi * 2^32 - hi (mod p)
    uint64_t hi_shifted = hi << 32;
    uint64_t correction = hi_shifted - hi;
    
    result = add_mod(result, correction);
    
    // Handle the case where hi_shifted overflows
    if (hi >> 32) {
        // Additional correction needed
        uint64_t extra = (hi >> 32);
        result = add_mod(result, (extra << 32) - extra);
    }
    
    // Final reduction
    if (result >= GOLDILOCKS_PRIME) {
        result -= GOLDILOCKS_PRIME;
    }
    
    return result;
}

/**
 * Modular exponentiation: a^exp mod p
 */
__device__ __forceinline__ uint64_t pow_mod(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = mul_mod(result, base);
        }
        base = mul_mod(base, base);
        exp >>= 1;
    }
    return result;
}

/**
 * Modular inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p
 */
__device__ __forceinline__ uint64_t inv_mod(uint64_t a) {
    return pow_mod(a, GOLDILOCKS_PRIME - 2);
}

// ============================================================================
// Thread/Block Configuration Helpers
// ============================================================================

/**
 * Calculate optimal grid dimensions for a given problem size
 */
inline void get_launch_config(
    size_t n,
    int& grid_size,
    int& block_size,
    int max_threads_per_block = 256
) {
    block_size = max_threads_per_block;
    grid_size = (n + block_size - 1) / block_size;
    
    // Limit grid size to avoid launch failures
    if (grid_size > 65535) {
        grid_size = 65535;
    }
}

/**
 * Get shared memory size for NTT kernels
 */
inline size_t get_ntt_shared_memory_size(size_t n, size_t element_size) {
    // Use shared memory for butterflies within a block
    return n * element_size;
}

// ============================================================================
// Multi-GPU Support
// ============================================================================

// Global flag for unified memory mode (set via environment or command line)
inline bool& use_unified_memory() {
    static bool enabled = false;
    return enabled;
}

// Get the number of GPUs to use (set via TRITON_GPU_COUNT env var)
// Returns -1 if not set (use all GPUs), or the limit if set
inline int get_gpu_count_limit() {
    static int limit = -2; // -2 = not initialized
    if (limit == -2) {
        const char* env = std::getenv("TRITON_GPU_COUNT");
        if (env) {
            limit = std::atoi(env);
            if (limit < 1) limit = -1; // invalid, use all
        } else {
            limit = -1; // not set, use all
        }
    }
    return limit;
}

// Get effective GPU count (respects TRITON_GPU_COUNT limit)
inline int get_effective_gpu_count() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int limit = get_gpu_count_limit();
    if (limit > 0 && limit < device_count) {
        return limit;
    }
    return device_count;
}

/**
 * Enable unified memory across multiple GPUs
 * Call this once at startup before any allocations
 * Respects TRITON_GPU_COUNT environment variable to limit GPU count
 */
inline void enable_multi_gpu_unified_memory() {
    int total_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&total_devices));
    
    // Apply GPU count limit from environment variable
    int device_count = get_effective_gpu_count();
    
    if (device_count < 2) {
        printf("[GPU] Single GPU mode - unified memory will use local memory\n");
        return;
    }
    
    if (device_count < total_devices) {
        printf("[GPU] Multi-GPU setup: Using %d of %d GPUs (TRITON_GPU_COUNT=%d)\n", 
               device_count, total_devices, device_count);
    } else {
        printf("[GPU] Multi-GPU setup: %d GPUs detected\n", device_count);
    }
    
    // Enable peer access between GPU pairs (only for GPUs we're using)
    for (int i = 0; i < device_count; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < device_count; j++) {
            if (i != j) {
                int can_access = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err == cudaSuccess) {
                        printf("[GPU] Enabled P2P access: GPU %d -> GPU %d\n", i, j);
                    } else if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        printf("[GPU] Warning: Could not enable P2P: GPU %d -> GPU %d\n", i, j);
                    }
                }
            }
        }
    }
    
    // Return to GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    printf("[GPU] Multi-GPU unified memory enabled (%d GPUs)\n", device_count);
}

/**
 * Get total memory across GPUs (respects TRITON_GPU_COUNT limit)
 */
inline size_t get_total_gpu_memory() {
    int device_count = get_effective_gpu_count();
    
    size_t total = 0;
    for (int i = 0; i < device_count; i++) {
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        total += total_mem;
    }
    CUDA_CHECK(cudaSetDevice(0));
    return total;
}

/**
 * Allocate memory - uses unified memory (cudaMallocManaged) when multi-GPU is enabled
 * to allow memory access across multiple GPUs, otherwise uses regular device memory
 */
#define CUDA_ALLOC(ptr, size)                                                  \
    do {                                                                       \
        if (use_unified_memory()) {                                            \
            CUDA_CHECK(cudaMallocManaged(ptr, size));                           \
        } else {                                                                \
            CUDA_CHECK(cudaMalloc(ptr, size));                                  \
        }                                                                      \
    } while (0)

/**
 * Set memory advice for unified memory to optimize access patterns
 * Call this after allocating unified memory to hint which GPU should access it
 */
inline void advise_unified_memory(void* ptr, size_t size, int device_id = -1) {
    if (!use_unified_memory()) {
        return; // Not using unified memory, no advice needed
    }
    
    int device_count = get_effective_gpu_count();
    if (device_id < 0) {
        // Set read/write access for all GPUs
        for (int i = 0; i < device_count; i++) {
            CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, i));
            CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, i));
        }
    } else {
        // Set preferred location for specific GPU
        CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id));
    }
}

// ============================================================================
// Memory Utilities
// ============================================================================

/**
 * Async memory copy with stream
 */
template<typename T>
__host__ inline void async_copy_to_device(
    T* d_dst,
    const T* h_src,
    size_t count,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
}

template<typename T>
__host__ inline void async_copy_to_host(
    T* h_dst,
    const T* d_src,
    size_t count,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
}

} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

