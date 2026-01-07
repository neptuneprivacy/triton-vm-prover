/**
 * NTT Co-Verification Tests
 * 
 * Compares CPU NTT implementation with GPU CUDA kernels
 * to ensure they produce identical results.
 */

#include "co_verify_framework.hpp"
#include "ntt/ntt.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/ntt_kernel.cuh"
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

namespace triton_vm {
namespace co_verify {

class NttCoVerifyTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "No CUDA device available";
        }
        gpu::kernels::ntt_init_constants();
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
    
    std::mt19937_64 rng_{42};
    
    // Generate random BFieldElement
    BFieldElement random_bfield() {
        std::uniform_int_distribution<uint64_t> dist(0, BFieldElement::MODULUS - 1);
        return BFieldElement(dist(rng_));
    }
    
    // Generate random vector of BFieldElements
    std::vector<BFieldElement> random_vector(size_t n) {
        std::vector<BFieldElement> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = random_bfield();
        }
        return v;
    }
    
    // Convert BFieldElement vector to uint64_t vector
    std::vector<uint64_t> to_u64(const std::vector<BFieldElement>& v) {
        std::vector<uint64_t> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i].value();
        }
        return result;
    }
    
    // Convert uint64_t vector to BFieldElement vector
    std::vector<BFieldElement> from_u64(const std::vector<uint64_t>& v) {
        std::vector<BFieldElement> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = BFieldElement(v[i]);
        }
        return result;
    }
    
    // Compare two BFieldElements
    static bool compare_bfield(const BFieldElement& a, const BFieldElement& b) {
        return a == b;
    }
};

// ============================================================================
// Forward NTT Tests
// ============================================================================

TEST_F(NttCoVerifyTest, Forward_Small) {
#ifdef TRITON_CUDA_ENABLED
    // Test small sizes: 2, 4, 8, 16
    for (size_t n : {2, 4, 8, 16}) {
        auto input = random_vector(n);
        
        // CPU computation
        auto cpu_result = input;
        NTT::forward(cpu_result);
        
        // GPU computation
        auto gpu_input = to_u64(input);
        uint64_t* d_data;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::ntt_forward_gpu(d_data, n);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_output(n);
        cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        auto gpu_result = from_u64(gpu_output);
        
        // Compare
        size_t mismatch;
        bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
        EXPECT_TRUE(match) << "Forward NTT mismatch at n=" << n << ", index " << mismatch
                           << ": CPU=" << cpu_result[mismatch] << ", GPU=" << gpu_result[mismatch];
        
        cudaFree(d_data);
    }
#endif
}

TEST_F(NttCoVerifyTest, Forward_Medium) {
#ifdef TRITON_CUDA_ENABLED
    // Test medium sizes: 256, 1024, 4096
    for (size_t n : {256, 1024, 4096}) {
        auto input = random_vector(n);
        
        // CPU computation
        auto cpu_result = input;
        NTT::forward(cpu_result);
        
        // GPU computation
        auto gpu_input = to_u64(input);
        uint64_t* d_data;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::ntt_forward_gpu(d_data, n);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_output(n);
        cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        auto gpu_result = from_u64(gpu_output);
        
        // Compare
        size_t mismatch;
        bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
        EXPECT_TRUE(match) << "Forward NTT mismatch at n=" << n << ", index " << mismatch
                           << ": CPU=" << cpu_result[mismatch] << ", GPU=" << gpu_result[mismatch];
        
        cudaFree(d_data);
    }
#endif
}

TEST_F(NttCoVerifyTest, Forward_Large) {
#ifdef TRITON_CUDA_ENABLED
    // Test larger sizes: 16K, 64K
    for (size_t n : {16384, 65536}) {
        auto input = random_vector(n);
        
        // CPU computation
        auto cpu_result = input;
        NTT::forward(cpu_result);
        
        // GPU computation
        auto gpu_input = to_u64(input);
        uint64_t* d_data;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::ntt_forward_gpu(d_data, n);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_output(n);
        cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        auto gpu_result = from_u64(gpu_output);
        
        // Compare
        size_t mismatch;
        bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
        EXPECT_TRUE(match) << "Forward NTT mismatch at n=" << n << ", index " << mismatch
                           << ": CPU=" << cpu_result[mismatch] << ", GPU=" << gpu_result[mismatch];
        
        cudaFree(d_data);
    }
#endif
}

// ============================================================================
// Inverse NTT Tests
// ============================================================================

TEST_F(NttCoVerifyTest, Inverse_Small) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t n : {2, 4, 8, 16}) {
        auto input = random_vector(n);
        
        // CPU computation
        auto cpu_result = input;
        NTT::inverse(cpu_result);
        
        // GPU computation
        auto gpu_input = to_u64(input);
        uint64_t* d_data;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::ntt_inverse_gpu(d_data, n);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_output(n);
        cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        auto gpu_result = from_u64(gpu_output);
        
        // Compare
        size_t mismatch;
        bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
        EXPECT_TRUE(match) << "Inverse NTT mismatch at n=" << n << ", index " << mismatch
                           << ": CPU=" << cpu_result[mismatch] << ", GPU=" << gpu_result[mismatch];
        
        cudaFree(d_data);
    }
#endif
}

TEST_F(NttCoVerifyTest, Inverse_Medium) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t n : {256, 1024, 4096}) {
        auto input = random_vector(n);
        
        // CPU computation
        auto cpu_result = input;
        NTT::inverse(cpu_result);
        
        // GPU computation
        auto gpu_input = to_u64(input);
        uint64_t* d_data;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::ntt_inverse_gpu(d_data, n);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_output(n);
        cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        auto gpu_result = from_u64(gpu_output);
        
        // Compare
        size_t mismatch;
        bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
        EXPECT_TRUE(match) << "Inverse NTT mismatch at n=" << n << ", index " << mismatch
                           << ": CPU=" << cpu_result[mismatch] << ", GPU=" << gpu_result[mismatch];
        
        cudaFree(d_data);
    }
#endif
}

// ============================================================================
// Roundtrip Tests (Forward then Inverse should give original)
// ============================================================================

TEST_F(NttCoVerifyTest, Roundtrip_GPU) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t n : {256, 1024, 4096}) {
        auto original = random_vector(n);
        
        // GPU: forward then inverse
        auto gpu_data = to_u64(original);
        uint64_t* d_data;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::ntt_forward_gpu(d_data, n);
        cudaDeviceSynchronize();
        gpu::kernels::ntt_inverse_gpu(d_data, n);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_output(n);
        cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        auto result = from_u64(gpu_output);
        
        // Compare with original
        size_t mismatch;
        bool match = compare_vectors(original, result, mismatch, compare_bfield);
        EXPECT_TRUE(match) << "Roundtrip mismatch at n=" << n << ", index " << mismatch
                           << ": original=" << original[mismatch] << ", result=" << result[mismatch];
        
        cudaFree(d_data);
    }
#endif
}

// ============================================================================
// Cross-Verification: CPU forward = GPU forward
// ============================================================================

TEST_F(NttCoVerifyTest, CrossVerify_Forward) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 4096;
    auto input = random_vector(n);
    
    // CPU
    auto cpu_result = input;
    NTT::forward(cpu_result);
    
    // GPU
    auto gpu_input = to_u64(input);
    uint64_t* d_data;
    cudaMalloc(&d_data, n * sizeof(uint64_t));
    cudaMemcpy(d_data, gpu_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::ntt_forward_gpu(d_data, n);
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> gpu_output(n);
    cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = from_u64(gpu_output);
    
    // Compare
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "CPU vs GPU forward NTT mismatch at index " << mismatch;
    
    cudaFree(d_data);
#endif
}

TEST_F(NttCoVerifyTest, CrossVerify_Inverse) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 4096;
    auto input = random_vector(n);
    
    // CPU
    auto cpu_result = input;
    NTT::inverse(cpu_result);
    
    // GPU
    auto gpu_input = to_u64(input);
    uint64_t* d_data;
    cudaMalloc(&d_data, n * sizeof(uint64_t));
    cudaMemcpy(d_data, gpu_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::ntt_inverse_gpu(d_data, n);
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> gpu_output(n);
    cudaMemcpy(gpu_output.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = from_u64(gpu_output);
    
    // Compare
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "CPU vs GPU inverse NTT mismatch at index " << mismatch;
    
    cudaFree(d_data);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(NttCoVerifyTest, Benchmark) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1 << 20;  // 1M elements
    
    auto input = random_vector(n);
    
    // CPU timing
    auto cpu_data = input;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    NTT::forward(cpu_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing (including transfer)
    auto gpu_data = to_u64(input);
    uint64_t* d_data;
    cudaMalloc(&d_data, n * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_data, gpu_data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    gpu::kernels::ntt_forward_gpu(d_data, n);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_data.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] NTT Forward (" << n << " elements):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify correctness
    auto gpu_result = from_u64(gpu_data);
    size_t mismatch;
    bool match = compare_vectors(cpu_data, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "Benchmark results mismatch at index " << mismatch;
    
    cudaFree(d_data);
#endif
}

} // namespace co_verify
} // namespace triton_vm
