/**
 * Tip5 Hash Co-Verification Tests
 * 
 * Compares CPU Tip5 implementation with GPU CUDA kernels
 * to ensure they produce identical results.
 */

#include "co_verify_framework.hpp"
#include "hash/tip5.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/tip5_kernel.cuh"
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

namespace triton_vm {
namespace co_verify {

class Tip5CoVerifyTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "No CUDA device available";
        }
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
    
    std::mt19937_64 rng_{42};
    
    BFieldElement random_bfield() {
        std::uniform_int_distribution<uint64_t> dist(0, BFieldElement::MODULUS - 1);
        return BFieldElement(dist(rng_));
    }
    
    Digest random_digest() {
        return Digest(
            random_bfield(), random_bfield(), random_bfield(),
            random_bfield(), random_bfield()
        );
    }
    
    // Convert Digest to flat array
    std::vector<uint64_t> to_flat(const Digest& d) {
        return {d[0].value(), d[1].value(), d[2].value(), d[3].value(), d[4].value()};
    }
    
    // Convert flat array to Digest
    Digest from_flat(const uint64_t* data) {
        return Digest(
            BFieldElement(data[0]), BFieldElement(data[1]), BFieldElement(data[2]),
            BFieldElement(data[3]), BFieldElement(data[4])
        );
    }
    
    static bool compare_digest(const Digest& a, const Digest& b) {
        for (size_t i = 0; i < 5; ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
};

// ============================================================================
// Hash Pair Tests
// ============================================================================

TEST_F(Tip5CoVerifyTest, HashPair_Single) {
#ifdef TRITON_CUDA_ENABLED
    Digest left = random_digest();
    Digest right = random_digest();
    
    // CPU
    Digest cpu_result = Tip5::hash_pair(left, right);
    
    // GPU
    auto left_flat = to_flat(left);
    auto right_flat = to_flat(right);
    std::vector<uint64_t> output_flat(5);
    
    uint64_t *d_left, *d_right, *d_output;
    cudaMalloc(&d_left, 5 * sizeof(uint64_t));
    cudaMalloc(&d_right, 5 * sizeof(uint64_t));
    cudaMalloc(&d_output, 5 * sizeof(uint64_t));
    
    cudaMemcpy(d_left, left_flat.data(), 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right_flat.data(), 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::hash_pairs_gpu(d_left, d_right, d_output, 1);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_flat.data(), d_output, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    Digest gpu_result = from_flat(output_flat.data());
    
    EXPECT_TRUE(compare_digest(cpu_result, gpu_result))
        << "Hash pair mismatch\n"
        << "  Left:  " << left << "\n"
        << "  Right: " << right << "\n"
        << "  CPU:   " << cpu_result << "\n"
        << "  GPU:   " << gpu_result;
    
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_output);
#endif
}

TEST_F(Tip5CoVerifyTest, HashPair_Batch) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 1000;
    
    std::vector<Digest> left(N), right(N), cpu_results(N);
    for (size_t i = 0; i < N; ++i) {
        left[i] = random_digest();
        right[i] = random_digest();
        cpu_results[i] = Tip5::hash_pair(left[i], right[i]);
    }
    
    // Flatten
    std::vector<uint64_t> left_flat(N * 5), right_flat(N * 5), output_flat(N * 5);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            left_flat[i * 5 + j] = left[i][j].value();
            right_flat[i * 5 + j] = right[i][j].value();
        }
    }
    
    uint64_t *d_left, *d_right, *d_output;
    cudaMalloc(&d_left, N * 5 * sizeof(uint64_t));
    cudaMalloc(&d_right, N * 5 * sizeof(uint64_t));
    cudaMalloc(&d_output, N * 5 * sizeof(uint64_t));
    
    cudaMemcpy(d_left, left_flat.data(), N * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right_flat.data(), N * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::hash_pairs_gpu(d_left, d_right, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_flat.data(), d_output, N * 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Compare
    for (size_t i = 0; i < N; ++i) {
        Digest gpu_result = from_flat(output_flat.data() + i * 5);
        EXPECT_TRUE(compare_digest(cpu_results[i], gpu_result))
            << "Mismatch at index " << i;
    }
    
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_output);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(Tip5CoVerifyTest, Benchmark_HashPairs) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 1 << 16;  // 64K pairs
    
    std::vector<Digest> left(N), right(N);
    for (size_t i = 0; i < N; ++i) {
        left[i] = random_digest();
        right[i] = random_digest();
    }
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<Digest> cpu_results(N);
    for (size_t i = 0; i < N; ++i) {
        cpu_results[i] = Tip5::hash_pair(left[i], right[i]);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    std::vector<uint64_t> left_flat(N * 5), right_flat(N * 5), output_flat(N * 5);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            left_flat[i * 5 + j] = left[i][j].value();
            right_flat[i * 5 + j] = right[i][j].value();
        }
    }
    
    uint64_t *d_left, *d_right, *d_output;
    cudaMalloc(&d_left, N * 5 * sizeof(uint64_t));
    cudaMalloc(&d_right, N * 5 * sizeof(uint64_t));
    cudaMalloc(&d_output, N * 5 * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_left, left_flat.data(), N * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right_flat.data(), N * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    gpu::kernels::hash_pairs_gpu(d_left, d_right, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(output_flat.data(), d_output, N * 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] Tip5 Hash Pairs (" << N << " pairs):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify first few results
    for (size_t i = 0; i < 10; ++i) {
        Digest gpu_result = from_flat(output_flat.data() + i * 5);
        EXPECT_TRUE(compare_digest(cpu_results[i], gpu_result))
            << "Benchmark mismatch at index " << i;
    }
    
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_output);
#endif
}

} // namespace co_verify
} // namespace triton_vm
