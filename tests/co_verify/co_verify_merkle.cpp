/**
 * Merkle Tree Co-Verification Tests
 * 
 * Compares CPU Merkle tree implementation with GPU CUDA kernels
 * to ensure they produce identical results.
 */

#include "co_verify_framework.hpp"
#include "merkle/merkle_tree.hpp"
#include "hash/tip5.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/merkle_kernel.cuh"
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

class MerkleCoVerifyTest : public ::testing::Test {
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
    
    // Generate random Digest
    Digest random_digest() {
        std::uniform_int_distribution<uint64_t> dist(0, BFieldElement::MODULUS - 1);
        return Digest(
            BFieldElement(dist(rng_)),
            BFieldElement(dist(rng_)),
            BFieldElement(dist(rng_)),
            BFieldElement(dist(rng_)),
            BFieldElement(dist(rng_))
        );
    }
    
    // Generate random leaves
    std::vector<Digest> random_leaves(size_t n) {
        std::vector<Digest> leaves(n);
        for (size_t i = 0; i < n; ++i) {
            leaves[i] = random_digest();
        }
        return leaves;
    }
    
    // Convert Digest vector to flat uint64_t array
    std::vector<uint64_t> to_flat(const std::vector<Digest>& digests) {
        std::vector<uint64_t> result;
        result.reserve(digests.size() * 5);
        for (const auto& d : digests) {
            for (size_t i = 0; i < 5; ++i) {
                result.push_back(d[i].value());
            }
        }
        return result;
    }
    
    // Convert flat uint64_t array to Digest
    Digest from_flat(const uint64_t* data) {
        return Digest(
            BFieldElement(data[0]),
            BFieldElement(data[1]),
            BFieldElement(data[2]),
            BFieldElement(data[3]),
            BFieldElement(data[4])
        );
    }
    
    // Compare digests
    static bool compare_digest(const Digest& a, const Digest& b) {
        for (size_t i = 0; i < 5; ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
};

// ============================================================================
// Tip5 Hash Pair Tests (foundation for Merkle)
// ============================================================================

TEST_F(MerkleCoVerifyTest, HashPair_Single) {
#ifdef TRITON_CUDA_ENABLED
    Digest left = random_digest();
    Digest right = random_digest();
    
    // CPU hash
    Digest cpu_result = Tip5::hash_pair(left, right);
    
    // GPU hash
    std::vector<uint64_t> left_flat = to_flat({left});
    std::vector<uint64_t> right_flat = to_flat({right});
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
        << "  CPU: " << cpu_result << "\n"
        << "  GPU: " << gpu_result;
    
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_output);
#endif
}

TEST_F(MerkleCoVerifyTest, HashPair_Batch) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 1000;
    
    std::vector<Digest> left(N), right(N), cpu_results(N);
    for (size_t i = 0; i < N; ++i) {
        left[i] = random_digest();
        right[i] = random_digest();
        cpu_results[i] = Tip5::hash_pair(left[i], right[i]);
    }
    
    // GPU
    auto left_flat = to_flat(left);
    auto right_flat = to_flat(right);
    std::vector<uint64_t> output_flat(N * 5);
    
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
            << "Hash pair mismatch at index " << i;
    }
    
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_output);
#endif
}

// ============================================================================
// Merkle Root Tests
// ============================================================================

TEST_F(MerkleCoVerifyTest, Root_Small) {
#ifdef TRITON_CUDA_ENABLED
    // Test small trees: 2, 4, 8, 16 leaves
    for (size_t num_leaves : {2, 4, 8, 16}) {
        auto leaves = random_leaves(num_leaves);
        
        // CPU Merkle tree
        MerkleTree cpu_tree(leaves);
        Digest cpu_root = cpu_tree.root();
        
        // GPU root computation
        auto leaves_flat = to_flat(leaves);
        std::vector<uint64_t> root_flat(5);
        
        uint64_t *d_leaves, *d_root;
        cudaMalloc(&d_leaves, num_leaves * 5 * sizeof(uint64_t));
        cudaMalloc(&d_root, 5 * sizeof(uint64_t));
        
        cudaMemcpy(d_leaves, leaves_flat.data(), num_leaves * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::merkle_root_gpu(d_leaves, d_root, num_leaves);
        cudaDeviceSynchronize();
        
        cudaMemcpy(root_flat.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        Digest gpu_root = from_flat(root_flat.data());
        
        EXPECT_TRUE(compare_digest(cpu_root, gpu_root))
            << "Merkle root mismatch for " << num_leaves << " leaves\n"
            << "  CPU: " << cpu_root << "\n"
            << "  GPU: " << gpu_root;
        
        cudaFree(d_leaves);
        cudaFree(d_root);
    }
#endif
}

TEST_F(MerkleCoVerifyTest, Root_Medium) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t num_leaves : {64, 256, 1024}) {
        auto leaves = random_leaves(num_leaves);
        
        MerkleTree cpu_tree(leaves);
        Digest cpu_root = cpu_tree.root();
        
        auto leaves_flat = to_flat(leaves);
        std::vector<uint64_t> root_flat(5);
        
        uint64_t *d_leaves, *d_root;
        cudaMalloc(&d_leaves, num_leaves * 5 * sizeof(uint64_t));
        cudaMalloc(&d_root, 5 * sizeof(uint64_t));
        
        cudaMemcpy(d_leaves, leaves_flat.data(), num_leaves * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::merkle_root_gpu(d_leaves, d_root, num_leaves);
        cudaDeviceSynchronize();
        
        cudaMemcpy(root_flat.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        Digest gpu_root = from_flat(root_flat.data());
        
        EXPECT_TRUE(compare_digest(cpu_root, gpu_root))
            << "Merkle root mismatch for " << num_leaves << " leaves";
        
        cudaFree(d_leaves);
        cudaFree(d_root);
    }
#endif
}

// ============================================================================
// Deterministic Tests
// ============================================================================

TEST_F(MerkleCoVerifyTest, Deterministic) {
#ifdef TRITON_CUDA_ENABLED
    // Same input should always produce same output
    auto leaves = random_leaves(16);
    
    std::vector<uint64_t> root1(5), root2(5);
    auto leaves_flat = to_flat(leaves);
    
    uint64_t *d_leaves, *d_root;
    cudaMalloc(&d_leaves, 16 * 5 * sizeof(uint64_t));
    cudaMalloc(&d_root, 5 * sizeof(uint64_t));
    
    cudaMemcpy(d_leaves, leaves_flat.data(), 16 * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Run twice
    gpu::kernels::merkle_root_gpu(d_leaves, d_root, 16);
    cudaDeviceSynchronize();
    cudaMemcpy(root1.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    gpu::kernels::merkle_root_gpu(d_leaves, d_root, 16);
    cudaDeviceSynchronize();
    cudaMemcpy(root2.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(root1[i], root2[i]) << "Non-deterministic at element " << i;
    }
    
    cudaFree(d_leaves);
    cudaFree(d_root);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(MerkleCoVerifyTest, Benchmark) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_leaves = 1 << 14;  // 16K leaves
    
    auto leaves = random_leaves(num_leaves);
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    MerkleTree cpu_tree(leaves);
    Digest cpu_root = cpu_tree.root();
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    auto leaves_flat = to_flat(leaves);
    std::vector<uint64_t> root_flat(5);
    
    uint64_t *d_leaves, *d_root;
    cudaMalloc(&d_leaves, num_leaves * 5 * sizeof(uint64_t));
    cudaMalloc(&d_root, 5 * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_leaves, leaves_flat.data(), num_leaves * 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    gpu::kernels::merkle_root_gpu(d_leaves, d_root, num_leaves);
    cudaDeviceSynchronize();
    cudaMemcpy(root_flat.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] Merkle Root (" << num_leaves << " leaves):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify correctness
    Digest gpu_root = from_flat(root_flat.data());
    EXPECT_TRUE(compare_digest(cpu_root, gpu_root)) << "Benchmark root mismatch";
    
    cudaFree(d_leaves);
    cudaFree(d_root);
#endif
}

} // namespace co_verify
} // namespace triton_vm
