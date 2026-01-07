/**
 * Fiat-Shamir Co-Verification Tests
 * 
 * Compares CPU ProofStream operations with GPU Fiat-Shamir kernels.
 */

#include "co_verify_framework.hpp"
#include "proof_stream/proof_stream.hpp"
#include "hash/tip5.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/fiat_shamir_kernel.cuh"
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

class FiatShamirCoVerifyTest : public ::testing::Test {
protected:
#ifdef TRITON_CUDA_ENABLED
    uint16_t* d_sbox_table_ = nullptr;
    uint64_t* d_mds_matrix_ = nullptr;
    uint64_t* d_round_constants_ = nullptr;
    uint64_t* d_state_ = nullptr;
#endif

    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "No CUDA device available";
        }
        
        // Allocate and upload Tip5 tables
        cudaMalloc(&d_sbox_table_, 65536 * sizeof(uint16_t));
        cudaMalloc(&d_mds_matrix_, 16 * sizeof(uint64_t));
        cudaMalloc(&d_round_constants_, 8 * 16 * sizeof(uint64_t));
        cudaMalloc(&d_state_, 16 * sizeof(uint64_t));
        
        // Get Tip5 tables from CPU implementation
        std::vector<uint16_t> sbox_table(65536);
        for (int i = 0; i < 65536; ++i) {
            sbox_table[i] = Tip5::LOOKUP_TABLE[i % 256];
            // Expand 8-bit table to 16-bit as expected by GPU
        }
        
        // Actually, we need the full 16-bit S-box table
        // The GPU tip5_kernel generates this internally
        // Let's use the proper initialization
        gpu::kernels::tip5_init_tables(
            d_sbox_table_, d_mds_matrix_, d_round_constants_
        );
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
    
    void TearDown() override {
#ifdef TRITON_CUDA_ENABLED
        if (d_sbox_table_) cudaFree(d_sbox_table_);
        if (d_mds_matrix_) cudaFree(d_mds_matrix_);
        if (d_round_constants_) cudaFree(d_round_constants_);
        if (d_state_) cudaFree(d_state_);
#endif
    }
    
    std::mt19937_64 rng_{42};
    
    BFieldElement random_bfield() {
        std::uniform_int_distribution<uint64_t> dist(0, BFieldElement::MODULUS - 1);
        return BFieldElement(dist(rng_));
    }
    
    std::vector<BFieldElement> random_vector(size_t n) {
        std::vector<BFieldElement> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = random_bfield();
        }
        return v;
    }
    
    std::vector<uint64_t> to_u64(const std::vector<BFieldElement>& v) {
        std::vector<uint64_t> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i].value();
        }
        return result;
    }
};

// ============================================================================
// Absorb Tests
// ============================================================================

TEST_F(FiatShamirCoVerifyTest, Absorb_Small) {
#ifdef TRITON_CUDA_ENABLED
    // Test absorbing a small amount of data
    auto data = random_vector(5);
    
    // CPU: Use ProofStream
    ProofStream cpu_stream;
    cpu_stream.alter_fiat_shamir_state_with(data);
    auto cpu_sponge = cpu_stream.sponge();
    
    // GPU
    gpu::kernels::fs_init_sponge_gpu(d_state_);
    cudaDeviceSynchronize();
    
    auto data_u64 = to_u64(data);
    gpu::kernels::fs_absorb_gpu(
        d_state_, data_u64.data(), data_u64.size(),
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    cudaDeviceSynchronize();
    
    // Download GPU state
    std::vector<uint64_t> gpu_state(16);
    cudaMemcpy(gpu_state.data(), d_state_, 16 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Compare
    bool match = true;
    for (int i = 0; i < 16; ++i) {
        if (cpu_sponge.state[i].value() != gpu_state[i]) {
            match = false;
            std::cout << "State mismatch at " << i 
                      << ": CPU=" << cpu_sponge.state[i].value()
                      << ", GPU=" << gpu_state[i] << std::endl;
        }
    }
    EXPECT_TRUE(match) << "Absorb state mismatch";
#endif
}

TEST_F(FiatShamirCoVerifyTest, Absorb_Medium) {
#ifdef TRITON_CUDA_ENABLED
    // Test absorbing data that spans multiple blocks
    auto data = random_vector(25);  // More than 2 rate blocks
    
    ProofStream cpu_stream;
    cpu_stream.alter_fiat_shamir_state_with(data);
    auto cpu_sponge = cpu_stream.sponge();
    
    gpu::kernels::fs_init_sponge_gpu(d_state_);
    auto data_u64 = to_u64(data);
    gpu::kernels::fs_absorb_gpu(
        d_state_, data_u64.data(), data_u64.size(),
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> gpu_state(16);
    cudaMemcpy(gpu_state.data(), d_state_, 16 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    bool match = true;
    for (int i = 0; i < 16; ++i) {
        if (cpu_sponge.state[i].value() != gpu_state[i]) {
            match = false;
        }
    }
    EXPECT_TRUE(match) << "Absorb medium data state mismatch";
#endif
}

// ============================================================================
// Sample Scalars Tests
// ============================================================================

TEST_F(FiatShamirCoVerifyTest, SampleScalars_Small) {
#ifdef TRITON_CUDA_ENABLED
    auto data = random_vector(10);
    size_t num_scalars = 5;
    
    // CPU
    ProofStream cpu_stream;
    cpu_stream.alter_fiat_shamir_state_with(data);
    auto cpu_scalars = cpu_stream.sample_scalars(num_scalars);
    
    // GPU
    gpu::kernels::fs_init_sponge_gpu(d_state_);
    auto data_u64 = to_u64(data);
    gpu::kernels::fs_absorb_gpu(
        d_state_, data_u64.data(), data_u64.size(),
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    
    uint64_t* d_output;
    cudaMalloc(&d_output, num_scalars * 3 * sizeof(uint64_t));
    
    gpu::kernels::fs_sample_scalars_gpu(
        d_state_, d_output, num_scalars,
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> gpu_output(num_scalars * 3);
    cudaMemcpy(gpu_output.data(), d_output, num_scalars * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Compare XFieldElements
    bool match = true;
    for (size_t i = 0; i < num_scalars; ++i) {
        XFieldElement cpu_x = cpu_scalars[i];
        XFieldElement gpu_x(
            BFieldElement(gpu_output[i * 3 + 0]),
            BFieldElement(gpu_output[i * 3 + 1]),
            BFieldElement(gpu_output[i * 3 + 2])
        );
        
        if (cpu_x != gpu_x) {
            match = false;
            std::cout << "Scalar mismatch at " << i 
                      << ": CPU=" << cpu_x.to_string()
                      << ", GPU=" << gpu_x.to_string() << std::endl;
        }
    }
    EXPECT_TRUE(match) << "Sample scalars mismatch";
    
    cudaFree(d_output);
#endif
}

TEST_F(FiatShamirCoVerifyTest, SampleScalars_Many) {
#ifdef TRITON_CUDA_ENABLED
    auto data = random_vector(20);
    size_t num_scalars = 59;  // Typical challenge count
    
    // CPU
    ProofStream cpu_stream;
    cpu_stream.alter_fiat_shamir_state_with(data);
    auto cpu_scalars = cpu_stream.sample_scalars(num_scalars);
    
    // GPU
    gpu::kernels::fs_init_sponge_gpu(d_state_);
    auto data_u64 = to_u64(data);
    gpu::kernels::fs_absorb_gpu(
        d_state_, data_u64.data(), data_u64.size(),
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    
    uint64_t* d_output;
    cudaMalloc(&d_output, num_scalars * 3 * sizeof(uint64_t));
    
    gpu::kernels::fs_sample_scalars_gpu(
        d_state_, d_output, num_scalars,
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> gpu_output(num_scalars * 3);
    cudaMemcpy(gpu_output.data(), d_output, num_scalars * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < num_scalars; ++i) {
        XFieldElement cpu_x = cpu_scalars[i];
        XFieldElement gpu_x(
            BFieldElement(gpu_output[i * 3 + 0]),
            BFieldElement(gpu_output[i * 3 + 1]),
            BFieldElement(gpu_output[i * 3 + 2])
        );
        
        if (cpu_x != gpu_x) {
            mismatches++;
            if (mismatches <= 3) {
                std::cout << "Scalar mismatch at " << i << std::endl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " scalar mismatches out of " << num_scalars;
    
    cudaFree(d_output);
#endif
}

// ============================================================================
// Sample Indices Tests
// ============================================================================

TEST_F(FiatShamirCoVerifyTest, SampleIndices) {
#ifdef TRITON_CUDA_ENABLED
    auto data = random_vector(15);
    size_t upper_bound = 4096;
    size_t num_indices = 160;  // Typical FRI query count
    
    // CPU
    ProofStream cpu_stream;
    cpu_stream.alter_fiat_shamir_state_with(data);
    auto cpu_indices = cpu_stream.sample_indices(upper_bound, num_indices);
    
    // GPU
    gpu::kernels::fs_init_sponge_gpu(d_state_);
    auto data_u64 = to_u64(data);
    gpu::kernels::fs_absorb_gpu(
        d_state_, data_u64.data(), data_u64.size(),
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    
    size_t* d_output;
    cudaMalloc(&d_output, num_indices * sizeof(size_t));
    
    gpu::kernels::fs_sample_indices_gpu(
        d_state_, d_output, upper_bound, num_indices,
        d_sbox_table_, d_mds_matrix_, d_round_constants_
    );
    cudaDeviceSynchronize();
    
    std::vector<size_t> gpu_indices(num_indices);
    cudaMemcpy(gpu_indices.data(), d_output, num_indices * sizeof(size_t), cudaMemcpyDeviceToHost);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < num_indices; ++i) {
        if (cpu_indices[i] != gpu_indices[i]) {
            mismatches++;
            if (mismatches <= 3) {
                std::cout << "Index mismatch at " << i 
                          << ": CPU=" << cpu_indices[i]
                          << ", GPU=" << gpu_indices[i] << std::endl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " index mismatches out of " << num_indices;
    
    cudaFree(d_output);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(FiatShamirCoVerifyTest, Benchmark) {
#ifdef TRITON_CUDA_ENABLED
    auto data = random_vector(100);
    size_t num_scalars = 200;
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        ProofStream cpu_stream;
        cpu_stream.alter_fiat_shamir_state_with(data);
        auto cpu_scalars = cpu_stream.sample_scalars(num_scalars);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    uint64_t* d_output;
    cudaMalloc(&d_output, num_scalars * 3 * sizeof(uint64_t));
    auto data_u64 = to_u64(data);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        gpu::kernels::fs_init_sponge_gpu(d_state_);
        gpu::kernels::fs_absorb_gpu(
            d_state_, data_u64.data(), data_u64.size(),
            d_sbox_table_, d_mds_matrix_, d_round_constants_
        );
        gpu::kernels::fs_sample_scalars_gpu(
            d_state_, d_output, num_scalars,
            d_sbox_table_, d_mds_matrix_, d_round_constants_
        );
    }
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] Fiat-Shamir (absorb " << data.size() 
              << " elements + sample " << num_scalars << " scalars) x 100:\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    cudaFree(d_output);
    SUCCEED();
#endif
}

} // namespace co_verify
} // namespace triton_vm

