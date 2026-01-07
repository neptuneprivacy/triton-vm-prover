/**
 * FRI Protocol Co-Verification Tests
 * 
 * Compares CPU FRI operations with GPU CUDA kernels.
 */

#include "co_verify_framework.hpp"
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "table/master_table.hpp"
#include "fri/fri.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/fri_kernel.cuh"
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

namespace triton_vm {
namespace co_verify {

class FriCoVerifyTest : public ::testing::Test {
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
        std::uniform_int_distribution<uint64_t> dist(1, BFieldElement::MODULUS - 1);
        return BFieldElement(dist(rng_));
    }
    
    XFieldElement random_xfield() {
        return XFieldElement(random_bfield(), random_bfield(), random_bfield());
    }
    
    std::vector<XFieldElement> random_xfield_vector(size_t n) {
        std::vector<XFieldElement> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = random_xfield();
        }
        return v;
    }
    
    std::vector<uint64_t> xfe_to_u64(const std::vector<XFieldElement>& v) {
        std::vector<uint64_t> result(v.size() * 3);
        for (size_t i = 0; i < v.size(); ++i) {
            result[i * 3 + 0] = v[i].coeff(0).value();
            result[i * 3 + 1] = v[i].coeff(1).value();
            result[i * 3 + 2] = v[i].coeff(2).value();
        }
        return result;
    }
    
    std::vector<XFieldElement> u64_to_xfe(const std::vector<uint64_t>& v) {
        std::vector<XFieldElement> result(v.size() / 3);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = XFieldElement(
                BFieldElement(v[i * 3 + 0]),
                BFieldElement(v[i * 3 + 1]),
                BFieldElement(v[i * 3 + 2])
            );
        }
        return result;
    }
    
    std::vector<uint64_t> bfe_to_u64(const std::vector<BFieldElement>& v) {
        std::vector<uint64_t> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i].value();
        }
        return result;
    }
};

// ============================================================================
// FRI Folding Tests
// ============================================================================

TEST_F(FriCoVerifyTest, Fold_Small) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 64;  // Must be power of 2
    const size_t half_n = n / 2;
    
    // Create a domain
    ArithmeticDomain domain = ArithmeticDomain::of_length(n);
    
    // Generate random codeword and challenge
    auto codeword = random_xfield_vector(n);
    XFieldElement challenge = random_xfield();
    
    // CPU: Compute domain point inverses (first half)
    std::vector<BFieldElement> domain_points;
    domain_points.reserve(n);
    BFieldElement current = domain.offset;
    for (size_t i = 0; i < n; ++i) {
        domain_points.push_back(current);
        current = current * domain.generator;
    }
    auto all_inv = BFieldElement::batch_inversion(domain_points);
    std::vector<BFieldElement> domain_inv(all_inv.begin(), all_inv.begin() + half_n);
    
    // CPU folding
    XFieldElement one = XFieldElement::one();
    XFieldElement two_inv = XFieldElement(BFieldElement(2)).inverse();
    
    std::vector<XFieldElement> cpu_folded(half_n);
    for (size_t i = 0; i < half_n; ++i) {
        XFieldElement scaled_offset_inv = challenge * domain_inv[i];
        XFieldElement left_summand = (one + scaled_offset_inv) * codeword[i];
        XFieldElement right_summand = (one - scaled_offset_inv) * codeword[half_n + i];
        cpu_folded[i] = (left_summand + right_summand) * two_inv;
    }
    
    // GPU
    auto codeword_u64 = xfe_to_u64(codeword);
    auto domain_inv_u64 = bfe_to_u64(domain_inv);
    std::vector<uint64_t> challenge_u64 = {
        challenge.coeff(0).value(),
        challenge.coeff(1).value(),
        challenge.coeff(2).value()
    };
    uint64_t two_inv_u64 = two_inv.coeff(0).value();  // XFE two_inv has only c0 set
    
    std::vector<uint64_t> gpu_folded_u64(half_n * 3);
    
    uint64_t *d_codeword, *d_challenge, *d_domain_inv, *d_folded;
    cudaMalloc(&d_codeword, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_challenge, 3 * sizeof(uint64_t));
    cudaMalloc(&d_domain_inv, half_n * sizeof(uint64_t));
    cudaMalloc(&d_folded, half_n * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_codeword, codeword_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_challenge, challenge_u64.data(), 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain_inv, domain_inv_u64.data(), half_n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::fri_fold_gpu(
        d_codeword, n, d_challenge, d_domain_inv, two_inv_u64, d_folded
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_folded_u64.data(), d_folded, half_n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_folded = u64_to_xfe(gpu_folded_u64);
    
    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < half_n; ++i) {
        if (cpu_folded[i] != gpu_folded[i]) {
            mismatches++;
            if (mismatches <= 3) {
                std::cout << "Mismatch at " << i << ": CPU=" << cpu_folded[i].to_string()
                          << ", GPU=" << gpu_folded[i].to_string() << std::endl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " fold mismatches out of " << half_n;
    
    cudaFree(d_codeword);
    cudaFree(d_challenge);
    cudaFree(d_domain_inv);
    cudaFree(d_folded);
#endif
}

TEST_F(FriCoVerifyTest, Fold_Medium) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1024;
    const size_t half_n = n / 2;
    
    ArithmeticDomain domain = ArithmeticDomain::of_length(n);
    auto codeword = random_xfield_vector(n);
    XFieldElement challenge = random_xfield();
    
    // CPU domain inverses
    std::vector<BFieldElement> domain_points;
    BFieldElement current = domain.offset;
    for (size_t i = 0; i < n; ++i) {
        domain_points.push_back(current);
        current = current * domain.generator;
    }
    auto all_inv = BFieldElement::batch_inversion(domain_points);
    std::vector<BFieldElement> domain_inv(all_inv.begin(), all_inv.begin() + half_n);
    
    // CPU folding
    XFieldElement one = XFieldElement::one();
    XFieldElement two_inv = XFieldElement(BFieldElement(2)).inverse();
    
    std::vector<XFieldElement> cpu_folded(half_n);
    for (size_t i = 0; i < half_n; ++i) {
        XFieldElement scaled_offset_inv = challenge * domain_inv[i];
        XFieldElement left_summand = (one + scaled_offset_inv) * codeword[i];
        XFieldElement right_summand = (one - scaled_offset_inv) * codeword[half_n + i];
        cpu_folded[i] = (left_summand + right_summand) * two_inv;
    }
    
    // GPU
    auto codeword_u64 = xfe_to_u64(codeword);
    auto domain_inv_u64 = bfe_to_u64(domain_inv);
    std::vector<uint64_t> challenge_u64 = {
        challenge.coeff(0).value(),
        challenge.coeff(1).value(),
        challenge.coeff(2).value()
    };
    uint64_t two_inv_u64 = two_inv.coeff(0).value();
    
    std::vector<uint64_t> gpu_folded_u64(half_n * 3);
    
    uint64_t *d_codeword, *d_challenge, *d_domain_inv, *d_folded;
    cudaMalloc(&d_codeword, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_challenge, 3 * sizeof(uint64_t));
    cudaMalloc(&d_domain_inv, half_n * sizeof(uint64_t));
    cudaMalloc(&d_folded, half_n * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_codeword, codeword_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_challenge, challenge_u64.data(), 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain_inv, domain_inv_u64.data(), half_n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::fri_fold_gpu(
        d_codeword, n, d_challenge, d_domain_inv, two_inv_u64, d_folded
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_folded_u64.data(), d_folded, half_n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_folded = u64_to_xfe(gpu_folded_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < half_n; ++i) {
        if (cpu_folded[i] != gpu_folded[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " fold mismatches";
    
    cudaFree(d_codeword);
    cudaFree(d_challenge);
    cudaFree(d_domain_inv);
    cudaFree(d_folded);
#endif
}

// ============================================================================
// Domain Inverse Tests
// ============================================================================

TEST_F(FriCoVerifyTest, DomainInverses) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 256;
    const size_t half_n = n / 2;
    
    ArithmeticDomain domain = ArithmeticDomain::of_length(n);
    
    // CPU: compute domain inverses
    std::vector<BFieldElement> domain_points;
    BFieldElement current = domain.offset;
    for (size_t i = 0; i < n; ++i) {
        domain_points.push_back(current);
        current = current * domain.generator;
    }
    auto all_inv = BFieldElement::batch_inversion(domain_points);
    std::vector<BFieldElement> cpu_inv(all_inv.begin(), all_inv.begin() + half_n);
    
    // GPU
    std::vector<uint64_t> gpu_inv_u64(half_n);
    
    uint64_t* d_domain_inv;
    cudaMalloc(&d_domain_inv, half_n * sizeof(uint64_t));
    
    gpu::kernels::compute_domain_inverses_gpu(
        domain.offset.value(),
        domain.generator.value(),
        d_domain_inv,
        half_n
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_inv_u64.data(), d_domain_inv, half_n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < half_n; ++i) {
        if (cpu_inv[i].value() != gpu_inv_u64[i]) {
            mismatches++;
            if (mismatches <= 3) {
                std::cout << "Mismatch at " << i << ": CPU=" << cpu_inv[i].value()
                          << ", GPU=" << gpu_inv_u64[i] << std::endl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " domain inverse mismatches";
    
    cudaFree(d_domain_inv);
#endif
}

// ============================================================================
// Multi-Round Folding Test
// ============================================================================

TEST_F(FriCoVerifyTest, MultiRoundFold) {
#ifdef TRITON_CUDA_ENABLED
    const size_t initial_n = 256;
    const size_t num_rounds = 4;  // 256 -> 128 -> 64 -> 32 -> 16
    
    auto codeword = random_xfield_vector(initial_n);
    std::vector<XFieldElement> challenges;
    for (size_t r = 0; r < num_rounds; ++r) {
        challenges.push_back(random_xfield());
    }
    
    // CPU multi-round folding
    std::vector<XFieldElement> cpu_current = codeword;
    for (size_t round = 0; round < num_rounds; ++round) {
        size_t n = cpu_current.size();
        size_t half_n = n / 2;
        
        ArithmeticDomain domain = ArithmeticDomain::of_length(n);
        
        std::vector<BFieldElement> domain_points;
        BFieldElement current = domain.offset;
        for (size_t i = 0; i < n; ++i) {
            domain_points.push_back(current);
            current = current * domain.generator;
        }
        auto all_inv = BFieldElement::batch_inversion(domain_points);
        std::vector<BFieldElement> domain_inv(all_inv.begin(), all_inv.begin() + half_n);
        
        XFieldElement one = XFieldElement::one();
        XFieldElement two_inv = XFieldElement(BFieldElement(2)).inverse();
        
        std::vector<XFieldElement> folded(half_n);
        for (size_t i = 0; i < half_n; ++i) {
            XFieldElement scaled_offset_inv = challenges[round] * domain_inv[i];
            XFieldElement left_summand = (one + scaled_offset_inv) * cpu_current[i];
            XFieldElement right_summand = (one - scaled_offset_inv) * cpu_current[half_n + i];
            folded[i] = (left_summand + right_summand) * two_inv;
        }
        cpu_current = folded;
    }
    
    // GPU multi-round folding
    std::vector<XFieldElement> gpu_current = codeword;
    for (size_t round = 0; round < num_rounds; ++round) {
        size_t n = gpu_current.size();
        size_t half_n = n / 2;
        
        ArithmeticDomain domain = ArithmeticDomain::of_length(n);
        
        // CPU domain inverses for this round
        std::vector<BFieldElement> domain_points;
        BFieldElement current = domain.offset;
        for (size_t i = 0; i < n; ++i) {
            domain_points.push_back(current);
            current = current * domain.generator;
        }
        auto all_inv = BFieldElement::batch_inversion(domain_points);
        std::vector<BFieldElement> domain_inv(all_inv.begin(), all_inv.begin() + half_n);
        
        auto codeword_u64 = xfe_to_u64(gpu_current);
        auto domain_inv_u64 = bfe_to_u64(domain_inv);
        std::vector<uint64_t> challenge_u64 = {
            challenges[round].coeff(0).value(),
            challenges[round].coeff(1).value(),
            challenges[round].coeff(2).value()
        };
        XFieldElement two_inv = XFieldElement(BFieldElement(2)).inverse();
        uint64_t two_inv_u64 = two_inv.coeff(0).value();
        
        uint64_t *d_codeword, *d_challenge, *d_domain_inv, *d_folded;
        cudaMalloc(&d_codeword, n * 3 * sizeof(uint64_t));
        cudaMalloc(&d_challenge, 3 * sizeof(uint64_t));
        cudaMalloc(&d_domain_inv, half_n * sizeof(uint64_t));
        cudaMalloc(&d_folded, half_n * 3 * sizeof(uint64_t));
        
        cudaMemcpy(d_codeword, codeword_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_challenge, challenge_u64.data(), 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_domain_inv, domain_inv_u64.data(), half_n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::fri_fold_gpu(
            d_codeword, n, d_challenge, d_domain_inv, two_inv_u64, d_folded
        );
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_folded_u64(half_n * 3);
        cudaMemcpy(gpu_folded_u64.data(), d_folded, half_n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        gpu_current = u64_to_xfe(gpu_folded_u64);
        
        cudaFree(d_codeword);
        cudaFree(d_challenge);
        cudaFree(d_domain_inv);
        cudaFree(d_folded);
    }
    
    // Compare final results
    ASSERT_EQ(cpu_current.size(), gpu_current.size());
    
    size_t mismatches = 0;
    for (size_t i = 0; i < cpu_current.size(); ++i) {
        if (cpu_current[i] != gpu_current[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " multi-round fold mismatches out of " << cpu_current.size();
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(FriCoVerifyTest, Benchmark) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1 << 16;  // 64K
    const size_t half_n = n / 2;
    
    auto codeword = random_xfield_vector(n);
    XFieldElement challenge = random_xfield();
    
    ArithmeticDomain domain = ArithmeticDomain::of_length(n);
    
    // CPU domain inverses
    std::vector<BFieldElement> domain_points;
    BFieldElement current = domain.offset;
    for (size_t i = 0; i < n; ++i) {
        domain_points.push_back(current);
        current = current * domain.generator;
    }
    auto all_inv = BFieldElement::batch_inversion(domain_points);
    std::vector<BFieldElement> domain_inv(all_inv.begin(), all_inv.begin() + half_n);
    
    // CPU timing
    XFieldElement one = XFieldElement::one();
    XFieldElement two_inv = XFieldElement(BFieldElement(2)).inverse();
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<XFieldElement> cpu_folded(half_n);
    for (size_t i = 0; i < half_n; ++i) {
        XFieldElement scaled_offset_inv = challenge * domain_inv[i];
        XFieldElement left_summand = (one + scaled_offset_inv) * codeword[i];
        XFieldElement right_summand = (one - scaled_offset_inv) * codeword[half_n + i];
        cpu_folded[i] = (left_summand + right_summand) * two_inv;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    auto codeword_u64 = xfe_to_u64(codeword);
    auto domain_inv_u64 = bfe_to_u64(domain_inv);
    std::vector<uint64_t> challenge_u64 = {
        challenge.coeff(0).value(),
        challenge.coeff(1).value(),
        challenge.coeff(2).value()
    };
    uint64_t two_inv_u64 = two_inv.coeff(0).value();
    
    uint64_t *d_codeword, *d_challenge, *d_domain_inv, *d_folded;
    cudaMalloc(&d_codeword, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_challenge, 3 * sizeof(uint64_t));
    cudaMalloc(&d_domain_inv, half_n * sizeof(uint64_t));
    cudaMalloc(&d_folded, half_n * 3 * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_codeword, codeword_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_challenge, challenge_u64.data(), 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain_inv, domain_inv_u64.data(), half_n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::fri_fold_gpu(
        d_codeword, n, d_challenge, d_domain_inv, two_inv_u64, d_folded
    );
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> gpu_folded_u64(half_n * 3);
    cudaMemcpy(gpu_folded_u64.data(), d_folded, half_n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] FRI Fold (" << n << " -> " << half_n << " XFEs):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify correctness
    auto gpu_folded = u64_to_xfe(gpu_folded_u64);
    size_t mismatches = 0;
    for (size_t i = 0; i < half_n; ++i) {
        if (cpu_folded[i] != gpu_folded[i]) mismatches++;
    }
    EXPECT_EQ(mismatches, 0) << "Benchmark has mismatches";
    
    cudaFree(d_codeword);
    cudaFree(d_challenge);
    cudaFree(d_domain_inv);
    cudaFree(d_folded);
#endif
}

} // namespace co_verify
} // namespace triton_vm

