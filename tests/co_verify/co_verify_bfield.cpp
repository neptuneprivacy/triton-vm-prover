/**
 * BFieldElement Co-Verification Tests
 * 
 * Verifies that CUDA BField arithmetic produces identical results to CPU.
 * This is the foundation test - all other GPU operations depend on correct field arithmetic.
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "types/b_field_element.hpp"

#ifdef TRITON_CUDA_ENABLED
#include <cuda_runtime.h>
#include "gpu/kernels/bfield_kernel.cuh"
#endif

namespace triton_vm {
namespace co_verify {

// ============================================================================
// Test Utilities
// ============================================================================

class BFieldCoVerify : public ::testing::Test {
protected:
    void SetUp() override {
#ifndef TRITON_CUDA_ENABLED
        GTEST_SKIP() << "CUDA not enabled";
#else
        // Check CUDA device
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
#endif
    }
    
    // Generate random field elements
    std::vector<uint64_t> random_field_elements(size_t n, uint64_t seed = 42) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, BFieldElement::MODULUS - 1);
        
        std::vector<uint64_t> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = dist(rng);
        }
        return result;
    }
    
    // Generate non-zero random field elements (for division/inverse tests)
    std::vector<uint64_t> random_nonzero_field_elements(size_t n, uint64_t seed = 42) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(1, BFieldElement::MODULUS - 1);
        
        std::vector<uint64_t> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = dist(rng);
        }
        return result;
    }
    
    // CPU reference: batch addition
    std::vector<uint64_t> cpu_add(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        std::vector<uint64_t> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = (BFieldElement(a[i]) + BFieldElement(b[i])).value();
        }
        return result;
    }
    
    // CPU reference: batch subtraction
    std::vector<uint64_t> cpu_sub(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        std::vector<uint64_t> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = (BFieldElement(a[i]) - BFieldElement(b[i])).value();
        }
        return result;
    }
    
    // CPU reference: batch multiplication
    std::vector<uint64_t> cpu_mul(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        std::vector<uint64_t> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = (BFieldElement(a[i]) * BFieldElement(b[i])).value();
        }
        return result;
    }
    
    // CPU reference: batch negation
    std::vector<uint64_t> cpu_neg(const std::vector<uint64_t>& a) {
        std::vector<uint64_t> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = (-BFieldElement(a[i])).value();
        }
        return result;
    }
    
    // CPU reference: batch inverse
    std::vector<uint64_t> cpu_inv(const std::vector<uint64_t>& a) {
        std::vector<uint64_t> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = BFieldElement(a[i]).inverse().value();
        }
        return result;
    }
    
    // CPU reference: batch power
    std::vector<uint64_t> cpu_pow(const std::vector<uint64_t>& a, uint64_t exp) {
        std::vector<uint64_t> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = BFieldElement(a[i]).pow(exp).value();
        }
        return result;
    }
    
#ifdef TRITON_CUDA_ENABLED
    // GPU: batch addition
    std::vector<uint64_t> gpu_add(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        size_t n = a.size();
        std::vector<uint64_t> result(n);
        
        uint64_t *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, n * sizeof(uint64_t));
        cudaMalloc(&d_b, n * sizeof(uint64_t));
        cudaMalloc(&d_out, n * sizeof(uint64_t));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::bfield_add_gpu(d_a, d_b, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(result.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);
        
        return result;
    }
    
    // GPU: batch subtraction
    std::vector<uint64_t> gpu_sub(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        size_t n = a.size();
        std::vector<uint64_t> result(n);
        
        uint64_t *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, n * sizeof(uint64_t));
        cudaMalloc(&d_b, n * sizeof(uint64_t));
        cudaMalloc(&d_out, n * sizeof(uint64_t));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::bfield_sub_gpu(d_a, d_b, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(result.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);
        
        return result;
    }
    
    // GPU: batch multiplication
    std::vector<uint64_t> gpu_mul(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        size_t n = a.size();
        std::vector<uint64_t> result(n);
        
        uint64_t *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, n * sizeof(uint64_t));
        cudaMalloc(&d_b, n * sizeof(uint64_t));
        cudaMalloc(&d_out, n * sizeof(uint64_t));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::bfield_mul_gpu(d_a, d_b, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(result.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);
        
        return result;
    }
    
    // GPU: batch negation
    std::vector<uint64_t> gpu_neg(const std::vector<uint64_t>& a) {
        size_t n = a.size();
        std::vector<uint64_t> result(n);
        
        uint64_t *d_a, *d_out;
        cudaMalloc(&d_a, n * sizeof(uint64_t));
        cudaMalloc(&d_out, n * sizeof(uint64_t));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::bfield_neg_gpu(d_a, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(result.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_out);
        
        return result;
    }
    
    // GPU: batch inverse
    std::vector<uint64_t> gpu_inv(const std::vector<uint64_t>& a) {
        size_t n = a.size();
        std::vector<uint64_t> result(n);
        
        uint64_t *d_a, *d_out;
        cudaMalloc(&d_a, n * sizeof(uint64_t));
        cudaMalloc(&d_out, n * sizeof(uint64_t));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::bfield_inv_gpu(d_a, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(result.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_out);
        
        return result;
    }
    
    // GPU: batch power
    std::vector<uint64_t> gpu_pow(const std::vector<uint64_t>& a, uint64_t exp) {
        size_t n = a.size();
        std::vector<uint64_t> result(n);
        
        uint64_t *d_a, *d_out;
        cudaMalloc(&d_a, n * sizeof(uint64_t));
        cudaMalloc(&d_out, n * sizeof(uint64_t));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::bfield_pow_gpu(d_a, exp, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(result.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_out);
        
        return result;
    }
#endif
    
    // Compare results and report mismatches
    bool compare_results(
        const std::vector<uint64_t>& cpu_result,
        const std::vector<uint64_t>& gpu_result,
        const std::string& op_name,
        size_t& first_mismatch
    ) {
        if (cpu_result.size() != gpu_result.size()) {
            std::cout << "[" << op_name << "] Size mismatch: CPU=" << cpu_result.size()
                      << " GPU=" << gpu_result.size() << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < cpu_result.size(); ++i) {
            if (cpu_result[i] != gpu_result[i]) {
                first_mismatch = i;
                std::cout << "[" << op_name << "] Mismatch at index " << i
                          << ": CPU=" << cpu_result[i]
                          << " GPU=" << gpu_result[i] << std::endl;
                return false;
            }
        }
        
        return true;
    }
    
    // Test sizes to verify
    std::vector<size_t> test_sizes = {1, 16, 256, 1024, 4096, 65536};
};

// ============================================================================
// Addition Tests
// ============================================================================

TEST_F(BFieldCoVerify, Addition_Random) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t n : test_sizes) {
        auto a = random_field_elements(n, 42);
        auto b = random_field_elements(n, 123);
        
        auto cpu_result = cpu_add(a, b);
        auto gpu_result = gpu_add(a, b);
        
        size_t mismatch;
        EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Add", mismatch))
            << "Failed at size " << n << ", index " << mismatch;
    }
#endif
}

TEST_F(BFieldCoVerify, Addition_EdgeCases) {
#ifdef TRITON_CUDA_ENABLED
    // Test edge cases: 0, 1, p-1, overflow scenarios
    std::vector<uint64_t> a = {
        0,                              // 0 + x
        1,                              // 1 + x
        BFieldElement::MODULUS - 1,     // (p-1) + 1 = 0
        BFieldElement::MODULUS - 1,     // (p-1) + 2 = 1
        BFieldElement::MODULUS / 2,     // Half modulus
    };
    std::vector<uint64_t> b = {
        0,
        BFieldElement::MODULUS - 1,
        1,
        2,
        BFieldElement::MODULUS / 2 + 1,
    };
    
    auto cpu_result = cpu_add(a, b);
    auto gpu_result = gpu_add(a, b);
    
    size_t mismatch;
    EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Add_Edge", mismatch));
#endif
}

// ============================================================================
// Subtraction Tests
// ============================================================================

TEST_F(BFieldCoVerify, Subtraction_Random) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t n : test_sizes) {
        auto a = random_field_elements(n, 42);
        auto b = random_field_elements(n, 123);
        
        auto cpu_result = cpu_sub(a, b);
        auto gpu_result = gpu_sub(a, b);
        
        size_t mismatch;
        EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Sub", mismatch))
            << "Failed at size " << n;
    }
#endif
}

TEST_F(BFieldCoVerify, Subtraction_EdgeCases) {
#ifdef TRITON_CUDA_ENABLED
    std::vector<uint64_t> a = {
        0,                              // 0 - 1 = p-1
        1,                              // 1 - 1 = 0
        BFieldElement::MODULUS - 1,     // (p-1) - (p-1) = 0
        0,                              // 0 - (p-1) = 1
        100,                            // 100 - 200 (underflow)
    };
    std::vector<uint64_t> b = {
        1,
        1,
        BFieldElement::MODULUS - 1,
        BFieldElement::MODULUS - 1,
        200,
    };
    
    auto cpu_result = cpu_sub(a, b);
    auto gpu_result = gpu_sub(a, b);
    
    size_t mismatch;
    EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Sub_Edge", mismatch));
#endif
}

// ============================================================================
// Multiplication Tests
// ============================================================================

TEST_F(BFieldCoVerify, Multiplication_Random) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t n : test_sizes) {
        auto a = random_field_elements(n, 42);
        auto b = random_field_elements(n, 123);
        
        auto cpu_result = cpu_mul(a, b);
        auto gpu_result = gpu_mul(a, b);
        
        size_t mismatch;
        EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Mul", mismatch))
            << "Failed at size " << n;
    }
#endif
}

TEST_F(BFieldCoVerify, Multiplication_EdgeCases) {
#ifdef TRITON_CUDA_ENABLED
    std::vector<uint64_t> a = {
        0,                              // 0 * x = 0
        1,                              // 1 * x = x
        2,                              // 2 * (p-1)/2
        BFieldElement::MODULUS - 1,     // (p-1) * (p-1)
        1ULL << 32,                     // 2^32 * 2^32 = 2^64 mod p
    };
    std::vector<uint64_t> b = {
        BFieldElement::MODULUS - 1,
        BFieldElement::MODULUS - 1,
        (BFieldElement::MODULUS - 1) / 2,
        BFieldElement::MODULUS - 1,
        1ULL << 32,
    };
    
    auto cpu_result = cpu_mul(a, b);
    auto gpu_result = gpu_mul(a, b);
    
    size_t mismatch;
    EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Mul_Edge", mismatch));
#endif
}

TEST_F(BFieldCoVerify, Multiplication_LargeValues) {
#ifdef TRITON_CUDA_ENABLED
    // Test with values close to modulus (stress test for reduction)
    std::vector<uint64_t> a, b;
    for (uint64_t i = 0; i < 1000; ++i) {
        a.push_back(BFieldElement::MODULUS - 1 - i);
        b.push_back(BFieldElement::MODULUS - 1 - i);
    }
    
    auto cpu_result = cpu_mul(a, b);
    auto gpu_result = gpu_mul(a, b);
    
    size_t mismatch;
    EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Mul_Large", mismatch));
#endif
}

// ============================================================================
// Negation Tests
// ============================================================================

TEST_F(BFieldCoVerify, Negation_Random) {
#ifdef TRITON_CUDA_ENABLED
    for (size_t n : test_sizes) {
        auto a = random_field_elements(n, 42);
        
        auto cpu_result = cpu_neg(a);
        auto gpu_result = gpu_neg(a);
        
        size_t mismatch;
        EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Neg", mismatch))
            << "Failed at size " << n;
    }
#endif
}

TEST_F(BFieldCoVerify, Negation_EdgeCases) {
#ifdef TRITON_CUDA_ENABLED
    std::vector<uint64_t> a = {
        0,                              // -0 = 0
        1,                              // -1 = p-1
        BFieldElement::MODULUS - 1,     // -(p-1) = 1
        BFieldElement::MODULUS / 2,     // -half
    };
    
    auto cpu_result = cpu_neg(a);
    auto gpu_result = gpu_neg(a);
    
    size_t mismatch;
    EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Neg_Edge", mismatch));
#endif
}

// ============================================================================
// Inverse Tests
// ============================================================================

TEST_F(BFieldCoVerify, Inverse_Random) {
#ifdef TRITON_CUDA_ENABLED
    // Use smaller sizes for inverse (it's slow)
    std::vector<size_t> inv_sizes = {1, 16, 256, 1024};
    
    for (size_t n : inv_sizes) {
        auto a = random_nonzero_field_elements(n, 42);
        
        auto cpu_result = cpu_inv(a);
        auto gpu_result = gpu_inv(a);
        
        size_t mismatch;
        EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Inv", mismatch))
            << "Failed at size " << n;
    }
#endif
}

TEST_F(BFieldCoVerify, Inverse_Verify) {
#ifdef TRITON_CUDA_ENABLED
    // Verify that a * a^{-1} = 1
    auto a = random_nonzero_field_elements(100, 42);
    auto inv_gpu = gpu_inv(a);
    auto product = gpu_mul(a, inv_gpu);
    
    for (size_t i = 0; i < product.size(); ++i) {
        EXPECT_EQ(product[i], 1) << "a * a^{-1} != 1 at index " << i;
    }
#endif
}

// ============================================================================
// Power Tests
// ============================================================================

TEST_F(BFieldCoVerify, Power_Random) {
#ifdef TRITON_CUDA_ENABLED
    // Test with various exponents
    std::vector<uint64_t> exponents = {0, 1, 2, 7, 64, 1000, BFieldElement::MODULUS - 2};
    
    for (uint64_t exp : exponents) {
        auto a = random_nonzero_field_elements(256, 42);
        
        auto cpu_result = cpu_pow(a, exp);
        auto gpu_result = gpu_pow(a, exp);
        
        size_t mismatch;
        EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Pow_" + std::to_string(exp), mismatch));
    }
#endif
}

TEST_F(BFieldCoVerify, Power_Fermat) {
#ifdef TRITON_CUDA_ENABLED
    // Fermat's little theorem: a^{p-1} = 1 for a != 0
    auto a = random_nonzero_field_elements(100, 42);
    auto result = gpu_pow(a, BFieldElement::MODULUS - 1);
    
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], 1) << "a^{p-1} != 1 at index " << i;
    }
#endif
}

// ============================================================================
// Performance Benchmark
// ============================================================================

TEST_F(BFieldCoVerify, Benchmark_Multiplication) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1 << 20;  // 1M elements
    auto a = random_field_elements(n, 42);
    auto b = random_field_elements(n, 123);
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_result = cpu_mul(a, b);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing (including transfer)
    auto gpu_start = std::chrono::high_resolution_clock::now();
    auto gpu_result = gpu_mul(a, b);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    // Verify correctness
    size_t mismatch;
    EXPECT_TRUE(compare_results(cpu_result, gpu_result, "Bench_Mul", mismatch));
    
    std::cout << "\n[Benchmark] BField Multiplication (" << n << " elements):" << std::endl;
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms" << std::endl;
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)" << std::endl;
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x" << std::endl;
#endif
}

} // namespace co_verify
} // namespace triton_vm

