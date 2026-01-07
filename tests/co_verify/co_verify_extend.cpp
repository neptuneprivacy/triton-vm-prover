/**
 * Co-Verification Tests for Auxiliary Table Extension
 * 
 * Tests GPU aux table extension against CPU implementation.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <random>

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/extend_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include <cuda_runtime.h>
#endif

using namespace triton_vm;

namespace {

template<typename Clock = std::chrono::high_resolution_clock>
double elapsed_ms(std::chrono::time_point<Clock> start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

class ExtendCoVerifyTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
};

#ifdef TRITON_CUDA_ENABLED

// ============================================================================
// XFE Prefix Product Tests
// ============================================================================

TEST_F(ExtendCoVerifyTest, XFE_PrefixProduct_Small) {
    // Test prefix product on small array
    size_t n = 16;
    
    // Generate random XFEs
    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_data(n * 3);
    for (size_t i = 0; i < n * 3; ++i) {
        h_data[i] = rng() % BFieldElement::MODULUS;
    }
    
    // CPU prefix product
    std::vector<XFieldElement> cpu_result(n);
    XFieldElement product = XFieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        XFieldElement elem(
            BFieldElement(h_data[i * 3]),
            BFieldElement(h_data[i * 3 + 1]),
            BFieldElement(h_data[i * 3 + 2])
        );
        product = product * elem;
        cpu_result[i] = product;
    }
    
    // GPU prefix product
    uint64_t* d_data;
    cudaMalloc(&d_data, n * 3 * sizeof(uint64_t));
    cudaMemcpy(d_data, h_data.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_prefix_product_gpu(d_data, n, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    // Download and compare
    std::vector<uint64_t> gpu_result_raw(n * 3);
    cudaMemcpy(gpu_result_raw.data(), d_data, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < n && mismatches < 5; ++i) {
        XFieldElement gpu_elem(
            BFieldElement(gpu_result_raw[i * 3]),
            BFieldElement(gpu_result_raw[i * 3 + 1]),
            BFieldElement(gpu_result_raw[i * 3 + 2])
        );
        
        if (cpu_result[i].coeff(0).value() != gpu_elem.coeff(0).value() ||
            cpu_result[i].coeff(1).value() != gpu_elem.coeff(1).value() ||
            cpu_result[i].coeff(2).value() != gpu_elem.coeff(2).value()) {
            std::cout << "Mismatch at " << i << ":" << std::endl;
            std::cout << "  CPU: " << cpu_result[i].to_string() << std::endl;
            std::cout << "  GPU: " << gpu_elem.to_string() << std::endl;
            ++mismatches;
        }
    }
    
    std::cout << "XFE Prefix Product (n=" << n << "): GPU=" << gpu_time << " ms" << std::endl;
    EXPECT_EQ(mismatches, 0);
    
    cudaFree(d_data);
}

TEST_F(ExtendCoVerifyTest, XFE_PrefixSum_Small) {
    // Test prefix sum on small array
    size_t n = 16;
    
    // Generate random XFEs
    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_data(n * 3);
    for (size_t i = 0; i < n * 3; ++i) {
        h_data[i] = rng() % BFieldElement::MODULUS;
    }
    
    // CPU prefix sum
    std::vector<XFieldElement> cpu_result(n);
    XFieldElement sum = XFieldElement::zero();
    for (size_t i = 0; i < n; ++i) {
        XFieldElement elem(
            BFieldElement(h_data[i * 3]),
            BFieldElement(h_data[i * 3 + 1]),
            BFieldElement(h_data[i * 3 + 2])
        );
        sum = sum + elem;
        cpu_result[i] = sum;
    }
    
    // GPU prefix sum
    uint64_t* d_data;
    cudaMalloc(&d_data, n * 3 * sizeof(uint64_t));
    cudaMemcpy(d_data, h_data.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_prefix_sum_gpu(d_data, n, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    // Download and compare
    std::vector<uint64_t> gpu_result_raw(n * 3);
    cudaMemcpy(gpu_result_raw.data(), d_data, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < n && mismatches < 5; ++i) {
        XFieldElement gpu_elem(
            BFieldElement(gpu_result_raw[i * 3]),
            BFieldElement(gpu_result_raw[i * 3 + 1]),
            BFieldElement(gpu_result_raw[i * 3 + 2])
        );
        
        if (cpu_result[i].coeff(0).value() != gpu_elem.coeff(0).value() ||
            cpu_result[i].coeff(1).value() != gpu_elem.coeff(1).value() ||
            cpu_result[i].coeff(2).value() != gpu_elem.coeff(2).value()) {
            std::cout << "Mismatch at " << i << ":" << std::endl;
            std::cout << "  CPU: " << cpu_result[i].to_string() << std::endl;
            std::cout << "  GPU: " << gpu_elem.to_string() << std::endl;
            ++mismatches;
        }
    }
    
    std::cout << "XFE Prefix Sum (n=" << n << "): GPU=" << gpu_time << " ms" << std::endl;
    EXPECT_EQ(mismatches, 0);
    
    cudaFree(d_data);
}

TEST_F(ExtendCoVerifyTest, XFE_BatchInverse) {
    // Test batch inverse computation
    size_t n = 256;
    
    // Generate random non-zero XFEs
    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_data(n * 3);
    for (size_t i = 0; i < n; ++i) {
        // Ensure non-zero
        h_data[i * 3] = 1 + (rng() % (BFieldElement::MODULUS - 1));
        h_data[i * 3 + 1] = rng() % BFieldElement::MODULUS;
        h_data[i * 3 + 2] = rng() % BFieldElement::MODULUS;
    }
    
    // CPU batch inverse
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<XFieldElement> cpu_inverses(n);
    for (size_t i = 0; i < n; ++i) {
        XFieldElement elem(
            BFieldElement(h_data[i * 3]),
            BFieldElement(h_data[i * 3 + 1]),
            BFieldElement(h_data[i * 3 + 2])
        );
        cpu_inverses[i] = elem.inverse();
    }
    double cpu_time = elapsed_ms(cpu_start);
    
    // GPU batch inverse
    uint64_t* d_input;
    uint64_t* d_output;
    cudaMalloc(&d_input, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_output, n * 3 * sizeof(uint64_t));
    cudaMemcpy(d_input, h_data.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_batch_inverse_gpu(d_input, d_output, n, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    // Download and compare
    std::vector<uint64_t> gpu_result_raw(n * 3);
    cudaMemcpy(gpu_result_raw.data(), d_output, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < n && mismatches < 5; ++i) {
        if (cpu_inverses[i].coeff(0).value() != gpu_result_raw[i * 3] ||
            cpu_inverses[i].coeff(1).value() != gpu_result_raw[i * 3 + 1] ||
            cpu_inverses[i].coeff(2).value() != gpu_result_raw[i * 3 + 2]) {
            std::cout << "Mismatch at " << i << ":" << std::endl;
            std::cout << "  CPU: " << cpu_inverses[i].to_string() << std::endl;
            std::cout << "  GPU: (" << gpu_result_raw[i * 3] << ", " 
                      << gpu_result_raw[i * 3 + 1] << ", " 
                      << gpu_result_raw[i * 3 + 2] << ")" << std::endl;
            ++mismatches;
        }
    }
    
    std::cout << "XFE Batch Inverse (n=" << n << "):" << std::endl;
    std::cout << "  CPU: " << cpu_time << " ms" << std::endl;
    std::cout << "  GPU: " << gpu_time << " ms" << std::endl;
    std::cout << "  Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
    
    EXPECT_EQ(mismatches, 0);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// Larger Scale Tests (for benchmarking)
// ============================================================================

TEST_F(ExtendCoVerifyTest, XFE_PrefixProduct_Large) {
    size_t n = 4096;  // Typical trace size for spin_input8
    
    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_data(n * 3);
    for (size_t i = 0; i < n * 3; ++i) {
        h_data[i] = rng() % BFieldElement::MODULUS;
    }
    
    // GPU prefix product
    uint64_t* d_data;
    cudaMalloc(&d_data, n * 3 * sizeof(uint64_t));
    cudaMemcpy(d_data, h_data.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Warmup
    gpu::kernels::xfe_prefix_product_gpu(d_data, n, 0);
    cudaDeviceSynchronize();
    
    // Reset data
    cudaMemcpy(d_data, h_data.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_prefix_product_gpu(d_data, n, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    std::cout << "XFE Prefix Product (n=" << n << "): GPU=" << gpu_time << " ms" << std::endl;
    std::cout << "  Throughput: " << (n / gpu_time * 1000.0 / 1e6) << " M elements/sec" << std::endl;
    
    cudaFree(d_data);
}

TEST_F(ExtendCoVerifyTest, XFE_BatchInverse_Large) {
    size_t n = 4096;
    
    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_data(n * 3);
    for (size_t i = 0; i < n; ++i) {
        h_data[i * 3] = 1 + (rng() % (BFieldElement::MODULUS - 1));
        h_data[i * 3 + 1] = rng() % BFieldElement::MODULUS;
        h_data[i * 3 + 2] = rng() % BFieldElement::MODULUS;
    }
    
    // CPU batch inverse
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        XFieldElement elem(
            BFieldElement(h_data[i * 3]),
            BFieldElement(h_data[i * 3 + 1]),
            BFieldElement(h_data[i * 3 + 2])
        );
        volatile auto inv = elem.inverse();  // Prevent optimization
        (void)inv;
    }
    double cpu_time = elapsed_ms(cpu_start);
    
    // GPU batch inverse
    uint64_t* d_input;
    uint64_t* d_output;
    cudaMalloc(&d_input, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_output, n * 3 * sizeof(uint64_t));
    cudaMemcpy(d_input, h_data.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Warmup
    gpu::kernels::xfe_batch_inverse_gpu(d_input, d_output, n, 0);
    cudaDeviceSynchronize();
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_batch_inverse_gpu(d_input, d_output, n, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    std::cout << "XFE Batch Inverse (n=" << n << "):" << std::endl;
    std::cout << "  CPU: " << cpu_time << " ms" << std::endl;
    std::cout << "  GPU: " << gpu_time << " ms" << std::endl;
    std::cout << "  Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
}

#endif // TRITON_CUDA_ENABLED

