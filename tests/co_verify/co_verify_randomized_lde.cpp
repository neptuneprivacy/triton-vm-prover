/**
 * Co-verification test for GPU Randomized LDE
 * 
 * Compares GPU randomized_lde_column_gpu against CPU RandomizedLDE::extend_column_with_randomizer
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

#include "co_verify_framework.hpp"
#include "types/b_field_element.hpp"
#include "lde/lde_randomized.hpp"
#include "table/master_table.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/randomized_lde_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include "ntt/ntt.hpp"
#include <cuda_runtime.h>
#endif

using namespace triton_vm;

class RandomizedLDECoVerifyTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(42);
    }
    
    std::vector<BFieldElement> random_bfield_vec(size_t n) {
        std::vector<BFieldElement> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = BFieldElement(dist_(rng_) % BFieldElement::MODULUS);
        }
        return result;
    }
    
    std::mt19937_64 rng_;
    std::uniform_int_distribution<uint64_t> dist_;
};

#ifdef TRITON_CUDA_ENABLED

TEST_F(RandomizedLDECoVerifyTest, SingleColumn_Small) {
    // Small test: trace_len = 16, target_len = 64, randomizer_len = 4
    const size_t trace_len = 16;
    const size_t target_len = 64;
    const size_t randomizer_len = 4;
    
    // Create trace column
    std::vector<BFieldElement> trace_column = random_bfield_vec(trace_len);
    
    // Create randomizer coefficients
    std::vector<BFieldElement> randomizer_coeffs = random_bfield_vec(randomizer_len);
    
    // Define domains - use offset 1 to simplify debugging
    BFieldElement trace_offset = BFieldElement::one();
    BFieldElement target_offset = BFieldElement(7);  // Standard FRI offset
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len)
        .with_offset(trace_offset);
    ArithmeticDomain target_domain = ArithmeticDomain::of_length(target_len)
        .with_offset(target_offset);
    
    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<BFieldElement> cpu_result = RandomizedLDE::extend_column_with_randomizer(
        trace_column, trace_domain, target_domain, randomizer_coeffs
    );
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU computation
    std::vector<uint64_t> h_trace(trace_len);
    std::vector<uint64_t> h_rand(randomizer_len);
    for (size_t i = 0; i < trace_len; ++i) h_trace[i] = trace_column[i].value();
    for (size_t i = 0; i < randomizer_len; ++i) h_rand[i] = randomizer_coeffs[i].value();
    
    uint64_t* d_trace;
    uint64_t* d_rand;
    uint64_t* d_output;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_rand, randomizer_len * sizeof(uint64_t));
    cudaMalloc(&d_output, target_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, h_trace.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand, h_rand.data(), randomizer_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::randomized_lde_column_gpu(
        d_trace, trace_len,
        d_rand, randomizer_len,
        trace_offset.value(), target_offset.value(),
        target_len,
        d_output, 0
    );
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::vector<uint64_t> h_output(target_len);
    cudaMemcpy(h_output.data(), d_output, target_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    std::vector<BFieldElement> gpu_result(target_len);
    for (size_t i = 0; i < target_len; ++i) {
        gpu_result[i] = BFieldElement(h_output[i]);
    }
    
    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < target_len; ++i) {
        if (cpu_result[i].value() != gpu_result[i].value()) {
            if (mismatches < 5) {
                std::cout << "Mismatch at " << i << ": CPU=" << cpu_result[i].value()
                          << ", GPU=" << gpu_result[i].value() << std::endl;
            }
            mismatches++;
        }
    }
    
    std::cout << "SingleColumn_Small: CPU=" << cpu_time << "ms, GPU=" << gpu_time << "ms" << std::endl;
    std::cout << "Mismatches: " << mismatches << " / " << target_len << std::endl;
    
    EXPECT_EQ(mismatches, 0) << "GPU and CPU randomized LDE results should match";
    
    cudaFree(d_trace);
    cudaFree(d_rand);
    cudaFree(d_output);
}

TEST_F(RandomizedLDECoVerifyTest, SingleColumn_TraceSize512) {
    // Realistic test: trace_len = 512, target_len = 4096
    const size_t trace_len = 512;
    const size_t target_len = 4096;
    const size_t randomizer_len = 8;
    
    std::vector<BFieldElement> trace_column = random_bfield_vec(trace_len);
    std::vector<BFieldElement> randomizer_coeffs = random_bfield_vec(randomizer_len);
    
    BFieldElement trace_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(trace_len)) + 1
    );
    BFieldElement target_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(target_len)) + 1
    );
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len)
        .with_offset(trace_offset);
    ArithmeticDomain target_domain = ArithmeticDomain::of_length(target_len)
        .with_offset(target_offset);
    
    // CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<BFieldElement> cpu_result = RandomizedLDE::extend_column_with_randomizer(
        trace_column, trace_domain, target_domain, randomizer_coeffs
    );
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU
    std::vector<uint64_t> h_trace(trace_len);
    std::vector<uint64_t> h_rand(randomizer_len);
    for (size_t i = 0; i < trace_len; ++i) h_trace[i] = trace_column[i].value();
    for (size_t i = 0; i < randomizer_len; ++i) h_rand[i] = randomizer_coeffs[i].value();
    
    uint64_t* d_trace;
    uint64_t* d_rand;
    uint64_t* d_output;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_rand, randomizer_len * sizeof(uint64_t));
    cudaMalloc(&d_output, target_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, h_trace.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand, h_rand.data(), randomizer_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::randomized_lde_column_gpu(
        d_trace, trace_len,
        d_rand, randomizer_len,
        trace_offset.value(), target_offset.value(),
        target_len,
        d_output, 0
    );
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::vector<uint64_t> h_output(target_len);
    cudaMemcpy(h_output.data(), d_output, target_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < target_len; ++i) {
        if (cpu_result[i].value() != h_output[i]) {
            if (mismatches < 5) {
                std::cout << "Mismatch at " << i << ": CPU=" << cpu_result[i].value()
                          << ", GPU=" << h_output[i] << std::endl;
            }
            mismatches++;
        }
    }
    
    std::cout << "SingleColumn_TraceSize512: CPU=" << cpu_time << "ms, GPU=" << gpu_time << "ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
    std::cout << "Mismatches: " << mismatches << " / " << target_len << std::endl;
    
    EXPECT_EQ(mismatches, 0) << "GPU and CPU randomized LDE results should match";
    
    cudaFree(d_trace);
    cudaFree(d_rand);
    cudaFree(d_output);
}

TEST_F(RandomizedLDECoVerifyTest, WithNonTrivialTraceOffset) {
    // Test with non-trivial trace offset to debug coset interpolation
    const size_t trace_len = 16;
    const size_t target_len = 64;
    const size_t randomizer_len = 4;
    
    std::vector<BFieldElement> trace_column = random_bfield_vec(trace_len);
    std::vector<BFieldElement> randomizer_coeffs = random_bfield_vec(randomizer_len);
    
    // Use a non-trivial trace offset
    BFieldElement trace_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(trace_len)) + 1
    );
    BFieldElement target_offset = BFieldElement(7);
    
    std::cout << "trace_offset = " << trace_offset.value() << std::endl;
    std::cout << "target_offset = " << target_offset.value() << std::endl;
    
    // Debug: Test GPU vs CPU coset interpolation step by step
    {
        // CPU: INTT
        std::vector<BFieldElement> cpu_after_intt = trace_column;
        NTT::inverse(cpu_after_intt);
        
        // GPU: INTT
        std::vector<uint64_t> gpu_data(trace_len);
        for (size_t i = 0; i < trace_len; ++i) gpu_data[i] = trace_column[i].value();
        
        uint64_t* d_data;
        cudaMalloc(&d_data, trace_len * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_data.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        gpu::kernels::ntt_inverse_gpu(d_data, trace_len, 0);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> gpu_after_intt(trace_len);
        cudaMemcpy(gpu_after_intt.data(), d_data, trace_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        // Compare INTT results
        size_t intt_mismatches = 0;
        for (size_t i = 0; i < trace_len; ++i) {
            if (cpu_after_intt[i].value() != gpu_after_intt[i]) {
                intt_mismatches++;
            }
        }
        std::cout << "INTT mismatches: " << intt_mismatches << " / " << trace_len << std::endl;
        
        // Now test coset scaling
        // CPU: Scale by offset^(-i)
        BFieldElement offset_inv = trace_offset.inverse();
        BFieldElement scale = BFieldElement::one();
        std::vector<BFieldElement> cpu_scaled = cpu_after_intt;
        for (size_t i = 0; i < trace_len; ++i) {
            cpu_scaled[i] = cpu_after_intt[i] * scale;
            scale = scale * offset_inv;
        }
        
        std::cout << "CPU interpolant[0..3]: " << cpu_scaled[0].value() << ", " 
                  << cpu_scaled[1].value() << ", " << cpu_scaled[2].value() << ", "
                  << cpu_scaled[3].value() << std::endl;
        
        // GPU: Scale by offset^(-i) using host function
        uint64_t offset_inv_val = gpu::kernels::bfield_inv_host(trace_offset.value());
        std::cout << "offset_inv CPU: " << offset_inv.value() << ", GPU host: " << offset_inv_val << std::endl;
        
        cudaFree(d_data);
    }
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len)
        .with_offset(trace_offset);
    ArithmeticDomain target_domain = ArithmeticDomain::of_length(target_len)
        .with_offset(target_offset);
    
    // CPU
    std::vector<BFieldElement> cpu_result = RandomizedLDE::extend_column_with_randomizer(
        trace_column, trace_domain, target_domain, randomizer_coeffs
    );
    
    // GPU
    std::vector<uint64_t> h_trace(trace_len);
    std::vector<uint64_t> h_rand(randomizer_len);
    for (size_t i = 0; i < trace_len; ++i) h_trace[i] = trace_column[i].value();
    for (size_t i = 0; i < randomizer_len; ++i) h_rand[i] = randomizer_coeffs[i].value();
    
    uint64_t* d_trace;
    uint64_t* d_rand;
    uint64_t* d_output;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_rand, randomizer_len * sizeof(uint64_t));
    cudaMalloc(&d_output, target_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, h_trace.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand, h_rand.data(), randomizer_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::randomized_lde_column_gpu(
        d_trace, trace_len,
        d_rand, randomizer_len,
        trace_offset.value(), target_offset.value(),
        target_len,
        d_output, 0
    );
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> h_output(target_len);
    cudaMemcpy(h_output.data(), d_output, target_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < target_len; ++i) {
        if (cpu_result[i].value() != h_output[i]) {
            mismatches++;
        }
    }
    
    EXPECT_EQ(mismatches, 0) << "GPU and CPU should match with offset=1";
    
    cudaFree(d_trace);
    cudaFree(d_rand);
    cudaFree(d_output);
}

#else

TEST_F(RandomizedLDECoVerifyTest, CUDANotEnabled) {
    GTEST_SKIP() << "CUDA not enabled";
}

#endif // TRITON_CUDA_ENABLED

