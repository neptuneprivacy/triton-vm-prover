/**
 * LDE (Low Degree Extension) Co-Verification Tests
 * 
 * Compares CPU LDE implementation with GPU CUDA kernels
 * to ensure they produce identical results.
 */

#include "co_verify_framework.hpp"
#include "ntt/ntt.hpp"
#include "table/master_table.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/lde_kernel.cuh"
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

class LdeCoVerifyTest : public ::testing::Test {
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
    
    std::vector<BFieldElement> from_u64(const std::vector<uint64_t>& v) {
        std::vector<BFieldElement> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = BFieldElement(v[i]);
        }
        return result;
    }
    
    static bool compare_bfield(const BFieldElement& a, const BFieldElement& b) {
        return a == b;
    }
};

// ============================================================================
// Basic LDE Tests
// ============================================================================

TEST_F(LdeCoVerifyTest, Column_Small) {
#ifdef TRITON_CUDA_ENABLED
    // Test small LDE: trace_len=8 -> extended_len=32
    const size_t trace_len = 8;
    const size_t extended_len = 32;
    
    // Create trace domain and quotient domain
    ArithmeticDomain trace_domain;
    trace_domain.length = trace_len;
    trace_domain.offset = BFieldElement(7);  // Common offset in Triton VM
    
    ArithmeticDomain quotient_domain;
    quotient_domain.length = extended_len;
    quotient_domain.offset = BFieldElement(7);  // Same offset for simplicity
    
    auto trace = random_vector(trace_len);
    
    // CPU LDE
    auto cpu_result = LDE::extend_column(trace, trace_domain, quotient_domain);
    
    // GPU LDE
    auto trace_u64 = to_u64(trace);
    std::vector<uint64_t> gpu_result_u64(extended_len);
    
    uint64_t *d_trace, *d_extended;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, extended_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, trace_u64.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::lde_column_gpu(
        d_trace, trace_len,
        d_extended, extended_len,
        trace_domain.offset.value(),
        quotient_domain.offset.value()
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_extended, extended_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = from_u64(gpu_result_u64);
    
    // Compare
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "LDE mismatch at index " << mismatch
                       << ": CPU=" << cpu_result[mismatch] << ", GPU=" << gpu_result[mismatch];
    
    cudaFree(d_trace);
    cudaFree(d_extended);
#endif
}

TEST_F(LdeCoVerifyTest, Column_Medium) {
#ifdef TRITON_CUDA_ENABLED
    // Test medium LDE: trace_len=256 -> extended_len=1024
    const size_t trace_len = 256;
    const size_t extended_len = 1024;
    
    ArithmeticDomain trace_domain;
    trace_domain.length = trace_len;
    trace_domain.offset = BFieldElement(7);
    
    ArithmeticDomain quotient_domain;
    quotient_domain.length = extended_len;
    quotient_domain.offset = BFieldElement(7);
    
    auto trace = random_vector(trace_len);
    
    // CPU LDE
    auto cpu_result = LDE::extend_column(trace, trace_domain, quotient_domain);
    
    // GPU LDE
    auto trace_u64 = to_u64(trace);
    std::vector<uint64_t> gpu_result_u64(extended_len);
    
    uint64_t *d_trace, *d_extended;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, extended_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, trace_u64.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::lde_column_gpu(
        d_trace, trace_len,
        d_extended, extended_len,
        trace_domain.offset.value(),
        quotient_domain.offset.value()
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_extended, extended_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = from_u64(gpu_result_u64);
    
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "LDE mismatch at index " << mismatch;
    
    cudaFree(d_trace);
    cudaFree(d_extended);
#endif
}

TEST_F(LdeCoVerifyTest, Column_Large) {
#ifdef TRITON_CUDA_ENABLED
    // Test larger LDE: trace_len=4096 -> extended_len=16384
    const size_t trace_len = 4096;
    const size_t extended_len = 16384;
    
    ArithmeticDomain trace_domain;
    trace_domain.length = trace_len;
    trace_domain.offset = BFieldElement(7);
    
    ArithmeticDomain quotient_domain;
    quotient_domain.length = extended_len;
    quotient_domain.offset = BFieldElement(7);
    
    auto trace = random_vector(trace_len);
    
    // CPU LDE
    auto cpu_result = LDE::extend_column(trace, trace_domain, quotient_domain);
    
    // GPU LDE
    auto trace_u64 = to_u64(trace);
    std::vector<uint64_t> gpu_result_u64(extended_len);
    
    uint64_t *d_trace, *d_extended;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, extended_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, trace_u64.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::lde_column_gpu(
        d_trace, trace_len,
        d_extended, extended_len,
        trace_domain.offset.value(),
        quotient_domain.offset.value()
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_extended, extended_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = from_u64(gpu_result_u64);
    
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "LDE mismatch at index " << mismatch;
    
    cudaFree(d_trace);
    cudaFree(d_extended);
#endif
}

// ============================================================================
// Different Expansion Ratios
// ============================================================================

TEST_F(LdeCoVerifyTest, ExpansionRatio_4x) {
#ifdef TRITON_CUDA_ENABLED
    const size_t trace_len = 256;
    const size_t extended_len = 1024;  // 4x expansion
    
    ArithmeticDomain trace_domain{trace_len, BFieldElement(7)};
    ArithmeticDomain quotient_domain{extended_len, BFieldElement(7)};
    
    auto trace = random_vector(trace_len);
    auto cpu_result = LDE::extend_column(trace, trace_domain, quotient_domain);
    
    auto trace_u64 = to_u64(trace);
    std::vector<uint64_t> gpu_result_u64(extended_len);
    
    uint64_t *d_trace, *d_extended;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, extended_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, trace_u64.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::lde_column_gpu(d_trace, trace_len, d_extended, extended_len,
                                  trace_domain.offset.value(), quotient_domain.offset.value());
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_extended, extended_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = from_u64(gpu_result_u64);
    
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "4x expansion mismatch at index " << mismatch;
    
    cudaFree(d_trace);
    cudaFree(d_extended);
#endif
}

TEST_F(LdeCoVerifyTest, ExpansionRatio_8x) {
#ifdef TRITON_CUDA_ENABLED
    const size_t trace_len = 256;
    const size_t extended_len = 2048;  // 8x expansion
    
    ArithmeticDomain trace_domain{trace_len, BFieldElement(7)};
    ArithmeticDomain quotient_domain{extended_len, BFieldElement(7)};
    
    auto trace = random_vector(trace_len);
    auto cpu_result = LDE::extend_column(trace, trace_domain, quotient_domain);
    
    auto trace_u64 = to_u64(trace);
    std::vector<uint64_t> gpu_result_u64(extended_len);
    
    uint64_t *d_trace, *d_extended;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, extended_len * sizeof(uint64_t));
    
    cudaMemcpy(d_trace, trace_u64.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::lde_column_gpu(d_trace, trace_len, d_extended, extended_len,
                                  trace_domain.offset.value(), quotient_domain.offset.value());
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_extended, extended_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = from_u64(gpu_result_u64);
    
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "8x expansion mismatch at index " << mismatch;
    
    cudaFree(d_trace);
    cudaFree(d_extended);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(LdeCoVerifyTest, Benchmark) {
#ifdef TRITON_CUDA_ENABLED
    const size_t trace_len = 1 << 14;     // 16K
    const size_t extended_len = 1 << 16;  // 64K (4x expansion)
    
    ArithmeticDomain trace_domain{trace_len, BFieldElement(7)};
    ArithmeticDomain quotient_domain{extended_len, BFieldElement(7)};
    
    auto trace = random_vector(trace_len);
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_result = LDE::extend_column(trace, trace_domain, quotient_domain);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    auto trace_u64 = to_u64(trace);
    std::vector<uint64_t> gpu_result_u64(extended_len);
    
    uint64_t *d_trace, *d_extended;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, extended_len * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_trace, trace_u64.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    gpu::kernels::lde_column_gpu(d_trace, trace_len, d_extended, extended_len,
                                  trace_domain.offset.value(), quotient_domain.offset.value());
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_result_u64.data(), d_extended, extended_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] LDE Column (trace=" << trace_len << " -> extended=" << extended_len << "):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify correctness
    auto gpu_result = from_u64(gpu_result_u64);
    size_t mismatch;
    bool match = compare_vectors(cpu_result, gpu_result, mismatch, compare_bfield);
    EXPECT_TRUE(match) << "Benchmark mismatch at index " << mismatch;
    
    cudaFree(d_trace);
    cudaFree(d_extended);
#endif
}

} // namespace co_verify
} // namespace triton_vm

