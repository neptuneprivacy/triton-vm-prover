/**
 * Quotient Computation Co-Verification Tests
 * 
 * Compares CPU quotient operations with GPU CUDA kernels.
 * Tests core mathematical operations used in quotient computation:
 * - Zerofier inverse batch computation
 * - XFieldElement weighted sums
 * - Element-wise XFE multiplication
 */

#include "co_verify_framework.hpp"
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "table/master_table.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/quotient_kernel.cuh"
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

namespace triton_vm {
namespace co_verify {

class QuotientCoVerifyTest : public ::testing::Test {
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
    
    std::vector<BFieldElement> random_bfield_vector(size_t n) {
        std::vector<BFieldElement> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = random_bfield();
        }
        return v;
    }
    
    std::vector<XFieldElement> random_xfield_vector(size_t n) {
        std::vector<XFieldElement> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = random_xfield();
        }
        return v;
    }
    
    std::vector<uint64_t> bfe_to_u64(const std::vector<BFieldElement>& v) {
        std::vector<uint64_t> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i].value();
        }
        return result;
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
    
    std::vector<BFieldElement> u64_to_bfe(const std::vector<uint64_t>& v) {
        std::vector<BFieldElement> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = BFieldElement(v[i]);
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
};

// ============================================================================
// Zerofier Inverse Tests
// ============================================================================

TEST_F(QuotientCoVerifyTest, ZerofierInv_Basic) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1024;
    auto points = random_bfield_vector(n);
    BFieldElement offset(1);  // Initial zerofier uses offset = 1
    
    // CPU: compute (x - 1)^{-1}
    std::vector<BFieldElement> cpu_result(n);
    for (size_t i = 0; i < n; ++i) {
        cpu_result[i] = (points[i] - offset).inverse();
    }
    
    // GPU
    auto points_u64 = bfe_to_u64(points);
    std::vector<uint64_t> gpu_result_u64(n);
    
    uint64_t *d_points, *d_output;
    cudaMalloc(&d_points, n * sizeof(uint64_t));
    cudaMalloc(&d_output, n * sizeof(uint64_t));
    
    cudaMemcpy(d_points, points_u64.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::zerofier_inv_gpu(d_points, offset.value(), d_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_bfe(gpu_result_u64);
    
    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
            if (mismatches <= 3) {
                std::cout << "Mismatch at " << i << ": CPU=" << cpu_result[i]
                          << ", GPU=" << gpu_result[i] << std::endl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " mismatches out of " << n;
    
    cudaFree(d_points);
    cudaFree(d_output);
#endif
}

TEST_F(QuotientCoVerifyTest, PowerZerofierInv) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1024;
    const size_t power = 256;  // Typical trace length
    auto points = random_bfield_vector(n);
    
    // CPU: compute (x^power - 1)^{-1}
    std::vector<BFieldElement> cpu_result(n);
    for (size_t i = 0; i < n; ++i) {
        cpu_result[i] = (points[i].pow(power) - BFieldElement::one()).inverse();
    }
    
    // GPU
    auto points_u64 = bfe_to_u64(points);
    std::vector<uint64_t> gpu_result_u64(n);
    
    uint64_t *d_points, *d_output;
    cudaMalloc(&d_points, n * sizeof(uint64_t));
    cudaMalloc(&d_output, n * sizeof(uint64_t));
    
    cudaMemcpy(d_points, points_u64.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::power_zerofier_inv_gpu(d_points, power, d_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_bfe(gpu_result_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " power zerofier mismatches";
    
    cudaFree(d_points);
    cudaFree(d_output);
#endif
}

// ============================================================================
// Weighted Sum Tests
// ============================================================================

TEST_F(QuotientCoVerifyTest, WeightedSum_XFE) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 100;
    auto values = random_xfield_vector(n);
    auto weights = random_xfield_vector(n);
    
    // CPU weighted sum
    XFieldElement cpu_sum = XFieldElement::zero();
    for (size_t i = 0; i < n; ++i) {
        cpu_sum = cpu_sum + values[i] * weights[i];
    }
    
    // GPU
    auto values_u64 = xfe_to_u64(values);
    auto weights_u64 = xfe_to_u64(weights);
    std::vector<uint64_t> gpu_sum_u64(3);
    
    uint64_t *d_values, *d_weights, *d_output;
    cudaMalloc(&d_values, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_weights, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_output, 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_values, values_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfe_weighted_sum_gpu(d_values, d_weights, d_output, n, true);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_sum_u64.data(), d_output, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    XFieldElement gpu_sum = XFieldElement(
        BFieldElement(gpu_sum_u64[0]),
        BFieldElement(gpu_sum_u64[1]),
        BFieldElement(gpu_sum_u64[2])
    );
    
    bool match = (cpu_sum == gpu_sum);
    EXPECT_TRUE(match) << "Weighted sum mismatch: CPU=" << cpu_sum.to_string()
                       << ", GPU=" << gpu_sum.to_string();
    
    cudaFree(d_values);
    cudaFree(d_weights);
    cudaFree(d_output);
#endif
}

TEST_F(QuotientCoVerifyTest, WeightedSumBatch) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 256;
    const size_t num_elements = 50;
    
    // Generate random values and weights
    std::vector<XFieldElement> all_values(num_rows * num_elements);
    for (auto& v : all_values) v = random_xfield();
    
    auto weights = random_xfield_vector(num_elements);
    
    // CPU: compute weighted sum for each row
    std::vector<XFieldElement> cpu_result(num_rows);
    for (size_t row = 0; row < num_rows; ++row) {
        XFieldElement sum = XFieldElement::zero();
        for (size_t i = 0; i < num_elements; ++i) {
            sum = sum + all_values[row * num_elements + i] * weights[i];
        }
        cpu_result[row] = sum;
    }
    
    // GPU
    auto values_u64 = xfe_to_u64(all_values);
    auto weights_u64 = xfe_to_u64(weights);
    std::vector<uint64_t> gpu_result_u64(num_rows * 3);
    
    uint64_t *d_values, *d_weights, *d_output;
    cudaMalloc(&d_values, num_rows * num_elements * 3 * sizeof(uint64_t));
    cudaMalloc(&d_weights, num_elements * 3 * sizeof(uint64_t));
    cudaMalloc(&d_output, num_rows * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_values, values_u64.data(), values_u64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_u64.data(), weights_u64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfe_weighted_sum_batch_gpu(d_values, d_weights, d_output, num_rows, num_elements);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, num_rows * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_xfe(gpu_result_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < num_rows; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
            if (mismatches <= 3) {
                std::cout << "Row " << i << " mismatch: CPU=" << cpu_result[i].to_string()
                          << ", GPU=" << gpu_result[i].to_string() << std::endl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " batch sum mismatches";
    
    cudaFree(d_values);
    cudaFree(d_weights);
    cudaFree(d_output);
#endif
}

// ============================================================================
// Element-wise Multiplication Tests
// ============================================================================

TEST_F(QuotientCoVerifyTest, ElementwiseMul) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1024;
    auto a = random_xfield_vector(n);
    auto b = random_xfield_vector(n);
    
    // CPU
    std::vector<XFieldElement> cpu_result(n);
    for (size_t i = 0; i < n; ++i) {
        cpu_result[i] = a[i] * b[i];
    }
    
    // GPU
    auto a_u64 = xfe_to_u64(a);
    auto b_u64 = xfe_to_u64(b);
    std::vector<uint64_t> gpu_result_u64(n * 3);
    
    uint64_t *d_a, *d_b, *d_output;
    cudaMalloc(&d_a, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_b, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_output, n * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfe_elementwise_mul_gpu(d_a, d_b, d_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_xfe(gpu_result_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " elementwise mul mismatches";
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);
#endif
}

TEST_F(QuotientCoVerifyTest, ScaleByInv) {
#ifdef TRITON_CUDA_ENABLED
    const size_t n = 1024;
    auto xfe_values = random_xfield_vector(n);
    auto bfe_inv = random_bfield_vector(n);
    
    // CPU: scale each XFE by corresponding BFE
    std::vector<XFieldElement> cpu_result(n);
    for (size_t i = 0; i < n; ++i) {
        cpu_result[i] = xfe_values[i] * XFieldElement(bfe_inv[i]);
    }
    
    // GPU
    auto xfe_u64 = xfe_to_u64(xfe_values);
    auto bfe_u64 = bfe_to_u64(bfe_inv);
    std::vector<uint64_t> gpu_result_u64(n * 3);
    
    uint64_t *d_xfe, *d_bfe, *d_output;
    cudaMalloc(&d_xfe, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_bfe, n * sizeof(uint64_t));
    cudaMalloc(&d_output, n * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_xfe, xfe_u64.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bfe, bfe_u64.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfe_scale_by_inv_gpu(d_xfe, d_bfe, d_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_xfe(gpu_result_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " scale by inv mismatches";
    
    cudaFree(d_xfe);
    cudaFree(d_bfe);
    cudaFree(d_output);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(QuotientCoVerifyTest, Benchmark) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 16384;
    const size_t num_constraints = 100;
    
    std::vector<XFieldElement> all_values(num_rows * num_constraints);
    for (auto& v : all_values) v = random_xfield();
    auto weights = random_xfield_vector(num_constraints);
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<XFieldElement> cpu_result(num_rows);
    for (size_t row = 0; row < num_rows; ++row) {
        XFieldElement sum = XFieldElement::zero();
        for (size_t i = 0; i < num_constraints; ++i) {
            sum = sum + all_values[row * num_constraints + i] * weights[i];
        }
        cpu_result[row] = sum;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    auto values_u64 = xfe_to_u64(all_values);
    auto weights_u64 = xfe_to_u64(weights);
    std::vector<uint64_t> gpu_result_u64(num_rows * 3);
    
    uint64_t *d_values, *d_weights, *d_output;
    cudaMalloc(&d_values, num_rows * num_constraints * 3 * sizeof(uint64_t));
    cudaMalloc(&d_weights, num_constraints * 3 * sizeof(uint64_t));
    cudaMalloc(&d_output, num_rows * 3 * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_values, values_u64.data(), values_u64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_u64.data(), weights_u64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfe_weighted_sum_batch_gpu(d_values, d_weights, d_output, num_rows, num_constraints);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, num_rows * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] Quotient weighted sum (" << num_rows << " rows x " 
              << num_constraints << " constraints):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify correctness
    auto gpu_result = u64_to_xfe(gpu_result_u64);
    size_t mismatches = 0;
    for (size_t i = 0; i < num_rows; ++i) {
        if (cpu_result[i] != gpu_result[i]) mismatches++;
    }
    EXPECT_EQ(mismatches, 0) << "Benchmark has mismatches";
    
    cudaFree(d_values);
    cudaFree(d_weights);
    cudaFree(d_output);
#endif
}

} // namespace co_verify
} // namespace triton_vm

