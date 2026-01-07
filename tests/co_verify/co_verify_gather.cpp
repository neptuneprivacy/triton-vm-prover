/**
 * Gather Co-Verification Tests
 * 
 * Compares CPU gather operations with GPU CUDA kernels.
 */

#include "co_verify_framework.hpp"
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/gather_kernel.cuh"
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <set>

namespace triton_vm {
namespace co_verify {

class GatherCoVerifyTest : public ::testing::Test {
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
    
    XFieldElement random_xfield() {
        return XFieldElement(random_bfield(), random_bfield(), random_bfield());
    }
    
    std::vector<size_t> random_indices(size_t num_indices, size_t max_val) {
        std::vector<size_t> indices(num_indices);
        std::uniform_int_distribution<size_t> dist(0, max_val - 1);
        for (auto& idx : indices) {
            idx = dist(rng_);
        }
        return indices;
    }
    
    std::vector<uint64_t> bfe_to_u64(const std::vector<BFieldElement>& v) {
        std::vector<uint64_t> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i].value();
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
};

// ============================================================================
// BField Gather Tests
// ============================================================================

TEST_F(GatherCoVerifyTest, GatherColumn_BField) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 1024;
    const size_t num_indices = 160;  // Typical FRI query count
    
    // Create column
    std::vector<BFieldElement> column(num_rows);
    for (auto& v : column) v = random_bfield();
    
    // Create random indices
    auto indices = random_indices(num_indices, num_rows);
    
    // CPU gather
    std::vector<BFieldElement> cpu_result(num_indices);
    for (size_t i = 0; i < num_indices; ++i) {
        cpu_result[i] = column[indices[i]];
    }
    
    // GPU gather
    auto column_u64 = bfe_to_u64(column);
    std::vector<uint64_t> gpu_result_u64(num_indices);
    
    uint64_t* d_column;
    size_t* d_indices;
    uint64_t* d_output;
    
    cudaMalloc(&d_column, num_rows * sizeof(uint64_t));
    cudaMalloc(&d_indices, num_indices * sizeof(size_t));
    cudaMalloc(&d_output, num_indices * sizeof(uint64_t));
    
    cudaMemcpy(d_column, column_u64.data(), num_rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), num_indices * sizeof(size_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::gather_column_gpu(d_column, d_indices, d_output, num_rows, num_indices);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, num_indices * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_bfe(gpu_result_u64);
    
    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < num_indices; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
            if (mismatches <= 3) {
                std::cout << "Mismatch at " << i << " (idx=" << indices[i] << "): CPU=" 
                          << cpu_result[i] << ", GPU=" << gpu_result[i] << std::endl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " column gather mismatches";
    
    cudaFree(d_column);
    cudaFree(d_indices);
    cudaFree(d_output);
#endif
}

TEST_F(GatherCoVerifyTest, GatherRows_BField) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 512;
    const size_t row_width = 64;
    const size_t num_indices = 100;
    
    // Create table (row-major)
    std::vector<BFieldElement> table(num_rows * row_width);
    for (auto& v : table) v = random_bfield();
    
    auto indices = random_indices(num_indices, num_rows);
    
    // CPU gather
    std::vector<BFieldElement> cpu_result(num_indices * row_width);
    for (size_t i = 0; i < num_indices; ++i) {
        for (size_t j = 0; j < row_width; ++j) {
            cpu_result[i * row_width + j] = table[indices[i] * row_width + j];
        }
    }
    
    // GPU gather
    auto table_u64 = bfe_to_u64(table);
    std::vector<uint64_t> gpu_result_u64(num_indices * row_width);
    
    uint64_t* d_table;
    size_t* d_indices;
    uint64_t* d_output;
    
    cudaMalloc(&d_table, num_rows * row_width * sizeof(uint64_t));
    cudaMalloc(&d_indices, num_indices * sizeof(size_t));
    cudaMalloc(&d_output, num_indices * row_width * sizeof(uint64_t));
    
    cudaMemcpy(d_table, table_u64.data(), num_rows * row_width * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), num_indices * sizeof(size_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::gather_bfield_rows_gpu(d_table, d_indices, d_output, num_rows, row_width, num_indices);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, num_indices * row_width * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_bfe(gpu_result_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " row gather mismatches";
    
    cudaFree(d_table);
    cudaFree(d_indices);
    cudaFree(d_output);
#endif
}

// ============================================================================
// XField Gather Tests
// ============================================================================

TEST_F(GatherCoVerifyTest, GatherColumn_XField) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 1024;
    const size_t num_indices = 160;
    
    std::vector<XFieldElement> column(num_rows);
    for (auto& v : column) v = random_xfield();
    
    auto indices = random_indices(num_indices, num_rows);
    
    // CPU gather
    std::vector<XFieldElement> cpu_result(num_indices);
    for (size_t i = 0; i < num_indices; ++i) {
        cpu_result[i] = column[indices[i]];
    }
    
    // GPU gather
    auto column_u64 = xfe_to_u64(column);
    std::vector<uint64_t> gpu_result_u64(num_indices * 3);
    
    uint64_t* d_column;
    size_t* d_indices;
    uint64_t* d_output;
    
    cudaMalloc(&d_column, num_rows * 3 * sizeof(uint64_t));
    cudaMalloc(&d_indices, num_indices * sizeof(size_t));
    cudaMalloc(&d_output, num_indices * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_column, column_u64.data(), num_rows * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), num_indices * sizeof(size_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::gather_xfield_column_gpu(d_column, d_indices, d_output, num_rows, num_indices);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, num_indices * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_xfe(gpu_result_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < num_indices; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " XField column gather mismatches";
    
    cudaFree(d_column);
    cudaFree(d_indices);
    cudaFree(d_output);
#endif
}

TEST_F(GatherCoVerifyTest, GatherRows_XField) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 256;
    const size_t row_width = 32;  // XFEs per row
    const size_t num_indices = 80;
    
    std::vector<XFieldElement> table(num_rows * row_width);
    for (auto& v : table) v = random_xfield();
    
    auto indices = random_indices(num_indices, num_rows);
    
    // CPU gather
    std::vector<XFieldElement> cpu_result(num_indices * row_width);
    for (size_t i = 0; i < num_indices; ++i) {
        for (size_t j = 0; j < row_width; ++j) {
            cpu_result[i * row_width + j] = table[indices[i] * row_width + j];
        }
    }
    
    // GPU gather
    auto table_u64 = xfe_to_u64(table);
    std::vector<uint64_t> gpu_result_u64(num_indices * row_width * 3);
    
    uint64_t* d_table;
    size_t* d_indices;
    uint64_t* d_output;
    
    cudaMalloc(&d_table, num_rows * row_width * 3 * sizeof(uint64_t));
    cudaMalloc(&d_indices, num_indices * sizeof(size_t));
    cudaMalloc(&d_output, num_indices * row_width * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_table, table_u64.data(), num_rows * row_width * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), num_indices * sizeof(size_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::gather_xfield_rows_gpu(d_table, d_indices, d_output, num_rows, row_width, num_indices);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, num_indices * row_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_result = u64_to_xfe(gpu_result_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " XField row gather mismatches";
    
    cudaFree(d_table);
    cudaFree(d_indices);
    cudaFree(d_output);
#endif
}

// ============================================================================
// Scatter Test (inverse of gather)
// ============================================================================

TEST_F(GatherCoVerifyTest, Scatter_BField) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 512;
    const size_t row_width = 32;
    const size_t num_indices = 50;
    
    // Create empty table
    std::vector<BFieldElement> table(num_rows * row_width, BFieldElement::zero());
    
    // Create values to scatter
    std::vector<BFieldElement> values(num_indices * row_width);
    for (auto& v : values) v = random_bfield();
    
    // Create unique indices for scatter
    std::vector<size_t> indices;
    std::set<size_t> used;
    while (indices.size() < num_indices) {
        size_t idx = rng_() % num_rows;
        if (used.find(idx) == used.end()) {
            indices.push_back(idx);
            used.insert(idx);
        }
    }
    
    // CPU scatter
    std::vector<BFieldElement> cpu_table = table;
    for (size_t i = 0; i < num_indices; ++i) {
        for (size_t j = 0; j < row_width; ++j) {
            cpu_table[indices[i] * row_width + j] = values[i * row_width + j];
        }
    }
    
    // GPU scatter
    auto table_u64 = bfe_to_u64(table);
    auto values_u64 = bfe_to_u64(values);
    
    uint64_t* d_table;
    size_t* d_indices;
    uint64_t* d_values;
    
    cudaMalloc(&d_table, num_rows * row_width * sizeof(uint64_t));
    cudaMalloc(&d_indices, num_indices * sizeof(size_t));
    cudaMalloc(&d_values, num_indices * row_width * sizeof(uint64_t));
    
    cudaMemcpy(d_table, table_u64.data(), num_rows * row_width * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), num_indices * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values_u64.data(), num_indices * row_width * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::scatter_bfield_gpu(d_values, d_indices, d_table, num_rows, row_width, num_indices);
    cudaDeviceSynchronize();
    
    cudaMemcpy(table_u64.data(), d_table, num_rows * row_width * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto gpu_table = u64_to_bfe(table_u64);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < cpu_table.size(); ++i) {
        if (cpu_table[i] != gpu_table[i]) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " scatter mismatches";
    
    cudaFree(d_table);
    cudaFree(d_indices);
    cudaFree(d_values);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(GatherCoVerifyTest, Benchmark) {
#ifdef TRITON_CUDA_ENABLED
    const size_t num_rows = 1 << 16;  // 64K rows
    const size_t row_width = 256;     // 256 columns
    const size_t num_indices = 160;   // Typical FRI query count
    
    std::vector<BFieldElement> table(num_rows * row_width);
    for (auto& v : table) v = random_bfield();
    
    auto indices = random_indices(num_indices, num_rows);
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<BFieldElement> cpu_result(num_indices * row_width);
    for (size_t i = 0; i < num_indices; ++i) {
        for (size_t j = 0; j < row_width; ++j) {
            cpu_result[i * row_width + j] = table[indices[i] * row_width + j];
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    auto table_u64 = bfe_to_u64(table);
    std::vector<uint64_t> gpu_result_u64(num_indices * row_width);
    
    uint64_t* d_table;
    size_t* d_indices;
    uint64_t* d_output;
    
    cudaMalloc(&d_table, num_rows * row_width * sizeof(uint64_t));
    cudaMalloc(&d_indices, num_indices * sizeof(size_t));
    cudaMalloc(&d_output, num_indices * row_width * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_table, table_u64.data(), num_rows * row_width * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), num_indices * sizeof(size_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::gather_bfield_rows_gpu(d_table, d_indices, d_output, num_rows, row_width, num_indices);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_result_u64.data(), d_output, num_indices * row_width * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] Gather (" << num_indices << " rows x " << row_width 
              << " cols from " << num_rows << " row table):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify
    auto gpu_result = u64_to_bfe(gpu_result_u64);
    size_t mismatches = 0;
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        if (cpu_result[i] != gpu_result[i]) mismatches++;
    }
    EXPECT_EQ(mismatches, 0) << "Benchmark has mismatches";
    
    cudaFree(d_table);
    cudaFree(d_indices);
    cudaFree(d_output);
#endif
}

} // namespace co_verify
} // namespace triton_vm

