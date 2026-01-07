/**
 * GPU Proof Integration Test
 * 
 * End-to-end test that runs proof generation steps on GPU and compares
 * with CPU at each checkpoint.
 * 
 * Uses spin_input8.tasm for testing.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "hash/tip5.hpp"
#include "ntt/ntt.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/lde_kernel.cuh"
#include "gpu/kernels/merkle_kernel.cuh"
#include "gpu/kernels/tip5_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include "gpu/kernels/extend_kernel.cuh"
#include <cuda_runtime.h>
#endif

// External FFI functions from Rust
extern "C" {
    int tvm_trace_and_pad_main_table_from_tasm_file(
        const char* tasm_path,
        const uint64_t* public_input,
        size_t public_input_len,
        uint64_t** out_flat_table,
        size_t* out_flat_table_len,
        size_t* out_num_rows,
        size_t* out_num_cols,
        uint64_t out_program_digest[5],
        uint32_t* out_version,
        uint64_t** out_claim_output_ptr,
        size_t* out_claim_output_len,
        uint64_t out_trace_domain[3],
        uint64_t out_quotient_domain[3],
        uint64_t out_fri_domain[3]
    );
    
    void tvm_main_table_free(uint64_t* flat_table, size_t flat_table_len);
    void tvm_claim_output_free(uint64_t* claim_output, size_t claim_output_len);
}

using namespace triton_vm;

namespace {

// Helper to get elapsed milliseconds
template<typename Clock = std::chrono::high_resolution_clock>
double elapsed_ms(std::chrono::time_point<Clock> start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

class GpuProofIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        // Verify CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "  Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Compute: " << prop.major << "." << prop.minor << std::endl;
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
};

#ifdef TRITON_CUDA_ENABLED

TEST_F(GpuProofIntegrationTest, LoadTraceFromRust) {
    // Load spin_input8.tasm trace
    std::string tasm_path = "spin_input8.tasm";
    uint64_t input_value = 8;
    
    uint64_t* flat_table = nullptr;
    size_t flat_table_len = 0;
    size_t num_rows = 0;
    size_t num_cols = 0;
    uint64_t digest[5] = {0};
    uint32_t version = 0;
    uint64_t* claim_output = nullptr;
    size_t claim_output_len = 0;
    uint64_t trace_dom[3] = {0};
    uint64_t quot_dom[3] = {0};
    uint64_t fri_dom[3] = {0};
    
    std::cout << "Loading trace from " << tasm_path << " with input=" << input_value << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    int rc = tvm_trace_and_pad_main_table_from_tasm_file(
        tasm_path.c_str(),
        &input_value,
        1,
        &flat_table,
        &flat_table_len,
        &num_rows,
        &num_cols,
        digest,
        &version,
        &claim_output,
        &claim_output_len,
        trace_dom,
        quot_dom,
        fri_dom
    );
    double load_time = elapsed_ms(start);
    
    ASSERT_EQ(rc, 0) << "Failed to load trace from Rust FFI";
    ASSERT_GT(num_rows, 0);
    ASSERT_GT(num_cols, 0);
    ASSERT_EQ(flat_table_len, num_rows * num_cols);
    
    std::cout << "  Trace dimensions: " << num_rows << " x " << num_cols << std::endl;
    std::cout << "  Trace elements: " << flat_table_len << std::endl;
    std::cout << "  Load time: " << load_time << " ms" << std::endl;
    // Domain structure: [length, offset, ...]
    std::cout << "  Trace domain: length=" << trace_dom[0] 
              << ", offset=" << trace_dom[1] << std::endl;
    std::cout << "  FRI domain: length=" << fri_dom[0] 
              << ", offset=" << fri_dom[1] << std::endl;
    
    // Clean up
    if (claim_output) tvm_claim_output_free(claim_output, claim_output_len);
    if (flat_table) tvm_main_table_free(flat_table, flat_table_len);
}

TEST_F(GpuProofIntegrationTest, LDE_SingleColumn_Comparison) {
    // Load trace
    std::string tasm_path = "spin_input8.tasm";
    uint64_t input_value = 8;
    
    uint64_t* flat_table = nullptr;
    size_t flat_table_len = 0;
    size_t num_rows = 0;
    size_t num_cols = 0;
    uint64_t digest[5] = {0};
    uint32_t version = 0;
    uint64_t* claim_output = nullptr;
    size_t claim_output_len = 0;
    uint64_t trace_dom[3] = {0};
    uint64_t quot_dom[3] = {0};
    uint64_t fri_dom[3] = {0};
    
    int rc = tvm_trace_and_pad_main_table_from_tasm_file(
        tasm_path.c_str(), &input_value, 1, &flat_table, &flat_table_len,
        &num_rows, &num_cols, digest, &version, &claim_output, &claim_output_len,
        trace_dom, quot_dom, fri_dom
    );
    ASSERT_EQ(rc, 0);
    
    // Extract first column for testing
    // Domain structure: [length, offset, ...]
    size_t trace_len = num_rows;
    size_t extended_len = fri_dom[0];  // FRI domain length
    uint64_t trace_offset = trace_dom[1];
    uint64_t fri_offset = fri_dom[1];
    
    std::vector<uint64_t> column(trace_len);
    for (size_t r = 0; r < trace_len; ++r) {
        column[r] = flat_table[r * num_cols + 0];  // First column
    }
    
    std::cout << "LDE test: " << trace_len << " -> " << extended_len << std::endl;
    
    // CPU LDE
    std::vector<BFieldElement> cpu_column(trace_len);
    for (size_t i = 0; i < trace_len; ++i) {
        cpu_column[i] = BFieldElement(column[i]);
    }
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    // Interpolate (INTT)
    NTT::inverse(cpu_column);
    // Pad to extended length
    cpu_column.resize(extended_len, BFieldElement::zero());
    // Scale by coset offset
    BFieldElement scale = BFieldElement::one();
    BFieldElement offset_bfe(fri_offset);
    for (size_t i = 0; i < extended_len; ++i) {
        cpu_column[i] = cpu_column[i] * scale;
        scale = scale * offset_bfe;
    }
    // Evaluate (NTT)
    NTT::forward(cpu_column);
    double cpu_time = elapsed_ms(cpu_start);
    
    // GPU LDE
    uint64_t* d_trace;
    uint64_t* d_extended;
    cudaMalloc(&d_trace, trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, extended_len * sizeof(uint64_t));
    cudaMemcpy(d_trace, column.data(), trace_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::lde_column_gpu(d_trace, trace_len, d_extended, extended_len,
                                   trace_offset, fri_offset, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    // Download GPU result
    std::vector<uint64_t> gpu_result(extended_len);
    cudaMemcpy(gpu_result.data(), d_extended, extended_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < extended_len && mismatches < 10; ++i) {
        if (gpu_result[i] != cpu_column[i].value()) {
            std::cout << "  Mismatch at " << i << ": GPU=" << gpu_result[i]
                      << " CPU=" << cpu_column[i].value() << std::endl;
            ++mismatches;
        }
    }
    
    std::cout << "  CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "  GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
    
    EXPECT_EQ(mismatches, 0) << "LDE results don't match!";
    
    cudaFree(d_trace);
    cudaFree(d_extended);
    if (claim_output) tvm_claim_output_free(claim_output, claim_output_len);
    if (flat_table) tvm_main_table_free(flat_table, flat_table_len);
}

TEST_F(GpuProofIntegrationTest, LDE_AllColumns_Benchmark) {
    // Load trace
    std::string tasm_path = "spin_input8.tasm";
    uint64_t input_value = 8;
    
    uint64_t* flat_table = nullptr;
    size_t flat_table_len = 0;
    size_t num_rows = 0;
    size_t num_cols = 0;
    uint64_t digest[5] = {0};
    uint32_t version = 0;
    uint64_t* claim_output = nullptr;
    size_t claim_output_len = 0;
    uint64_t trace_dom[3] = {0};
    uint64_t quot_dom[3] = {0};
    uint64_t fri_dom[3] = {0};
    
    int rc = tvm_trace_and_pad_main_table_from_tasm_file(
        tasm_path.c_str(), &input_value, 1, &flat_table, &flat_table_len,
        &num_rows, &num_cols, digest, &version, &claim_output, &claim_output_len,
        trace_dom, quot_dom, fri_dom
    );
    ASSERT_EQ(rc, 0);
    
    // Domain structure: [length, offset, ...]
    size_t trace_len = num_rows;
    size_t extended_len = fri_dom[0];  // FRI domain length
    uint64_t trace_offset = trace_dom[1];
    uint64_t fri_offset = fri_dom[1];
    
    std::cout << "\nLDE Benchmark: " << num_cols << " columns, " 
              << trace_len << " -> " << extended_len << std::endl;
    
    // Allocate GPU memory
    uint64_t* d_traces;
    uint64_t* d_extended;
    cudaMalloc(&d_traces, num_cols * trace_len * sizeof(uint64_t));
    cudaMalloc(&d_extended, num_cols * extended_len * sizeof(uint64_t));
    
    // Upload all columns (transposed to column-major)
    std::vector<uint64_t> columns(num_cols * trace_len);
    for (size_t c = 0; c < num_cols; ++c) {
        for (size_t r = 0; r < trace_len; ++r) {
            columns[c * trace_len + r] = flat_table[r * num_cols + c];
        }
    }
    
    auto upload_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_traces, columns.data(), num_cols * trace_len * sizeof(uint64_t), 
               cudaMemcpyHostToDevice);
    double upload_time = elapsed_ms(upload_start);
    
    // GPU LDE for all columns
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::lde_batch_gpu(d_traces, num_cols, trace_len, d_extended, extended_len,
                                  trace_offset, fri_offset, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    // Report
    std::cout << "  Upload time: " << upload_time << " ms" << std::endl;
    std::cout << "  GPU LDE time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Total GPU time: " << (upload_time + gpu_time) << " ms" << std::endl;
    std::cout << "  Throughput: " << (num_cols * trace_len / (gpu_time / 1000.0) / 1e6) 
              << " M elements/sec" << std::endl;
    
    cudaFree(d_traces);
    cudaFree(d_extended);
    if (claim_output) tvm_claim_output_free(claim_output, claim_output_len);
    if (flat_table) tvm_main_table_free(flat_table, flat_table_len);
}

TEST_F(GpuProofIntegrationTest, MerkleTree_Comparison) {
    // Create test data: 1024 random digests
    size_t num_leaves = 1024;
    
    std::vector<Digest> leaves(num_leaves);
    std::mt19937_64 rng(12345);
    for (auto& d : leaves) {
        for (size_t i = 0; i < 5; ++i) {
            d[i] = BFieldElement(rng() % BFieldElement::MODULUS);
        }
    }
    
    std::cout << "\nMerkle Tree Test: " << num_leaves << " leaves" << std::endl;
    
    // CPU Merkle root
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<Digest> current = leaves;
    while (current.size() > 1) {
        std::vector<Digest> next(current.size() / 2);
        for (size_t i = 0; i < next.size(); ++i) {
            next[i] = Tip5::hash_pair(current[2*i], current[2*i + 1]);
        }
        current = std::move(next);
    }
    Digest cpu_root = current[0];
    double cpu_time = elapsed_ms(cpu_start);
    
    // GPU Merkle root
    uint64_t* d_leaves;
    uint64_t* d_root;
    cudaMalloc(&d_leaves, num_leaves * 5 * sizeof(uint64_t));
    cudaMalloc(&d_root, 5 * sizeof(uint64_t));
    
    // Flatten leaves
    std::vector<uint64_t> flat_leaves(num_leaves * 5);
    for (size_t i = 0; i < num_leaves; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            flat_leaves[i * 5 + j] = leaves[i][j].value();
        }
    }
    cudaMemcpy(d_leaves, flat_leaves.data(), num_leaves * 5 * sizeof(uint64_t), 
               cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu::kernels::merkle_root_gpu(d_leaves, d_root, num_leaves, 0);
    cudaDeviceSynchronize();
    double gpu_time = elapsed_ms(gpu_start);
    
    // Download GPU root
    uint64_t gpu_root_data[5];
    cudaMemcpy(gpu_root_data, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    Digest gpu_root;
    for (size_t i = 0; i < 5; ++i) {
        gpu_root[i] = BFieldElement(gpu_root_data[i]);
    }
    
    // Compare
    bool match = true;
    for (size_t i = 0; i < 5; ++i) {
        if (cpu_root[i].value() != gpu_root[i].value()) {
            match = false;
            std::cout << "  Mismatch at " << i << ": CPU=" << cpu_root[i].value()
                      << " GPU=" << gpu_root[i].value() << std::endl;
        }
    }
    
    std::cout << "  CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "  GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
    
    EXPECT_TRUE(match) << "Merkle roots don't match!";
    
    cudaFree(d_leaves);
    cudaFree(d_root);
}

TEST_F(GpuProofIntegrationTest, FullPipeline_Timing) {
    // Load trace
    std::string tasm_path = "spin_input8.tasm";
    uint64_t input_value = 8;
    
    uint64_t* flat_table = nullptr;
    size_t flat_table_len = 0;
    size_t num_rows = 0;
    size_t num_cols = 0;
    uint64_t digest[5] = {0};
    uint32_t version = 0;
    uint64_t* claim_output = nullptr;
    size_t claim_output_len = 0;
    uint64_t trace_dom[3] = {0};
    uint64_t quot_dom[3] = {0};
    uint64_t fri_dom[3] = {0};
    
    int rc = tvm_trace_and_pad_main_table_from_tasm_file(
        tasm_path.c_str(), &input_value, 1, &flat_table, &flat_table_len,
        &num_rows, &num_cols, digest, &version, &claim_output, &claim_output_len,
        trace_dom, quot_dom, fri_dom
    );
    ASSERT_EQ(rc, 0);
    
    // Domain structure: [length, offset, ...]
    size_t trace_len = num_rows;
    size_t extended_len = fri_dom[0];  // FRI domain length
    uint64_t trace_offset = trace_dom[1];
    uint64_t fri_offset = fri_dom[1];
    
    std::cout << "\n=== Full Pipeline Timing ===" << std::endl;
    std::cout << "Trace: " << num_rows << " x " << num_cols << std::endl;
    std::cout << "Extended: " << extended_len << " x " << num_cols << std::endl;
    
    // Allocate GPU memory
    size_t trace_bytes = num_cols * trace_len * sizeof(uint64_t);
    size_t extended_bytes = num_cols * extended_len * sizeof(uint64_t);
    size_t digest_bytes = extended_len * 5 * sizeof(uint64_t);
    
    std::cout << "\nMemory allocation:" << std::endl;
    std::cout << "  Trace table: " << (trace_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Extended table: " << (extended_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Digests: " << (digest_bytes / (1024 * 1024)) << " MB" << std::endl;
    
    uint64_t* d_traces;
    uint64_t* d_extended;
    uint64_t* d_digests;
    uint64_t* d_root;
    
    cudaMalloc(&d_traces, trace_bytes);
    cudaMalloc(&d_extended, extended_bytes);
    cudaMalloc(&d_digests, digest_bytes);
    cudaMalloc(&d_root, 5 * sizeof(uint64_t));
    
    // Transpose to column-major
    std::vector<uint64_t> columns(num_cols * trace_len);
    for (size_t c = 0; c < num_cols; ++c) {
        for (size_t r = 0; r < trace_len; ++r) {
            columns[c * trace_len + r] = flat_table[r * num_cols + c];
        }
    }
    
    std::cout << "\n=== GPU Pipeline Execution ===" << std::endl;
    
    // Step 1: Upload
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_traces, columns.data(), trace_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    double upload_time = elapsed_ms(t1);
    std::cout << "1. H2D Upload: " << upload_time << " ms" << std::endl;
    
    // Step 2: LDE all columns
    auto t2 = std::chrono::high_resolution_clock::now();
    gpu::kernels::lde_batch_gpu(d_traces, num_cols, trace_len, d_extended, extended_len,
                                  trace_offset, fri_offset, 0);
    cudaDeviceSynchronize();
    double lde_time = elapsed_ms(t2);
    std::cout << "2. LDE (" << num_cols << " columns): " << lde_time << " ms" << std::endl;
    
    // Step 3: Merkle tree of FRI domain
    auto t3 = std::chrono::high_resolution_clock::now();
    // We need row digests first - for now just time the merkle root on digest data
    // Initialize digest buffer with dummy data for timing
    cudaMemset(d_digests, 0x42, digest_bytes);
    gpu::kernels::merkle_root_gpu(d_digests, d_root, extended_len, 0);
    cudaDeviceSynchronize();
    double merkle_time = elapsed_ms(t3);
    std::cout << "3. Merkle tree: " << merkle_time << " ms" << std::endl;
    
    double total_gpu = upload_time + lde_time + merkle_time;
    std::cout << "\nTotal GPU time (main table steps): " << total_gpu << " ms" << std::endl;
    
    cudaFree(d_traces);
    cudaFree(d_extended);
    cudaFree(d_digests);
    cudaFree(d_root);
    
    if (claim_output) tvm_claim_output_free(claim_output, claim_output_len);
    if (flat_table) tvm_main_table_free(flat_table, flat_table_len);
}

TEST_F(GpuProofIntegrationTest, AuxTableExtension_Benchmark) {
    // Load trace
    std::string tasm_path = "spin_input8.tasm";
    uint64_t input_value = 8;
    
    uint64_t* flat_table = nullptr;
    size_t flat_table_len = 0;
    size_t num_rows = 0;
    size_t num_cols = 0;
    uint64_t digest[5] = {0};
    uint32_t version = 0;
    uint64_t* claim_output = nullptr;
    size_t claim_output_len = 0;
    uint64_t trace_dom[3] = {0};
    uint64_t quot_dom[3] = {0};
    uint64_t fri_dom[3] = {0};
    
    int rc = tvm_trace_and_pad_main_table_from_tasm_file(
        tasm_path.c_str(), &input_value, 1, &flat_table, &flat_table_len,
        &num_rows, &num_cols, digest, &version, &claim_output, &claim_output_len,
        trace_dom, quot_dom, fri_dom
    );
    ASSERT_EQ(rc, 0);
    
    std::cout << "\n=== Aux Table Extension Benchmark ===" << std::endl;
    std::cout << "Trace: " << num_rows << " rows x " << num_cols << " cols" << std::endl;
    std::cout << "Aux table: " << num_rows << " rows x 88 XFE cols" << std::endl;
    
    // For now, just benchmark the GPU primitives used in aux extension
    size_t n = num_rows;
    
    // Allocate GPU memory for XFE operations
    uint64_t* d_xfes;
    uint64_t* d_inverses;
    cudaMalloc(&d_xfes, n * 3 * sizeof(uint64_t));
    cudaMalloc(&d_inverses, n * 3 * sizeof(uint64_t));
    
    // Initialize with random data
    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_xfes(n * 3);
    for (size_t i = 0; i < n; ++i) {
        h_xfes[i * 3] = 1 + (rng() % (BFieldElement::MODULUS - 1));  // Non-zero
        h_xfes[i * 3 + 1] = rng() % BFieldElement::MODULUS;
        h_xfes[i * 3 + 2] = rng() % BFieldElement::MODULUS;
    }
    cudaMemcpy(d_xfes, h_xfes.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Benchmark batch inverse (key operation for log derivatives)
    auto t1 = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_batch_inverse_gpu(d_xfes, d_inverses, n, 0);
    cudaDeviceSynchronize();
    double inverse_time = elapsed_ms(t1);
    
    // Benchmark prefix sum (for accumulating log derivatives)
    cudaMemcpy(d_xfes, h_xfes.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    auto t2 = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_prefix_sum_gpu(d_xfes, n, 0);
    cudaDeviceSynchronize();
    double sum_time = elapsed_ms(t2);
    
    // Benchmark prefix product (for running products)
    cudaMemcpy(d_xfes, h_xfes.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    auto t3 = std::chrono::high_resolution_clock::now();
    gpu::kernels::xfe_prefix_product_gpu(d_xfes, n, 0);
    cudaDeviceSynchronize();
    double product_time = elapsed_ms(t3);
    
    std::cout << "\nGPU Aux Extension Primitives (n=" << n << "):" << std::endl;
    std::cout << "  Batch inverse: " << inverse_time << " ms" << std::endl;
    std::cout << "  Prefix sum: " << sum_time << " ms" << std::endl;
    std::cout << "  Prefix product: " << product_time << " ms" << std::endl;
    
    // Estimate total aux extension time
    // Aux table has 9 sub-tables, each needs ~2 columns of log derivs/products
    // Total ~18 prefix operations + batch inverses for each
    double estimated_gpu_time = 9 * (inverse_time + sum_time);
    std::cout << "\nEstimated GPU aux extension time: " << estimated_gpu_time << " ms" << std::endl;
    
    cudaFree(d_xfes);
    cudaFree(d_inverses);
    
    if (claim_output) tvm_claim_output_free(claim_output, claim_output_len);
    if (flat_table) tvm_main_table_free(flat_table, flat_table_len);
}

#endif // TRITON_CUDA_ENABLED
