/**
 * GPU-Accelerated Triton VM Prover
 * 
 * Uses GPU for:
 * - LDE (Low-Degree Extension) of tables
 * - Merkle tree construction
 * - FRI folding
 * 
 * Uses CPU for:
 * - Aux table extension (constraint-specific logic)
 * - Fiat-Shamir transcript
 * 
 * Uses Rust FFI for:
 * - Tracing and padding
 * - Proof encoding
 */

#include "common/debug_control.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "hash/tip5.hpp"
#include "ntt/ntt.hpp"
#include "table/master_table.hpp"
#include "table/table_commitment.hpp"
#include "stark.hpp"
#include "proof_stream/proof_stream.hpp"
#include "stark/challenges.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/lde_kernel.cuh"
#include "gpu/kernels/randomized_lde_kernel.cuh"
#include "gpu/kernels/merkle_kernel.cuh"
#include "gpu/kernels/tip5_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include "gpu/kernels/row_hash_kernel.cuh"
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
    
    int proof_stream_encode_and_serialize(
        const char* json_proof_items,
        size_t json_len,
        const char* output_path
    );
}

using namespace triton_vm;

namespace {

template<typename Clock = std::chrono::high_resolution_clock>
double elapsed_ms(std::chrono::time_point<Clock> start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<uint64_t> parse_u64_list(const std::string& s) {
    std::vector<uint64_t> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t'))
            token.erase(0, 1);
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t'))
            token.pop_back();
        if (token.empty()) continue;
        uint64_t val = 0;
        if (token.rfind("0x", 0) == 0 || token.rfind("0X", 0) == 0)
            val = std::stoull(token, nullptr, 16);
        else
            val = std::stoull(token, nullptr, 10);
        result.push_back(val);
    }
    return result;
}

}  // namespace

int main(int argc, char* argv[]) {
    // Configure OpenMP threads for maximum CPU utilization
    // Threadripper 9995WX: 96 cores, 192 threads (Zen 5)
    // Default to 96 threads to match physical cores if OMP_NUM_THREADS is not set
#ifdef _OPENMP
    const char* omp_threads_env = std::getenv("OMP_NUM_THREADS");
    int target_threads = 96;
    if (omp_threads_env) {
        target_threads = std::atoi(omp_threads_env);
    }
    omp_set_num_threads(target_threads);
    
    // Verify OpenMP is actually working
    int max_threads = omp_get_max_threads();
    int num_procs = omp_get_num_procs();
    std::cout << "OpenMP Configuration:" << std::endl;
    std::cout << "  Requested threads: " << target_threads << std::endl;
    std::cout << "  Max threads available: " << max_threads << std::endl;
    std::cout << "  Processors detected: " << num_procs << std::endl;
    std::cout << "  OpenMP version: " << _OPENMP << std::endl;
    if (max_threads != target_threads) {
        std::cout << "  WARNING: Requested " << target_threads << " but only " << max_threads << " available!" << std::endl;
    }
#else
    std::cout << "WARNING: OpenMP is not available - CPU parallelization disabled!" << std::endl;
#endif

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <program.tasm> <public_input> <output_claim> <output_proof>\n";
        return 1;
    }
    
    std::string tasm_path = argv[1];
    std::string public_input_str = argv[2];
    std::string output_claim = argv[3];
    std::string output_proof = argv[4];
    
    std::vector<uint64_t> public_input = parse_u64_list(public_input_str);
    
#ifndef TRITON_CUDA_ENABLED
    std::cerr << "Error: CUDA not enabled. Build with -DENABLE_CUDA=ON\n";
    return 1;
#else
    // Check CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found\n";
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "=== GPU-Accelerated Triton VM Prover ===" << std::endl;
    std::cout << "GPU: " << prop.name << " (" << (prop.totalGlobalMem / (1024*1024)) << " MB)" << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        // =====================================================================
        // Step 1: Load trace from Rust FFI
        // =====================================================================
        auto step1_start = std::chrono::high_resolution_clock::now();
        
        uint64_t* flat_table = nullptr;
        size_t flat_table_len = 0;
        size_t num_rows = 0;
        size_t num_cols = 0;
        uint64_t digest_arr[5] = {0};
        uint32_t version = 0;
        uint64_t* claim_output_ptr = nullptr;
        size_t claim_output_len = 0;
        uint64_t trace_dom[3] = {0};
        uint64_t quot_dom[3] = {0};
        uint64_t fri_dom[3] = {0};
        
        uint64_t dummy_input = 0;
        const uint64_t* input_ptr = public_input.empty() ? &dummy_input : public_input.data();
        
        int rc = tvm_trace_and_pad_main_table_from_tasm_file(
            tasm_path.c_str(),
            input_ptr,
            public_input.size(),
            &flat_table,
            &flat_table_len,
            &num_rows,
            &num_cols,
            digest_arr,
            &version,
            &claim_output_ptr,
            &claim_output_len,
            trace_dom,
            quot_dom,
            fri_dom
        );
        
        if (rc != 0) {
            std::cerr << "Error: Rust FFI failed\n";
            return 1;
        }
        
        double step1_time = elapsed_ms(step1_start);
        std::cout << "\n[Step 1] Load trace: " << step1_time << " ms" << std::endl;
        std::cout << "  Trace: " << num_rows << " x " << num_cols << std::endl;
        
        // Domain info
        size_t trace_len = trace_dom[0];
        size_t fri_len = fri_dom[0];
        uint64_t trace_offset = trace_dom[1];
        uint64_t fri_offset = fri_dom[1];
        
        std::cout << "  FRI domain: " << fri_len << " points" << std::endl;
        
        // =====================================================================
        // Step 2: GPU Randomized LDE of main table
        // =====================================================================
        auto step2_start = std::chrono::high_resolution_clock::now();
        
        // Create main table first to get randomizer coefficients
        ArithmeticDomain trace_domain_tmp = ArithmeticDomain::of_length(trace_len)
            .with_offset(BFieldElement(trace_offset));
        ArithmeticDomain quotient_domain_tmp = ArithmeticDomain::of_length(quot_dom[0])
            .with_offset(BFieldElement(quot_dom[1]));
        ArithmeticDomain fri_domain_tmp = ArithmeticDomain::of_length(fri_len)
            .with_offset(BFieldElement(fri_offset));
        
        MasterMainTable main_table_for_rand(num_rows, num_cols, trace_domain_tmp, quotient_domain_tmp, fri_domain_tmp);
        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                main_table_for_rand.set(r, c, BFieldElement(flat_table[r * num_cols + c]));
            }
        }
        
        // Set trace randomizers
        auto stark = Stark::default_stark();
        main_table_for_rand.set_num_trace_randomizers(stark.num_trace_randomizers());
        
        // Get randomizer coefficients for all columns
        size_t randomizer_len = stark.num_trace_randomizers();
        std::vector<uint64_t> all_randomizers(num_cols * randomizer_len);
        for (size_t c = 0; c < num_cols; ++c) {
            auto coeffs = main_table_for_rand.trace_randomizer_for_column(c);
            for (size_t i = 0; i < coeffs.size() && i < randomizer_len; ++i) {
                all_randomizers[c * randomizer_len + i] = coeffs[i].value();
            }
        }
        
        // Allocate GPU memory
        size_t trace_bytes = num_cols * num_rows * sizeof(uint64_t);
        size_t lde_bytes = num_cols * fri_len * sizeof(uint64_t);
        size_t rand_bytes = num_cols * randomizer_len * sizeof(uint64_t);
        
        uint64_t* d_trace;
        uint64_t* d_lde;
        uint64_t* d_randomizers;
        cudaMalloc(&d_trace, trace_bytes);
        cudaMalloc(&d_lde, lde_bytes);
        cudaMalloc(&d_randomizers, rand_bytes);
        
        // Transpose to column-major for GPU
        std::vector<uint64_t> columns(num_cols * num_rows);
        for (size_t c = 0; c < num_cols; ++c) {
            for (size_t r = 0; r < num_rows; ++r) {
                columns[c * num_rows + r] = flat_table[r * num_cols + c];
            }
        }
        
        // Upload trace, randomizers
        cudaMemcpy(d_trace, columns.data(), trace_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_randomizers, all_randomizers.data(), rand_bytes, cudaMemcpyHostToDevice);
        
        // GPU Randomized LDE
        gpu::kernels::randomized_lde_batch_gpu(
            d_trace, num_cols, num_rows,
            d_randomizers, randomizer_len,
            trace_offset, fri_offset, fri_len,
            d_lde, 0
        );
        cudaDeviceSynchronize();
        
        double step2_time = elapsed_ms(step2_start);
        std::cout << "\n[Step 2] GPU Randomized LDE main table: " << step2_time << " ms" << std::endl;
        
        // Download LDE result for CPU processing
        std::vector<uint64_t> lde_data(num_cols * fri_len);
        cudaMemcpy(lde_data.data(), d_lde, lde_bytes, cudaMemcpyDeviceToHost);
        
        cudaFree(d_randomizers);
        
        // =====================================================================
        // Step 3: GPU Row Hashing + GPU Merkle tree for main table
        // =====================================================================
        auto step3_start = std::chrono::high_resolution_clock::now();
        
        // GPU row hashing (LDE data is already column-major on GPU from step 2)
        uint64_t* d_digests;
        cudaMalloc(&d_digests, fri_len * 5 * sizeof(uint64_t));
        
        // Hash all rows on GPU using Tip5 sponge
        std::cout << "  Starting GPU row hashing: " << fri_len << " rows x " << num_cols << " cols" << std::endl;
        auto hash_start = std::chrono::high_resolution_clock::now();
        gpu::kernels::hash_bfield_rows_gpu(d_lde, fri_len, num_cols, d_digests, 0);
        cudaDeviceSynchronize();
        auto step3_hash_time = elapsed_ms(hash_start);
        std::cout << "  GPU row hashing complete: " << step3_hash_time << " ms" << std::endl;
        
        // Download digests for CPU Merkle tree building (faster than GPU Merkle)
        std::vector<uint64_t> digests_flat(fri_len * 5);
        cudaMemcpy(digests_flat.data(), d_digests, fri_len * 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::vector<Digest> main_fri_digests;
        main_fri_digests.reserve(fri_len);
        for (size_t i = 0; i < fri_len; ++i) {
            Digest d;
            for (size_t j = 0; j < 5; ++j) {
                d[j] = BFieldElement(digests_flat[i * 5 + j]);
            }
            main_fri_digests.push_back(d);
        }
        
        // CPU Merkle tree (parallelized with OpenMP - faster than current GPU implementation)
        auto step3_merkle_start = std::chrono::high_resolution_clock::now();
        auto main_commitment = TableCommitment::from_digests(main_fri_digests);
        Digest main_root = main_commitment.root();
        auto step3_merkle_time = elapsed_ms(step3_merkle_start);
        
        double step3_time = elapsed_ms(step3_start);
        std::cout << "\n[Step 3] GPU Row hashing + CPU Merkle: " << step3_time << " ms" << std::endl;
        std::cout << "  Row hashing: " << step3_hash_time << " ms, Merkle: " << step3_merkle_time << " ms" << std::endl;
        std::cout << "  Main root: " << main_root << std::endl;
        
        // =====================================================================
        // Now use the CPU prover with GPU-computed LDE (skip CPU LDE)
        // =====================================================================
        
        std::cout << "\n[Step 4+] Running CPU prover with GPU LDE..." << std::endl;
        
        // Create main table from the flat data
        ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len)
            .with_offset(BFieldElement(trace_offset));
        ArithmeticDomain quotient_domain = ArithmeticDomain::of_length(quot_dom[0])
            .with_offset(BFieldElement(quot_dom[1]));
        ArithmeticDomain fri_domain_obj = ArithmeticDomain::of_length(fri_len)
            .with_offset(BFieldElement(fri_offset));
        
        MasterMainTable main_table(num_rows, num_cols, trace_domain, quotient_domain, fri_domain_obj);
        
        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                main_table.set(r, c, BFieldElement(flat_table[r * num_cols + c]));
            }
        }
        
        // Set GPU-computed LDE table in the main table (skip CPU LDE recomputation!)
        std::vector<std::vector<BFieldElement>> lde_table(fri_len, std::vector<BFieldElement>(num_cols));
        for (size_t r = 0; r < fri_len; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                lde_table[r][c] = BFieldElement(lde_data[c * fri_len + r]);
            }
        }
        main_table.set_lde_table(std::move(lde_table));
        main_table.set_num_trace_randomizers(stark.num_trace_randomizers());
        main_table.set_fri_digests(std::move(main_fri_digests));
        std::cout << "  Set GPU-computed randomized LDE + digests in main table" << std::endl;
        
        // Create claim
        Claim claim;
        claim.version = version;
        for (size_t i = 0; i < 5; ++i) {
            claim.program_digest[i] = BFieldElement(digest_arr[i]);
        }
        for (size_t i = 0; i < public_input.size(); ++i) {
            claim.input.push_back(BFieldElement(public_input[i]));
        }
        for (size_t i = 0; i < claim_output_len; ++i) {
            claim.output.push_back(BFieldElement(claim_output_ptr[i]));
        }
        
        // Run CPU prover (stark was defined earlier)
        ProofStream proof_stream;
        
        // Encode claim for Fiat-Shamir
        std::vector<BFieldElement> claim_encoding;
        auto encode_vec = [&](const std::vector<BFieldElement>& v) {
            claim_encoding.push_back(BFieldElement(1 + v.size()));
            claim_encoding.push_back(BFieldElement(v.size()));
            for (const auto& e : v) claim_encoding.push_back(e);
        };
        encode_vec(claim.output);
        encode_vec(claim.input);
        claim_encoding.push_back(BFieldElement(claim.version));
        for (size_t i = 0; i < 5; ++i) {
            claim_encoding.push_back(claim.program_digest[i]);
        }
        proof_stream.alter_fiat_shamir_state_with(claim_encoding);
        
        // Generate proof using CPU prover
        auto cpu_start = std::chrono::high_resolution_clock::now();
        Proof proof = stark.prove_with_table(claim, main_table, proof_stream, output_proof, "");
        double cpu_time = elapsed_ms(cpu_start);
        
        std::cout << "\n[CPU Steps] Proof generation: " << cpu_time << " ms" << std::endl;
        
        // Save claim
        claim.save_to_file(output_claim);
        
        // Cleanup
        cudaFree(d_trace);
        cudaFree(d_lde);
        cudaFree(d_digests);
        
        if (claim_output_ptr) tvm_claim_output_free(claim_output_ptr, claim_output_len);
        if (flat_table) tvm_main_table_free(flat_table, flat_table_len);
        
        double total_time = elapsed_ms(total_start);
        
        std::cout << "\n=== GPU Proof Generation Complete ===" << std::endl;
        std::cout << "Total time: " << total_time << " ms" << std::endl;
        std::cout << "  - Step 1 (trace): " << step1_time << " ms" << std::endl;
        std::cout << "  - Step 2 (GPU LDE): " << step2_time << " ms" << std::endl;
        std::cout << "  - Step 3 (GPU Merkle): " << step3_time << " ms" << std::endl;
        std::cout << "  - Steps 4+ (CPU): " << cpu_time << " ms" << std::endl;
        std::cout << "\nClaim saved to: " << output_claim << std::endl;
        std::cout << "Proof saved to: " << output_proof << std::endl;
        std::cout << "\nTo verify: triton-cli verify --claim " << output_claim 
                  << " --proof " << output_proof << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
#endif
}

