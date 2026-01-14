/**
 * CPU Auxiliary Table Computation for GpuStark
 * 
 * This file contains the CPU implementation of aux table computation.
 * It's separated from gpu_stark.cu to allow TBB usage without AMX intrinsics errors.
 * 
 * TBB headers can be safely included here since this is a pure C++ file (not .cu).
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/gpu_stark.hpp"
#include "gpu/gpu_proof_context.hpp"
#include "gpu/kernels/extend_kernel.cuh"
#include "table/extend_helpers.hpp"
#include "stark/challenges.hpp"
#include "chacha12_rng.hpp"
#include "types/b_field_element.hpp"
#include "common/debug_control.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>

// CUDA_CHECK macro (from cuda_common.cuh, but avoiding the constant definitions)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
} while(0)

#include <iostream>
#include <chrono>
#include <vector>
#include <atomic>
#include <cstring>
#include <thread>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TVM_USE_TBB
#include <tbb/parallel_invoke.h>
#endif

// =============================================================================
// mt19937_64 (std::mt19937_64 compatible) for aux randomizer column (col 87)
// CPU version of the GPU kernel functions
// =============================================================================
namespace {
uint64_t mt19937_64_temper(uint64_t x) {
    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);
    return x;
}

void mt19937_64_twist(uint64_t* mt) {
    constexpr int N = 312;
    constexpr int M = 156;
    constexpr uint64_t MATRIX_A = 0xB5026F5AA96619E9ULL;
    constexpr uint64_t UM = 0xFFFFFFFF80000000ULL; // upper 33 bits
    constexpr uint64_t LM = 0x7FFFFFFFULL;         // lower 31 bits

    for (int i = 0; i < N; i++) {
        uint64_t x = (mt[i] & UM) | (mt[(i + 1) % N] & LM);
        uint64_t xA = x >> 1;
        if (x & 1ULL) xA ^= MATRIX_A;
        mt[i] = mt[(i + M) % N] ^ xA;
    }
}

void mt19937_64_init(uint64_t* mt, uint64_t seed) {
    constexpr int N = 312;
    constexpr uint64_t F = 6364136223846793005ULL;
    mt[0] = seed;
    for (int i = 1; i < N; i++) {
        uint64_t x = mt[i - 1] ^ (mt[i - 1] >> 62);
        mt[i] = F * x + static_cast<uint64_t>(i);
    }
}

uint64_t mt19937_64_next(uint64_t* mt, int& idx) {
    constexpr int N = 312;
    if (idx >= N) {
        mt19937_64_twist(mt);
        idx = 0;
    }
    return mt19937_64_temper(mt[idx++]);
}
} // namespace

namespace triton_vm {
namespace gpu {

void GpuStark::compute_aux_table_cpu(
    const Challenges& challenges,
    const ChaCha12Rng::Seed& aux_seed,
    uint64_t* d_aux_trace,
    cudaStream_t stream
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const size_t num_rows = dims_.padded_height;
    const size_t main_width = dims_.main_width;

    if (h_main_table_data_ == nullptr) {
        throw std::runtime_error("[CPU AUX] h_main_table_data_ is null (hybrid aux requires a host main-table buffer)");
    }

    // NEW: operate directly on flat row-major u64 BFEs (no pre-conversion to vector<vector<BFieldElement>>).
    MainTableFlatView main_table_vec;
    main_table_vec.data = h_main_table_data_;
    main_table_vec.num_rows = num_rows;
    main_table_vec.num_cols = main_width;
    
    // Create aux table in CPU format (parallelized initialization, optional)
    // Control via TRITON_OMP_INIT (default: enabled if OpenMP available)
    static int omp_init_enabled = -1;
    if (omp_init_enabled == -1) {
        const char* env = std::getenv("TRITON_OMP_INIT");
        omp_init_enabled = (env == nullptr || (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        TRITON_PROFILE_COUT("[CPU AUX] OpenMP init parallelization: " << (omp_init_enabled ? "enabled" : "disabled") << std::endl);
    }
    
    auto t_init_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<XFieldElement>> aux_table_vec(num_rows);
#ifdef _OPENMP
    if (omp_init_enabled) {
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < num_rows; ++r) {
            aux_table_vec[r].resize(dims_.aux_width, XFieldElement::zero());
        }
    } else {
        for (size_t r = 0; r < num_rows; ++r) {
            aux_table_vec[r].resize(dims_.aux_width, XFieldElement::zero());
        }
    }
#else
    for (size_t r = 0; r < num_rows; ++r) {
        aux_table_vec[r].resize(dims_.aux_width, XFieldElement::zero());
    }
#endif
    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
    if (init_ms > 1.0) {
        TRITON_PROFILE_COUT("[CPU AUX] Aux table init: " << init_ms << " ms" << std::endl);
    }
    
    // Run CPU extend functions with CONSERVATIVE parallelization
    // Phase 1: Run small/stateless tables in parallel (write to non-overlapping columns)
    // Phase 2: Run stateful tables sequentially (Hash, Processor, DegreeLowering)
    TRITON_PROFILE_COUT("[CPU AUX] Running hybrid parallel/sequential CPU extend..." << std::endl);
    auto t_extend_start = std::chrono::high_resolution_clock::now();
    
    // Timing variables
    std::atomic<double> t_program{0}, t_opstack{0}, t_jumpstack{0}, t_ram{0};
    std::atomic<double> t_lookup{0}, t_u32{0}, t_cascade{0};
    double t_hash = 0, t_processor = 0, t_degree = 0;
    
    // Phase 1: Parallel small tables (all write to non-overlapping columns in aux_table_vec)
    // These tables have simple sequential algorithms and write to separate column ranges
    // Using TBB parallel_invoke for better work stealing on irregular workloads
    auto t_phase1_start = std::chrono::high_resolution_clock::now();
    {
#ifdef TVM_USE_TBB
        // Use TBB parallel_invoke for better work stealing
        if (std::getenv("TVM_VERIFY_TBB")) {
            std::cout << "[CPU AUX][TBB] Using TBB parallel_invoke for Phase 1 (7 tasks)" << std::endl;
        }
        tbb::parallel_invoke(
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_program_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_program = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_op_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_opstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_jump_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_jumpstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_ram_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_ram = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_lookup_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_lookup = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_u32_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_u32 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_cascade_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_cascade = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            }
        );
#else
        // Fallback to std::thread if TBB not available
        std::vector<std::thread> threads;
        
        // Program table (cols 0-2)
        threads.emplace_back([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_program_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_program = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // OpStack table (cols 14-15)
        threads.emplace_back([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_op_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_opstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // JumpStack table (cols 22-23)
        threads.emplace_back([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_jump_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_jumpstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // RAM table (cols 16-21)
        threads.emplace_back([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_ram_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_ram = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // Lookup table (cols 46-47)
        threads.emplace_back([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_lookup_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_lookup = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // U32 table (col 48)
        threads.emplace_back([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_u32_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_u32 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // Cascade table (cols 44-45)
        threads.emplace_back([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_cascade_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_cascade = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        for (auto& th : threads) {
            th.join();
        }
#endif
    }
    double t_phase1 = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_phase1_start).count();
    
    TRITON_IF_PROFILE {
        std::cout << "[CPU AUX] Phase 1 (parallel small tables):" << std::endl;
        std::cout << "[CPU AUX]   Program: " << t_program.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   OpStack: " << t_opstack.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   JumpStack: " << t_jumpstack.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   RAM: " << t_ram.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   Lookup: " << t_lookup.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   U32: " << t_u32.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   Cascade: " << t_cascade.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   Phase 1 wall time: " << t_phase1 << " ms" << std::endl;
    }
    
    // Allocate host buffer for aux table (device pointer cannot be written from CPU)
    // Cast to size_t before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    const size_t aux_width = static_cast<size_t>(dims_.aux_width);
    const size_t aux_size = static_cast<size_t>(num_rows) * aux_width * 3;
    std::vector<uint64_t> h_aux_buffer(aux_size, 0);
    uint64_t* h_aux_ptr = h_aux_buffer.data();
    
    // Helper lambda for parallel column upload (OpenMP parallelized, optional)
    // Control via TRITON_OMP_UPLOAD (default: enabled if OpenMP available)
    static int omp_upload_enabled = -1;
    if (omp_upload_enabled == -1) {
        const char* env = std::getenv("TRITON_OMP_UPLOAD");
        omp_upload_enabled = (env == nullptr || (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        TRITON_PROFILE_COUT("[CPU AUX] OpenMP upload parallelization: " << (omp_upload_enabled ? "enabled" : "disabled") << std::endl);
    }
    
    auto upload_columns = [&](size_t col_start, size_t col_end) {
        auto t_upload_start = std::chrono::high_resolution_clock::now();
        // Use aux_width from outer scope (captured by reference)
#ifdef _OPENMP
        if (omp_upload_enabled) {
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < num_rows; ++r) {
                for (size_t c = col_start; c < col_end && c < aux_width; ++c) {
                    // Cast before multiply to prevent 32-bit overflow
                    size_t idx = (static_cast<size_t>(r) * aux_width + static_cast<size_t>(c)) * 3;
                    const auto& xfe = aux_table_vec[r][c];
                    h_aux_ptr[idx + 0] = xfe.coeff(0).value();
                    h_aux_ptr[idx + 1] = xfe.coeff(1).value();
                    h_aux_ptr[idx + 2] = xfe.coeff(2).value();
                }
            }
        } else {
            for (size_t r = 0; r < num_rows; ++r) {
                for (size_t c = col_start; c < col_end && c < aux_width; ++c) {
                    // Cast before multiply to prevent 32-bit overflow
                    size_t idx = (static_cast<size_t>(r) * aux_width + static_cast<size_t>(c)) * 3;
                    const auto& xfe = aux_table_vec[r][c];
                    h_aux_ptr[idx + 0] = xfe.coeff(0).value();
                    h_aux_ptr[idx + 1] = xfe.coeff(1).value();
                    h_aux_ptr[idx + 2] = xfe.coeff(2).value();
                }
            }
        }
#else
        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = col_start; c < col_end && c < aux_width; ++c) {
                // Cast before multiply to prevent 32-bit overflow
                size_t idx = (static_cast<size_t>(r) * aux_width + static_cast<size_t>(c)) * 3;
                const auto& xfe = aux_table_vec[r][c];
                h_aux_ptr[idx + 0] = xfe.coeff(0).value();
                h_aux_ptr[idx + 1] = xfe.coeff(1).value();
                h_aux_ptr[idx + 2] = xfe.coeff(2).value();
            }
        }
#endif
        auto t_upload_end = std::chrono::high_resolution_clock::now();
        double upload_ms = std::chrono::duration<double, std::milli>(t_upload_end - t_upload_start).count();
        // Logging removed: upload timing messages
        (void)upload_ms;  // Suppress unused variable warning
    };
    
    // Phase 2: Hash + Processor + OVERLAPPED UPLOAD of Phase 1 columns
    // Using TBB parallel_invoke for better work stealing
    TRITON_PROFILE_COUT("[CPU AUX] Phase 2 (Hash + Processor + Upload P1 cols parallel):" << std::endl);
    auto t_phase2_start = std::chrono::high_resolution_clock::now();
    std::atomic<double> t_upload_p1{0};
    {
        std::atomic<double> t_hash_atomic{0}, t_processor_atomic{0};
        
#ifdef TVM_USE_TBB
        // Use TBB parallel_invoke for better work stealing
        if (std::getenv("TVM_VERIFY_TBB")) {
            std::cout << "[CPU AUX][TBB] Using TBB parallel_invoke for Phase 2 (3 tasks)" << std::endl;
        }
        tbb::parallel_invoke(
            [&]() {
                // Hash table (cols 24-43)
                auto t = std::chrono::high_resolution_clock::now();
                extend_hash_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_hash_atomic = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                // Processor table (cols 3-13)
                auto t = std::chrono::high_resolution_clock::now();
                extend_processor_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_processor_atomic = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                // Upload Phase 1 columns (already computed!) while Hash/Processor run
                auto t = std::chrono::high_resolution_clock::now();
                upload_columns(0, 3);      // Program
                upload_columns(14, 24);    // OpStack, RAM, JumpStack
                upload_columns(44, 49);    // Cascade, Lookup, U32
                t_upload_p1 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            }
        );
#else
        // Fallback to std::thread if TBB not available
        // Thread 1: Hash table (cols 24-43)
        std::thread th_hash([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_hash_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_hash_atomic = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // Thread 2: Processor table (cols 3-13)
        std::thread th_proc([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            extend_processor_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_processor_atomic = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        // Thread 3: Upload Phase 1 columns (already computed!) while Hash/Processor run
        std::thread th_upload([&]() {
            auto t = std::chrono::high_resolution_clock::now();
            upload_columns(0, 3);      // Program
            upload_columns(14, 24);    // OpStack, RAM, JumpStack
            upload_columns(44, 49);    // Cascade, Lookup, U32
            t_upload_p1 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        });
        
        th_hash.join();
        th_proc.join();
        th_upload.join();
#endif
        t_hash = t_hash_atomic;
        t_processor = t_processor_atomic;
    }
    double t_phase2 = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_phase2_start).count();
    TRITON_IF_PROFILE {
        std::cout << "[CPU AUX]   Hash: " << t_hash << " ms" << std::endl;
        std::cout << "[CPU AUX]   Processor: " << t_processor << " ms" << std::endl;
        std::cout << "[CPU AUX]   Upload P1 cols (overlapped): " << t_upload_p1.load() << " ms" << std::endl;
        std::cout << "[CPU AUX]   Phase 2 wall time: " << t_phase2 << " ms" << std::endl;
    }
    
    // Upload remaining columns (Hash and Processor from Phase 2)
    upload_columns(3, 14);      // Processor (cols 3-13)
    upload_columns(24, 44);     // Hash (cols 24-43)
    
    // Phase 3: Apply randomizers BEFORE uploading to GPU (eliminates redundant download/upload)
    // Cast to size_t before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    const size_t aux_width_dl = static_cast<size_t>(dims_.aux_width);
    constexpr size_t RANDOMIZER_COL = 87;

    // Apply randomizers directly to h_aux_buffer (no download needed!)
    TRITON_PROFILE_COUT("[CPU AUX] Applying aux table randomizers..." << std::endl);
    auto t_randomizer_start = std::chrono::high_resolution_clock::now();

    // Try to load randomizer column values from Rust test data (for deterministic comparison)
    // Skip loading if TVM_DISABLE_RANDOMIZER_LOAD is set (use ChaCha12 RNG instead)
    bool loaded_from_test_data = false;
    
    const char* disable_load_env = std::getenv("TVM_DISABLE_RANDOMIZER_LOAD");
    bool skip_loading = (disable_load_env && (strcmp(disable_load_env, "1") == 0 || strcmp(disable_load_env, "true") == 0));
    
    const char* test_data_dir_env = std::getenv("TVM_RUST_TEST_DATA_DIR");
    if (test_data_dir_env && !skip_loading) {
        std::string test_data_dir = test_data_dir_env;
        std::string aux_create_path = test_data_dir + "/07_aux_tables_create.json";
        std::cout << "[CPU AUX] Attempting to load randomizer column from: " << aux_create_path << std::endl;
        std::ifstream file(aux_create_path);
        
        if (file.is_open()) {
            std::cout << "[CPU AUX] Successfully opened test data file" << std::endl;
            try {
                nlohmann::json data = nlohmann::json::parse(file);
                
                // Parse XFieldElement strings (reusable lambda)
                auto parse_xfe_string = [](const std::string& s) -> std::tuple<uint64_t, uint64_t, uint64_t> {
                    if (s == "0_xfe" || s == "0") return {0, 0, 0};
                    if (s == "1_xfe" || s == "1") return {1, 0, 0};
                    
                    if (s.size() >= 5 && s.substr(s.size() - 4) == "_xfe") {
                        std::string num_str = s.substr(0, s.size() - 4);
                        try {
                            uint64_t val = std::stoull(num_str);
                            return {val, 0, 0};
                        } catch (...) {
                            return {0, 0, 0};
                        }
                    }
                    
                    if (s.empty() || s.front() != '(' || s.back() != ')') return {0, 0, 0};
                    std::string inner = s.substr(1, s.size() - 2);
                    
                    size_t x2_pos = std::string::npos;
                    size_t x_pos = std::string::npos;
                    for (size_t i = 0; i < inner.size(); ++i) {
                        if (inner[i] == 'x') {
                            if (i + 4 < inner.size() && inner.substr(i, 4) == "x + ") {
                                if (x_pos == std::string::npos) x_pos = i;
                            } else if (i + 1 < inner.size() && 
                                      static_cast<unsigned char>(inner[i+1]) == 0xC2 &&
                                      static_cast<unsigned char>(inner[i+2]) == 0xB2 &&
                                      i + 4 < inner.size() && inner.substr(i+3, 3) == " + ") {
                                x2_pos = i;
                            }
                        }
                    }
                    
                    if (x2_pos == std::string::npos || x_pos == std::string::npos) return {0, 0, 0};
                    
                    try {
                        size_t c2_start = 0;
                        while (c2_start < x2_pos && (inner[c2_start] == ' ' || inner[c2_start] == '\t')) c2_start++;
                        size_t c2_end = x2_pos;
                        while (c2_end > c2_start && inner[c2_end - 1] != ' ' && (inner[c2_end - 1] < '0' || inner[c2_end - 1] > '9')) c2_end--;
                        
                        size_t c1_start = x2_pos + 6;
                        while (c1_start < x_pos && (inner[c1_start] == ' ' || inner[c1_start] == '\t')) c1_start++;
                        size_t c1_end = x_pos;
                        while (c1_end > c1_start && inner[c1_end - 1] != ' ' && (inner[c1_end - 1] < '0' || inner[c1_end - 1] > '9')) c1_end--;
                        
                        size_t c0_start = x_pos + 4;
                        while (c0_start < inner.size() && (inner[c0_start] == ' ' || inner[c0_start] == '\t')) c0_start++;
                        size_t c0_end = inner.size();
                        while (c0_end > c0_start && (inner[c0_end - 1] == ' ' || inner[c0_end - 1] == '\t')) c0_end--;
                        
                        std::string c2_str = inner.substr(c2_start, c2_end - c2_start);
                        std::string c1_str = inner.substr(c1_start, c1_end - c1_start);
                        std::string c0_str = inner.substr(c0_start, c0_end - c0_start);
                        
                        c2_str.erase(std::remove_if(c2_str.begin(), c2_str.end(), [](char c) { return c < '0' || c > '9'; }), c2_str.end());
                        c1_str.erase(std::remove_if(c1_str.begin(), c1_str.end(), [](char c) { return c < '0' || c > '9'; }), c1_str.end());
                        c0_str.erase(std::remove_if(c0_str.begin(), c0_str.end(), [](char c) { return c < '0' || c > '9'; }), c0_str.end());
                        
                        if (c2_str.empty()) c2_str = "0";
                        if (c1_str.empty()) c1_str = "0";
                        if (c0_str.empty()) c0_str = "0";
                        
                        uint64_t c2 = std::stoull(c2_str);
                        uint64_t c1 = std::stoull(c1_str);
                        uint64_t c0 = std::stoull(c0_str);
                        return {c0, c1, c2};
                    } catch (...) {
                        return {0, 0, 0};
                    }
                };
                
                // Try to load all rows from sampled_rows (if available and matches row count)
                if (data.contains("sampled_rows") && data["sampled_rows"].is_array()) {
                    auto sampled_rows = data["sampled_rows"];
                    size_t rust_row_count = sampled_rows.size();
                    
                    if (rust_row_count == num_rows && sampled_rows[0].is_array() && sampled_rows[0].size() > RANDOMIZER_COL) {
                        // Load all rows for column 87 directly into h_aux_buffer
                        for (size_t r = 0; r < num_rows && r < rust_row_count; ++r) {
                            std::string col87_str = sampled_rows[r][RANDOMIZER_COL].get<std::string>();
                            auto [c0, c1, c2] = parse_xfe_string(col87_str);
                            size_t idx = (static_cast<size_t>(r) * aux_width_dl + static_cast<size_t>(RANDOMIZER_COL)) * 3;
                            h_aux_ptr[idx + 0] = c0;
                            h_aux_ptr[idx + 1] = c1;
                            h_aux_ptr[idx + 2] = c2;
                        }
                        loaded_from_test_data = true;
                        std::cout << "[CPU AUX] Loaded all " << num_rows << " rows of randomizer column 87 from Rust test data" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                // If parsing fails, fall back to RNG generation
                std::cerr << "[CPU AUX] Warning: Failed to load randomizer from test data: " << e.what() << std::endl;
            }
        } else {
            // Logging removed: Test data file not found message
        }
    } else {
        // Logging removed: TVM_RUST_TEST_DATA_DIR message
    }
    
    // Generate randomizer values using RNG if not loaded from test data
    if (!loaded_from_test_data) {
        // Apply column 87 randomizer using ChaCha12Rng to match Rust exactly
        ChaCha12Rng rng(aux_seed);

        // Match rand-0.9.x `UniformInt::sample_single_inclusive` (biased Canon's method) used by
        // `rng.random_range(0..=BFieldElement::MAX)` in twenty-first.
        auto sample_bfe_u64 = [&rng]() -> uint64_t {
            constexpr uint64_t P = BFieldElement::MODULUS;
            // range = P, so `range.wrapping_neg()` equals 2^32 - 1 for Goldilocks.
            constexpr uint64_t NEG_RANGE = static_cast<uint64_t>(0) - P;
            const uint64_t x = rng.next_u64();
            const __uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(P);
            uint64_t result = static_cast<uint64_t>(m >> 64);
            const uint64_t lo_order = static_cast<uint64_t>(m);
            if (lo_order > NEG_RANGE) {
                const uint64_t y = rng.next_u64();
                const __uint128_t my = static_cast<__uint128_t>(y) * static_cast<__uint128_t>(P);
                const uint64_t new_hi = static_cast<uint64_t>(my >> 64);
                // Overflow check: (lo_order + new_hi) overflowed u64?
                if (lo_order + new_hi < lo_order) {
                    result += 1;
                }
            }
            return result;
        };

        // Use aux_width_dl (already declared earlier in function)
        // Apply randomizers directly to h_aux_buffer (no separate buffer needed)
        for (size_t r = 0; r < num_rows; ++r) {
            // Rust XFieldElement coefficients are [c0, c1, c2] = [const, x, x^2].
            const uint64_t c0 = sample_bfe_u64();
            const uint64_t c1 = sample_bfe_u64();
            const uint64_t c2 = sample_bfe_u64();
            // Cast before multiply to prevent 32-bit overflow
            size_t idx = (static_cast<size_t>(r) * aux_width_dl + static_cast<size_t>(RANDOMIZER_COL)) * 3;
            h_aux_ptr[idx + 0] = c0;
            h_aux_ptr[idx + 1] = c1;
            h_aux_ptr[idx + 2] = c2;
        }
    }

    // OPTIMIZATION: Upload to GPU once (randomizers already applied to h_aux_buffer)
    // This eliminates the redundant download/upload cycle
    auto t_copy_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_aux_trace, h_aux_ptr, aux_size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    
    // Synchronize to ensure copy is complete (cudaStreamSynchronize is sufficient)
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    double t_copy = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_copy_start).count();
    TRITON_PROFILE_COUT("[CPU AUX]   Copy to device (with randomizers): " << t_copy << " ms" << std::endl);

    auto t_randomizer_end = std::chrono::high_resolution_clock::now();
    double randomizer_ms = std::chrono::duration<double, std::milli>(t_randomizer_end - t_randomizer_start).count();
    TRITON_PROFILE_COUT("[CPU AUX]   Randomizers applied: " << randomizer_ms << " ms" << std::endl);

    // Phase 4: Run DegreeLowering on GPU (much faster than Rust FFI)
    TRITON_PROFILE_COUT("[CPU AUX] Running DegreeLowering on GPU..." << std::endl);
    auto t_degree_start = std::chrono::high_resolution_clock::now();

    kernels::degree_lowering_only_gpu(
        ctx_->d_main_trace(),
        dims_.main_width,
        num_rows,
        ctx_->d_challenges(),
        d_aux_trace,
        stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t_degree_end = std::chrono::high_resolution_clock::now();
    double degree_ms = std::chrono::duration<double, std::milli>(t_degree_end - t_degree_start).count();
    TRITON_PROFILE_COUT("[CPU AUX]   DegreeLowering (GPU): " << degree_ms << " ms" << std::endl);
}

void GpuStark::compute_aux_table_cpu_extension_only(
    const Challenges& challenges,
    uint64_t* d_aux_trace
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const size_t num_rows = dims_.padded_height;
    const size_t main_width = dims_.main_width;

    if (h_main_table_data_ == nullptr) {
        throw std::runtime_error("[CPU AUX] h_main_table_data_ is null (hybrid aux requires a host main-table buffer)");
    }

    // Create main table view for extension functions
    MainTableFlatView main_table_vec;
    main_table_vec.data = h_main_table_data_;
    main_table_vec.num_rows = num_rows;
    main_table_vec.num_cols = main_width;

    // Create aux table in CPU format (parallelized initialization)
    static int omp_init_enabled = -1;
    if (omp_init_enabled == -1) {
        const char* env = std::getenv("TRITON_OMP_INIT");
        omp_init_enabled = (env == nullptr || (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        std::cout << "[CPU AUX DEBUG] OpenMP init parallelization: " << (omp_init_enabled ? "enabled" : "disabled") << std::endl;
    }

    auto t_init_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<XFieldElement>> aux_table_vec(num_rows);
#ifdef _OPENMP
    if (omp_init_enabled) {
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < num_rows; ++r) {
            aux_table_vec[r].resize(dims_.aux_width, XFieldElement::zero());
        }
    } else {
        for (size_t r = 0; r < num_rows; ++r) {
            aux_table_vec[r].resize(dims_.aux_width, XFieldElement::zero());
        }
    }
#else
    for (size_t r = 0; r < num_rows; ++r) {
        aux_table_vec[r].resize(dims_.aux_width, XFieldElement::zero());
    }
#endif
    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
    if (init_ms > 1.0) {
        std::cout << "[CPU AUX DEBUG] Aux table init: " << init_ms << " ms" << std::endl;
    }

    // Run CPU extend functions with CONSERVATIVE parallelization (extension only, no degree lowering)
    std::cout << "[CPU AUX DEBUG] Running CPU extension only..." << std::endl;
    auto t_extend_start = std::chrono::high_resolution_clock::now();

    // Timing variables (only for extension phase)
    std::atomic<double> t_program{0}, t_opstack{0}, t_jumpstack{0}, t_ram{0};
    std::atomic<double> t_lookup{0}, t_u32{0}, t_cascade{0};
    double t_hash = 0, t_processor = 0;

    // Phase 1: Parallel small tables (all write to non-overlapping columns in aux_table_vec)
    auto t_phase1_start = std::chrono::high_resolution_clock::now();
    {
#ifdef TVM_USE_TBB
        // Use TBB parallel_invoke for better work stealing
        if (std::getenv("TVM_VERIFY_TBB")) {
            std::cout << "[CPU AUX DEBUG][TBB] Using TBB parallel_invoke for Phase 1 (7 tasks)" << std::endl;
        }
        tbb::parallel_invoke(
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_program_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_program = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_op_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_opstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_jump_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_jumpstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_ram_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_ram = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_lookup_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_lookup = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_cascade_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_cascade = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            },
            [&]() {
                auto t = std::chrono::high_resolution_clock::now();
                extend_u32_table(main_table_vec, aux_table_vec, challenges, num_rows);
                t_u32 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
            }
        );
#else
        // Fallback to sequential execution
        {
            auto t = std::chrono::high_resolution_clock::now();
            extend_program_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_program = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        }
        {
            auto t = std::chrono::high_resolution_clock::now();
            extend_op_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_opstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        }
        {
            auto t = std::chrono::high_resolution_clock::now();
            extend_jump_stack_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_jumpstack = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        }
        {
            auto t = std::chrono::high_resolution_clock::now();
            extend_ram_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_ram = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        }
        {
            auto t = std::chrono::high_resolution_clock::now();
            extend_lookup_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_lookup = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        }
        {
            auto t = std::chrono::high_resolution_clock::now();
            extend_cascade_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_cascade = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        }
        {
            auto t = std::chrono::high_resolution_clock::now();
            extend_u32_table(main_table_vec, aux_table_vec, challenges, num_rows);
            t_u32 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        }
#endif
    }
    auto t_phase1_end = std::chrono::high_resolution_clock::now();
    double phase1_ms = std::chrono::duration<double, std::milli>(t_phase1_end - t_phase1_start).count();

    // Phase 2: Sequential stateful tables (Hash and Processor)
    auto t_phase2_start = std::chrono::high_resolution_clock::now();
    {
        auto t = std::chrono::high_resolution_clock::now();
        extend_hash_table(main_table_vec, aux_table_vec, challenges, num_rows);
        t_hash = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
    }
    {
        auto t = std::chrono::high_resolution_clock::now();
        extend_processor_table(main_table_vec, aux_table_vec, challenges, num_rows);
        t_processor = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
    }
    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    double phase2_ms = std::chrono::duration<double, std::milli>(t_phase2_end - t_phase2_start).count();

    auto t_extend_end = std::chrono::high_resolution_clock::now();
    double extend_ms = std::chrono::duration<double, std::milli>(t_extend_end - t_extend_start).count();

    std::cout << "[CPU AUX DEBUG] Extension Phase 1 (parallel): " << phase1_ms << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG] Extension Phase 2 (sequential): " << phase2_ms << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG] Total extension: " << extend_ms << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   Program: " << t_program << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   OpStack: " << t_opstack << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   JumpStack: " << t_jumpstack << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   RAM: " << t_ram << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   Lookup: " << t_lookup << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   Cascade: " << t_cascade << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   U32: " << t_u32 << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   Hash: " << t_hash << " ms" << std::endl;
    std::cout << "[CPU AUX DEBUG]   Processor: " << t_processor << " ms" << std::endl;

    // Convert to GPU format and upload (extension only)
    // Cast to size_t before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    const size_t aux_width_flat = static_cast<size_t>(dims_.aux_width);
    std::vector<uint64_t> h_aux_flat(static_cast<size_t>(num_rows) * aux_width_flat * 3);
    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t c = 0; c < aux_width_flat; ++c) {
            // Cast before multiply to prevent 32-bit overflow
            size_t dst_idx = (static_cast<size_t>(r) * aux_width_flat + static_cast<size_t>(c)) * 3;
            h_aux_flat[dst_idx + 0] = aux_table_vec[r][c].coeff(0).value();
            h_aux_flat[dst_idx + 1] = aux_table_vec[r][c].coeff(1).value();
            h_aux_flat[dst_idx + 2] = aux_table_vec[r][c].coeff(2).value();
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(
        d_aux_trace,
        h_aux_flat.data(),
        h_aux_flat.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        ctx_->stream()
    ));
    CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "[CPU AUX DEBUG] Extension-only total: " << total_ms << " ms" << std::endl;
}

void GpuStark::apply_cpu_aux_randomizer(
    uint64_t* d_aux_trace,
    uint64_t aux_seed_value,
    size_t num_rows,
    size_t aux_width
) {
    // Download aux table, apply randomizer, upload back
    // Cast to size_t before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    std::vector<uint64_t> h_aux(static_cast<size_t>(num_rows) * static_cast<size_t>(aux_width) * 3);
    CUDA_CHECK(cudaMemcpy(h_aux.data(), d_aux_trace, h_aux.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Apply column 87 randomizer (matches GPU implementation)
    uint64_t mt[312];
    int mt_idx = 312;
    mt19937_64_init(mt, aux_seed_value);

    constexpr size_t RANDOMIZER_COL = 87;
    for (size_t r = 0; r < num_rows; r++) {
        uint64_t a0 = mt19937_64_next(mt, mt_idx) % 0xFFFFFFFF00000001ULL;
        uint64_t a1 = mt19937_64_next(mt, mt_idx) % 0xFFFFFFFF00000001ULL;
        uint64_t a2 = mt19937_64_next(mt, mt_idx) % 0xFFFFFFFF00000001ULL;

        // Match GPU coefficient layout: (a2, a1, a0)
        // Cast before multiply to prevent 32-bit overflow
        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(aux_width) + static_cast<size_t>(RANDOMIZER_COL)) * 3;
        h_aux[idx + 0] = a2;
        h_aux[idx + 1] = a1;
        h_aux[idx + 2] = a0;
    }

    // Upload back to GPU
    CUDA_CHECK(cudaMemcpy(d_aux_trace, h_aux.data(), h_aux.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

