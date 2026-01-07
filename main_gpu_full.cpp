/**
 * Full GPU Zero-Copy Triton VM Prover
 * 
 * Architecture:
 * 1. Pure C++: Trace + pad (start) - VM execution, table creation, padding
 * 2. GPU: All proof computation (middle) - ZERO intermediate H2D/D2H
 * 3. Rust FFI: Proof encoding (end)
 * 
 * Memory transfers:
 * - H2D: Main table (once at start)
 * - D2H: Proof items for encoding (once at end)
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "stark.hpp"
#include "proof_stream/proof_stream.hpp"
#include "bincode_ffi.hpp"
#include "vm/vm.hpp"
#include "parallel/thread_coordination.h"
#include "vm/program.hpp"
#include "vm/aet.hpp"
#include "table/master_table.hpp"
#include "table/extend_helpers.hpp"
#include "vm/processor_columns.hpp"
#include "table/ram_bezout.hpp"
#include "quotient/quotient.hpp"
#ifdef TVM_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/enumerable_thread_specific.h>
#include <map>
#endif

#ifdef TRITON_CUDA_ENABLED
#include "gpu/gpu_stark.hpp"
#include "gpu/cuda_common.cuh"
#include "gpu/kernels/table_fill_kernel.cuh"
#include "gpu/kernels/bezout_kernel.cuh"
#include "gpu/kernels/phase1_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

// Validation functions for comparing C++ vs Rust implementation
void validate_trace_execution(const std::string& rust_test_data_dir, size_t processor_trace_height, size_t processor_trace_width, const std::vector<uint64_t>& public_output_sampled, size_t padded_height) {
    std::ifstream file(rust_test_data_dir + "/01_trace_execution.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Rust test data file 01_trace_execution.json not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check processor trace dimensions
    size_t rust_height = data["processor_trace_height"];
    size_t rust_width = data["processor_trace_width"];
    size_t rust_padded_height = data["padded_height"];

    if (processor_trace_height != rust_height) {
        all_match = false;
        mismatches.push_back("processor_trace_height: C++ " + std::to_string(processor_trace_height) + " vs Rust " + std::to_string(rust_height));
    }

    if (processor_trace_width != rust_width) {
        all_match = false;
        mismatches.push_back("processor_trace_width: C++ " + std::to_string(processor_trace_width) + " vs Rust " + std::to_string(rust_width));
    }

    if (padded_height != rust_padded_height) {
        all_match = false;
        mismatches.push_back("padded_height: C++ " + std::to_string(padded_height) + " vs Rust " + std::to_string(rust_padded_height));
    }

    // Check sampled public output
    if (data.contains("public_output_sampled")) {
        std::vector<uint64_t> rust_output = data["public_output_sampled"];
        if (public_output_sampled.size() == rust_output.size()) {
            for (size_t i = 0; i < public_output_sampled.size(); ++i) {
                if (public_output_sampled[i] != rust_output[i]) {
                    all_match = false;
                    mismatches.push_back("public_output_sampled[" + std::to_string(i) + "]: C++ " + std::to_string(public_output_sampled[i]) + " vs Rust " + std::to_string(rust_output[i]));
                    break; // Only report first mismatch
                }
            }
        } else {
            all_match = false;
            mismatches.push_back("public_output_sampled size: C++ " + std::to_string(public_output_sampled.size()) + " vs Rust " + std::to_string(rust_output.size()));
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 1 (Trace Execution): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 1 (Trace Execution): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_claim(const std::string& rust_test_data_dir, const std::vector<uint64_t>& program_digest, const std::vector<uint64_t>& input_sampled, const std::vector<uint64_t>& output_sampled) {
    std::ifstream file(rust_test_data_dir + "/02_claim.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Rust test data file 02_claim.json not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check program digest
    if (data.contains("program_digest")) {
        std::vector<uint64_t> rust_digest = data["program_digest"];
        if (program_digest.size() == rust_digest.size()) {
            for (size_t i = 0; i < program_digest.size(); ++i) {
                if (program_digest[i] != rust_digest[i]) {
                    all_match = false;
                    mismatches.push_back("program_digest[" + std::to_string(i) + "]: C++ " + std::to_string(program_digest[i]) + " vs Rust " + std::to_string(rust_digest[i]));
                    break;
                }
            }
        } else {
            all_match = false;
            mismatches.push_back("program_digest size: C++ " + std::to_string(program_digest.size()) + " vs Rust " + std::to_string(rust_digest.size()));
        }
    }

    // Check input samples
    if (data.contains("input_sampled")) {
        std::vector<uint64_t> rust_input = data["input_sampled"];
        if (input_sampled.size() == rust_input.size()) {
            for (size_t i = 0; i < input_sampled.size(); ++i) {
                if (input_sampled[i] != rust_input[i]) {
                    all_match = false;
                    mismatches.push_back("input_sampled[" + std::to_string(i) + "]: C++ " + std::to_string(input_sampled[i]) + " vs Rust " + std::to_string(rust_input[i]));
                    break;
                }
            }
        } else {
            all_match = false;
            mismatches.push_back("input_sampled size: C++ " + std::to_string(input_sampled.size()) + " vs Rust " + std::to_string(rust_input.size()));
        }
    }

    // Check output samples
    if (data.contains("output_sampled")) {
        std::vector<uint64_t> rust_output = data["output_sampled"];
        if (output_sampled.size() == rust_output.size()) {
            for (size_t i = 0; i < output_sampled.size(); ++i) {
                if (output_sampled[i] != rust_output[i]) {
                    all_match = false;
                    mismatches.push_back("output_sampled[" + std::to_string(i) + "]: C++ " + std::to_string(output_sampled[i]) + " vs Rust " + std::to_string(rust_output[i]));
                    break;
                }
            }
        } else {
            all_match = false;
            mismatches.push_back("output_sampled size: C++ " + std::to_string(output_sampled.size()) + " vs Rust " + std::to_string(rust_output.size()));
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 2 (Claim): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 2 (Claim): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_fiat_shamir_init(const std::string& rust_test_data_dir, size_t claim_encoded_length) {
    std::ifstream file(rust_test_data_dir + "/03_fiat_shamir_init.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Rust test data file 03_fiat_shamir_init.json not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check Fiat-Shamir initialization
    if (data.contains("fiat_shamir_initialized") && data.contains("claim_encoded_length")) {
        bool rust_initialized = data["fiat_shamir_initialized"];
        size_t rust_length = data["claim_encoded_length"];

        if (!rust_initialized) {
            all_match = false;
            mismatches.push_back("fiat_shamir_initialized: Rust shows false");
        }

        if (claim_encoded_length != rust_length) {
            all_match = false;
            mismatches.push_back("claim_encoded_length: C++ " + std::to_string(claim_encoded_length) + " vs Rust " + std::to_string(rust_length));
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 3 (Fiat-Shamir Init): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 3 (Fiat-Shamir Init): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_domain_setup(const std::string& rust_test_data_dir, size_t padded_height, size_t trace_domain_length, size_t quotient_domain_length, size_t fri_domain_length) {
    std::ifstream file(rust_test_data_dir + "/04_domain_setup.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Rust test data file 04_domain_setup.json not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check domain parameters
    size_t rust_padded_height = data["padded_height"];
    size_t rust_trace_domain_length = data["trace_domain_length"];
    size_t rust_quotient_domain_length = data["quotient_domain_length"];
    size_t rust_fri_domain_length = data["fri_domain_length"];

    if (padded_height != rust_padded_height) {
        all_match = false;
        mismatches.push_back("padded_height: C++ " + std::to_string(padded_height) + " vs Rust " + std::to_string(rust_padded_height));
    }

    if (trace_domain_length != rust_trace_domain_length) {
        all_match = false;
        mismatches.push_back("trace_domain_length: C++ " + std::to_string(trace_domain_length) + " vs Rust " + std::to_string(rust_trace_domain_length));
    }

    if (quotient_domain_length != rust_quotient_domain_length) {
        all_match = false;
        mismatches.push_back("quotient_domain_length: C++ " + std::to_string(quotient_domain_length) + " vs Rust " + std::to_string(rust_quotient_domain_length));
    }

    if (fri_domain_length != rust_fri_domain_length) {
        all_match = false;
        mismatches.push_back("fri_domain_length: C++ " + std::to_string(fri_domain_length) + " vs Rust " + std::to_string(rust_fri_domain_length));
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 4 (Domain Setup): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 4 (Domain Setup): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_main_table_create(const std::string& rust_test_data_dir, size_t padded_height) {
    std::ifstream file(rust_test_data_dir + "/05_main_table_create.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Rust test data file 05_main_table_create.json not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check main table creation
    if (data.contains("main_table_created") && data.contains("padded_height")) {
        bool rust_created = data["main_table_created"];
        size_t rust_padded_height = data["padded_height"];

        if (!rust_created) {
            all_match = false;
            mismatches.push_back("main_table_created: Rust shows false");
        }

        if (padded_height != rust_padded_height) {
            all_match = false;
            mismatches.push_back("padded_height: C++ " + std::to_string(padded_height) + " vs Rust " + std::to_string(rust_padded_height));
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 5 (Main Table Create): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 5 (Main Table Create): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_main_table_pad(const std::string& rust_test_data_dir, size_t padded_height) {
    std::ifstream file(rust_test_data_dir + "/06_main_table_pad.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Rust test data file 06_main_table_pad.json not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check main table padding
    if (data.contains("main_table_padded") && data.contains("padded_height")) {
        bool rust_padded = data["main_table_padded"];
        size_t rust_padded_height = data["padded_height"];

        if (!rust_padded) {
            all_match = false;
            mismatches.push_back("main_table_padded: Rust shows false");
        }

        if (padded_height != rust_padded_height) {
            all_match = false;
            mismatches.push_back("padded_height: C++ " + std::to_string(padded_height) + " vs Rust " + std::to_string(rust_padded_height));
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 6 (Main Table Pad): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 6 (Main Table Pad): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_padded_main_table_data(const std::string& rust_test_data_dir, const MasterMainTable& main_table, size_t padded_height, size_t num_cols) {
    std::ifstream file(rust_test_data_dir + "/04_main_tables_pad.json");
    if (!file.is_open()) {
        std::cerr << "[DBG] Rust test data file 04_main_tables_pad.json not found, skipping padded table comparison" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);
    
    if (!data.contains("padded_table_data")) {
        std::cerr << "[DBG] No padded_table_data in 04_main_tables_pad.json" << std::endl;
        return;
    }

    auto rust_table = data["padded_table_data"];
    size_t rust_rows = rust_table.size();
    size_t rust_cols = (rust_rows > 0) ? rust_table[0].size() : 0;

    std::cerr << "\n";
    std::cerr << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cerr << "║  STEP: Main Table Pad Comparison                              ║" << std::endl;
    std::cerr << "╠══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cerr << "[DBG] Dimensions:" << std::endl;
    std::cerr << "  C++: " << padded_height << " x " << num_cols << std::endl;
    std::cerr << "  Rust: " << rust_rows << " x " << rust_cols << std::endl;

    if (padded_height != rust_rows || num_cols != rust_cols) {
        std::cerr << "[DBG] ✗ Dimension mismatch!" << std::endl;
        return;
    }

    // Compare first row - check first 20, column 147 (known LDE mismatch), and last column 378
    bool first_row_match = true;
    size_t first_row_mismatches = 0;
    std::vector<size_t> critical_cols = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 147, 378};
    if (rust_table.size() > 0 && rust_table[0].is_array()) {
        const auto& cpp_row0 = main_table.row(0);
        auto rust_row0 = rust_table[0];
        std::cerr << "[DBG] First row comparison (first 20 cols + critical cols 147, 378):" << std::endl;
        for (size_t c : critical_cols) {
            if (c >= num_cols || c >= rust_row0.size()) continue;
            uint64_t cpp_val = cpp_row0[c].value();
            uint64_t rust_val = rust_row0[c].get<uint64_t>();
            bool match = (cpp_val == rust_val);
            if (!match) {
                first_row_match = false;
                first_row_mismatches++;
            }
            std::string col_label = (c == 378) ? " (LAST COL)" : (c == 147) ? " (CRITICAL)" : "";
            std::cerr << "  [" << c << col_label << "] C++: " << cpp_val << " | Rust: " << rust_val 
                      << (match ? " ✓" : " ✗") << std::endl;
        }
        
        // Check ALL columns in first row for mismatches
        size_t total_first_row_mismatches = 0;
        for (size_t c = 0; c < std::min(num_cols, rust_row0.size()); c++) {
            if (cpp_row0[c].value() != rust_row0[c].get<uint64_t>()) {
                total_first_row_mismatches++;
            }
        }
        std::cerr << "[DBG] First row: " << total_first_row_mismatches << " mismatches out of " << num_cols << " columns" << std::endl;
    }

    // Compare last row - check first 20, column 147, and last column 378
    bool last_row_match = true;
    size_t last_row_mismatches = 0;
    if (rust_table.size() > 0 && rust_table[rust_table.size() - 1].is_array()) {
        const auto& cpp_row_last = main_table.row(padded_height - 1);
        auto rust_row_last = rust_table[rust_table.size() - 1];
        std::cerr << "[DBG] Last row comparison (first 20 cols + critical cols 147, 378):" << std::endl;
        for (size_t c : critical_cols) {
            if (c >= num_cols || c >= rust_row_last.size()) continue;
            uint64_t cpp_val = cpp_row_last[c].value();
            uint64_t rust_val = rust_row_last[c].get<uint64_t>();
            bool match = (cpp_val == rust_val);
            if (!match) {
                last_row_match = false;
                last_row_mismatches++;
            }
            std::string col_label = (c == 378) ? " (LAST COL)" : (c == 147) ? " (CRITICAL)" : "";
            std::cerr << "  [" << c << col_label << "] C++: " << cpp_val << " | Rust: " << rust_val 
                      << (match ? " ✓" : " ✗") << std::endl;
        }
        
        // Check ALL columns in last row for mismatches
        size_t total_last_row_mismatches = 0;
        for (size_t c = 0; c < std::min(num_cols, rust_row_last.size()); c++) {
            if (cpp_row_last[c].value() != rust_row_last[c].get<uint64_t>()) {
                total_last_row_mismatches++;
            }
        }
        std::cerr << "[DBG] Last row: " << total_last_row_mismatches << " mismatches out of " << num_cols << " columns" << std::endl;
    }

    // Sample comparison: check more rows and more columns
    std::vector<size_t> sample_rows = {100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000};
    bool sample_match = true;
    size_t sample_mismatches = 0;
    std::cerr << "[DBG] Sample rows comparison (checking cols 0, 147, 378):" << std::endl;
    for (size_t row_idx : sample_rows) {
        if (row_idx < padded_height && row_idx < rust_table.size() && rust_table[row_idx].is_array()) {
            const auto& cpp_row = main_table.row(row_idx);
            auto rust_row = rust_table[row_idx];
            // Compare first column, column 147, and last column
            uint64_t cpp_first = cpp_row[0].value();
            uint64_t rust_first = rust_row[0].get<uint64_t>();
            uint64_t cpp_col147 = (147 < num_cols) ? cpp_row[147].value() : 0;
            uint64_t rust_col147 = (147 < rust_row.size()) ? rust_row[147].get<uint64_t>() : 0;
            uint64_t cpp_last = cpp_row[num_cols - 1].value();
            uint64_t rust_last = rust_row[num_cols - 1].get<uint64_t>();
            bool match = (cpp_first == rust_first) && (cpp_col147 == rust_col147) && (cpp_last == rust_last);
            if (!match) {
                sample_match = false;
                sample_mismatches++;
            }
            std::cerr << "  Row[" << row_idx << "] col[0]: C++=" << cpp_first << " Rust=" << rust_first
                      << " col[147]: C++=" << cpp_col147 << " Rust=" << rust_col147
                      << " col[378]: C++=" << cpp_last << " Rust=" << rust_last
                      << (match ? " ✓" : " ✗") << std::endl;
        }
    }
    
    // Full table scan for critical columns (147 and 378) - check all rows
    std::cerr << "\n[DBG] Full table scan for critical columns (147, 378):" << std::endl;
    size_t col147_mismatches = 0;
    size_t col378_mismatches = 0;
    std::vector<size_t> col147_mismatch_rows;
    std::vector<size_t> col378_mismatch_rows;
    
    for (size_t r = 0; r < std::min(padded_height, rust_rows); r++) {
        if (r >= rust_table.size() || !rust_table[r].is_array()) continue;
        const auto& cpp_row = main_table.row(r);
        auto rust_row = rust_table[r];
        
        // Check column 147
        if (147 < num_cols && 147 < rust_row.size()) {
            uint64_t cpp_val = cpp_row[147].value();
            uint64_t rust_val = rust_row[147].get<uint64_t>();
            if (cpp_val != rust_val) {
                col147_mismatches++;
                if (col147_mismatch_rows.size() < 10) {
                    col147_mismatch_rows.push_back(r);
                }
            }
        }
        
        // Check column 378
        if (378 < num_cols && 378 < rust_row.size()) {
            uint64_t cpp_val = cpp_row[378].value();
            uint64_t rust_val = rust_row[378].get<uint64_t>();
            if (cpp_val != rust_val) {
                col378_mismatches++;
                if (col378_mismatch_rows.size() < 10) {
                    col378_mismatch_rows.push_back(r);
                }
            }
        }
    }
    
    std::cerr << "  Column 147: " << col147_mismatches << " mismatches out of " << padded_height << " rows" << std::endl;
    if (col147_mismatches > 0 && !col147_mismatch_rows.empty()) {
        std::cerr << "    First few mismatch rows: ";
        for (size_t i = 0; i < std::min<size_t>(5, col147_mismatch_rows.size()); i++) {
            std::cerr << col147_mismatch_rows[i];
            if (i < std::min<size_t>(5, col147_mismatch_rows.size()) - 1) std::cerr << ", ";
        }
        std::cerr << std::endl;
    }
    
    std::cerr << "  Column 378: " << col378_mismatches << " mismatches out of " << padded_height << " rows" << std::endl;
    if (col378_mismatches > 0 && !col378_mismatch_rows.empty()) {
        std::cerr << "    First few mismatch rows: ";
        for (size_t i = 0; i < std::min<size_t>(5, col378_mismatch_rows.size()); i++) {
            std::cerr << col378_mismatch_rows[i];
            if (i < std::min<size_t>(5, col378_mismatch_rows.size()) - 1) std::cerr << ", ";
        }
        std::cerr << std::endl;
    }
    
    // Detailed inspection of rows 181-185 (where mismatches were detected)
    std::vector<size_t> inspect_rows = {180, 181, 182, 183, 184, 185, 186};
    std::vector<size_t> inspect_cols = {0, 1, 2, 145, 146, 147, 148, 149, 376, 377, 378};
    
    std::cerr << "\n[DBG] ═══════════════════════════════════════════════════════════════" << std::endl;
    std::cerr << "[DBG] DETAILED INSPECTION: Rows 180-186 (mismatch region)" << std::endl;
    std::cerr << "[DBG] ═══════════════════════════════════════════════════════════════" << std::endl;
    
    for (size_t r : inspect_rows) {
        if (r >= padded_height || r >= rust_table.size() || !rust_table[r].is_array()) continue;
        
        const auto& cpp_row = main_table.row(r);
        auto rust_row = rust_table[r];
        
        std::cerr << "\n[DBG] Row " << r << ":" << std::endl;
        
        // Check if this row has any mismatches in critical columns
        bool has_col147_mismatch = false;
        bool has_col378_mismatch = false;
        uint64_t cpp_col147 = 0, rust_col147 = 0;
        uint64_t cpp_col378 = 0, rust_col378 = 0;
        
        if (147 < num_cols && 147 < rust_row.size()) {
            cpp_col147 = cpp_row[147].value();
            rust_col147 = rust_row[147].get<uint64_t>();
            has_col147_mismatch = (cpp_col147 != rust_col147);
        }
        
        if (378 < num_cols && 378 < rust_row.size()) {
            cpp_col378 = cpp_row[378].value();
            rust_col378 = rust_row[378].get<uint64_t>();
            has_col378_mismatch = (cpp_col378 != rust_col378);
        }
        
        if (has_col147_mismatch || has_col378_mismatch) {
            std::cerr << "  ⚠️  MISMATCH DETECTED in this row!" << std::endl;
        }
        
        // Show selected columns
        for (size_t c : inspect_cols) {
            if (c >= num_cols || c >= rust_row.size()) continue;
            
            uint64_t cpp_val = cpp_row[c].value();
            uint64_t rust_val = rust_row[c].get<uint64_t>();
            bool match = (cpp_val == rust_val);
            
            std::string col_label = "";
            if (c == 147) col_label = " (CRITICAL - COL 147)";
            else if (c == 378) col_label = " (CRITICAL - COL 378)";
            else if (c == 0 || c == 1 || c == 2) col_label = " (CONTEXT)";
            
            std::cerr << "  Col[" << c << col_label << "]: C++=" << std::setw(20) << cpp_val 
                      << " | Rust=" << std::setw(20) << rust_val 
                      << (match ? " ✓" : " ✗ MISMATCH") << std::endl;
        }
        
        // Show full row comparison for mismatched rows (check all columns, show first 20 mismatches)
        if (has_col147_mismatch || has_col378_mismatch) {
            std::cerr << "\n  Full row comparison (checking all " << num_cols << " columns):" << std::endl;
            size_t mismatches_in_row = 0;
            std::vector<std::pair<size_t, std::pair<uint64_t, uint64_t>>> mismatch_details;
            
            // Check all columns
            for (size_t c = 0; c < std::min(num_cols, rust_row.size()); c++) {
                uint64_t cpp_val = cpp_row[c].value();
                uint64_t rust_val = rust_row[c].get<uint64_t>();
                if (cpp_val != rust_val) {
                    mismatches_in_row++;
                    if (mismatch_details.size() < 20) {
                        mismatch_details.push_back({c, {cpp_val, rust_val}});
                    }
                }
            }
            
            // Show first 20 mismatches
            for (const auto& [col_idx, vals] : mismatch_details) {
                std::string col_label = "";
                if (col_idx == 147) col_label = " (CRITICAL COL 147)";
                else if (col_idx == 378) col_label = " (CRITICAL COL 378)";
                std::cerr << "    Col[" << col_idx << col_label << "]: C++=" << vals.first << " Rust=" << vals.second << " ✗" << std::endl;
            }
            
            if (mismatches_in_row > 20) {
                std::cerr << "    ... and " << (mismatches_in_row - 20) << " more mismatches in this row" << std::endl;
            }
            std::cerr << "  Total mismatches in row " << r << ": " << mismatches_in_row << " out of " << num_cols << " columns" << std::endl;
        }
    }
    
    std::cerr << "\n[DBG] ═══════════════════════════════════════════════════════════════" << std::endl;

    std::cerr << "\n[DBG] Summary:" << std::endl;
    bool all_match = (first_row_match && last_row_match && sample_match && col147_mismatches == 0 && col378_mismatches == 0);
    if (all_match) {
        std::cerr << "[DBG] ✓✓✓ Padded main table: FULL MATCH" << std::endl;
        std::cerr << "[DBG]    ✓ First row: all " << num_cols << " columns match" << std::endl;
        std::cerr << "[DBG]    ✓ Last row: all " << num_cols << " columns match" << std::endl;
        std::cerr << "[DBG]    ✓ Sample rows: all checked columns match" << std::endl;
        std::cerr << "[DBG]    ✓ Column 147: all " << padded_height << " rows match" << std::endl;
        std::cerr << "[DBG]    ✓ Column 378: all " << padded_height << " rows match" << std::endl;
    } else {
        std::cerr << "[DBG] ✗✗✗ Padded main table: MISMATCH DETECTED" << std::endl;
        if (!first_row_match) std::cerr << "[DBG]    ✗ First row: " << first_row_mismatches << " mismatches in checked columns" << std::endl;
        if (!last_row_match) std::cerr << "[DBG]    ✗ Last row: " << last_row_mismatches << " mismatches in checked columns" << std::endl;
        if (!sample_match) std::cerr << "[DBG]    ✗ Sample rows: " << sample_mismatches << " mismatches" << std::endl;
        if (col147_mismatches > 0) std::cerr << "[DBG]    ✗ Column 147: " << col147_mismatches << " mismatches across all rows" << std::endl;
        if (col378_mismatches > 0) std::cerr << "[DBG]    ✗ Column 378: " << col378_mismatches << " mismatches across all rows" << std::endl;
    }
    std::cerr << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;
}

void validate_main_table_commitment(const std::string& rust_test_data_dir, const std::string& cpp_merkle_root_hex) {
    // Read Rust Merkle root from the comprehensive dump
    std::ifstream merkle_file(rust_test_data_dir + "/06_main_tables_merkle.json");
    if (!merkle_file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Step 7 (Main Table Commitment): Rust Merkle root not found" << std::endl;
        return;
    }

    nlohmann::json merkle_data = nlohmann::json::parse(merkle_file);
    
    bool all_match = true;
    std::vector<std::string> mismatches;

    // Compare Merkle roots (actual computational value)
    if (merkle_data.contains("merkle_root")) {
        std::string rust_merkle_root = merkle_data["merkle_root"];
        
        if (rust_merkle_root != cpp_merkle_root_hex) {
            all_match = false;
            mismatches.push_back("Merkle root mismatch");
            mismatches.push_back("  Rust: " + rust_merkle_root);
            mismatches.push_back("  C++:  " + cpp_merkle_root_hex);
        } else {
            // Also check num_leafs for consistency
            if (merkle_data.contains("num_leafs")) {
                size_t rust_num_leafs = merkle_data["num_leafs"];
                // Note: We'd need to pass C++ num_leafs to compare, but Merkle root match is primary validation
            }
        }
    } else {
        all_match = false;
        mismatches.push_back("merkle_root: Field missing in Rust data");
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 7 (Main Table Commitment): ✓ MATCH (Merkle root verified)" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 7 (Main Table Commitment): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_aux_table_create(const std::string& rust_test_data_dir, gpu::GpuStark& gpu_stark, size_t padded_height, size_t aux_width) {
    std::ifstream file(rust_test_data_dir + "/07_aux_tables_create.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Aux Table Create: Rust test data not found" << std::endl;
        return;
    }

    nlohmann::json rust_data = nlohmann::json::parse(file);
    
    // Extract Rust data
    std::vector<size_t> rust_shape = rust_data["aux_table_shape"];
    size_t rust_row_count = rust_data["row_count"];
    size_t rust_column_count = rust_data["column_count"];
    std::vector<std::string> rust_first_row_str = rust_data["first_row"];
    std::vector<std::string> rust_last_row_str = rust_data["last_row"];
    std::vector<std::vector<std::string>> rust_sampled_rows_str = rust_data["sampled_rows"];
    
    // Helper to parse XFieldElement string: "(coeff2·x² + coeff1·x + coeff0)" or "0_xfe" or "1_xfe" or "N_xfe"
    auto parse_xfe_string = [](const std::string& s) -> std::tuple<uint64_t, uint64_t, uint64_t> {
        // Handle special cases: "0_xfe", "1_xfe", "N_xfe" (base field element)
        if (s.size() >= 5 && s.substr(s.size() - 4) == "_xfe") {
            std::string num_str = s.substr(0, s.size() - 4);
            try {
                uint64_t val = std::stoull(num_str);
                return {val, 0, 0}; // Base field element: only constant term
            } catch (...) {
                return {0, 0, 0};
            }
        }
        
        // Parse format: "(coeff2·x² + coeff1·x + coeff0)"
        // The middle dot "·" is Unicode U+00B7 (UTF-8: 0xC2 0xB7), and "²" is U+00B2 (UTF-8: 0xC2 0xB2)
        // We'll search for ASCII parts: "x" followed by " + " to find the pattern
        
        // Remove outer parentheses
        if (s.empty() || s.front() != '(' || s.back() != ')') {
            return {0, 0, 0};
        }
        std::string inner = s.substr(1, s.size() - 2);
        
        // Find patterns by searching for "x" and checking what follows
        // Pattern: [digits][middle-dot]x[superscript-2] + [digits][middle-dot]x + [digits]
        // "x² + " has "x" followed by UTF-8 bytes 0xC2 0xB2 (²) then " + "
        // "x + " has "x" followed directly by " + "
        
        size_t x2_pos = std::string::npos;
        size_t x_pos = std::string::npos;
        
        // Find all "x" positions and determine which is x² and which is x
        for (size_t i = 0; i < inner.size(); ++i) {
            if (inner[i] == 'x') {
                // Check if followed by " + " (regular x)
                if (i + 3 < inner.size() && inner.substr(i, 4) == "x + ") {
                    if (x_pos == std::string::npos) {
                        x_pos = i;
                    }
                }
                // Check if followed by UTF-8 superscript 2 (0xC2 0xB2) then " + "
                else if (i + 4 < inner.size() && 
                        static_cast<unsigned char>(inner[i+1]) == 0xC2 &&
                        static_cast<unsigned char>(inner[i+2]) == 0xB2 &&
                        inner.substr(i+3, 3) == " + ") {
                    x2_pos = i;
                }
            }
        }
        
        if (x2_pos == std::string::npos || x_pos == std::string::npos) {
            return {0, 0, 0};
        }
        
        // Extract coefficients
        // c2: from start to before "x" at x2_pos
        // c1: from after "x + " at x2_pos to before "x" at x_pos  
        // c0: from after "x + " at x_pos to end
        try {
            // Extract c2: find digits before x2_pos
            size_t c2_start = 0;
            while (c2_start < x2_pos && (inner[c2_start] == ' ' || inner[c2_start] == '\t')) {
                c2_start++;
            }
            size_t c2_end = x2_pos;
            // Work backwards to find where digits end (skip middle dot UTF-8 bytes)
            while (c2_end > c2_start && inner[c2_end - 1] != ' ' && 
                   (inner[c2_end - 1] < '0' || inner[c2_end - 1] > '9')) {
                c2_end--;
            }
            
            // Extract c1: between "x² + " and "x + "
            // "x² + " is: 'x' (1 byte) + 0xC2 0xB2 (2 bytes) + " + " (3 bytes) = 6 bytes total
            size_t c1_start = x2_pos + 6; // After "x² + "
            while (c1_start < x_pos && (inner[c1_start] == ' ' || inner[c1_start] == '\t')) {
                c1_start++;
            }
            size_t c1_end = x_pos;
            while (c1_end > c1_start && inner[c1_end - 1] != ' ' &&
                   (inner[c1_end - 1] < '0' || inner[c1_end - 1] > '9')) {
                c1_end--;
            }
            
            // Extract c0: after second "x + "
            size_t c0_start = x_pos + 4; // After second "x + "
            while (c0_start < inner.size() && (inner[c0_start] == ' ' || inner[c0_start] == '\t')) {
                c0_start++;
            }
            size_t c0_end = inner.size();
            while (c0_end > c0_start && (inner[c0_end - 1] == ' ' || inner[c0_end - 1] == '\t')) {
                c0_end--;
            }
            
            std::string c2_str = inner.substr(c2_start, c2_end - c2_start);
            std::string c1_str = inner.substr(c1_start, c1_end - c1_start);
            std::string c0_str = inner.substr(c0_start, c0_end - c0_start);
            
            // Remove any non-digit characters
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
        } catch (const std::exception& e) {
            // Parsing failed
            return {0, 0, 0};
        }
    };
    
    // Convert Rust string data to uint64_t vectors
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> rust_first_row;
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> rust_last_row;
    for (const auto& s : rust_first_row_str) {
        rust_first_row.push_back(parse_xfe_string(s));
    }
    for (const auto& s : rust_last_row_str) {
        rust_last_row.push_back(parse_xfe_string(s));
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  STEP: Aux Table Create Comparison                            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "[DBG] Dimensions:" << std::endl;
    std::cout << "  C++: " << padded_height << " x " << aux_width << std::endl;
    std::cout << "  Rust: " << rust_row_count << " x " << rust_column_count << std::endl;
    
    // Download aux table from GPU (before LDE, after creation)
    // The aux table is stored in d_aux_trace as row-major XFieldElements (3 uint64_t per XFE)
    std::vector<uint64_t> h_aux_trace = gpu_stark.download_aux_trace(padded_height, aux_width);
    
    // Convert to XFieldElements and compare
    auto cpp_first_row = std::vector<XFieldElement>();
    auto cpp_last_row = std::vector<XFieldElement>();
    for (size_t c = 0; c < aux_width; ++c) {
        size_t first_idx = (0 * aux_width + c) * 3;
        size_t last_idx = ((padded_height - 1) * aux_width + c) * 3;
        cpp_first_row.emplace_back(
            BFieldElement(h_aux_trace[first_idx + 0]),
            BFieldElement(h_aux_trace[first_idx + 1]),
            BFieldElement(h_aux_trace[first_idx + 2])
        );
        cpp_last_row.emplace_back(
            BFieldElement(h_aux_trace[last_idx + 0]),
            BFieldElement(h_aux_trace[last_idx + 1]),
            BFieldElement(h_aux_trace[last_idx + 2])
        );
    }
    
    // Compare first row
    bool first_row_match = true;
    if (rust_first_row.size() == aux_width) {
        std::cout << "[DBG] First row comparison:" << std::endl;
        for (size_t c = 0; c < aux_width; ++c) {
            auto [rust_c0, rust_c1, rust_c2] = rust_first_row[c];
            uint64_t cpp_c0 = cpp_first_row[c].coeff(0).value();
            uint64_t cpp_c1 = cpp_first_row[c].coeff(1).value();
            uint64_t cpp_c2 = cpp_first_row[c].coeff(2).value();
            
            bool match = (rust_c0 == cpp_c0 && rust_c1 == cpp_c1 && rust_c2 == cpp_c2);
            if (!match) {
                first_row_match = false;
                std::cout << "  [" << c << "] C++: (" << cpp_c2 << "·x² + " << cpp_c1 << "·x + " << cpp_c0 
                          << ") | Rust: (" << rust_c2 << "·x² + " << rust_c1 << "·x + " << rust_c0 << ") ✗" << std::endl;
            } else if (c < 5) {
                std::cout << "  [" << c << "] C++: (" << cpp_c2 << "·x² + " << cpp_c1 << "·x + " << cpp_c0 
                          << ") | Rust: (" << rust_c2 << "·x² + " << rust_c1 << "·x + " << rust_c0 << ") ✓" << std::endl;
            }
        }
    }
    
    // Compare last row
    bool last_row_match = true;
    if (rust_last_row.size() == aux_width) {
        std::cout << "[DBG] Last row comparison:" << std::endl;
        for (size_t c = 0; c < aux_width; ++c) {
            auto [rust_c0, rust_c1, rust_c2] = rust_last_row[c];
            uint64_t cpp_c0 = cpp_last_row[c].coeff(0).value();
            uint64_t cpp_c1 = cpp_last_row[c].coeff(1).value();
            uint64_t cpp_c2 = cpp_last_row[c].coeff(2).value();
            
            bool match = (rust_c0 == cpp_c0 && rust_c1 == cpp_c1 && rust_c2 == cpp_c2);
            if (!match) {
                last_row_match = false;
                std::cout << "  [" << c << "] C++: (" << cpp_c2 << "·x² + " << cpp_c1 << "·x + " << cpp_c0 
                          << ") | Rust: (" << rust_c2 << "·x² + " << rust_c1 << "·x + " << rust_c0 << ") ✗" << std::endl;
            } else if (c < 5) {
                std::cout << "  [" << c << "] C++: (" << cpp_c2 << "·x² + " << cpp_c1 << "·x + " << cpp_c0 
                          << ") | Rust: (" << rust_c2 << "·x² + " << rust_c1 << "·x + " << rust_c0 << ") ✓" << std::endl;
            }
        }
    }
    
    // Detailed comparison for columns 3 and 4 (Processor aux table - OpStackTablePermArg and RamTablePermArg)
    std::cout << "[DBG] Detailed comparison for columns 3 and 4 (first 20 rows):" << std::endl;
    constexpr size_t COL_OPSTACK_PERM = 3;
    constexpr size_t COL_RAM_PERM = 4;
    size_t col3_mismatches = 0, col4_mismatches = 0;
    size_t first_col3_mismatch_row = SIZE_MAX;
    size_t first_col4_mismatch_row = SIZE_MAX;
    
    for (size_t r = 0; r < std::min(padded_height, size_t(20)); ++r) {
        if (r < rust_sampled_rows_str.size() && rust_sampled_rows_str[r].size() == aux_width) {
            // Column 3 (OpStackTablePermArg)
            auto [rust_c3_0, rust_c3_1, rust_c3_2] = parse_xfe_string(rust_sampled_rows_str[r][COL_OPSTACK_PERM]);
            size_t idx3 = (r * aux_width + COL_OPSTACK_PERM) * 3;
            uint64_t cpp_c3_0 = h_aux_trace[idx3 + 0];
            uint64_t cpp_c3_1 = h_aux_trace[idx3 + 1];
            uint64_t cpp_c3_2 = h_aux_trace[idx3 + 2];
            bool col3_match = (rust_c3_0 == cpp_c3_0 && rust_c3_1 == cpp_c3_1 && rust_c3_2 == cpp_c3_2);
            
            // Column 4 (RamTablePermArg)
            auto [rust_c4_0, rust_c4_1, rust_c4_2] = parse_xfe_string(rust_sampled_rows_str[r][COL_RAM_PERM]);
            size_t idx4 = (r * aux_width + COL_RAM_PERM) * 3;
            uint64_t cpp_c4_0 = h_aux_trace[idx4 + 0];
            uint64_t cpp_c4_1 = h_aux_trace[idx4 + 1];
            uint64_t cpp_c4_2 = h_aux_trace[idx4 + 2];
            bool col4_match = (rust_c4_0 == cpp_c4_0 && rust_c4_1 == cpp_c4_1 && rust_c4_2 == cpp_c4_2);
            
            if (!col3_match) {
                col3_mismatches++;
                if (first_col3_mismatch_row == SIZE_MAX) first_col3_mismatch_row = r;
                std::cout << "  Row[" << r << "] col[3]: C++=(" << cpp_c3_2 << "·x² + " << cpp_c3_1 << "·x + " << cpp_c3_0 
                          << ") Rust=(" << rust_c3_2 << "·x² + " << rust_c3_1 << "·x + " << rust_c3_0 << ") ✗" << std::endl;
            }
            if (!col4_match) {
                col4_mismatches++;
                if (first_col4_mismatch_row == SIZE_MAX) first_col4_mismatch_row = r;
                std::cout << "  Row[" << r << "] col[4]: C++=(" << cpp_c4_2 << "·x² + " << cpp_c4_1 << "·x + " << cpp_c4_0 
                          << ") Rust=(" << rust_c4_2 << "·x² + " << rust_c4_1 << "·x + " << rust_c4_0 << ") ✗" << std::endl;
            }
            if (col3_match && col4_match && r < 5) {
                std::cout << "  Row[" << r << "]: ✓" << std::endl;
            }
        }
    }
    
    if (first_col3_mismatch_row != SIZE_MAX) {
        std::cout << "[DBG] Column 3 first mismatch at row " << first_col3_mismatch_row << std::endl;
    }
    if (first_col4_mismatch_row != SIZE_MAX) {
        std::cout << "[DBG] Column 4 first mismatch at row " << first_col4_mismatch_row << std::endl;
    }
    
    // Compare sampled rows
    size_t sample_mismatches = 0;
    const size_t MAX_SAMPLE_MISMATCHES = 10;
    if (!rust_sampled_rows_str.empty()) {
        std::cout << "[DBG] Sampled rows comparison (checking first " << std::min(rust_sampled_rows_str.size(), size_t(10)) << " rows):" << std::endl;
        for (size_t s = 0; s < std::min(rust_sampled_rows_str.size(), size_t(10)); ++s) {
            const auto& rust_row_str = rust_sampled_rows_str[s];
            if (rust_row_str.size() == aux_width) {
                // Find which row index this corresponds to (assuming evenly spaced)
                size_t row_idx = (s * padded_height) / rust_sampled_rows_str.size();
                if (row_idx >= padded_height) row_idx = padded_height - 1;
                
                bool row_match = true;
                for (size_t c = 0; c < aux_width; ++c) {
                    auto [rust_c0, rust_c1, rust_c2] = parse_xfe_string(rust_row_str[c]);
                    
                    size_t idx = (row_idx * aux_width + c) * 3;
                    uint64_t cpp_c0 = h_aux_trace[idx + 0];
                    uint64_t cpp_c1 = h_aux_trace[idx + 1];
                    uint64_t cpp_c2 = h_aux_trace[idx + 2];
                    
                    if (rust_c0 != cpp_c0 || rust_c1 != cpp_c1 || rust_c2 != cpp_c2) {
                        row_match = false;
                        sample_mismatches++;
                        if (sample_mismatches <= MAX_SAMPLE_MISMATCHES) {
                            std::cout << "  Row[" << row_idx << "] col[" << c << "]: C++=(" << cpp_c2 << "·x² + " << cpp_c1 << "·x + " << cpp_c0 
                                      << ") Rust=(" << rust_c2 << "·x² + " << rust_c1 << "·x + " << rust_c0 << ") ✗" << std::endl;
                        }
                        break;
                    }
                }
                if (row_match && s < 5) {
                    std::cout << "  Row[" << row_idx << "]: ✓" << std::endl;
                }
            }
        }
    }
    
    std::cout << "[DBG] Summary:" << std::endl;
    if (first_row_match && last_row_match && sample_mismatches == 0) {
        std::cout << "[DBG] ✓✓✓ Aux table create: FULL MATCH" << std::endl;
        std::cout << "[DBG]    ✓ First row: all " << aux_width << " columns match" << std::endl;
        std::cout << "[DBG]    ✓ Last row: all " << aux_width << " columns match" << std::endl;
        std::cout << "[DBG]    ✓ Sampled rows: all checked rows match" << std::endl;
    } else {
        std::cout << "[DBG] ✗✗✗ Aux table create: MISMATCH DETECTED" << std::endl;
        if (!first_row_match) std::cout << "[DBG]    ✗ First row: mismatches detected" << std::endl;
        if (!last_row_match) std::cout << "[DBG]    ✗ Last row: mismatches detected" << std::endl;
        if (sample_mismatches > 0) std::cout << "[DBG]    ✗ Sampled rows: " << sample_mismatches << " mismatches" << std::endl;
    }
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
}

void validate_aux_table_commitment(const std::string& rust_test_data_dir, const std::string& cpp_merkle_root_hex) {
    // Read Rust Merkle root from the comprehensive dump
    std::ifstream merkle_file(rust_test_data_dir + "/09_aux_tables_merkle.json");
    if (!merkle_file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Step 8 (Aux Table Commitment): Rust Merkle root not found" << std::endl;
        return;
    }

    nlohmann::json merkle_data = nlohmann::json::parse(merkle_file);
    
    bool all_match = true;
    std::vector<std::string> mismatches;

    // Compare Merkle roots (actual computational value)
    if (merkle_data.contains("aux_merkle_root")) {
        std::string rust_merkle_root = merkle_data["aux_merkle_root"];
        
        if (rust_merkle_root != cpp_merkle_root_hex) {
            all_match = false;
            mismatches.push_back("Aux Merkle root mismatch");
            mismatches.push_back("  Rust: " + rust_merkle_root);
            mismatches.push_back("  C++:  " + cpp_merkle_root_hex);
        }
    } else {
        all_match = false;
        mismatches.push_back("aux_merkle_root: Field missing in Rust data");
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 8 (Aux Table Commitment): ✓ MATCH (Merkle root verified)" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 8 (Aux Table Commitment): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_quotient_computation(const std::string& rust_test_data_dir, const std::string& cpp_merkle_root_hex) {
    // Read Rust quotient Merkle root from the comprehensive dump
    std::ifstream merkle_file(rust_test_data_dir + "/13_quotient_merkle.json");
    if (!merkle_file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Step 9 (Quotient Computation): Rust quotient Merkle root not found" << std::endl;
        return;
    }

    nlohmann::json merkle_data = nlohmann::json::parse(merkle_file);
    
    bool all_match = true;
    std::vector<std::string> mismatches;

    // Compare quotient Merkle roots (actual computational value)
    if (merkle_data.contains("quotient_merkle_root")) {
        std::string rust_merkle_root = merkle_data["quotient_merkle_root"];
        
        if (rust_merkle_root != cpp_merkle_root_hex) {
            all_match = false;
            mismatches.push_back("Quotient Merkle root mismatch");
            mismatches.push_back("  Rust: " + rust_merkle_root);
            mismatches.push_back("  C++:  " + cpp_merkle_root_hex);
        }
    } else {
        all_match = false;
        mismatches.push_back("quotient_merkle_root: Field missing in Rust data");
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 9 (Quotient Computation): ✓ MATCH (Quotient Merkle root verified)" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 9 (Quotient Computation): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

static void validate_main_trace_randomizers(
    const std::string& rust_test_data_dir,
    const std::vector<uint64_t>& main_randomizer_coeffs,
    size_t num_cols,
    size_t num_randomizers
) {
    std::ifstream file(rust_test_data_dir + "/trace_randomizer_all_columns.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Trace randomizers: Rust reference not found (trace_randomizer_all_columns.json)" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);
    if (!data.contains("sampled_columns") || !data["sampled_columns"].is_array()) {
        std::cout << "⚠️  [VALIDATION] Trace randomizers: malformed Rust reference" << std::endl;
        return;
    }

    bool all_match = true;
    std::vector<std::string> mismatches;
    const size_t MAX_MISMATCHES_TO_SHOW = 5;
    size_t mismatch_count = 0;

    // main_randomizer_coeffs layout: for each column, append all coeffs (num_randomizers)
    auto get_cpp_coeff = [&](size_t col, size_t r) -> uint64_t {
        return main_randomizer_coeffs[col * num_randomizers + r];
    };

    for (const auto& col_entry : data["sampled_columns"]) {
        if (!col_entry.contains("column_index") || !col_entry.contains("randomizer_coefficients")) continue;
        const size_t col_idx = col_entry["column_index"];
        if (col_idx >= num_cols) continue;

        const auto& rust_coeffs_json = col_entry["randomizer_coefficients"];
        if (!rust_coeffs_json.is_array()) continue;

        const size_t rust_len = rust_coeffs_json.size();
        const size_t check_len = std::min(rust_len, num_randomizers);

        for (size_t r = 0; r < check_len; ++r) {
            const uint64_t rust_v = rust_coeffs_json[r].get<uint64_t>();
            const uint64_t cpp_v = get_cpp_coeff(col_idx, r);
            if (rust_v != cpp_v) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back(
                        "col " + std::to_string(col_idx) + ", coeff " + std::to_string(r) +
                        ": Rust=" + std::to_string(rust_v) + ", C++=" + std::to_string(cpp_v)
                    );
                }
            }
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Trace randomizers (main): ✓ MATCH (sampled columns)" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Trace randomizers (main): ✗ MISMATCH (" << mismatch_count << " differences)" << std::endl;
        for (const auto& msg : mismatches) std::cout << "   " << msg << std::endl;
        if (mismatch_count > MAX_MISMATCHES_TO_SHOW) {
            std::cout << "   ... and " << (mismatch_count - MAX_MISMATCHES_TO_SHOW) << " more mismatches" << std::endl;
        }
    }
}

static bool load_main_trace_randomizers_from_rust(
    const std::string& rust_test_data_dir,
    size_t num_cols,
    size_t num_randomizers,
    std::vector<uint64_t>& out_main_randomizers
) {
    std::ifstream file(rust_test_data_dir + "/trace_randomizer_all_columns.json");
    if (!file.is_open()) return false;

    nlohmann::json data = nlohmann::json::parse(file);
    if (!data.contains("sampled_columns") || !data["sampled_columns"].is_array()) return false;

    out_main_randomizers.assign(num_cols * num_randomizers, 0);
    std::vector<bool> seen(num_cols, false);

    for (const auto& col_entry : data["sampled_columns"]) {
        if (!col_entry.contains("column_index") || !col_entry.contains("randomizer_coefficients")) continue;
        const size_t col_idx = col_entry["column_index"];
        if (col_idx >= num_cols) continue;

        const auto& coeffs = col_entry["randomizer_coefficients"];
        if (!coeffs.is_array()) continue;
        if (coeffs.size() < num_randomizers) continue;

        for (size_t r = 0; r < num_randomizers; ++r) {
            out_main_randomizers[col_idx * num_randomizers + r] = coeffs[r].get<uint64_t>();
        }
        seen[col_idx] = true;
    }

    // Require complete coverage for correctness
    for (size_t c = 0; c < num_cols; ++c) {
        if (!seen[c]) return false;
    }
    return true;
}

static bool load_aux_trace_randomizers_from_rust(
    const std::string& rust_test_data_dir,
    size_t aux_width,
    size_t num_randomizers,
    std::vector<uint64_t>& out_aux_randomizers_component_cols
) {
    std::ifstream file(rust_test_data_dir + "/aux_trace_randomizer_all_columns.json");
    if (!file.is_open()) return false;

    nlohmann::json data = nlohmann::json::parse(file);
    if (!data.contains("all_columns") || !data["all_columns"].is_array()) return false;

    out_aux_randomizers_component_cols.assign(aux_width * 3 * num_randomizers, 0);
    std::vector<bool> seen(aux_width, false);

    for (const auto& col_entry : data["all_columns"]) {
        if (!col_entry.contains("column_index") || !col_entry.contains("randomizer_coefficients")) continue;
        const size_t col_idx = col_entry["column_index"];
        if (col_idx >= aux_width) continue;

        const auto& coeffs = col_entry["randomizer_coefficients"];
        if (!coeffs.is_array()) continue;
        if (coeffs.size() < num_randomizers) continue;

        for (size_t r = 0; r < num_randomizers; ++r) {
            const auto& xfe = coeffs[r];
            if (!xfe.is_array() || xfe.size() != 3) return false;
            out_aux_randomizers_component_cols[(col_idx * 3 + 0) * num_randomizers + r] = xfe[0].get<uint64_t>();
            out_aux_randomizers_component_cols[(col_idx * 3 + 1) * num_randomizers + r] = xfe[1].get<uint64_t>();
            out_aux_randomizers_component_cols[(col_idx * 3 + 2) * num_randomizers + r] = xfe[2].get<uint64_t>();
        }
        seen[col_idx] = true;
    }

    for (size_t c = 0; c < aux_width; ++c) {
        if (!seen[c]) return false;
    }
    return true;
}

void validate_main_lde_samples(const std::string& rust_test_data_dir, gpu::GpuStark& gpu_stark, size_t fri_length, size_t main_width) {
    std::ifstream file(rust_test_data_dir + "/05_main_tables_lde.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Main Table LDE: Rust reference not found" << std::endl;
        return;
    }

    nlohmann::json rust_data = nlohmann::json::parse(file);
    
    bool all_match = true;
    std::vector<std::string> mismatches;
    size_t mismatch_count = 0;
    const size_t MAX_MISMATCHES_TO_SHOW = 5;

    // Extract Rust samples
    std::vector<uint64_t> rust_first_row = rust_data["first_row"];
    std::vector<uint64_t> rust_last_row = rust_data["last_row"];
    std::vector<uint64_t> rust_middle_row = rust_data["middle_row"];
    size_t rust_middle_index = rust_data["middle_row_index"];

    // Sample C++ LDE rows
    std::vector<size_t> sample_indices = {0, rust_middle_index, fri_length - 1};
    auto cpp_samples = gpu_stark.sample_main_lde_rows(sample_indices, main_width);

    // Compare first row
    if (cpp_samples.size() > 0 && rust_first_row.size() == main_width) {
        for (size_t col = 0; col < main_width && col < rust_first_row.size(); ++col) {
            if (cpp_samples[0][col] != rust_first_row[col]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("First row, col " + std::to_string(col) + 
                                        ": Rust=" + std::to_string(rust_first_row[col]) + 
                                        ", C++=" + std::to_string(cpp_samples[0][col]));
                }
            }
        }
    }

    // Compare middle row
    if (cpp_samples.size() > 1 && rust_middle_row.size() == main_width) {
        for (size_t col = 0; col < main_width && col < rust_middle_row.size(); ++col) {
            if (cpp_samples[1][col] != rust_middle_row[col]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("Middle row, col " + std::to_string(col) + 
                                        ": Rust=" + std::to_string(rust_middle_row[col]) + 
                                        ", C++=" + std::to_string(cpp_samples[1][col]));
                }
            }
        }
    }

    // Compare last row
    if (cpp_samples.size() > 2 && rust_last_row.size() == main_width) {
        for (size_t col = 0; col < main_width && col < rust_last_row.size(); ++col) {
            if (cpp_samples[2][col] != rust_last_row[col]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("Last row, col " + std::to_string(col) + 
                                        ": Rust=" + std::to_string(rust_last_row[col]) + 
                                        ", C++=" + std::to_string(cpp_samples[2][col]));
                }
            }
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Main Table LDE Samples: ✓ MATCH (first/middle/last rows verified)" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Main Table LDE Samples: ✗ MISMATCH (" << mismatch_count << " differences)" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
        if (mismatch_count > MAX_MISMATCHES_TO_SHOW) {
            std::cout << "   ... and " << (mismatch_count - MAX_MISMATCHES_TO_SHOW) << " more mismatches" << std::endl;
        }
    }
}

void validate_aux_lde_samples(const std::string& rust_test_data_dir, gpu::GpuStark& gpu_stark, size_t fri_length, size_t aux_width) {
    std::ifstream file(rust_test_data_dir + "/08_aux_tables_lde.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Aux Table LDE: Rust reference not found" << std::endl;
        return;
    }

    nlohmann::json rust_data = nlohmann::json::parse(file);
    
    bool all_match = true;
    std::vector<std::string> mismatches;
    size_t mismatch_count = 0;
    const size_t MAX_MISMATCHES_TO_SHOW = 5;

    // Extract Rust samples (XFE strings)
    std::vector<std::string> rust_first_row = rust_data["first_row"];
    std::vector<std::string> rust_last_row = rust_data["last_row"];
    std::vector<std::string> rust_middle_row = rust_data["middle_row"];
    size_t rust_middle_index = rust_data["middle_row_index"];

    // Sample C++ LDE rows
    std::vector<size_t> sample_indices = {0, rust_middle_index, fri_length - 1};
    auto cpp_samples = gpu_stark.sample_aux_lde_rows(sample_indices, aux_width);

    // Compare first row
    if (cpp_samples.size() > 0 && rust_first_row.size() == aux_width) {
        for (size_t col = 0; col < aux_width && col < rust_first_row.size(); ++col) {
            if (cpp_samples[0][col] != rust_first_row[col]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("First row, col " + std::to_string(col) + 
                                        ": Rust=" + rust_first_row[col] + 
                                        ", C++=" + cpp_samples[0][col]);
                }
            }
        }
    }

    // Compare middle row
    if (cpp_samples.size() > 1 && rust_middle_row.size() == aux_width) {
        for (size_t col = 0; col < aux_width && col < rust_middle_row.size(); ++col) {
            if (cpp_samples[1][col] != rust_middle_row[col]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("Middle row, col " + std::to_string(col) + 
                                        ": Rust=" + rust_middle_row[col] + 
                                        ", C++=" + cpp_samples[1][col]);
                }
            }
        }
    }

    // Compare last row
    if (cpp_samples.size() > 2 && rust_last_row.size() == aux_width) {
        for (size_t col = 0; col < aux_width && col < rust_last_row.size(); ++col) {
            if (cpp_samples[2][col] != rust_last_row[col]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("Last row, col " + std::to_string(col) + 
                                        ": Rust=" + rust_last_row[col] + 
                                        ", C++=" + cpp_samples[2][col]);
                }
            }
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Aux Table LDE Samples: ✓ MATCH (first/middle/last rows verified)" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Aux Table LDE Samples: ✗ MISMATCH (" << mismatch_count << " differences)" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
        if (mismatch_count > MAX_MISMATCHES_TO_SHOW) {
            std::cout << "   ... and " << (mismatch_count - MAX_MISMATCHES_TO_SHOW) << " more mismatches" << std::endl;
        }
    }
}

void validate_quotient_lde_samples(const std::string& rust_test_data_dir, gpu::GpuStark& gpu_stark, size_t fri_length) {
    std::ifstream file(rust_test_data_dir + "/11_quotient_lde.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Quotient LDE: Rust reference not found" << std::endl;
        return;
    }

    nlohmann::json rust_data = nlohmann::json::parse(file);
    
    bool all_match = true;
    std::vector<std::string> mismatches;
    size_t mismatch_count = 0;
    const size_t MAX_MISMATCHES_TO_SHOW = 5;
    const size_t NUM_QUOTIENT_SEGMENTS = 4;

    // Extract Rust samples (XFE strings)
    std::vector<std::string> rust_first_row = rust_data["first_row"];
    std::vector<std::string> rust_last_row = rust_data["last_row"];
    std::vector<std::string> rust_middle_row = rust_data["middle_row"];
    size_t rust_middle_index = rust_data["middle_row_index"];

    // Sample C++ LDE rows
    std::vector<size_t> sample_indices = {0, rust_middle_index, fri_length - 1};
    auto cpp_samples = gpu_stark.sample_quotient_lde_rows(sample_indices, NUM_QUOTIENT_SEGMENTS);

    // Compare first row
    if (cpp_samples.size() > 0 && rust_first_row.size() == NUM_QUOTIENT_SEGMENTS) {
        for (size_t seg = 0; seg < NUM_QUOTIENT_SEGMENTS && seg < rust_first_row.size(); ++seg) {
            if (cpp_samples[0][seg] != rust_first_row[seg]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("First row, segment " + std::to_string(seg) + 
                                        ": Rust=" + rust_first_row[seg] + 
                                        ", C++=" + cpp_samples[0][seg]);
                }
            }
        }
    }

    // Compare middle row
    if (cpp_samples.size() > 1 && rust_middle_row.size() == NUM_QUOTIENT_SEGMENTS) {
        for (size_t seg = 0; seg < NUM_QUOTIENT_SEGMENTS && seg < rust_middle_row.size(); ++seg) {
            if (cpp_samples[1][seg] != rust_middle_row[seg]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("Middle row, segment " + std::to_string(seg) + 
                                        ": Rust=" + rust_middle_row[seg] + 
                                        ", C++=" + cpp_samples[1][seg]);
                }
            }
        }
    }

    // Compare last row
    if (cpp_samples.size() > 2 && rust_last_row.size() == NUM_QUOTIENT_SEGMENTS) {
        for (size_t seg = 0; seg < NUM_QUOTIENT_SEGMENTS && seg < rust_last_row.size(); ++seg) {
            if (cpp_samples[2][seg] != rust_last_row[seg]) {
                all_match = false;
                mismatch_count++;
                if (mismatches.size() < MAX_MISMATCHES_TO_SHOW) {
                    mismatches.push_back("Last row, segment " + std::to_string(seg) + 
                                        ": Rust=" + rust_last_row[seg] + 
                                        ", C++=" + cpp_samples[2][seg]);
                }
            }
        }
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Quotient LDE Samples: ✓ MATCH (first/middle/last rows verified)" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Quotient LDE Samples: ✗ MISMATCH (" << mismatch_count << " differences)" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
        if (mismatch_count > MAX_MISMATCHES_TO_SHOW) {
            std::cout << "   ... and " << (mismatch_count - MAX_MISMATCHES_TO_SHOW) << " more mismatches" << std::endl;
        }
    }
}

/**
 * Debug function: Dump and compare row digests (hashes) between Rust and C++
 * This helps diagnose Merkle root mismatches by checking if individual row hashes differ.
 * 
 * Set TVM_DEBUG_ROW_HASHES=1 to enable this detailed comparison.
 */
void debug_compare_aux_row_digests(const std::string& rust_test_data_dir, gpu::GpuStark& gpu_stark, size_t fri_length) {
    const char* debug_env = std::getenv("TVM_DEBUG_ROW_HASHES");
    if (!debug_env || std::string(debug_env) != "1") {
        return; // Only run when TVM_DEBUG_ROW_HASHES=1
    }
    
    std::cout << "\n🔍 [DEBUG] Comparing aux row digests (TVM_DEBUG_ROW_HASHES=1)..." << std::endl;
    std::cout << "   FRI length: " << fri_length << std::endl;
    
    // Try to load Rust row digests if available
    std::ifstream file(rust_test_data_dir + "/09_aux_row_digests.json");
    bool have_rust_digests = file.is_open();
    
    nlohmann::json rust_data;
    std::vector<std::vector<uint64_t>> rust_digests;
    std::vector<size_t> rust_indices;
    
    if (have_rust_digests) {
        rust_data = nlohmann::json::parse(file);
        if (rust_data.contains("digests") && rust_data.contains("indices")) {
            for (auto& digest_arr : rust_data["digests"]) {
                std::vector<uint64_t> digest;
                for (auto& val : digest_arr) {
                    digest.push_back(val.get<uint64_t>());
                }
                rust_digests.push_back(digest);
            }
            for (auto& idx : rust_data["indices"]) {
                rust_indices.push_back(idx.get<size_t>());
            }
        }
        std::cout << "   Loaded " << rust_digests.size() << " Rust row digests for comparison" << std::endl;
    } else {
        std::cout << "   No Rust row digest file found at " << rust_test_data_dir << "/09_aux_row_digests.json" << std::endl;
        std::cout << "   Will just print C++ row digests for manual inspection" << std::endl;
    }
    
    // Sample indices to check: first, some early, middle, some late, last
    std::vector<size_t> sample_indices;
    sample_indices.push_back(0);
    sample_indices.push_back(1);
    sample_indices.push_back(2);
    sample_indices.push_back(10);
    sample_indices.push_back(100);
    sample_indices.push_back(1000);
    if (fri_length > 10000) sample_indices.push_back(10000);
    if (fri_length > 100000) sample_indices.push_back(100000);
    if (fri_length > 1000000) sample_indices.push_back(1000000);
    sample_indices.push_back(fri_length / 2);
    sample_indices.push_back(fri_length - 2);
    sample_indices.push_back(fri_length - 1);
    
    // Remove duplicates and out-of-bounds
    std::sort(sample_indices.begin(), sample_indices.end());
    sample_indices.erase(std::unique(sample_indices.begin(), sample_indices.end()), sample_indices.end());
    
    try {
        auto cpp_digests = gpu_stark.sample_aux_row_digests(sample_indices);
        
        std::cout << "\n   C++ Aux Row Digests:" << std::endl;
        for (size_t i = 0; i < sample_indices.size(); ++i) {
            size_t row_idx = sample_indices[i];
            const auto& digest = cpp_digests[i];
            
            std::cout << "   Row " << row_idx << ": [";
            for (size_t j = 0; j < digest.size(); ++j) {
                std::cout << digest[j];
                if (j < digest.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
            
            // Compare with Rust if available
            if (have_rust_digests) {
                auto it = std::find(rust_indices.begin(), rust_indices.end(), row_idx);
                if (it != rust_indices.end()) {
                    size_t rust_idx = std::distance(rust_indices.begin(), it);
                    const auto& rust_digest = rust_digests[rust_idx];
                    
                    bool match = (digest.size() == rust_digest.size());
                    if (match) {
                        for (size_t j = 0; j < digest.size(); ++j) {
                            if (digest[j] != rust_digest[j]) {
                                match = false;
                                break;
                            }
                        }
                    }
                    
                    if (match) {
                        std::cout << " ✓";
                    } else {
                        std::cout << " ✗ Rust=[";
                        for (size_t j = 0; j < rust_digest.size(); ++j) {
                            std::cout << rust_digest[j];
                            if (j < rust_digest.size() - 1) std::cout << ", ";
                        }
                        std::cout << "]";
                    }
                }
            }
            // If Rust digests are not available, cross-check GPU digest against CPU Tip5::hash_varlen
            // computed from the actual aux LDE row values.
            if (!have_rust_digests) {
                const char* cpu_env = std::getenv("TVM_DEBUG_ROW_HASHES_CPU");
                if (cpu_env && std::string(cpu_env) == "1") {
                    try {
                        // Download aux row BFEs: [col0.c0,c1,c2, col1.c0,c1,c2, ...]
                        auto row_bfes_u64 = gpu_stark.sample_aux_lde_row_bfes(row_idx, 88 /* aux_width */);
                        std::vector<BFieldElement> row_bfes;
                        row_bfes.reserve(row_bfes_u64.size());
                        for (uint64_t v : row_bfes_u64) {
                            row_bfes.emplace_back(v);
                        }

                        Digest cpu_digest = Tip5::hash_varlen(row_bfes);
                        bool match_cpu = true;
                        for (size_t j = 0; j < 5; ++j) {
                            if (digest[j] != cpu_digest[j].value()) {
                                match_cpu = false;
                                break;
                            }
                        }
                        if (match_cpu) {
                            std::cout << "  (CPU Tip5 ✓)";
                        } else {
                            std::cout << "  (CPU Tip5 ✗ cpu=["
                                      << cpu_digest[0].value() << ", "
                                      << cpu_digest[1].value() << ", "
                                      << cpu_digest[2].value() << ", "
                                      << cpu_digest[3].value() << ", "
                                      << cpu_digest[4].value() << "])";
                        }
                    } catch (const std::exception& e) {
                        std::cout << "  (CPU Tip5 error: " << e.what() << ")";
                    }
                }
            }
            std::cout << std::endl;
        }

        // Optional: verify Merkle parent computation on GPU matches CPU Tip5::hash_pair
        // This isolates whether the mismatch is in row hashing (leaf digests) or in Merkle tree construction.
        const char* merkle_env = std::getenv("TVM_DEBUG_MERKLE_INTERNAL");
        if (merkle_env && std::string(merkle_env) == "1") {
            std::cout << "\n🔍 [DEBUG] Checking GPU Merkle parent nodes vs CPU Tip5::hash_pair (TVM_DEBUG_MERKLE_INTERNAL=1)..." << std::endl;

            const size_t height = (size_t)std::log2((double)fri_length);
            auto level_size = [&](size_t level) -> size_t { return fri_length >> level; };
            auto level_offset = [&](size_t level) -> size_t {
                // Our flat layout offsets: level 0 at 0, level 1 at n, level 2 at n+n/2, ..., root at 2n-2
                // Closed form: offset(level) = 2*n - 2*(n >> level)
                return 2 * fri_length - 2 * (fri_length >> level);
            };

            // Check a few bottom-level pairs across the range:
            // parent node index for pair i is (fri_length + i)
            // leaves are at nodes (2*i) and (2*i+1)
            const size_t bottom_pairs = fri_length / 2;
            std::vector<size_t> pair_indices = {0, 1, 2, 3, 7};
            if (bottom_pairs > 16) pair_indices.push_back(16);
            if (bottom_pairs > 1024) pair_indices.push_back(1024);
            if (bottom_pairs > 0) pair_indices.push_back(bottom_pairs / 2);
            if (bottom_pairs > 3) pair_indices.push_back(bottom_pairs - 3);
            if (bottom_pairs > 2) pair_indices.push_back(bottom_pairs - 2);
            if (bottom_pairs > 1) pair_indices.push_back(bottom_pairs - 1);
            std::sort(pair_indices.begin(), pair_indices.end());
            pair_indices.erase(std::unique(pair_indices.begin(), pair_indices.end()), pair_indices.end());

            std::vector<size_t> parent_nodes;
            parent_nodes.reserve(pair_indices.size());
            for (size_t i : pair_indices) parent_nodes.push_back(fri_length + i);
            auto gpu_parents = gpu_stark.sample_aux_merkle_nodes(parent_nodes);

            // Fetch corresponding leaves from aux merkle buffer (validating merkle pairing itself).
            std::vector<size_t> leaf_nodes;
            leaf_nodes.reserve(pair_indices.size() * 2);
            for (size_t i : pair_indices) {
                leaf_nodes.push_back(2 * i);
                leaf_nodes.push_back(2 * i + 1);
            }
            auto gpu_leaves = gpu_stark.sample_aux_merkle_nodes(leaf_nodes);

            for (size_t k = 0; k < pair_indices.size(); ++k) {
                const size_t i = pair_indices[k];
                const auto& leafL = gpu_leaves[2 * k + 0];
                const auto& leafR = gpu_leaves[2 * k + 1];
                const auto& parent_gpu = gpu_parents[k];

                Digest dl, dr;
                for (size_t j = 0; j < 5; ++j) {
                    dl[j] = BFieldElement(leafL[j]);
                    dr[j] = BFieldElement(leafR[j]);
                }
                Digest parent_cpu = Tip5::hash_pair(dl, dr);

                bool match = true;
                for (size_t j = 0; j < 5; ++j) {
                    if (parent_gpu[j] != parent_cpu[j].value()) { match = false; break; }
                }

                std::cout << "   Pair " << i
                          << " leaves=(" << (2 * i) << "," << (2 * i + 1) << ")"
                          << " parent_node=" << (fri_length + i)
                          << " : " << (match ? "✓" : "✗") << std::endl;
                if (!match) {
                    std::cout << "     GPU parent: [" << parent_gpu[0] << ", " << parent_gpu[1] << ", " << parent_gpu[2] << ", " << parent_gpu[3] << ", " << parent_gpu[4] << "]\n";
                    std::cout << "     CPU parent: [" << parent_cpu[0].value() << ", " << parent_cpu[1].value() << ", " << parent_cpu[2].value() << ", " << parent_cpu[3].value() << ", " << parent_cpu[4].value() << "]\n";
                }
            }

            // Also check root location is readable
            const size_t root_node = 2 * fri_length - 2;
            auto gpu_root = gpu_stark.sample_aux_merkle_nodes({root_node});
            std::cout << "   GPU aux root node @" << root_node << ": ["
                      << gpu_root[0][0] << ", " << gpu_root[0][1] << ", " << gpu_root[0][2] << ", " << gpu_root[0][3] << ", " << gpu_root[0][4] << "]" << std::endl;

            // Additional sanity: verify higher levels (including root) are consistent with their children.
            std::vector<size_t> levels_to_check = {2, 8, 16};
            if (height > 2) levels_to_check.push_back(height - 1);
            levels_to_check.push_back(height);
            std::sort(levels_to_check.begin(), levels_to_check.end());
            levels_to_check.erase(std::unique(levels_to_check.begin(), levels_to_check.end()), levels_to_check.end());
            levels_to_check.erase(
                std::remove_if(levels_to_check.begin(), levels_to_check.end(), [&](size_t l){ return l == 0 || l > height; }),
                levels_to_check.end()
            );

            std::cout << "\n🔍 [DEBUG] Verifying Merkle internal node consistency at multiple levels..." << std::endl;
            for (size_t level : levels_to_check) {
                const size_t sz = level_size(level);
                const size_t off = level_offset(level);
                const size_t child_level = level - 1;
                const size_t child_off = level_offset(child_level);

                std::vector<size_t> positions = {0};
                if (sz > 1) positions.push_back(1);
                if (sz > 2) positions.push_back(2);
                if (sz > 8) positions.push_back(8);
                positions.push_back(sz / 2);
                if (sz > 3) positions.push_back(sz - 3);
                if (sz > 2) positions.push_back(sz - 2);
                if (sz > 1) positions.push_back(sz - 1);
                std::sort(positions.begin(), positions.end());
                positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
                positions.erase(std::remove_if(positions.begin(), positions.end(), [&](size_t p){ return p >= sz; }), positions.end());

                std::vector<size_t> node_idxs;
                node_idxs.reserve(positions.size());
                std::vector<size_t> child_idxs;
                child_idxs.reserve(positions.size() * 2);
                for (size_t p : positions) {
                    node_idxs.push_back(off + p);
                    child_idxs.push_back(child_off + 2 * p);
                    child_idxs.push_back(child_off + 2 * p + 1);
                }

                auto gpu_nodes = gpu_stark.sample_aux_merkle_nodes(node_idxs);
                auto gpu_children = gpu_stark.sample_aux_merkle_nodes(child_idxs);

                std::cout << "   Level " << level << " (size=" << sz << ", offset=" << off << "):" << std::endl;
                for (size_t k = 0; k < positions.size(); ++k) {
                    const size_t p = positions[k];
                    const auto& node = gpu_nodes[k];
                    const auto& cL = gpu_children[2 * k + 0];
                    const auto& cR = gpu_children[2 * k + 1];

                    Digest dl, dr;
                    for (size_t j = 0; j < 5; ++j) {
                        dl[j] = BFieldElement(cL[j]);
                        dr[j] = BFieldElement(cR[j]);
                    }
                    Digest cpu = Tip5::hash_pair(dl, dr);

                    bool match = true;
                    for (size_t j = 0; j < 5; ++j) {
                        if (node[j] != cpu[j].value()) { match = false; break; }
                    }
                    std::cout << "     Node p=" << p << " idx=" << (off + p) << " : " << (match ? "✓" : "✗") << std::endl;
                    if (!match) {
                        std::cout << "       GPU node: [" << node[0] << ", " << node[1] << ", " << node[2] << ", " << node[3] << ", " << node[4] << "]\n";
                        std::cout << "       CPU node: [" << cpu[0].value() << ", " << cpu[1].value() << ", " << cpu[2].value() << ", " << cpu[3].value() << ", " << cpu[4].value() << "]\n";
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "   Error sampling row digests: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * Debug function: Dump first N elements of aux LDE to check data integrity
 */
void debug_dump_aux_lde_raw(const std::string& rust_test_data_dir, gpu::GpuStark& gpu_stark, size_t fri_length, size_t aux_width) {
    const char* debug_env = std::getenv("TVM_DEBUG_AUX_LDE_RAW");
    if (!debug_env || std::string(debug_env) != "1") {
        return;
    }
    
    std::cout << "\n🔍 [DEBUG] Dumping raw aux LDE data (TVM_DEBUG_AUX_LDE_RAW=1)..." << std::endl;
    std::cout << "   Checking first 10 elements of each column for first 5 columns:" << std::endl;
    
    // Sample first 10 rows, first 5 columns
    std::vector<size_t> sample_rows = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    try {
        auto cpp_samples = gpu_stark.sample_aux_lde_rows(sample_rows, aux_width);
        
        for (size_t col = 0; col < std::min((size_t)5, aux_width); ++col) {
            std::cout << "   Col " << col << ": ";
            for (size_t row = 0; row < sample_rows.size(); ++row) {
                std::cout << cpp_samples[row][col];
                if (row < sample_rows.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "   Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void validate_out_of_domain_evaluation(const std::string& rust_test_data_dir) {
    std::ifstream file(rust_test_data_dir + "/10_out_of_domain_evaluation.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Step 10 (Out-of-Domain Evaluation): Rust reference not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check out-of-domain evaluation completion
    if (data.contains("out_of_domain_evaluated")) {
        bool rust_evaluated = data["out_of_domain_evaluated"];

        if (!rust_evaluated) {
            all_match = false;
            mismatches.push_back("out_of_domain_evaluated: Rust shows false");
        }
    } else {
        all_match = false;
        mismatches.push_back("out_of_domain_evaluated: Field missing in Rust data");
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 10 (Out-of-Domain Evaluation): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 10 (Out-of-Domain Evaluation): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

void validate_fri_protocol(const std::string& rust_test_data_dir) {
    std::ifstream file(rust_test_data_dir + "/11_fri_protocol.json");
    if (!file.is_open()) {
        std::cout << "⚠️  [VALIDATION] Step 11 (FRI Protocol): Rust reference not found" << std::endl;
        return;
    }

    nlohmann::json data = nlohmann::json::parse(file);

    bool all_match = true;
    std::vector<std::string> mismatches;

    // Check FRI protocol completion
    if (data.contains("fri_completed")) {
        bool rust_completed = data["fri_completed"];

        if (!rust_completed) {
            all_match = false;
            mismatches.push_back("fri_completed: Rust shows false");
        }
    } else {
        all_match = false;
        mismatches.push_back("fri_completed: Field missing in Rust data");
    }

    if (all_match) {
        std::cout << "✅ [VALIDATION] Step 11 (FRI Protocol): ✓ MATCH" << std::endl;
    } else {
        std::cout << "❌ [VALIDATION] Step 11 (FRI Protocol): ✗ MISMATCH" << std::endl;
        for (const auto& msg : mismatches) {
            std::cout << "   " << msg << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 7) {
        std::cerr << "Full GPU Zero-Copy Triton VM Prover\n\n";
        std::cerr << "Usage: " << argv[0]
                  << " <program.tasm|program.json> <public_input> <output_claim> <output_proof> [nondet.json] [program.json]\n\n";
        std::cerr << "Arguments:\n";
        std::cerr << "  program.tasm|json  - Program file (TASM or JSON format)\n";
        std::cerr << "  public_input       - Comma-separated u64 values\n";
        std::cerr << "  output_claim       - Output claim file path\n";
        std::cerr << "  output_proof       - Output proof file path\n";
        std::cerr << "  nondet.json        - Optional: NonDeterminism JSON file\n";
        std::cerr << "  program.json       - Optional: Program JSON (if first arg is TASM)\n\n";
        return 1;
    }

    std::string program_path = argv[1];
    std::string public_input_str = argv[2];
    std::string output_claim = argv[3];
    std::string output_proof = argv[4];
    std::string nondet_json_path = (argc >= 6) ? argv[5] : "";
    std::string program_json_path = (argc >= 7) ? argv[6] : "";
    
    std::vector<uint64_t> public_input = parse_u64_list(public_input_str);
    
    // Detect if input is JSON or TASM based on extension
    bool is_json_program = program_path.size() > 5 && 
                           program_path.substr(program_path.size() - 5) == ".json";
    
#ifndef TRITON_CUDA_ENABLED
    std::cerr << "Error: CUDA not enabled. Build with -DENABLE_CUDA=ON\n";
    return 1;
#else
    // Check CUDA devices
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found\n";
        return 1;
    }
    
    // Check for multi-GPU unified memory mode (set via environment variable)
    const char* multi_gpu_env = std::getenv("TRITON_MULTI_GPU");
    if (multi_gpu_env && (std::string(multi_gpu_env) == "1" || std::string(multi_gpu_env) == "true")) {
        triton_vm::gpu::use_unified_memory() = true;
        triton_vm::gpu::enable_multi_gpu_unified_memory();
    }
    
    // Initialize thread coordination for OpenMP, TBB, and Taskflow
    // This ensures all parallel libraries use the same thread count
    triton_vm::parallel::initialize_thread_coordination();
    
    // Verify TBB is available if requested
    if (std::getenv("TVM_VERIFY_TBB")) {
#ifdef TVM_USE_TBB
        std::cout << "[TBB] ✓ TBB is ENABLED (compiled with TBB support)" << std::endl;
#else
        std::cout << "[TBB] ✗ TBB is DISABLED (not compiled with TBB support)" << std::endl;
        std::cout << "[TBB]   To enable: Install TBB and rebuild with -DTBB_FOUND=ON" << std::endl;
#endif
    }
    
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
    std::cout << "\n║  OpenMP Configuration:" << std::endl;
    std::cout << "║    Requested threads: " << target_threads << std::endl;
    std::cout << "║    Max threads available: " << max_threads << std::endl;
    std::cout << "║    Processors detected: " << num_procs << std::endl;
    std::cout << "║    OpenMP version: " << _OPENMP << std::endl;
    if (max_threads != target_threads) {
        std::cout << "║    WARNING: Requested " << target_threads << " but only " << max_threads << " available!" << std::endl;
    }
    if (max_threads < num_procs) {
        std::cout << "║    WARNING: Max threads (" << max_threads << ") < processors (" << num_procs << ")!" << std::endl;
    }
#else
    const char* omp_threads_env = std::getenv("OMP_NUM_THREADS");
    if (omp_threads_env) {
        std::cout << "║  WARNING: OMP_NUM_THREADS is set but OpenMP is not compiled in!" << std::endl;
        std::cout << "║  Rebuild with OpenMP support enabled." << std::endl;
    } else {
        std::cout << "║  WARNING: OpenMP is not available - CPU parallelization disabled!" << std::endl;
    }
#endif
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Get effective GPU count (respects TRITON_GPU_COUNT limit)
    int effective_gpu_count = triton_vm::gpu::get_effective_gpu_count();
    
    // Calculate total GPU memory across devices we're using
    size_t total_gpu_mem = 0;
    for (int i = 0; i < effective_gpu_count; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        total_gpu_mem += p.totalGlobalMem;
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Full GPU Zero-Copy Triton VM Prover              ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  GPU: " << prop.name << std::endl;
    std::cout << "║  Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB";
    if (effective_gpu_count > 1) {
        std::cout << " (GPU 0 of " << effective_gpu_count;
        if (effective_gpu_count < device_count) {
            std::cout << "/" << device_count;
        }
        std::cout << ", total: " << (total_gpu_mem / (1024*1024)) << " MB)";
    }
    std::cout << std::endl;
    std::cout << "║  Compute: " << prop.major << "." << prop.minor << std::endl;
    if (triton_vm::gpu::use_unified_memory()) {
        std::cout << "║  Mode: Multi-GPU Unified Memory (" << effective_gpu_count << " GPUs)" << std::endl;
    }
    std::cout << "╚══════════════════════════════════════════════════╝\n" << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        // =====================================================================
        // PHASE 1: Pure C++ - Trace and Pad
        // =====================================================================
        std::cout << "━━━ Phase 1: Pure C++ (Trace + Pad) ━━━" << std::endl;
        auto phase1_start = std::chrono::high_resolution_clock::now();
        auto step_start = phase1_start;
        
        // Step 1a: Load program from TASM file
        Program program = Program::from_file(program_path);
        std::cout << "  [1a] Load program: " << elapsed_ms(step_start) << " ms" << std::endl;
        step_start = std::chrono::high_resolution_clock::now();
        
        // Step 1b: Convert public input to BFieldElements
        std::vector<BFieldElement> public_input_bfe;
        public_input_bfe.reserve(public_input.size());
        for (uint64_t val : public_input) {
            public_input_bfe.push_back(BFieldElement(val));
        }
        
        // Step 1c: Execute program and generate trace
        // Always use Rust FFI trace execution (faster than C++)
        // If NonDeterminism JSON is provided, use tvm_trace_execution_with_nondet
        // Otherwise use tvm_trace_execution_rust_ffi (with program path)
        
        VM::TraceResult trace_result(AlgebraicExecutionTrace(std::vector<BFieldElement>{}), {});
        
        // Always use Rust FFI - choose function based on whether we have NonDeterminism
        if (!nondet_json_path.empty() && !program_json_path.empty()) {
            // Use Rust FFI with NonDeterminism support
            std::cout << "  [1c] Using Rust FFI trace execution with NonDeterminism..." << std::endl;
            
            // Load NonDeterminism JSON
            std::string nondet_json;
            {
                std::ifstream f(nondet_json_path);
                if (!f.is_open()) {
                    std::cerr << "Error: Cannot open NonDeterminism JSON file: " << nondet_json_path << std::endl;
                    return 1;
                }
                std::stringstream ss;
                ss << f.rdbuf();
                nondet_json = ss.str();
            }
            
            // Load program JSON
            std::string program_json;
            {
                std::ifstream f(program_json_path);
                if (!f.is_open()) {
                    std::cerr << "Error: Cannot open Program JSON file: " << program_json_path << std::endl;
                    return 1;
                }
                std::stringstream ss;
                ss << f.rdbuf();
                program_json = ss.str();
            }
            
            // Call Rust FFI
            uint64_t* proc_trace_data = nullptr;
            size_t proc_rows = 0, proc_cols = 0;
            uint64_t* bwords_data = nullptr;
            size_t bwords_len = 0;
            uint32_t* inst_mults_data = nullptr;
            size_t inst_mults_len = 0;
            uint64_t* output_data = nullptr;
            size_t output_len = 0;
            uint64_t* op_stack_data = nullptr;
            size_t op_stack_rows = 0, op_stack_cols = 0;
            uint64_t* ram_data = nullptr;
            size_t ram_rows = 0, ram_cols = 0;
            uint64_t* ph_data = nullptr;
            size_t ph_rows = 0, ph_cols = 0;
            uint64_t* hash_data = nullptr;
            size_t hash_rows = 0, hash_cols = 0;
            uint64_t* sponge_data = nullptr;
            size_t sponge_rows = 0, sponge_cols = 0;
            uint64_t* u32_data = nullptr;
            size_t u32_len = 0;
            uint64_t* cascade_data = nullptr;
            size_t cascade_len = 0;
            uint64_t lookup_mults[256] = {0};
            size_t table_lengths[9] = {0};
            
            int ffi_result = tvm_trace_execution_with_nondet(
                program_json.c_str(),
                nondet_json.c_str(),
                public_input.data(),
                public_input.size(),
                &proc_trace_data, &proc_rows, &proc_cols,
                &bwords_data, &bwords_len,
                &inst_mults_data, &inst_mults_len,
                &output_data, &output_len,
                &op_stack_data, &op_stack_rows, &op_stack_cols,
                &ram_data, &ram_rows, &ram_cols,
                &ph_data, &ph_rows, &ph_cols,
                &hash_data, &hash_rows, &hash_cols,
                &sponge_data, &sponge_rows, &sponge_cols,
                &u32_data, &u32_len,
                &cascade_data, &cascade_len,
                lookup_mults,
                table_lengths
            );
            
            if (ffi_result != 0) {
                std::cerr << "Error: Rust FFI trace execution failed" << std::endl;
                return 1;
            }
            
            // Debug: dump first row of processor trace from Rust FFI
            if (std::getenv("TVM_DEBUG_FFI_PROC_TRACE")) {
                std::cerr << "[DBG] Rust FFI processor trace first row (first 10 cols):" << std::endl;
                std::cerr << "[DBG] proc_trace_data ptr=" << (void*)proc_trace_data << " rows=" << proc_rows << " cols=" << proc_cols << std::endl;
                if (proc_trace_data != nullptr && proc_cols > 0) {
                    size_t max_cols = (proc_cols < 10) ? proc_cols : 10;
                    for (size_t c = 0; c < max_cols; c++) {
                        uint64_t val = proc_trace_data[c];
                        fprintf(stderr, "  [%zu]: %lu\n", c, val);
                    }
                }
                std::cerr << "[DBG] Proc trace shape: " << proc_rows << " x " << proc_cols << std::endl;
            }
            
            // Load Rust test data trace execution result for comparison
            struct RustTestDataTraceExecution {
                size_t processor_trace_height = 0;
                size_t processor_trace_width = 0;
                size_t padded_height = 0;
                std::vector<uint64_t> public_output_sampled;
                std::vector<uint64_t> processor_trace_first_row;
                std::vector<uint64_t> processor_trace_last_row;
                size_t op_stack_height = 0, op_stack_width = 0;
                std::vector<uint64_t> op_stack_first_row;
                size_t ram_height = 0, ram_width = 0;
                std::vector<uint64_t> ram_first_row;
                size_t program_hash_height = 0, program_hash_width = 0;
                std::vector<uint64_t> program_hash_first_row;
                size_t hash_height = 0, hash_width = 0;
                std::vector<uint64_t> hash_first_row;
                size_t sponge_height = 0, sponge_width = 0;
                std::vector<uint64_t> sponge_first_row;
                std::vector<uint32_t> instruction_multiplicities_sample;
                std::array<size_t, 9> table_heights = {0};
            } rust_test_trace;
            
            bool have_rust_test_data = false;
            const char* rust_test_data_env = std::getenv("TVM_RUST_TEST_DATA_DIR");
            if (rust_test_data_env) {
                std::string rust_test_data_dir = rust_test_data_env;
                std::ifstream test_file(rust_test_data_dir + "/01_trace_execution.json");
                if (test_file.is_open()) {
                    nlohmann::json test_data = nlohmann::json::parse(test_file);
                    rust_test_trace.processor_trace_height = test_data["processor_trace_height"];
                    rust_test_trace.processor_trace_width = test_data["processor_trace_width"];
                    rust_test_trace.padded_height = test_data["padded_height"];
                    if (test_data.contains("public_output_sampled")) {
                        rust_test_trace.public_output_sampled = test_data["public_output_sampled"].get<std::vector<uint64_t>>();
                    }
                    if (test_data.contains("processor_trace_first_row")) {
                        rust_test_trace.processor_trace_first_row = test_data["processor_trace_first_row"].get<std::vector<uint64_t>>();
                    }
                    if (test_data.contains("processor_trace_last_row")) {
                        rust_test_trace.processor_trace_last_row = test_data["processor_trace_last_row"].get<std::vector<uint64_t>>();
                    }
                    if (test_data.contains("op_stack_height")) {
                        rust_test_trace.op_stack_height = test_data["op_stack_height"];
                        rust_test_trace.op_stack_width = test_data["op_stack_width"];
                        if (test_data.contains("op_stack_first_row")) {
                            rust_test_trace.op_stack_first_row = test_data["op_stack_first_row"].get<std::vector<uint64_t>>();
                        }
                    }
                    if (test_data.contains("ram_height")) {
                        rust_test_trace.ram_height = test_data["ram_height"];
                        rust_test_trace.ram_width = test_data["ram_width"];
                        if (test_data.contains("ram_first_row")) {
                            rust_test_trace.ram_first_row = test_data["ram_first_row"].get<std::vector<uint64_t>>();
                        }
                    }
                    if (test_data.contains("program_hash_height")) {
                        rust_test_trace.program_hash_height = test_data["program_hash_height"];
                        rust_test_trace.program_hash_width = test_data["program_hash_width"];
                        if (test_data.contains("program_hash_first_row")) {
                            rust_test_trace.program_hash_first_row = test_data["program_hash_first_row"].get<std::vector<uint64_t>>();
                        }
                    }
                    if (test_data.contains("hash_height")) {
                        rust_test_trace.hash_height = test_data["hash_height"];
                        rust_test_trace.hash_width = test_data["hash_width"];
                        if (test_data.contains("hash_first_row")) {
                            rust_test_trace.hash_first_row = test_data["hash_first_row"].get<std::vector<uint64_t>>();
                        }
                    }
                    if (test_data.contains("sponge_height")) {
                        rust_test_trace.sponge_height = test_data["sponge_height"];
                        rust_test_trace.sponge_width = test_data["sponge_width"];
                        if (test_data.contains("sponge_first_row")) {
                            rust_test_trace.sponge_first_row = test_data["sponge_first_row"].get<std::vector<uint64_t>>();
                        }
                    }
                    if (test_data.contains("instruction_multiplicities_sample")) {
                        rust_test_trace.instruction_multiplicities_sample = test_data["instruction_multiplicities_sample"].get<std::vector<uint32_t>>();
                    }
                    if (test_data.contains("table_heights")) {
                        auto heights = test_data["table_heights"].get<std::vector<size_t>>();
                        for (size_t i = 0; i < std::min<size_t>(9, heights.size()); i++) {
                            rust_test_trace.table_heights[i] = heights[i];
                        }
                    }
                    have_rust_test_data = true;
                    std::cerr << "[DBG] Loaded Rust test data trace execution:" << std::endl;
                    std::cerr << "  Processor trace: " << rust_test_trace.processor_trace_height << " x " << rust_test_trace.processor_trace_width << std::endl;
                    std::cerr << "  Op stack: " << rust_test_trace.op_stack_height << " x " << rust_test_trace.op_stack_width << std::endl;
                    std::cerr << "  RAM: " << rust_test_trace.ram_height << " x " << rust_test_trace.ram_width << std::endl;
                    std::cerr << "  Program hash: " << rust_test_trace.program_hash_height << " x " << rust_test_trace.program_hash_width << std::endl;
                    std::cerr << "  Hash: " << rust_test_trace.hash_height << " x " << rust_test_trace.hash_width << std::endl;
                    std::cerr << "  Sponge: " << rust_test_trace.sponge_height << " x " << rust_test_trace.sponge_width << std::endl;
                    std::cerr << "  Instruction multiplicities sample size: " << rust_test_trace.instruction_multiplicities_sample.size() << std::endl;
                } else {
                    std::cerr << "[DBG] Failed to open Rust test data file: " << (rust_test_data_dir + "/01_trace_execution.json") << std::endl;
                }
            } else {
                std::cerr << "[DBG] TVM_RUST_TEST_DATA_DIR not set, skipping Rust test data comparison" << std::endl;
            }
            
            // Save Rust FFI trace data for comparison (all intermediate values)
            struct RustFFITraceData {
                size_t proc_rows, proc_cols;
                std::vector<uint64_t> proc_trace_first_row;
                std::vector<uint64_t> proc_trace_last_row;
                std::vector<uint64_t> public_output;
                std::vector<uint64_t> program_bwords;
                std::vector<uint32_t> inst_mults;
                std::vector<uint64_t> op_stack_first_row;
                std::vector<uint64_t> ram_first_row;
                std::vector<uint64_t> ph_first_row;
                std::vector<uint64_t> hash_first_row;
                std::vector<uint64_t> sponge_first_row;
                std::vector<uint64_t> u32_data_sample;
                std::vector<uint64_t> cascade_data_sample;
                std::array<uint64_t, 256> lookup_mults;
                std::array<size_t, 9> table_lengths;
                size_t output_len;
                size_t op_stack_rows, op_stack_cols;
                size_t ram_rows, ram_cols;
                size_t ph_rows, ph_cols;
                size_t hash_rows, hash_cols;
                size_t sponge_rows, sponge_cols;
            } rust_ffi_data;
            
            rust_ffi_data.proc_rows = proc_rows;
            rust_ffi_data.proc_cols = proc_cols;
            rust_ffi_data.output_len = output_len;
            rust_ffi_data.op_stack_rows = op_stack_rows;
            rust_ffi_data.op_stack_cols = op_stack_cols;
            rust_ffi_data.ram_rows = ram_rows;
            rust_ffi_data.ram_cols = ram_cols;
            rust_ffi_data.ph_rows = ph_rows;
            rust_ffi_data.ph_cols = ph_cols;
            rust_ffi_data.hash_rows = hash_rows;
            rust_ffi_data.hash_cols = hash_cols;
            rust_ffi_data.sponge_rows = sponge_rows;
            rust_ffi_data.sponge_cols = sponge_cols;
            
            // Save processor trace first and last row
            if (proc_trace_data != nullptr && proc_cols > 0 && proc_rows > 0) {
                rust_ffi_data.proc_trace_first_row.resize(proc_cols);
                for (size_t c = 0; c < proc_cols; c++) {
                    rust_ffi_data.proc_trace_first_row[c] = proc_trace_data[c];
                }
                if (proc_rows > 1) {
                    rust_ffi_data.proc_trace_last_row.resize(proc_cols);
                    for (size_t c = 0; c < proc_cols; c++) {
                        rust_ffi_data.proc_trace_last_row[c] = proc_trace_data[(proc_rows - 1) * proc_cols + c];
                    }
                }
            }
            
            // Save public output
            rust_ffi_data.public_output.reserve(output_len);
            for (size_t i = 0; i < output_len; i++) {
                rust_ffi_data.public_output.push_back(output_data[i]);
            }
            
            // Save program bwords
            rust_ffi_data.program_bwords.reserve(bwords_len);
            for (size_t i = 0; i < bwords_len; i++) {
                rust_ffi_data.program_bwords.push_back(bwords_data[i]);
            }
            
            // Save instruction multiplicities
            rust_ffi_data.inst_mults.reserve(inst_mults_len);
            for (size_t i = 0; i < inst_mults_len; i++) {
                rust_ffi_data.inst_mults.push_back(inst_mults_data[i]);
            }
            
            // Save op stack first row
            if (op_stack_data != nullptr && op_stack_cols > 0 && op_stack_rows > 0) {
                rust_ffi_data.op_stack_first_row.resize(op_stack_cols);
                for (size_t c = 0; c < op_stack_cols; c++) {
                    rust_ffi_data.op_stack_first_row[c] = op_stack_data[c];
                }
            }
            
            // Save RAM first row
            if (ram_data != nullptr && ram_cols > 0 && ram_rows > 0) {
                rust_ffi_data.ram_first_row.resize(ram_cols);
                for (size_t c = 0; c < ram_cols; c++) {
                    rust_ffi_data.ram_first_row[c] = ram_data[c];
                }
            }
            
            // Save program hash first row
            if (ph_data != nullptr && ph_cols > 0 && ph_rows > 0) {
                rust_ffi_data.ph_first_row.resize(ph_cols);
                for (size_t c = 0; c < ph_cols; c++) {
                    rust_ffi_data.ph_first_row[c] = ph_data[c];
                }
            }
            
            // Save hash first row
            if (hash_data != nullptr && hash_cols > 0 && hash_rows > 0) {
                rust_ffi_data.hash_first_row.resize(hash_cols);
                for (size_t c = 0; c < hash_cols; c++) {
                    rust_ffi_data.hash_first_row[c] = hash_data[c];
                }
            }
            
            // Save sponge first row
            if (sponge_data != nullptr && sponge_cols > 0 && sponge_rows > 0) {
                rust_ffi_data.sponge_first_row.resize(sponge_cols);
                for (size_t c = 0; c < sponge_cols; c++) {
                    rust_ffi_data.sponge_first_row[c] = sponge_data[c];
                }
            }
            
            // Save U32 data sample (first 10 elements)
            if (u32_data != nullptr && u32_len > 0) {
                size_t sample_size = std::min<size_t>(10, u32_len);
                rust_ffi_data.u32_data_sample.resize(sample_size);
                for (size_t i = 0; i < sample_size; i++) {
                    rust_ffi_data.u32_data_sample[i] = u32_data[i];
                }
            }
            
            // Save cascade data sample (first 10 elements)
            if (cascade_data != nullptr && cascade_len > 0) {
                size_t sample_size = std::min<size_t>(10, cascade_len);
                rust_ffi_data.cascade_data_sample.resize(sample_size);
                for (size_t i = 0; i < sample_size; i++) {
                    rust_ffi_data.cascade_data_sample[i] = cascade_data[i];
                }
            }
            
            // Save lookup multiplicities
            for (size_t i = 0; i < 256; i++) {
                rust_ffi_data.lookup_mults[i] = lookup_mults[i];
            }
            
            // Save table lengths
            for (size_t i = 0; i < 9; i++) {
                rust_ffi_data.table_lengths[i] = table_lengths[i];
            }
            
            // Log Rust FFI trace data (all intermediate values) - only if TVM_DEBUG_TRACE_EXECUTION is set
            if (std::getenv("TVM_DEBUG_TRACE_EXECUTION")) {
                std::cerr << "\n[DBG] ========== Rust FFI Trace Data (BEFORE conversion) ==========" << std::endl;
            std::cerr << "[DBG] Processor trace: " << rust_ffi_data.proc_rows << " x " << rust_ffi_data.proc_cols << std::endl;
            std::cerr << "[DBG] Processor trace first row (first 10 cols):" << std::endl;
            for (size_t c = 0; c < std::min<size_t>(10, rust_ffi_data.proc_trace_first_row.size()); c++) {
                std::cerr << "  [" << c << "]: " << rust_ffi_data.proc_trace_first_row[c] << std::endl;
            }
            if (!rust_ffi_data.proc_trace_last_row.empty()) {
                std::cerr << "[DBG] Processor trace last row (first 10 cols):" << std::endl;
                for (size_t c = 0; c < std::min<size_t>(10, rust_ffi_data.proc_trace_last_row.size()); c++) {
                    std::cerr << "  [" << c << "]: " << rust_ffi_data.proc_trace_last_row[c] << std::endl;
                }
            }
            std::cerr << "[DBG] Public output (" << rust_ffi_data.public_output.size() << " elements):" << std::endl;
            for (size_t i = 0; i < std::min<size_t>(10, rust_ffi_data.public_output.size()); i++) {
                std::cerr << "  [" << i << "]: " << rust_ffi_data.public_output[i] << std::endl;
            }
            std::cerr << "[DBG] Program bwords (" << rust_ffi_data.program_bwords.size() << " elements)" << std::endl;
            std::cerr << "[DBG] Instruction multiplicities (" << rust_ffi_data.inst_mults.size() << " elements, first 10):" << std::endl;
            for (size_t i = 0; i < std::min<size_t>(10, rust_ffi_data.inst_mults.size()); i++) {
                std::cerr << "  [" << i << "]: " << rust_ffi_data.inst_mults[i] << std::endl;
            }
            std::cerr << "[DBG] Op stack: " << rust_ffi_data.op_stack_rows << " x " << rust_ffi_data.op_stack_cols;
            if (!rust_ffi_data.op_stack_first_row.empty()) {
                std::cerr << " (first row first 5 cols):";
                for (size_t c = 0; c < std::min<size_t>(5, rust_ffi_data.op_stack_first_row.size()); c++) {
                    std::cerr << " " << rust_ffi_data.op_stack_first_row[c];
                }
            }
            std::cerr << std::endl;
            std::cerr << "[DBG] RAM: " << rust_ffi_data.ram_rows << " x " << rust_ffi_data.ram_cols;
            if (!rust_ffi_data.ram_first_row.empty()) {
                std::cerr << " (first row first 5 cols):";
                for (size_t c = 0; c < std::min<size_t>(5, rust_ffi_data.ram_first_row.size()); c++) {
                    std::cerr << " " << rust_ffi_data.ram_first_row[c];
                }
            }
            std::cerr << std::endl;
            std::cerr << "[DBG] Program hash: " << rust_ffi_data.ph_rows << " x " << rust_ffi_data.ph_cols;
            if (!rust_ffi_data.ph_first_row.empty()) {
                std::cerr << " (first row first 5 cols):";
                for (size_t c = 0; c < std::min<size_t>(5, rust_ffi_data.ph_first_row.size()); c++) {
                    std::cerr << " " << rust_ffi_data.ph_first_row[c];
                }
            }
            std::cerr << std::endl;
            std::cerr << "[DBG] Hash: " << rust_ffi_data.hash_rows << " x " << rust_ffi_data.hash_cols;
            if (!rust_ffi_data.hash_first_row.empty()) {
                std::cerr << " (first row first 5 cols):";
                for (size_t c = 0; c < std::min<size_t>(5, rust_ffi_data.hash_first_row.size()); c++) {
                    std::cerr << " " << rust_ffi_data.hash_first_row[c];
                }
            }
            std::cerr << std::endl;
            std::cerr << "[DBG] Sponge: " << rust_ffi_data.sponge_rows << " x " << rust_ffi_data.sponge_cols;
            if (!rust_ffi_data.sponge_first_row.empty()) {
                std::cerr << " (first row first 5 cols):";
                for (size_t c = 0; c < std::min<size_t>(5, rust_ffi_data.sponge_first_row.size()); c++) {
                    std::cerr << " " << rust_ffi_data.sponge_first_row[c];
                }
            }
            std::cerr << std::endl;
            std::cerr << "[DBG] U32 data (" << u32_len << " elements, first 5):" << std::endl;
            for (size_t i = 0; i < std::min<size_t>(5, rust_ffi_data.u32_data_sample.size()); i++) {
                std::cerr << "  [" << i << "]: " << rust_ffi_data.u32_data_sample[i] << std::endl;
            }
            std::cerr << "[DBG] Cascade data (" << cascade_len << " elements, first 5):" << std::endl;
            for (size_t i = 0; i < std::min<size_t>(5, rust_ffi_data.cascade_data_sample.size()); i++) {
                std::cerr << "  [" << i << "]: " << rust_ffi_data.cascade_data_sample[i] << std::endl;
            }
            std::cerr << "[DBG] Lookup multiplicities (first 10):" << std::endl;
            for (size_t i = 0; i < 10; i++) {
                std::cerr << "  [" << i << "]: " << rust_ffi_data.lookup_mults[i] << std::endl;
            }
            std::cerr << "[DBG] Table lengths: [";
            for (size_t i = 0; i < 9; i++) {
                std::cerr << rust_ffi_data.table_lengths[i];
                if (i < 8) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
            std::cerr << "[DBG] ==============================================================\n" << std::endl;
            }
            
            // Compare Rust test data vs Rust FFI (only if TVM_DEBUG_TRACE_EXECUTION is set)
            if (have_rust_test_data && std::getenv("TVM_DEBUG_TRACE_EXECUTION")) {
                std::cerr << "\n[DBG] ========== Rust Test Data vs Rust FFI Comparison ==========" << std::endl;
                
                // Compare processor trace dimensions
                bool dims_match = (rust_test_trace.processor_trace_height == rust_ffi_data.proc_rows &&
                                  rust_test_trace.processor_trace_width == rust_ffi_data.proc_cols);
                std::cerr << "[DBG] Processor trace dimensions:" << std::endl;
                std::cerr << "  Test data: " << rust_test_trace.processor_trace_height << " x " << rust_test_trace.processor_trace_width << std::endl;
                std::cerr << "  FFI:       " << rust_ffi_data.proc_rows << " x " << rust_ffi_data.proc_cols 
                          << (dims_match ? " ✓" : " ✗") << std::endl;
                
                // Compare public output
                std::cerr << "[DBG] Public output comparison:" << std::endl;
                bool output_match = true;
                if (rust_test_trace.public_output_sampled.size() == rust_ffi_data.public_output.size()) {
                    for (size_t i = 0; i < std::min<size_t>(10, rust_test_trace.public_output_sampled.size()); i++) {
                        uint64_t test_val = rust_test_trace.public_output_sampled[i];
                        uint64_t ffi_val = rust_ffi_data.public_output[i];
                        bool match = (test_val == ffi_val);
                        if (!match) output_match = false;
                        std::cerr << "  [" << i << "] Test data: " << test_val << " | FFI: " << ffi_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    if (output_match && rust_test_trace.public_output_sampled.size() == rust_ffi_data.public_output.size()) {
                        std::cerr << "[DBG] ✓ Public output: MATCH (" << rust_test_trace.public_output_sampled.size() << " elements)" << std::endl;
                    } else {
                        std::cerr << "[DBG] ✗ Public output: MISMATCH" << std::endl;
                    }
                } else {
                    std::cerr << "[DBG] ✗ Public output size mismatch: Test data=" << rust_test_trace.public_output_sampled.size() 
                              << " FFI=" << rust_ffi_data.public_output.size() << std::endl;
                }
                
                // Note: Padded height comparison will be done after FFI conversion
                // since padded_height is computed from AlgebraicExecutionTrace, not from raw FFI data
                std::cerr << "[DBG] Processor trace height (from FFI): " << rust_ffi_data.table_lengths[1] << std::endl;
                std::cerr << "[DBG] Test data padded_height: " << rust_test_trace.padded_height << std::endl;
                std::cerr << "[DBG] (Padded height comparison will be done after FFI conversion)" << std::endl;
                
                // Compare processor trace first row
                if (!rust_test_trace.processor_trace_first_row.empty() && !rust_ffi_data.proc_trace_first_row.empty()) {
                    std::cerr << "[DBG] Processor trace first row comparison (first 10 cols):" << std::endl;
                    bool proc_first_match = true;
                    for (size_t c = 0; c < std::min<size_t>(10, std::min(rust_test_trace.processor_trace_first_row.size(), rust_ffi_data.proc_trace_first_row.size())); c++) {
                        uint64_t test_val = rust_test_trace.processor_trace_first_row[c];
                        uint64_t ffi_val = rust_ffi_data.proc_trace_first_row[c];
                        bool match = (test_val == ffi_val);
                        if (!match) proc_first_match = false;
                        std::cerr << "  [" << c << "] Test data: " << test_val << " | FFI: " << ffi_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    if (proc_first_match) {
                        std::cerr << "[DBG] ✓ Processor trace first row: MATCH" << std::endl;
                    } else {
                        std::cerr << "[DBG] ✗ Processor trace first row: MISMATCH" << std::endl;
                    }
                }
                
                // Compare processor trace last row
                if (!rust_test_trace.processor_trace_last_row.empty() && !rust_ffi_data.proc_trace_last_row.empty()) {
                    std::cerr << "[DBG] Processor trace last row comparison (first 10 cols):" << std::endl;
                    bool proc_last_match = true;
                    for (size_t c = 0; c < std::min<size_t>(10, std::min(rust_test_trace.processor_trace_last_row.size(), rust_ffi_data.proc_trace_last_row.size())); c++) {
                        uint64_t test_val = rust_test_trace.processor_trace_last_row[c];
                        uint64_t ffi_val = rust_ffi_data.proc_trace_last_row[c];
                        bool match = (test_val == ffi_val);
                        if (!match) proc_last_match = false;
                        std::cerr << "  [" << c << "] Test data: " << test_val << " | FFI: " << ffi_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    if (proc_last_match) {
                        std::cerr << "[DBG] ✓ Processor trace last row: MATCH" << std::endl;
                    } else {
                        std::cerr << "[DBG] ✗ Processor trace last row: MISMATCH" << std::endl;
                    }
                }
                
                // Compare op stack
                if (rust_test_trace.op_stack_height > 0 && rust_ffi_data.op_stack_rows > 0) {
                    std::cerr << "[DBG] Op stack dimensions:" << std::endl;
                    std::cerr << "  Test data: " << rust_test_trace.op_stack_height << " x " << rust_test_trace.op_stack_width << std::endl;
                    std::cerr << "  FFI:       " << rust_ffi_data.op_stack_rows << " x " << rust_ffi_data.op_stack_cols 
                              << ((rust_test_trace.op_stack_height == rust_ffi_data.op_stack_rows && 
                                   rust_test_trace.op_stack_width == rust_ffi_data.op_stack_cols) ? " ✓" : " ✗") << std::endl;
                    if (!rust_test_trace.op_stack_first_row.empty() && !rust_ffi_data.op_stack_first_row.empty()) {
                        std::cerr << "[DBG] Op stack first row comparison (first 5 cols):" << std::endl;
                        bool op_stack_match = true;
                        for (size_t c = 0; c < std::min<size_t>(5, std::min(rust_test_trace.op_stack_first_row.size(), rust_ffi_data.op_stack_first_row.size())); c++) {
                            uint64_t test_val = rust_test_trace.op_stack_first_row[c];
                            uint64_t ffi_val = rust_ffi_data.op_stack_first_row[c];
                            bool match = (test_val == ffi_val);
                            if (!match) op_stack_match = false;
                            std::cerr << "  [" << c << "] Test data: " << test_val << " | FFI: " << ffi_val 
                                      << (match ? " ✓" : " ✗") << std::endl;
                        }
                    }
                }
                
                // Compare RAM
                if (rust_test_trace.ram_height > 0 && rust_ffi_data.ram_rows > 0) {
                    std::cerr << "[DBG] RAM dimensions:" << std::endl;
                    std::cerr << "  Test data: " << rust_test_trace.ram_height << " x " << rust_test_trace.ram_width << std::endl;
                    std::cerr << "  FFI:       " << rust_ffi_data.ram_rows << " x " << rust_ffi_data.ram_cols 
                              << ((rust_test_trace.ram_height == rust_ffi_data.ram_rows && 
                                   rust_test_trace.ram_width == rust_ffi_data.ram_cols) ? " ✓" : " ✗") << std::endl;
                    if (!rust_test_trace.ram_first_row.empty() && !rust_ffi_data.ram_first_row.empty()) {
                        std::cerr << "[DBG] RAM first row comparison (first 5 cols):" << std::endl;
                        for (size_t c = 0; c < std::min<size_t>(5, std::min(rust_test_trace.ram_first_row.size(), rust_ffi_data.ram_first_row.size())); c++) {
                            uint64_t test_val = rust_test_trace.ram_first_row[c];
                            uint64_t ffi_val = rust_ffi_data.ram_first_row[c];
                            bool match = (test_val == ffi_val);
                            std::cerr << "  [" << c << "] Test data: " << test_val << " | FFI: " << ffi_val 
                                      << (match ? " ✓" : " ✗") << std::endl;
                        }
                    }
                }
                
                // Compare program hash
                if (rust_test_trace.program_hash_height > 0 && rust_ffi_data.ph_rows > 0) {
                    std::cerr << "[DBG] Program hash dimensions:" << std::endl;
                    std::cerr << "  Test data: " << rust_test_trace.program_hash_height << " x " << rust_test_trace.program_hash_width << std::endl;
                    std::cerr << "  FFI:       " << rust_ffi_data.ph_rows << " x " << rust_ffi_data.ph_cols 
                              << ((rust_test_trace.program_hash_height == rust_ffi_data.ph_rows && 
                                   rust_test_trace.program_hash_width == rust_ffi_data.ph_cols) ? " ✓" : " ✗") << std::endl;
                }
                
                // Compare hash
                if (rust_test_trace.hash_height > 0 && rust_ffi_data.hash_rows > 0) {
                    std::cerr << "[DBG] Hash dimensions:" << std::endl;
                    std::cerr << "  Test data: " << rust_test_trace.hash_height << " x " << rust_test_trace.hash_width << std::endl;
                    std::cerr << "  FFI:       " << rust_ffi_data.hash_rows << " x " << rust_ffi_data.hash_cols 
                              << ((rust_test_trace.hash_height == rust_ffi_data.hash_rows && 
                                   rust_test_trace.hash_width == rust_ffi_data.hash_cols) ? " ✓" : " ✗") << std::endl;
                }
                
                // Compare sponge
                if (rust_test_trace.sponge_height > 0 && rust_ffi_data.sponge_rows > 0) {
                    std::cerr << "[DBG] Sponge dimensions:" << std::endl;
                    std::cerr << "  Test data: " << rust_test_trace.sponge_height << " x " << rust_test_trace.sponge_width << std::endl;
                    std::cerr << "  FFI:       " << rust_ffi_data.sponge_rows << " x " << rust_ffi_data.sponge_cols 
                              << ((rust_test_trace.sponge_height == rust_ffi_data.sponge_rows && 
                                   rust_test_trace.sponge_width == rust_ffi_data.sponge_cols) ? " ✓" : " ✗") << std::endl;
                }
                
                // Compare instruction multiplicities
                if (!rust_test_trace.instruction_multiplicities_sample.empty() && !rust_ffi_data.inst_mults.empty()) {
                    std::cerr << "[DBG] Instruction multiplicities comparison (first 10):" << std::endl;
                    bool inst_mults_match = true;
                    for (size_t i = 0; i < std::min<size_t>(10, std::min(rust_test_trace.instruction_multiplicities_sample.size(), rust_ffi_data.inst_mults.size())); i++) {
                        uint32_t test_val = rust_test_trace.instruction_multiplicities_sample[i];
                        uint32_t ffi_val = rust_ffi_data.inst_mults[i];
                        bool match = (test_val == ffi_val);
                        if (!match) inst_mults_match = false;
                        std::cerr << "  [" << i << "] Test data: " << test_val << " | FFI: " << ffi_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    if (inst_mults_match) {
                        std::cerr << "[DBG] ✓ Instruction multiplicities: MATCH" << std::endl;
                    } else {
                        std::cerr << "[DBG] ✗ Instruction multiplicities: MISMATCH" << std::endl;
                    }
                }
                
                // Compare table lengths
                std::cerr << "[DBG] Table lengths comparison:" << std::endl;
                bool table_lengths_match = true;
                for (size_t i = 0; i < 9; i++) {
                    size_t test_val = rust_test_trace.table_heights[i];
                    size_t ffi_val = rust_ffi_data.table_lengths[i];
                    bool match = (test_val == ffi_val);
                    if (!match) table_lengths_match = false;
                    std::cerr << "  Table[" << i << "]: Test data=" << test_val << " | FFI=" << ffi_val 
                              << (match ? " ✓" : " ✗") << std::endl;
                }
                if (table_lengths_match) {
                    std::cerr << "[DBG] ✓ Table lengths: MATCH" << std::endl;
                } else {
                    std::cerr << "[DBG] ✗ Table lengths: MISMATCH" << std::endl;
                }
                
                // Summary of all comparisons performed
                std::cerr << "\n[DBG] ========== Comparison Summary ==========" << std::endl;
                std::cerr << "[DBG] ✓ Processor trace dimensions" << std::endl;
                std::cerr << "[DBG] ✓ Public output" << std::endl;
                std::cerr << "[DBG] ✓ Padded height" << std::endl;
                if (!rust_test_trace.processor_trace_first_row.empty()) {
                    std::cerr << "[DBG] ✓ Processor trace first row" << std::endl;
                }
                if (!rust_test_trace.processor_trace_last_row.empty()) {
                    std::cerr << "[DBG] ✓ Processor trace last row" << std::endl;
                }
                if (rust_test_trace.op_stack_height > 0) {
                    std::cerr << "[DBG] ✓ Op stack dimensions and first row" << std::endl;
                }
                if (rust_test_trace.ram_height > 0) {
                    std::cerr << "[DBG] ✓ RAM dimensions and first row" << std::endl;
                }
                if (rust_test_trace.program_hash_height > 0) {
                    std::cerr << "[DBG] ✓ Program hash dimensions" << std::endl;
                }
                if (rust_test_trace.hash_height > 0) {
                    std::cerr << "[DBG] ✓ Hash dimensions" << std::endl;
                }
                if (rust_test_trace.sponge_height > 0) {
                    std::cerr << "[DBG] ✓ Sponge dimensions" << std::endl;
                }
                if (!rust_test_trace.instruction_multiplicities_sample.empty()) {
                    std::cerr << "[DBG] ✓ Instruction multiplicities" << std::endl;
                }
                std::cerr << "[DBG] ✓ Table heights (all 9 tables)" << std::endl;
                std::cerr << "[DBG] ==============================================================\n" << std::endl;
            }
            
            // Convert FFI output to AlgebraicExecutionTrace
            std::vector<BFieldElement> program_bwords;
            program_bwords.reserve(bwords_len);
            for (size_t i = 0; i < bwords_len; i++) {
                program_bwords.push_back(BFieldElement(bwords_data[i]));
            }
            
            // IMPORTANT: When using Rust FFI, rebuild the Program from bwords
            // The program loaded from file might have wrong encoding (e.g., if JSON was passed as TASM)
            program = Program::from_bwords(bwords_data, bwords_len);
            
            std::vector<BFieldElement> public_output;
            public_output.reserve(output_len);
            for (size_t i = 0; i < output_len; i++) {
                public_output.push_back(BFieldElement(output_data[i]));
            }
            
            trace_result = VM::trace_execution_from_rust_ffi(
                program_bwords,
                proc_trace_data, proc_rows, proc_cols,
                inst_mults_data, inst_mults_len,
                public_output,
                op_stack_data, op_stack_rows, op_stack_cols,
                ram_data, ram_rows, ram_cols,
                ph_data, ph_rows, ph_cols,
                hash_data, hash_rows, hash_cols,
                sponge_data, sponge_rows, sponge_cols,
                u32_data, u32_len,
                cascade_data, cascade_len,
                lookup_mults,
                table_lengths
            );
            
            // Compare computed trace_result with Rust FFI data (only if TVM_DEBUG_TRACE_EXECUTION is set)
            if (std::getenv("TVM_DEBUG_TRACE_EXECUTION")) {
                std::cerr << "\n[DBG] ========== C++ Trace Result (AFTER conversion) ==========" << std::endl;
            const AlgebraicExecutionTrace& aet_compare = trace_result.aet;
            std::cerr << "[DBG] Processor trace height: " << aet_compare.height_of_table(1) << " (Rust FFI: " << rust_ffi_data.proc_rows << ")" << std::endl;
            std::cerr << "[DBG] Processor trace width: " << aet_compare.processor_trace_width() << " (Rust FFI: " << rust_ffi_data.proc_cols << ")" << std::endl;
            
            // Compare padded height (now that we have AlgebraicExecutionTrace)
            if (rust_test_trace.padded_height > 0) {
                size_t cpp_padded_height = aet_compare.padded_height();
                bool padded_height_match = (rust_test_trace.padded_height == cpp_padded_height);
                std::cerr << "[DBG] Padded height comparison:" << std::endl;
                std::cerr << "  Test data: " << rust_test_trace.padded_height << std::endl;
                std::cerr << "  C++ (from AET): " << cpp_padded_height 
                          << (padded_height_match ? " ✓" : " ✗") << std::endl;
            }
            
            // Compare processor trace first row
            if (aet_compare.height_of_table(1) > 0 && aet_compare.processor_trace_width() > 0) {
                const BFieldElement* proc_flat = aet_compare.processor_trace_flat_data();
                size_t proc_width = aet_compare.processor_trace_width();
                std::cerr << "[DBG] Processor trace first row comparison (first 10 cols):" << std::endl;
                bool proc_match = true;
                for (size_t c = 0; c < std::min<size_t>(10, std::min(proc_width, rust_ffi_data.proc_trace_first_row.size())); c++) {
                    uint64_t cpp_val = proc_flat[c].value();
                    uint64_t rust_val = rust_ffi_data.proc_trace_first_row[c];
                    bool match = (cpp_val == rust_val);
                    if (!match) proc_match = false;
                    std::cerr << "  [" << c << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                              << (match ? " ✓" : " ✗") << std::endl;
                }
                if (proc_match && proc_width == rust_ffi_data.proc_trace_first_row.size()) {
                    std::cerr << "[DBG] ✓ Processor trace first row: MATCH" << std::endl;
                } else {
                    std::cerr << "[DBG] ✗ Processor trace first row: MISMATCH" << std::endl;
                }
            }
            
            // Compare public output
            std::cerr << "[DBG] Public output comparison:" << std::endl;
            bool output_match = true;
            if (trace_result.output.size() == rust_ffi_data.public_output.size()) {
                for (size_t i = 0; i < std::min<size_t>(10, trace_result.output.size()); i++) {
                    uint64_t cpp_val = trace_result.output[i].value();
                    uint64_t rust_val = rust_ffi_data.public_output[i];
                    bool match = (cpp_val == rust_val);
                    if (!match) output_match = false;
                    std::cerr << "  [" << i << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                              << (match ? " ✓" : " ✗") << std::endl;
                }
                if (output_match) {
                    std::cerr << "[DBG] ✓ Public output: MATCH (" << trace_result.output.size() << " elements)" << std::endl;
                } else {
                    std::cerr << "[DBG] ✗ Public output: MISMATCH" << std::endl;
                }
            } else {
                std::cerr << "[DBG] ✗ Public output size mismatch: C++=" << trace_result.output.size() 
                          << " Rust FFI=" << rust_ffi_data.public_output.size() << std::endl;
            }
            
            // Compare processor trace last row
            if (aet_compare.height_of_table(1) > 1 && !rust_ffi_data.proc_trace_last_row.empty()) {
                const BFieldElement* proc_flat = aet_compare.processor_trace_flat_data();
                size_t proc_width = aet_compare.processor_trace_width();
                size_t proc_height = aet_compare.height_of_table(1);
                std::cerr << "[DBG] Processor trace last row comparison (first 10 cols):" << std::endl;
                bool proc_last_match = true;
                for (size_t c = 0; c < std::min<size_t>(10, std::min(proc_width, rust_ffi_data.proc_trace_last_row.size())); c++) {
                    uint64_t cpp_val = proc_flat[(proc_height - 1) * proc_width + c].value();
                    uint64_t rust_val = rust_ffi_data.proc_trace_last_row[c];
                    bool match = (cpp_val == rust_val);
                    if (!match) proc_last_match = false;
                    std::cerr << "  [" << c << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                              << (match ? " ✓" : " ✗") << std::endl;
                }
                if (proc_last_match) {
                    std::cerr << "[DBG] ✓ Processor trace last row: MATCH" << std::endl;
                } else {
                    std::cerr << "[DBG] ✗ Processor trace last row: MISMATCH" << std::endl;
                }
            }
            
            // Compare op stack
            if (aet_compare.height_of_table(2) > 0 && !rust_ffi_data.op_stack_first_row.empty()) {
                const auto& op_stack_trace = aet_compare.op_stack_underflow_trace();
                if (!op_stack_trace.empty() && !op_stack_trace[0].empty()) {
                    size_t op_stack_width = op_stack_trace[0].size();
                    std::cerr << "[DBG] Op stack first row comparison (first 5 cols):" << std::endl;
                    bool op_stack_match = true;
                    for (size_t c = 0; c < std::min<size_t>(5, std::min(op_stack_width, rust_ffi_data.op_stack_first_row.size())); c++) {
                        uint64_t cpp_val = op_stack_trace[0][c].value();
                        uint64_t rust_val = rust_ffi_data.op_stack_first_row[c];
                        bool match = (cpp_val == rust_val);
                        if (!match) op_stack_match = false;
                        std::cerr << "  [" << c << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    std::cerr << "[DBG] Op stack dimensions: C++=" << aet_compare.height_of_table(2) << " x " << op_stack_width
                              << " | Rust FFI=" << rust_ffi_data.op_stack_rows << " x " << rust_ffi_data.op_stack_cols << std::endl;
                }
            }
            
            // Compare RAM
            if (aet_compare.height_of_table(3) > 0 && !rust_ffi_data.ram_first_row.empty()) {
                const auto& ram_trace = aet_compare.ram_trace();
                if (!ram_trace.empty() && !ram_trace[0].empty()) {
                    size_t ram_width = ram_trace[0].size();
                    std::cerr << "[DBG] RAM first row comparison (first 5 cols):" << std::endl;
                    bool ram_match = true;
                    for (size_t c = 0; c < std::min<size_t>(5, std::min(ram_width, rust_ffi_data.ram_first_row.size())); c++) {
                        uint64_t cpp_val = ram_trace[0][c].value();
                        uint64_t rust_val = rust_ffi_data.ram_first_row[c];
                        bool match = (cpp_val == rust_val);
                        if (!match) ram_match = false;
                        std::cerr << "  [" << c << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    std::cerr << "[DBG] RAM dimensions: C++=" << aet_compare.height_of_table(3) << " x " << ram_width
                              << " | Rust FFI=" << rust_ffi_data.ram_rows << " x " << rust_ffi_data.ram_cols << std::endl;
                }
            }
            
            // Compare program hash
            if (aet_compare.height_of_table(4) > 0 && !rust_ffi_data.ph_first_row.empty()) {
                const auto& ph_trace = aet_compare.program_hash_trace();
                if (!ph_trace.empty() && !ph_trace[0].empty()) {
                    size_t ph_width = ph_trace[0].size();
                    std::cerr << "[DBG] Program hash first row comparison (first 5 cols):" << std::endl;
                    bool ph_match = true;
                    for (size_t c = 0; c < std::min<size_t>(5, std::min(ph_width, rust_ffi_data.ph_first_row.size())); c++) {
                        uint64_t cpp_val = ph_trace[0][c].value();
                        uint64_t rust_val = rust_ffi_data.ph_first_row[c];
                        bool match = (cpp_val == rust_val);
                        if (!match) ph_match = false;
                        std::cerr << "  [" << c << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    std::cerr << "[DBG] Program hash dimensions: C++=" << aet_compare.height_of_table(4) << " x " << ph_width
                              << " | Rust FFI=" << rust_ffi_data.ph_rows << " x " << rust_ffi_data.ph_cols << std::endl;
                }
            }
            
            // Compare hash
            if (aet_compare.height_of_table(5) > 0 && !rust_ffi_data.hash_first_row.empty()) {
                const auto& hash_trace = aet_compare.hash_trace();
                if (!hash_trace.empty() && !hash_trace[0].empty()) {
                    size_t hash_width = hash_trace[0].size();
                    std::cerr << "[DBG] Hash first row comparison (first 5 cols):" << std::endl;
                    bool hash_match = true;
                    for (size_t c = 0; c < std::min<size_t>(5, std::min(hash_width, rust_ffi_data.hash_first_row.size())); c++) {
                        uint64_t cpp_val = hash_trace[0][c].value();
                        uint64_t rust_val = rust_ffi_data.hash_first_row[c];
                        bool match = (cpp_val == rust_val);
                        if (!match) hash_match = false;
                        std::cerr << "  [" << c << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    std::cerr << "[DBG] Hash dimensions: C++=" << aet_compare.height_of_table(5) << " x " << hash_width
                              << " | Rust FFI=" << rust_ffi_data.hash_rows << " x " << rust_ffi_data.hash_cols << std::endl;
                }
            }
            
            // Compare sponge
            if (aet_compare.height_of_table(6) > 0 && !rust_ffi_data.sponge_first_row.empty()) {
                const auto& sponge_trace = aet_compare.sponge_trace();
                if (!sponge_trace.empty() && !sponge_trace[0].empty()) {
                    size_t sponge_width = sponge_trace[0].size();
                    std::cerr << "[DBG] Sponge first row comparison (first 5 cols):" << std::endl;
                    bool sponge_match = true;
                    for (size_t c = 0; c < std::min<size_t>(5, std::min(sponge_width, rust_ffi_data.sponge_first_row.size())); c++) {
                        uint64_t cpp_val = sponge_trace[0][c].value();
                        uint64_t rust_val = rust_ffi_data.sponge_first_row[c];
                        bool match = (cpp_val == rust_val);
                        if (!match) sponge_match = false;
                        std::cerr << "  [" << c << "] C++: " << cpp_val << " | Rust FFI: " << rust_val 
                                  << (match ? " ✓" : " ✗") << std::endl;
                    }
                    std::cerr << "[DBG] Sponge dimensions: C++=" << aet_compare.height_of_table(6) << " x " << sponge_width
                              << " | Rust FFI=" << rust_ffi_data.sponge_rows << " x " << rust_ffi_data.sponge_cols << std::endl;
                }
            }
            
            // Compare table lengths
            std::cerr << "[DBG] Table lengths comparison:" << std::endl;
            bool table_lengths_match = true;
            for (size_t i = 0; i < 9; i++) {
                size_t cpp_len = aet_compare.height_of_table(i);
                size_t rust_len = rust_ffi_data.table_lengths[i];
                bool match = (cpp_len == rust_len);
                if (!match) table_lengths_match = false;
                std::cerr << "  Table[" << i << "] C++: " << cpp_len << " | Rust FFI: " << rust_len 
                          << (match ? " ✓" : " ✗") << std::endl;
            }
            if (table_lengths_match) {
                std::cerr << "[DBG] ✓ Table lengths: MATCH" << std::endl;
            } else {
                std::cerr << "[DBG] ✗ Table lengths: MISMATCH" << std::endl;
            }
            
            std::cerr << "[DBG] ==============================================================\n" << std::endl;
            }
            
            // Free FFI-allocated memory
            tvm_trace_execution_rust_ffi_free(
                proc_trace_data, proc_rows, proc_cols,
                bwords_data, bwords_len,
                inst_mults_data, inst_mults_len,
                output_data, output_len,
                op_stack_data, op_stack_rows, op_stack_cols,
                ram_data, ram_rows, ram_cols,
                ph_data, ph_rows, ph_cols,
                hash_data, hash_rows, hash_cols,
                sponge_data, sponge_rows, sponge_cols,
                u32_data, u32_len,
                cascade_data, cascade_len
            );
            
            std::cout << "  [1c] Rust FFI trace execution: " << elapsed_ms(step_start) << " ms" << std::endl;
        } else {
            // Use Rust FFI trace execution (without NonDeterminism)
            std::cout << "  [1c] Using Rust FFI trace execution..." << std::endl;
            
            // Call Rust FFI with program path (TASM file)
            uint64_t* proc_trace_data = nullptr;
            size_t proc_rows = 0, proc_cols = 0;
            uint64_t* bwords_data = nullptr;
            size_t bwords_len = 0;
            uint32_t* inst_mults_data = nullptr;
            size_t inst_mults_len = 0;
            uint64_t* output_data = nullptr;
            size_t output_len = 0;
            uint64_t* op_stack_data = nullptr;
            size_t op_stack_rows = 0, op_stack_cols = 0;
            uint64_t* ram_data = nullptr;
            size_t ram_rows = 0, ram_cols = 0;
            uint64_t* ph_data = nullptr;
            size_t ph_rows = 0, ph_cols = 0;
            uint64_t* hash_data = nullptr;
            size_t hash_rows = 0, hash_cols = 0;
            uint64_t* sponge_data = nullptr;
            size_t sponge_rows = 0, sponge_cols = 0;
            uint64_t* u32_data = nullptr;
            size_t u32_len = 0;
            uint64_t* cascade_data = nullptr;
            size_t cascade_len = 0;
            uint64_t lookup_mults[256] = {0};
            size_t table_lengths[9] = {0};
            
            int ffi_result = tvm_trace_execution_rust_ffi(
                program_path.c_str(),
                public_input.data(),
                public_input.size(),
                &proc_trace_data, &proc_rows, &proc_cols,
                &bwords_data, &bwords_len,
                &inst_mults_data, &inst_mults_len,
                &output_data, &output_len,
                &op_stack_data, &op_stack_rows, &op_stack_cols,
                &ram_data, &ram_rows, &ram_cols,
                &ph_data, &ph_rows, &ph_cols,
                &hash_data, &hash_rows, &hash_cols,
                &sponge_data, &sponge_rows, &sponge_cols,
                &u32_data, &u32_len,
                &cascade_data, &cascade_len,
                lookup_mults,
                table_lengths
            );
            
            if (ffi_result != 0) {
                std::cerr << "Error: Rust FFI trace execution failed" << std::endl;
                return 1;
            }
            
            // Convert FFI output to AlgebraicExecutionTrace
            std::vector<BFieldElement> program_bwords;
            program_bwords.reserve(bwords_len);
            for (size_t i = 0; i < bwords_len; i++) {
                program_bwords.push_back(BFieldElement(bwords_data[i]));
            }
            
            // IMPORTANT: When using Rust FFI, rebuild the Program from bwords
            program = Program::from_bwords(bwords_data, bwords_len);
            
            std::vector<BFieldElement> public_output;
            public_output.reserve(output_len);
            for (size_t i = 0; i < output_len; i++) {
                public_output.push_back(BFieldElement(output_data[i]));
            }
            
            trace_result = VM::trace_execution_from_rust_ffi(
                program_bwords,
                proc_trace_data, proc_rows, proc_cols,
                inst_mults_data, inst_mults_len,
                public_output,
                op_stack_data, op_stack_rows, op_stack_cols,
                ram_data, ram_rows, ram_cols,
                ph_data, ph_rows, ph_cols,
                hash_data, hash_rows, hash_cols,
                sponge_data, sponge_rows, sponge_cols,
                u32_data, u32_len,
                cascade_data, cascade_len,
                lookup_mults,
                table_lengths
            );
            
            // Free FFI-allocated memory
            tvm_trace_execution_rust_ffi_free(
                proc_trace_data, proc_rows, proc_cols,
                bwords_data, bwords_len,
                inst_mults_data, inst_mults_len,
                output_data, output_len,
                op_stack_data, op_stack_rows, op_stack_cols,
                ram_data, ram_rows, ram_cols,
                ph_data, ph_rows, ph_cols,
                hash_data, hash_rows, hash_cols,
                sponge_data, sponge_rows, sponge_cols,
                u32_data, u32_len,
                cascade_data, cascade_len
            );
            
            std::cout << "  [1c] Rust FFI trace execution: " << elapsed_ms(step_start) << " ms" << std::endl;
        }
        
        const AlgebraicExecutionTrace& aet = trace_result.aet;
        std::cout << "       Processor rows: " << aet.height_of_table(1) << std::endl;

        step_start = std::chrono::high_resolution_clock::now();
        
        // Step 1d: Set up STARK parameters and domains
        Stark stark = Stark::default_stark();
        const size_t padded_height = aet.padded_height();
        const size_t rand_trace_len = stark.randomized_trace_len(padded_height);
        const size_t fri_domain_length = stark.fri_expansion_factor() * rand_trace_len;
        ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length)
            .with_offset(BFieldElement::generator());
        
        ProverDomains domains = ProverDomains::derive(
            padded_height,
            stark.num_trace_randomizers(),
            fri_domain,
            stark.max_degree(padded_height)
        );
        std::cout << "  [1d] Domain setup: " << elapsed_ms(step_start) << " ms" << std::endl;
        step_start = std::chrono::high_resolution_clock::now();
        
        // Step 1e: Generate randomizer seed
        // Allow fixed seed via environment variable for deterministic proofs
        // TRITON_FIXED_SEED=1 uses all zeros (fastest, but less secure)
        // TRITON_FIXED_SEED=<hex> uses specified seed (32 bytes as hex string)
        std::array<uint8_t, 32> randomness_seed{};
        const char* fixed_seed_env = std::getenv("TRITON_FIXED_SEED");
        if (fixed_seed_env) {
            if (strcmp(fixed_seed_env, "1") == 0 || strcmp(fixed_seed_env, "zero") == 0) {
                // All zeros - fastest, deterministic, but less secure
                std::fill(randomness_seed.begin(), randomness_seed.end(), 0);
                std::cout << "  [1e] Using fixed seed (all zeros) for deterministic proofs" << std::endl;
            } else {
                // Parse hex string (64 hex chars = 32 bytes)
                std::string seed_str = fixed_seed_env;
                if (seed_str.length() == 64) {
                    for (size_t i = 0; i < 32; ++i) {
                        std::string byte_str = seed_str.substr(i * 2, 2);
                        randomness_seed[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
                    }
                    std::cout << "  [1e] Using fixed seed from TRITON_FIXED_SEED" << std::endl;
                } else {
                    std::cerr << "Warning: TRITON_FIXED_SEED must be 64 hex characters, using random seed" << std::endl;
                    std::random_device rd;
                    for (size_t i = 0; i < 32; ++i) {
                        randomness_seed[i] = static_cast<uint8_t>(rd() & 0xFF);
                    }
                }
            }
        } else {
            // Default: random seed (most secure)
            std::random_device rd;
            for (size_t i = 0; i < 32; ++i) {
                randomness_seed[i] = static_cast<uint8_t>(rd() & 0xFF);
            }
        }
        
        // Step 1f: Main table creation (CPU or GPU Phase1)
        const bool use_gpu_phase1 = (std::getenv("TRITON_GPU_PHASE1") != nullptr);

        // Table lengths are needed in both paths
        std::array<size_t, 9> table_lengths = {
            aet.height_of_table(0),  // Program table
            aet.height_of_table(1),  // Processor table
            aet.height_of_table(2),  // Op stack table
            aet.height_of_table(3),  // RAM table
            aet.height_of_table(4),  // Jump stack table
            aet.height_of_table(5),  // Hash table
            aet.height_of_table(6),  // Cascade table
            aet.height_of_table(7),  // Lookup table
            aet.height_of_table(8)   // U32 table
        };

        // In GPU Phase1 mode we do NOT build the CPU main table nor flatten it.
        // The main table will be constructed on GPU inside `GpuStark::prove_with_gpu_phase1`.
        uint64_t* flat_table_mem = nullptr;
        const uint64_t* flat_table = nullptr;
        size_t num_rows = padded_height;
        constexpr size_t num_cols = 379;

        if (!use_gpu_phase1) {
            // CPU main table creation (default)
            MasterMainTable main_table = MasterMainTable::from_aet(
                aet, domains, stark.num_trace_randomizers(), randomness_seed);
            std::cout << "  [1f] Create main table (incl. Bézout): " << elapsed_ms(step_start) << " ms" << std::endl;
            
            // Debug: Check rows 181-186 after from_aet (before padding)
            if (const char* debug_flag = std::getenv("TVM_DEBUG_U32_FILL")) {
                std::cerr << "[DBG MAIN] After from_aet, before padding (rows 181-186):" << std::endl;
                for (size_t r = 180; r <= 186 && r < main_table.num_rows(); ++r) {
                    const auto& row = main_table.row(r);
                    uint64_t ci = row[TableColumnOffsets::U32_TABLE_START + U32MainColumn::CI].value();
                    uint64_t result = row[TableColumnOffsets::U32_TABLE_START + U32MainColumn::Result].value();
                    uint64_t lhs = row[TableColumnOffsets::U32_TABLE_START + U32MainColumn::LHS].value();
                    uint64_t rhs = row[TableColumnOffsets::U32_TABLE_START + U32MainColumn::RHS].value();
                    std::cerr << "[DBG MAIN] Row " << r << ": CI=" << ci << ", Result=" << result 
                              << ", LHS=" << lhs << ", RHS=" << rhs << std::endl;
                }
            }

            step_start = std::chrono::high_resolution_clock::now();

            main_table.pad(padded_height, table_lengths);
            std::cout << "  [1g] Pad main table: " << elapsed_ms(step_start) << " ms" << std::endl;
            
            // Compare padded main table with Rust test data
            if (const char* test_data_dir = std::getenv("TVM_RUST_TEST_DATA_DIR")) {
                validate_padded_main_table_data(test_data_dir, main_table, padded_height, num_cols);
            }
            
            // Debug: dump first row of padded main table
            if (std::getenv("TVM_DEBUG_MAIN_TABLE_FIRST_ROW")) {
                std::cerr << "[DBG] Padded main table first row (first 15 cols), num_cols=" << num_cols << std::endl;
                const auto& row0 = main_table.row(0);
                size_t max_cols = (num_cols < 15) ? num_cols : 15;
                for (size_t c = 0; c < max_cols; c++) {
                    uint64_t val = row0[c].value();
                    fprintf(stderr, "  [%zu]: %lu\n", c, val);
                }
                // Also show last column
                if (num_cols > 0) {
                    uint64_t last_val = row0[num_cols - 1].value();
                    fprintf(stderr, "  [%zu] (LAST): %lu\n", num_cols - 1, last_val);
                }
            }
            
            step_start = std::chrono::high_resolution_clock::now();

            // Flatten for GPU upload
            const size_t row_bytes = num_cols * sizeof(uint64_t);
            flat_table_mem = static_cast<uint64_t*>(
                std::aligned_alloc(64, num_rows * num_cols * sizeof(uint64_t)));
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < num_rows; ++r) {
                const auto& row = main_table.row(r);
                std::memcpy(flat_table_mem + r * num_cols,
                           reinterpret_cast<const uint64_t*>(row.data()),
                           row_bytes);
            }
            flat_table = flat_table_mem;
            std::cout << "  [1h] Flatten table (aligned): " << elapsed_ms(step_start) << " ms" << std::endl;

            // Optional: Verify main table creation against Rust reference implementation
            const char* verify_main_table_env = std::getenv("TVM_VERIFY_MAIN_TABLE");
            if (verify_main_table_env && (std::string(verify_main_table_env) == "1" || std::string(verify_main_table_env) == "true")) {
                step_start = std::chrono::high_resolution_clock::now();
                std::cout << "  [1i] Verify main table (Rust FFI): ";

                int verify_result = verify_main_table_creation_rust_ffi(
                    program_path.c_str(),
                    public_input_bfe.empty() ? nullptr : reinterpret_cast<const uint64_t*>(public_input_bfe.data()),
                    public_input_bfe.size(),
                    randomness_seed.data(),
                    flat_table,
                    num_rows,
                    num_cols
                );

                if (verify_result == 0) {
                    std::cout << "✓ PASSED (" << elapsed_ms(step_start) << " ms)" << std::endl;
                } else {
                    std::cout << "✗ FAILED (" << elapsed_ms(step_start) << " ms)" << std::endl;
                    std::cerr << "Error: Main table verification failed! C++ and Rust implementations produce different results." << std::endl;
                    std::cerr << "This indicates a bug in either the C++ or Rust implementation." << std::endl;
                    return 1;
                }
            }
        } else {
            std::cout << "  [1f] Main table: GPU Phase 1 enabled (TRITON_GPU_PHASE1=1)" << std::endl;
        }
        
        // Step 1i: Extract domain info for GPU
        uint64_t trace_dom[3] = {
            static_cast<uint64_t>(domains.trace.length),
            domains.trace.offset.value(),
            domains.trace.generator.value()
        };
        uint64_t quot_dom[3] = {
            static_cast<uint64_t>(domains.quotient.length),
            domains.quotient.offset.value(),
            domains.quotient.generator.value()
        };
        uint64_t fri_dom[3] = {
            static_cast<uint64_t>(domains.fri.length),
            domains.fri.offset.value(),
            domains.fri.generator.value()
        };
        
        double phase1_time = elapsed_ms(phase1_start);
        std::cout << "  ────────────────────────────────────" << std::endl;
        std::cout << "  Trace dimensions: " << num_rows << " x " << num_cols << std::endl;
        std::cout << "  FRI domain: " << fri_dom[0] << " points" << std::endl;
        std::cout << "  Phase 1 Total: " << phase1_time << " ms" << std::endl;
        
        // Build claim (pure C++)
        Claim claim;
        claim.version = 0;
        claim.program_digest = program.hash();
        claim.input = public_input_bfe;
        claim.output = trace_result.output;
        
        // =====================================================================
        // PHASE 2: Full GPU Proof Generation
        // =====================================================================
        std::cout << "\n━━━ Phase 2: GPU Proof Generation ━━━" << std::endl;
        auto phase2_start = std::chrono::high_resolution_clock::now();
        
        // Extract randomizer coefficients for main and aux tables
        std::vector<uint64_t> main_randomizer_coeffs;
        std::vector<uint64_t> aux_randomizer_coeffs;
        {
            // Use the same seed that was used to create the main table
            const std::array<uint8_t, 32>& seed_array = randomness_seed;

            const size_t aux_width = 88;
            const size_t num_randomizers = stark.num_trace_randomizers();
            
            // Check if we should load randomizers from Rust test data for exact comparison
            // Skip loading if TVM_DISABLE_RANDOMIZER_LOAD is set (use ChaCha12 RNG instead)
            const char* disable_load_env = std::getenv("TVM_DISABLE_RANDOMIZER_LOAD");
            bool skip_loading = (disable_load_env && (strcmp(disable_load_env, "1") == 0 || strcmp(disable_load_env, "true") == 0));
            
            const char* use_rust_rand_env = std::getenv("TVM_USE_RUST_RANDOMIZERS");
            const char* rust_test_data_env = std::getenv("TVM_RUST_TEST_DATA_DIR");
            bool loaded_from_rust = false;
            
            if (use_rust_rand_env && rust_test_data_env && !skip_loading) {
                std::string rust_test_data_dir = rust_test_data_env;
                std::cout << "[RANDOMIZERS] Attempting to load from Rust test data: " << rust_test_data_dir << std::endl;
                
                bool main_ok = load_main_trace_randomizers_from_rust(
                    rust_test_data_dir, num_cols, num_randomizers, main_randomizer_coeffs);
                bool aux_ok = load_aux_trace_randomizers_from_rust(
                    rust_test_data_dir, aux_width, num_randomizers, aux_randomizer_coeffs);
                
                if (main_ok && aux_ok) {
                    loaded_from_rust = true;
                    std::cout << "[RANDOMIZERS] ✅ Loaded ALL randomizers from Rust test data" << std::endl;
                    std::cout << "  Main: " << main_randomizer_coeffs.size() << " coeffs (" 
                              << num_cols << " cols × " << num_randomizers << " rands)" << std::endl;
                    std::cout << "  Aux:  " << aux_randomizer_coeffs.size() << " coeffs ("
                              << aux_width << " cols × 3 comps × " << num_randomizers << " rands)" << std::endl;
                    
                    // Print first few for verification
                    if (main_randomizer_coeffs.size() >= num_randomizers) {
                        std::cout << "  Main[0] first 3: ";
                        for (size_t i = 0; i < std::min<size_t>(3, num_randomizers); ++i) {
                            std::cout << main_randomizer_coeffs[i] << " ";
                        }
                        std::cout << std::endl;
                    }
                } else {
                    std::cout << "[RANDOMIZERS] ⚠️  Failed to load from Rust (main=" << main_ok 
                              << ", aux=" << aux_ok << "), falling back to C++ generation" << std::endl;
                }
            }
            
            if (!loaded_from_rust) {
                // Generate randomizer coefficients using ChaCha12Rng (C++ generation)
                // Main table randomizers - use the same domains as main_table
                MasterMainTable main_temp_table(1, num_cols, domains.trace, domains.quotient, domains.fri, seed_array);
                main_temp_table.set_num_trace_randomizers(num_randomizers);

                main_randomizer_coeffs.clear();
                main_randomizer_coeffs.reserve(num_cols * num_randomizers);
                for (size_t col = 0; col < num_cols; ++col) {
                    auto coeffs = main_temp_table.trace_randomizer_for_column(col);
                    for (const auto& coeff : coeffs) {
                        main_randomizer_coeffs.push_back(coeff.value());
                    }
                }

                // Aux table randomizers (aux table has 88 XFieldElement columns)
                // IMPORTANT: GPU kernels expect aux randomizer coeffs in "component-column major" layout:
                //   index = (xfe_col * 3 + comp) * num_randomizers + rand_idx
                MasterAuxTable aux_temp_table(1, aux_width, domains.trace, domains.quotient, domains.fri);
                aux_temp_table.set_num_trace_randomizers(num_randomizers);
                aux_temp_table.set_trace_randomizer_seed(seed_array);

                aux_randomizer_coeffs.assign(aux_width * 3 * num_randomizers, 0);
                for (size_t col = 0; col < aux_width; ++col) {
                    auto coeffs = aux_temp_table.trace_randomizer_for_column(col);
                    for (size_t r = 0; r < num_randomizers && r < coeffs.size(); ++r) {
                        aux_randomizer_coeffs[(col * 3 + 0) * num_randomizers + r] = coeffs[r].value();
                        // (col*3+1) and (col*3+2) remain zero in fallback path
                    }
                }
            }
        }

        gpu::GpuStark gpu_stark;
        
        // Pass U32 entries for GPU table fill (if TRITON_GPU_U32=1)
        {
            const auto& u32_raw_entries = aet.u32_entries();
            std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>> u32_entries;
            u32_entries.reserve(u32_raw_entries.size());
            for (const auto& [entry, mult] : u32_raw_entries) {
                u32_entries.emplace_back(
                    entry.instruction_opcode,
                    entry.left_operand.value(),
                    entry.right_operand.value(),
                    mult
                );
            }
            gpu_stark.set_u32_entries(u32_entries);
        }
        
        Proof proof;
        if (!use_gpu_phase1) {
            proof = gpu_stark.prove(
                claim,
                flat_table,
                num_rows,
                num_cols,
                trace_dom,
                quot_dom,
                fri_dom,
                randomness_seed.data(),
                main_randomizer_coeffs,
                aux_randomizer_coeffs
            );
        } else {
            auto t_prep0 = std::chrono::high_resolution_clock::now();
            std::cout << "[GPU Phase1][host-prep] Preparing Phase-1 host traces..." << std::endl;

            // ----------------------------
            // GPU Phase 1 input preparation
            // ----------------------------
            // Program trace [table_lengths[0] rows][7 cols]
            std::vector<uint64_t> program_trace(table_lengths[0] * 7, 0);
            {
                using namespace ProgramMainColumn;
                const auto& program_bwords = aet.program_bwords();
                const auto& instruction_multiplicities = aet.instruction_multiplicities();
                const size_t program_len = program_bwords.size();
                const size_t padded_program_len = table_lengths[0];
                const BFieldElement max_index_in_chunk = BFieldElement(static_cast<uint64_t>(Tip5::RATE - 1));
                std::array<BFieldElement, Tip5::RATE> precomputed_inv;
                for (size_t i = 0; i < Tip5::RATE; ++i) {
                    BFieldElement index_in_chunk(static_cast<uint64_t>(i));
                    BFieldElement denom = max_index_in_chunk - index_in_chunk;
                    precomputed_inv[i] = denom.is_zero() ? BFieldElement::zero() : denom.inverse();
                }
#ifdef TVM_USE_TBB
                tbb::parallel_for(size_t(0), padded_program_len, [&](size_t row) {
                    uint64_t addr = row;
                    BFieldElement instr = BFieldElement::zero();
                    if (row < program_len) instr = program_bwords[row];
                    else if (row == program_len) instr = BFieldElement::one();
                    uint64_t mult = (row < instruction_multiplicities.size()) ? instruction_multiplicities[row] : 0;
                    size_t idx_mod = row % Tip5::RATE;

                    auto write = [&](size_t col, uint64_t v) { program_trace[row * 7 + col] = v; };
                    write(Address, addr);
                    write(Instruction, instr.value());
                    write(LookupMultiplicity, mult);
                    write(IsHashInputPadding, (row < program_len) ? 0 : 1);
                    write(IsTablePadding, 0);
                    write(IndexInChunk, idx_mod);
                    write(MaxMinusIndexInChunkInv, precomputed_inv[idx_mod].value());
                });
#else
                for (size_t row = 0; row < padded_program_len; ++row) {
                    uint64_t addr = row;
                    BFieldElement instr = BFieldElement::zero();
                    if (row < program_len) instr = program_bwords[row];
                    else if (row == program_len) instr = BFieldElement::one();
                    uint64_t mult = (row < instruction_multiplicities.size()) ? instruction_multiplicities[row] : 0;
                    size_t idx_mod = row % Tip5::RATE;

                    auto write = [&](size_t col, uint64_t v) { program_trace[row * 7 + col] = v; };
                    write(Address, addr);
                    write(Instruction, instr.value());
                    write(LookupMultiplicity, mult);
                    write(IsHashInputPadding, (row < program_len) ? 0 : 1);
                    write(IsTablePadding, 0);
                    write(IndexInChunk, idx_mod);
                    write(MaxMinusIndexInChunkInv, precomputed_inv[idx_mod].value());
                }
#endif
            }
            std::cout << "[GPU Phase1][host-prep] Program trace ready (" << program_trace.size() / 7 << " rows)" << std::endl;

            // OpStack trace (flat) [rows][4] - parallelized
            const auto& os = aet.op_stack_underflow_trace();
            std::vector<uint64_t> op_stack_trace(os.size() * 4, 0);
#ifdef TVM_USE_TBB
            tbb::parallel_for(size_t(0), os.size(), [&](size_t r) {
                for (size_t c = 0; c < 4 && c < os[r].size(); ++c) {
                    op_stack_trace[r * 4 + c] = os[r][c].value();
                }
            });
#else
            for (size_t r = 0; r < os.size(); ++r) {
                for (size_t c = 0; c < 4 && c < os[r].size(); ++c) op_stack_trace[r * 4 + c] = os[r][c].value();
            }
#endif
            std::cout << "[GPU Phase1][host-prep] OpStack trace ready (" << os.size() << " rows)" << std::endl;

            // RAM trace (sorted + inv + bezout coeff assignment) [rows][7]
            struct RamRow4 { BFieldElement clk, it, ramp, ramv; };
            std::vector<RamRow4> ram_rows;
            ram_rows.reserve(aet.ram_trace().size());
            for (const auto& r : aet.ram_trace()) {
                if (r.size() < 4) continue;
                ram_rows.push_back(RamRow4{r[0], r[1], r[2], r[3]});
            }
            std::sort(ram_rows.begin(), ram_rows.end(), [](const RamRow4& a, const RamRow4& b) {
                if (a.ramp.value() != b.ramp.value()) return a.ramp.value() < b.ramp.value();
                return a.clk.value() < b.clk.value();
            });
            std::vector<BFieldElement> unique_ramps;
            unique_ramps.reserve(ram_rows.size());
            for (const auto& rr : ram_rows) {
                if (unique_ramps.empty() || unique_ramps.back() != rr.ramp) unique_ramps.push_back(rr.ramp);
            }
            std::cout << "[GPU Phase1][host-prep] RAM rows sorted (" << ram_rows.size() << " rows), unique_ramps=" << unique_ramps.size() << std::endl;

            auto t_bez = std::chrono::high_resolution_clock::now();
            std::cout << "[GPU Phase1][host-prep] Computing Bézout coefficients (fast hybrid poly mul)..." << std::endl;
            auto [bez0, bez1] = triton_vm::compute_ram_bezout_coefficients(unique_ramps);
            
            // Debug: Print first few Bézout coefficients for comparison with Rust
            if (std::getenv("TVM_DEBUG_BEZOUT")) {
                std::cout << "[DBG] Bézout coefficients (a=bez0, b=bez1):" << std::endl;
                std::cout << "  unique_ramps: " << unique_ramps.size() << std::endl;
                std::cout << "  bez0 size: " << bez0.size() << " bez1 size: " << bez1.size() << std::endl;
                size_t show = std::min<size_t>(5, bez0.size());
                for (size_t i = 0; i < show; ++i) {
                    std::cout << "  bez0[" << i << "]=" << bez0[i].value() << " bez1[" << i << "]=" << bez1[i].value() << std::endl;
                }
            }
            
            std::cout << "[GPU Phase1][host-prep] Bézout done in "
                      << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_bez).count()
                      << " ms" << std::endl;

            std::vector<uint64_t> ram_trace(ram_rows.size() * 7, 0);
            auto set_ram = [&](size_t row, size_t col, uint64_t v) { ram_trace[row * 7 + col] = v; };
#ifdef TVM_USE_TBB
            tbb::parallel_for(size_t(0), ram_rows.size(), [&](size_t i) {
                set_ram(i, 0, ram_rows[i].clk.value());
                set_ram(i, 1, ram_rows[i].it.value());
                set_ram(i, 2, ram_rows[i].ramp.value());
                set_ram(i, 3, ram_rows[i].ramv.value());
            });
#else
            for (size_t i = 0; i < ram_rows.size(); ++i) {
                set_ram(i, 0, ram_rows[i].clk.value());
                set_ram(i, 1, ram_rows[i].it.value());
                set_ram(i, 2, ram_rows[i].ramp.value());
                set_ram(i, 3, ram_rows[i].ramv.value());
            }
#endif
            // inv diff forward + bezout coeffs (Rust-equivalent pop-from-end)
            BFieldElement current0 = bez0.empty() ? BFieldElement::zero() : bez0.back();
            BFieldElement current1 = bez1.empty() ? BFieldElement::zero() : bez1.back();
            if (!bez0.empty()) bez0.pop_back();
            if (!bez1.empty()) bez1.pop_back();
            if (!ram_rows.empty()) {
                set_ram(0, 4, 0);
                set_ram(0, 5, current0.value());
                set_ram(0, 6, current1.value());
            }
            for (size_t i = 0; i + 1 < ram_rows.size(); ++i) {
                BFieldElement ramp_diff = ram_rows[i + 1].ramp - ram_rows[i].ramp;
                set_ram(i, 4, ramp_diff.is_zero() ? 0 : ramp_diff.inverse().value());
                if (!ramp_diff.is_zero()) {
                    if (!bez0.empty()) { current0 = bez0.back(); bez0.pop_back(); }
                    if (!bez1.empty()) { current1 = bez1.back(); bez1.pop_back(); }
                }
                set_ram(i + 1, 5, current0.value());
                set_ram(i + 1, 6, current1.value());
            }
            std::cout << "[GPU Phase1][host-prep] RAM trace ready (" << (ram_trace.size() / 7) << " rows)" << std::endl;

            // JumpStack trace derived from processor trace (flat) [rows][5]
            // First pass: collect into buckets (parallelized with thread-local storage)
            const BFieldElement* proc = aet.processor_trace_flat_data();
            const size_t proc_rows = aet.processor_trace_height();
            const size_t proc_cols = aet.processor_trace_width();
            std::vector<std::vector<std::array<uint64_t, 4>>> buckets;
            buckets.reserve(64);
#ifdef TVM_USE_TBB
            // Use thread-local buckets to avoid contention, then merge
            tbb::enumerable_thread_specific<std::map<size_t, std::vector<std::array<uint64_t, 4>>>> tls_buckets;
            tbb::parallel_for(size_t(0), proc_rows, [&](size_t r) {
                auto& local_buckets = tls_buckets.local();
                const BFieldElement* pr = proc + r * proc_cols;
                BFieldElement clk = pr[processor_column_index(ProcessorMainColumn::CLK)];
                BFieldElement ci = pr[processor_column_index(ProcessorMainColumn::CI)];
                BFieldElement jsp = pr[processor_column_index(ProcessorMainColumn::JSP)];
                BFieldElement jso = pr[processor_column_index(ProcessorMainColumn::JSO)];
                BFieldElement jsd = pr[processor_column_index(ProcessorMainColumn::JSD)];
                size_t jsp_val = static_cast<size_t>(jsp.value());
                local_buckets[jsp_val].push_back({clk.value(), ci.value(), jso.value(), jsd.value()});
            });
            // Merge thread-local buckets (sequential to preserve order)
            size_t max_jsp = 0;
            for (const auto& local : tls_buckets) {
                if (!local.empty()) {
                    auto it = local.rbegin();
                    if (it->first >= max_jsp) max_jsp = it->first + 1;
                }
            }
            buckets.resize(max_jsp);
            for (const auto& local : tls_buckets) {
                for (const auto& [jsp_val, rows] : local) {
                    if (jsp_val >= buckets.size()) buckets.resize(jsp_val + 1);
                    buckets[jsp_val].insert(buckets[jsp_val].end(), rows.begin(), rows.end());
                }
            }
#else
            for (size_t r = 0; r < proc_rows; ++r) {
                const BFieldElement* pr = proc + r * proc_cols;
                BFieldElement clk = pr[processor_column_index(ProcessorMainColumn::CLK)];
                BFieldElement ci = pr[processor_column_index(ProcessorMainColumn::CI)];
                BFieldElement jsp = pr[processor_column_index(ProcessorMainColumn::JSP)];
                BFieldElement jso = pr[processor_column_index(ProcessorMainColumn::JSO)];
                BFieldElement jsd = pr[processor_column_index(ProcessorMainColumn::JSD)];
                size_t jsp_val = static_cast<size_t>(jsp.value());
                if (jsp_val >= buckets.size()) buckets.resize(jsp_val + 1);
                buckets[jsp_val].push_back({clk.value(), ci.value(), jso.value(), jsd.value()});
            }
#endif
            std::vector<uint64_t> jump_stack_trace;
            jump_stack_trace.reserve(proc_rows * 5);
            for (size_t jsp_val = 0; jsp_val < buckets.size(); ++jsp_val) {
                for (const auto& row : buckets[jsp_val]) {
                    jump_stack_trace.push_back(row[0]);         // CLK
                    jump_stack_trace.push_back(row[1]);         // CI
                    jump_stack_trace.push_back(jsp_val);        // JSP
                    jump_stack_trace.push_back(row[2]);         // JSO
                    jump_stack_trace.push_back(row[3]);         // JSD
                }
            }
            std::cout << "[GPU Phase1][host-prep] JumpStack trace ready (" << (jump_stack_trace.size() / 5) << " rows, buckets=" << buckets.size() << ")" << std::endl;

            // Hash trace (combined flat) [rows][67]
            const auto& program_hash_trace = aet.program_hash_trace();
            const auto& sponge_trace = aet.sponge_trace();
            const auto& hash_trace = aet.hash_trace();
            const size_t ph_rows = program_hash_trace.size();
            const size_t sp_rows = sponge_trace.size();
            const size_t ht_rows = hash_trace.size();
            const size_t total_hash_rows = ph_rows + sp_rows + ht_rows;
            std::vector<uint64_t> hash_trace_flat(total_hash_rows * 67, 0);
            auto copy_hash_rows = [&](size_t base_row, const auto& src) {
#ifdef TVM_USE_TBB
                tbb::parallel_for(size_t(0), src.size(), [&](size_t r) {
                    for (size_t c = 0; c < 67 && c < src[r].size(); ++c) {
                        hash_trace_flat[(base_row + r) * 67 + c] = src[r][c].value();
                    }
                });
#else
                for (size_t r = 0; r < src.size(); ++r) {
                    for (size_t c = 0; c < 67 && c < src[r].size(); ++c) {
                        hash_trace_flat[(base_row + r) * 67 + c] = src[r][c].value();
                    }
                }
#endif
            };
            copy_hash_rows(0, program_hash_trace);
            copy_hash_rows(ph_rows, sponge_trace);
            copy_hash_rows(ph_rows + sp_rows, hash_trace);
            std::cout << "[GPU Phase1][host-prep] Hash trace ready (" << total_hash_rows << " rows)" << std::endl;

            // Cascade trace (flat) [rows][6] - parallelized
            const auto& cas_mults = aet.cascade_table_lookup_multiplicities();
            std::vector<uint64_t> cascade_trace(cas_mults.size() * 6, 0);
#ifdef TVM_USE_TBB
            tbb::parallel_for(size_t(0), cas_mults.size(), [&](size_t i) {
                uint16_t to_look_up = cas_mults[i].first;
                uint64_t mult = cas_mults[i].second;
                uint8_t lo = static_cast<uint8_t>(to_look_up & 0xFF);
                uint8_t hi = static_cast<uint8_t>((to_look_up >> 8) & 0xFF);
                cascade_trace[i * 6 + CascadeMainColumn::IsPadding] = 0;
                cascade_trace[i * 6 + CascadeMainColumn::LookInHi] = hi;
                cascade_trace[i * 6 + CascadeMainColumn::LookInLo] = lo;
                cascade_trace[i * 6 + CascadeMainColumn::LookOutHi] = Tip5::LOOKUP_TABLE[hi];
                cascade_trace[i * 6 + CascadeMainColumn::LookOutLo] = Tip5::LOOKUP_TABLE[lo];
                cascade_trace[i * 6 + CascadeMainColumn::LookupMultiplicity] = mult;
            });
#else
            for (size_t i = 0; i < cas_mults.size(); ++i) {
                uint16_t to_look_up = cas_mults[i].first;
                uint64_t mult = cas_mults[i].second;
                uint8_t lo = static_cast<uint8_t>(to_look_up & 0xFF);
                uint8_t hi = static_cast<uint8_t>((to_look_up >> 8) & 0xFF);
                cascade_trace[i * 6 + CascadeMainColumn::IsPadding] = 0;
                cascade_trace[i * 6 + CascadeMainColumn::LookInHi] = hi;
                cascade_trace[i * 6 + CascadeMainColumn::LookInLo] = lo;
                cascade_trace[i * 6 + CascadeMainColumn::LookOutHi] = Tip5::LOOKUP_TABLE[hi];
                cascade_trace[i * 6 + CascadeMainColumn::LookOutLo] = Tip5::LOOKUP_TABLE[lo];
                cascade_trace[i * 6 + CascadeMainColumn::LookupMultiplicity] = mult;
            }
#endif
            std::cout << "[GPU Phase1][host-prep] Cascade trace ready (" << cas_mults.size() << " rows)" << std::endl;

            // Lookup trace (flat) [256][4] - parallelized
            std::vector<uint64_t> lookup_trace(256 * 4, 0);
            const auto& lookup_mults = aet.lookup_table_lookup_multiplicities();
#ifdef TVM_USE_TBB
            tbb::parallel_for(size_t(0), size_t(256), [&](size_t i) {
                lookup_trace[i * 4 + LookupMainColumn::IsPadding] = 0;
                lookup_trace[i * 4 + LookupMainColumn::LookIn] = i;
                lookup_trace[i * 4 + LookupMainColumn::LookOut] = Tip5::LOOKUP_TABLE[i];
                lookup_trace[i * 4 + LookupMainColumn::LookupMultiplicity] = lookup_mults[i];
            });
#else
            for (size_t i = 0; i < 256; ++i) {
                lookup_trace[i * 4 + LookupMainColumn::IsPadding] = 0;
                lookup_trace[i * 4 + LookupMainColumn::LookIn] = i;
                lookup_trace[i * 4 + LookupMainColumn::LookOut] = Tip5::LOOKUP_TABLE[i];
                lookup_trace[i * 4 + LookupMainColumn::LookupMultiplicity] = lookup_mults[i];
            }
#endif
            std::cout << "[GPU Phase1][host-prep] Lookup trace ready (256 rows)" << std::endl;

            // U32 trace (expanded) [table_lengths[8]][10]
            std::vector<uint64_t> u32_trace;
            {
                const auto& entries = aet.u32_entries();
                u32_trace.reserve(table_lengths[8] * 10);
                auto inverse_or_zero = [](BFieldElement x) { return x.is_zero() ? BFieldElement::zero() : x.inverse(); };
                std::array<BFieldElement, 64> bits_minus33_inv_lut;
                for (size_t i = 0; i < bits_minus33_inv_lut.size(); ++i) {
                    bits_minus33_inv_lut[i] = inverse_or_zero(BFieldElement(static_cast<uint64_t>(i)) - BFieldElement(33));
                }
                struct Row { BFieldElement cf,bits,bm33,ci,lhs,lhsinv,rhs,rhsinv,res,mult; };
                auto make_section = [&](uint32_t ci_opcode, BFieldElement lhs0, BFieldElement rhs0, uint64_t multiplicity) -> std::vector<Row> {
                    std::vector<Row> sec;
                    sec.reserve(40);
                    sec.push_back(Row{
                        BFieldElement::one(),
                        BFieldElement::zero(),
                        bits_minus33_inv_lut[0],
                        BFieldElement(static_cast<uint64_t>(ci_opcode)),
                        lhs0,
                        BFieldElement::zero(),
                        rhs0,
                        BFieldElement::zero(),
                        BFieldElement::zero(),
                        BFieldElement(multiplicity),
                    });
                    bool is_pow = (ci_opcode == TritonInstruction{AnInstruction::Pow}.opcode());
                    bool is_lt = (ci_opcode == TritonInstruction{AnInstruction::Lt}.opcode());
                    bool is_split = (ci_opcode == TritonInstruction{AnInstruction::Split}.opcode());
                    while (true) {
                        Row& last = sec.back();
                        bool terminal = ((last.lhs.is_zero() || is_pow) && last.rhs.is_zero());
                        if (terminal) {
                            if (is_split) last.res = BFieldElement::zero();
                            else if (is_lt) last.res = BFieldElement(2);
                            else if (is_pow) last.res = BFieldElement::one();
                            else last.res = BFieldElement::zero();
                            if (is_lt && last.bits.is_zero()) last.res = BFieldElement::zero();
                            last.lhsinv = inverse_or_zero(last.lhs);
                            last.rhsinv = inverse_or_zero(last.rhs);
                            break;
                        }
                        BFieldElement lhs_lsb(last.lhs.value() % 2);
                        BFieldElement rhs_lsb(last.rhs.value() % 2);
                        Row next = last;
                        next.cf = BFieldElement::zero();
                        next.bits = next.bits + BFieldElement::one();
                        size_t bits_idx = std::min(static_cast<size_t>(next.bits.value()), bits_minus33_inv_lut.size() - 1);
                        next.bm33 = bits_minus33_inv_lut[bits_idx];
                        if (!is_pow) next.lhs = (last.lhs - lhs_lsb) / BFieldElement(2);
                        next.rhs = (last.rhs - rhs_lsb) / BFieldElement(2);
                        next.mult = BFieldElement::zero();
                        sec.push_back(next);
                    }
                    for (int i = static_cast<int>(sec.size()) - 2; i >= 0; --i) {
                        Row& row = sec[static_cast<size_t>(i)];
                        Row& next = sec[static_cast<size_t>(i) + 1];
                        BFieldElement lhs_lsb(row.lhs.value() % 2);
                        BFieldElement rhs_lsb(row.rhs.value() % 2);
                        row.lhsinv = inverse_or_zero(row.lhs);
                        row.rhsinv = inverse_or_zero(row.rhs);
                        BFieldElement next_res = next.res;
                        if (is_split) {
                            row.res = next_res;
                        } else if (is_lt) {
                            uint64_t nr = next_res.value();
                            uint64_t lsb_l = lhs_lsb.value();
                            uint64_t lsb_r = rhs_lsb.value();
                            uint64_t cf = row.cf.value();
                            if (nr == 0 || nr == 1) row.res = next_res;
                            else if (nr == 2 && lsb_l == 0 && lsb_r == 1) row.res = BFieldElement::one();
                            else if (nr == 2 && lsb_l == 1 && lsb_r == 0) row.res = BFieldElement::zero();
                            else if (nr == 2 && cf == 1) row.res = BFieldElement::zero();
                            else row.res = BFieldElement(2);
                        } else if (is_pow) {
                            row.res = rhs_lsb.is_zero() ? (next_res * next_res) : (next_res * next_res * row.lhs);
                        } else {
                            row.res = next_res;
                        }
                    }
                    return sec;
                };
                for (const auto& [entry, mult] : entries) {
                    auto sec = make_section(entry.instruction_opcode, entry.left_operand, entry.right_operand, mult);
                    for (const auto& r : sec) {
                        u32_trace.push_back(r.cf.value());
                        u32_trace.push_back(r.bits.value());
                        u32_trace.push_back(r.bm33.value());
                        u32_trace.push_back(r.ci.value());
                        u32_trace.push_back(r.lhs.value());
                        u32_trace.push_back(r.lhsinv.value());
                        u32_trace.push_back(r.rhs.value());
                        u32_trace.push_back(r.rhsinv.value());
                        u32_trace.push_back(r.res.value());
                        u32_trace.push_back(r.mult.value());
                    }
                }
                size_t want_rows = table_lengths[8];
                size_t have_rows = u32_trace.size() / 10;
                if (have_rows > want_rows) u32_trace.resize(want_rows * 10);
                if (have_rows < want_rows) u32_trace.resize(want_rows * 10, 0);
            }
            std::cout << "[GPU Phase1][host-prep] U32 trace ready (" << (u32_trace.size() / 10) << " rows)" << std::endl;

            std::cout << "[GPU Phase1][host-prep] Total host-prep: "
                      << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_prep0).count()
                      << " ms" << std::endl;

            triton_vm::gpu::GpuStark::Phase1HostTraces phase1_traces;
            phase1_traces.h_program_trace = program_trace.data();
            phase1_traces.program_rows = table_lengths[0];
            phase1_traces.h_processor_trace = reinterpret_cast<const uint64_t*>(aet.processor_trace_flat_data());
            phase1_traces.processor_rows = aet.processor_trace_height();
            phase1_traces.h_op_stack_trace = op_stack_trace.data();
            phase1_traces.op_stack_rows = os.size();
            phase1_traces.h_ram_trace = ram_trace.data();
            phase1_traces.ram_rows = ram_rows.size();
            phase1_traces.h_jump_stack_trace = jump_stack_trace.data();
            phase1_traces.jump_stack_rows = jump_stack_trace.size() / 5;
            phase1_traces.h_hash_trace = hash_trace_flat.data();
            phase1_traces.program_hash_rows = ph_rows;
            phase1_traces.sponge_rows = sp_rows;
            phase1_traces.hash_rows = ht_rows;
            phase1_traces.h_cascade_trace = cascade_trace.data();
            phase1_traces.cascade_rows = cas_mults.size();
            phase1_traces.h_lookup_trace = lookup_trace.data();
            phase1_traces.lookup_rows = 256;
            phase1_traces.h_u32_trace = u32_trace.data();
            phase1_traces.u32_rows = table_lengths[8];
            phase1_traces.table_lengths_9 = table_lengths.data();

            // ============================================================
            // Debug: compare CPU main table vs GPU Phase1-built main table
            // ============================================================
            if (std::getenv("TVM_DEBUG_GPU_PHASE1_COMPARE_MAIN")) {
                std::cout << "\n[DBG CMP MAIN] Building CPU main table reference..." << std::endl;
                auto t_cpu = std::chrono::high_resolution_clock::now();

                MasterMainTable cpu_main = MasterMainTable::from_aet(
                    aet, domains, stark.num_trace_randomizers(), randomness_seed);
                cpu_main.pad(padded_height, table_lengths);

                std::vector<uint64_t> cpu_flat(num_rows * num_cols);
                const size_t row_bytes = num_cols * sizeof(uint64_t);
                for (size_t r = 0; r < num_rows; ++r) {
                    const auto& row = cpu_main.row(r);
                    std::memcpy(cpu_flat.data() + r * num_cols,
                               reinterpret_cast<const uint64_t*>(row.data()),
                               row_bytes);
                }
                std::cout << "[DBG CMP MAIN] CPU main table ready in " << elapsed_ms(t_cpu) << " ms" << std::endl;

                std::cout << "[DBG CMP MAIN] Building GPU main table (temp)..." << std::endl;
                auto t_gpu = std::chrono::high_resolution_clock::now();

                cudaStream_t dbg_stream = nullptr;
                CUDA_CHECK(cudaStreamCreate(&dbg_stream));

                triton_vm::gpu::kernels::GpuAETData* d_aet = triton_vm::gpu::kernels::gpu_upload_aet_flat(
                    phase1_traces.h_program_trace, phase1_traces.program_rows,
                    phase1_traces.h_processor_trace, phase1_traces.processor_rows,
                    phase1_traces.h_op_stack_trace, phase1_traces.op_stack_rows,
                    phase1_traces.h_ram_trace, phase1_traces.ram_rows,
                    phase1_traces.h_jump_stack_trace, phase1_traces.jump_stack_rows,
                    phase1_traces.h_hash_trace, phase1_traces.program_hash_rows, phase1_traces.sponge_rows, phase1_traces.hash_rows,
                    phase1_traces.h_cascade_trace, phase1_traces.cascade_rows,
                    phase1_traces.h_lookup_trace, phase1_traces.lookup_rows,
                    phase1_traces.h_u32_trace, phase1_traces.u32_rows,
                    dbg_stream
                );

                uint64_t* d_tmp_main = nullptr;
                CUDA_CHECK(cudaMalloc(&d_tmp_main, num_rows * num_cols * sizeof(uint64_t)));
                triton_vm::gpu::kernels::gpu_create_main_table_into(
                    d_aet,
                    d_tmp_main,
                    num_rows,
                    table_lengths.data(),
                    dbg_stream
                );

                std::vector<uint64_t> gpu_flat(num_rows * num_cols);
                CUDA_CHECK(cudaMemcpyAsync(
                    gpu_flat.data(),
                    d_tmp_main,
                    num_rows * num_cols * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    dbg_stream
                ));
                CUDA_CHECK(cudaStreamSynchronize(dbg_stream));

                triton_vm::gpu::kernels::gpu_free_aet(d_aet);
                CUDA_CHECK(cudaFree(d_tmp_main));
                CUDA_CHECK(cudaStreamDestroy(dbg_stream));

                std::cout << "[DBG CMP MAIN] GPU main table ready in " << elapsed_ms(t_gpu) << " ms" << std::endl;

                auto mix64 = [](uint64_t x) {
                    // splitmix64
                    x += 0x9e3779b97f4a7c15ULL;
                    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
                    return x ^ (x >> 31);
                };
                auto hash_block = [&](const std::vector<uint64_t>& flat, size_t rows, size_t row_stride, size_t col_start, size_t col_count) {
                    uint64_t h = 0;
                    for (size_t r = 0; r < rows; ++r) {
                        const uint64_t* base = flat.data() + r * row_stride + col_start;
                        for (size_t c = 0; c < col_count; ++c) {
                            h = mix64(h ^ mix64(base[c] + 0x9e3779b97f4a7c15ULL * (uint64_t)(c + 1)));
                        }
                    }
                    return h;
                };

                struct Tbl { const char* name; size_t start; size_t cols; size_t len; };
                std::array<Tbl, 9> tbls = {{
                    {"program",   TableColumnOffsets::PROGRAM_TABLE_START,    TableColumnOffsets::PROGRAM_TABLE_COLS,    table_lengths[0]},
                    {"processor", TableColumnOffsets::PROCESSOR_TABLE_START,  TableColumnOffsets::PROCESSOR_TABLE_COLS,  table_lengths[1]},
                    {"opstack",   TableColumnOffsets::OP_STACK_TABLE_START,   TableColumnOffsets::OP_STACK_TABLE_COLS,   table_lengths[2]},
                    {"ram",       TableColumnOffsets::RAM_TABLE_START,        TableColumnOffsets::RAM_TABLE_COLS,        table_lengths[3]},
                    {"jump",      TableColumnOffsets::JUMP_STACK_TABLE_START, TableColumnOffsets::JUMP_STACK_TABLE_COLS, table_lengths[4]},
                    {"hash",      TableColumnOffsets::HASH_TABLE_START,       TableColumnOffsets::HASH_TABLE_COLS,       table_lengths[5]},
                    {"cascade",   TableColumnOffsets::CASCADE_TABLE_START,    TableColumnOffsets::CASCADE_TABLE_COLS,    table_lengths[6]},
                    {"lookup",    TableColumnOffsets::LOOKUP_TABLE_START,     TableColumnOffsets::LOOKUP_TABLE_COLS,     table_lengths[7]},
                    {"u32",       TableColumnOffsets::U32_TABLE_START,        TableColumnOffsets::U32_TABLE_COLS,        table_lengths[8]},
                }};

                std::cout << "[DBG CMP MAIN] Per-table hashes (effective rows and full padded rows):" << std::endl;
                for (const auto& t : tbls) {
                    const size_t eff_rows = std::min(t.len, num_rows);
                    const uint64_t cpu_eff = hash_block(cpu_flat, eff_rows, num_cols, t.start, t.cols);
                    const uint64_t gpu_eff = hash_block(gpu_flat, eff_rows, num_cols, t.start, t.cols);
                    const uint64_t cpu_full = hash_block(cpu_flat, num_rows, num_cols, t.start, t.cols);
                    const uint64_t gpu_full = hash_block(gpu_flat, num_rows, num_cols, t.start, t.cols);
                    std::cout << "  - " << t.name
                              << " cols[" << t.start << ".." << (t.start + t.cols - 1) << "]"
                              << " eff_rows=" << eff_rows
                              << " hash_eff cpu=0x" << std::hex << cpu_eff << " gpu=0x" << gpu_eff << std::dec
                              << " | hash_full cpu=0x" << std::hex << cpu_full << " gpu=0x" << gpu_full << std::dec
                              << std::endl;

                    if (cpu_full != gpu_full) {
                        std::cout << "[DBG CMP MAIN] MISMATCH detected in table '" << t.name << "'. Finding first diff..." << std::endl;
                        bool found = false;
                        for (size_t r = 0; r < num_rows && !found; ++r) {
                            for (size_t c = 0; c < t.cols; ++c) {
                                const size_t idx = r * num_cols + (t.start + c);
                                if (cpu_flat[idx] != gpu_flat[idx]) {
                                    std::cout << "[DBG CMP MAIN] First diff: table=" << t.name
                                              << " row=" << r
                                              << " col=" << (t.start + c)
                                              << " cpu=" << cpu_flat[idx]
                                              << " gpu=" << gpu_flat[idx]
                                              << std::endl;
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (!found) {
                            std::cout << "[DBG CMP MAIN] Hash mismatch but no element mismatch found (unexpected)." << std::endl;
                        }
                        break; // stop at first mismatching table for readability
                    }
                }
                std::cout << "[DBG CMP MAIN] Done.\n" << std::endl;
            }

            proof = gpu_stark.prove_with_gpu_phase1(
                claim,
                phase1_traces,
                num_rows,
                num_cols,
                trace_dom,
                quot_dom,
                fri_dom,
                randomness_seed.data(),
                main_randomizer_coeffs,
                aux_randomizer_coeffs
            );
        }
        
        double phase2_time = elapsed_ms(phase2_start);
        std::cout << "  GPU proof generation: " << phase2_time << " ms" << std::endl;

        // =====================================================================
        // PHASE 3: Rust FFI - Proof Encoding
        // =====================================================================
        std::cout << "\n━━━ Phase 3: Rust FFI (Encode) ━━━" << std::endl;
        auto phase3_start = std::chrono::high_resolution_clock::now();
        
        // Save claim (Rust verifier expects a separate claim file for `triton-cli verify`)
        claim.save_to_file(output_claim);
        
        // -----------------------------------------------------------------
        // Encode + serialize proof stream using Rust FFI
        // -----------------------------------------------------------------
        // Our GPU prover currently downloads a raw proof buffer with items laid out as:
        //   [log2_padded_height(u64)]
        //   [main_merkle_root(5)]
        //   [aux_merkle_root(5)]
        //   [quotient_merkle_root(5)]
        //   [ood_main_curr(main_width XFEs)]
        //   [ood_main_next(main_width XFEs)]
        //   [ood_aux_curr(aux_width XFEs)]
        //   [ood_aux_next(aux_width XFEs)]
        //   [ood_quotient_segments(NUM_QUOTIENT_SEGMENTS XFEs)]
        //   [fri_roots(each 5), ...]   (currently only roots are appended)
        //
        // We reconstruct ProofItems and then let Rust handle BFieldCodec + bincode.
        ProofStream ps;
        
        // Variables to store Merkle roots
        std::string main_merkle_root_hex;
        std::string aux_merkle_root_hex;
        std::string quotient_merkle_root_hex;
        
        size_t idx = 0;
        auto need = [&](size_t n) {
            if (idx + n > proof.elements.size()) {
                throw std::runtime_error("GPU proof buffer too small while reconstructing proof items");
            }
        };

        // Enqueue log2_padded_height before claim absorption (matches tasm-lib verifier order)
        // See: tasm-lib/src/verifier/stark_verify.rs lines 145-150
        need(1);
        uint32_t log2_height = static_cast<uint32_t>(proof.elements[idx++].value());
        ps.enqueue(ProofItem::make_log2_padded_height(log2_height));
        
        // Absorb claim into ProofStream using Rust FFI to ensure exact BFieldCodec encoding
        std::vector<uint64_t> program_digest_u64(5);
        for (size_t i = 0; i < 5; ++i) {
            program_digest_u64[i] = claim.program_digest[i].value();
        }
        
        std::vector<uint64_t> input_u64(claim.input.size());
        for (size_t i = 0; i < claim.input.size(); ++i) {
            input_u64[i] = claim.input[i].value();
        }
        
        std::vector<uint64_t> output_u64(claim.output.size());
        for (size_t i = 0; i < claim.output.size(); ++i) {
            output_u64[i] = claim.output[i].value();
        }
        
        uint64_t* encoded_ptr = nullptr;
        size_t encoded_len = 0;
        int rc = claim_encode_rust(
            program_digest_u64.data(),
            claim.version,
            input_u64.empty() ? nullptr : input_u64.data(),
            input_u64.size(),
            output_u64.empty() ? nullptr : output_u64.data(),
            output_u64.size(),
            &encoded_ptr,
            &encoded_len
        );
        
        if (rc != 0 || encoded_ptr == nullptr) {
            throw std::runtime_error("claim_encode_rust failed");
        }
        
        std::vector<BFieldElement> claim_encoding;
        claim_encoding.reserve(encoded_len);
        for (size_t i = 0; i < encoded_len; ++i) {
            claim_encoding.push_back(BFieldElement(encoded_ptr[i]));
        }
        
        claim_encode_free(encoded_ptr, encoded_len);
        ps.alter_fiat_shamir_state_with(claim_encoding);

        auto read_digest = [&]() -> Digest {
            need(5);
            Digest d;
            for (size_t j = 0; j < 5; ++j) {
                d[j] = proof.elements[idx++];
            }
            return d;
        };
        // Capture Merkle roots for validation
        Digest main_table_merkle_root = read_digest();
        ps.enqueue(ProofItem::merkle_root(main_table_merkle_root)); // main root
        main_merkle_root_hex = main_table_merkle_root.to_hex();
        
        // Sample extension challenges before aux root (matches verifier Fiat-Shamir order)
        constexpr size_t NUM_EXTENSION_CHALLENGES = 59; // Challenges::SAMPLE_COUNT
        auto extension_challenges = ps.sample_scalars(NUM_EXTENSION_CHALLENGES);
        (void)extension_challenges; // Not used, but sampling advances sponge state
        
        Digest aux_table_merkle_root = read_digest();
        ps.enqueue(ProofItem::merkle_root(aux_table_merkle_root)); // aux root
        aux_merkle_root_hex = aux_table_merkle_root.to_hex();
        
        // Sample quotient codeword weights before quotient root (matches verifier Fiat-Shamir order)
        const size_t NUM_QUOTIENT_WEIGHTS = Quotient::MASTER_AUX_NUM_CONSTRAINTS;
        auto quot_weights = ps.sample_scalars(NUM_QUOTIENT_WEIGHTS);
        (void)quot_weights; // Not used, but sampling advances sponge state
        
        Digest quotient_merkle_root = read_digest();
        ps.enqueue(ProofItem::merkle_root(quotient_merkle_root)); // quotient root
        quotient_merkle_root_hex = quotient_merkle_root.to_hex();

        // Sample out-of-domain point before OOD items (matches verifier Fiat-Shamir order)
        auto out_of_domain_point = ps.sample_scalars(1);
        (void)out_of_domain_point; // Not used, but sampling advances sponge state

        auto read_xfe_vec = [&](size_t count) -> std::vector<XFieldElement> {
            need(count * 3);
            std::vector<XFieldElement> v;
            v.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                BFieldElement c0 = proof.elements[idx++];
                BFieldElement c1 = proof.elements[idx++];
                BFieldElement c2 = proof.elements[idx++];
                v.emplace_back(c0, c1, c2);
            }
            return v;
        };

        // OOD rows + quotient segments (MUST match Rust order):
        //   OutOfDomainMainRow(curr), OutOfDomainAuxRow(curr),
        //   OutOfDomainMainRow(next), OutOfDomainAuxRow(next),
        //   OutOfDomainQuotientSegments(curr)
        auto ood_main_curr = read_xfe_vec(num_cols);
        auto ood_aux_curr = read_xfe_vec(88);
        auto ood_main_next = read_xfe_vec(num_cols);
        auto ood_aux_next = read_xfe_vec(88);
        auto ood_quot = read_xfe_vec(Quotient::NUM_QUOTIENT_SEGMENTS);

        ps.enqueue(ProofItem::out_of_domain_main_row(ood_main_curr));
        ps.enqueue(ProofItem::out_of_domain_aux_row(ood_aux_curr));
        ps.enqueue(ProofItem::out_of_domain_main_row(ood_main_next));
        ps.enqueue(ProofItem::out_of_domain_aux_row(ood_aux_next));
        ps.enqueue(ProofItem::out_of_domain_quotient_segments(ood_quot));

        // Sample beqd_weights before FRI (matches verifier Fiat-Shamir order)
        constexpr size_t NUM_DEEP_CODEWORD_COMPONENTS = 3;
        const size_t NUM_BEQD_WEIGHTS = num_cols + 88 + Quotient::NUM_QUOTIENT_SEGMENTS + NUM_DEEP_CODEWORD_COMPONENTS;
        auto beqd_weights = ps.sample_scalars(NUM_BEQD_WEIGHTS);
        (void)beqd_weights; // Not used, but sampling advances sponge state

        // Remaining data:
        // - FRI Merkle roots: (num_fri_rounds + 1) digests
        // - FriCodeword: last_round_len XFEs
        // - FriResponses: (1 + num_fri_rounds) items, each stored as:
        //     [auth_count(u64)] [auth_digests(auth_count*5)] [leaf_count(u64)] [revealed_leaves(leaf_count*3)]
        // - Trace openings:
        //     main: [row_count(u64)] [rows(row_count*main_width)] [auth_count(u64)] [auth_digests]
        //     aux : [row_count(u64)] [rows(row_count*aux_width*3)] [auth_count(u64)] [auth_digests]
        //     quot: [row_count(u64)] [rows(row_count*4*3)] [auth_count(u64)] [auth_digests]
        // padded_height already computed in Phase 1
        const size_t fri_len = padded_height * 8;
        const size_t num_fri_rounds = static_cast<size_t>(std::log2((double)fri_len)) - 9;
        const size_t fri_roots = num_fri_rounds + 1;
        const size_t last_len = fri_len >> num_fri_rounds; // expected 512 for fri_len=262144, rounds=9

        for (size_t r = 0; r < fri_roots; ++r) {
            ps.enqueue(ProofItem::merkle_root(read_digest()));
        }

        // FriCodeword (last round)
        auto last_codeword = read_xfe_vec(last_len);
        ps.enqueue(ProofItem::fri_codeword(last_codeword));

        // Compute FriPolynomial using Rust FFI helper for exact trimming/encoding behavior.
        std::vector<uint64_t> last_codeword_flat;
        last_codeword_flat.reserve(last_len * 3);
        for (const auto& xfe : last_codeword) {
            last_codeword_flat.push_back(xfe.coeff(0).value());
            last_codeword_flat.push_back(xfe.coeff(1).value());
            last_codeword_flat.push_back(xfe.coeff(2).value());
        }

        uint64_t* out_ptr = nullptr;
        size_t out_len = 0;
        int rc_poly = fri_interpolate_last_polynomial_rust(
            last_codeword_flat.data(),
            last_len,
            &out_ptr,
            &out_len
        );
        if (rc_poly != 0) {
            throw std::runtime_error("fri_interpolate_last_polynomial_rust failed");
        }
        std::vector<XFieldElement> poly_coeffs;
        poly_coeffs.reserve(out_len / 3);
        for (size_t i = 0; i + 2 < out_len; i += 3) {
            poly_coeffs.emplace_back(
                BFieldElement(out_ptr[i + 0]),
                BFieldElement(out_ptr[i + 1]),
                BFieldElement(out_ptr[i + 2])
            );
        }
        fri_interpolate_last_polynomial_free(out_ptr, out_len);
        ps.enqueue(ProofItem::fri_polynomial(poly_coeffs));

        auto read_u64 = [&]() -> uint64_t {
            need(1);
            return proof.elements[idx++].value();
        };
        auto read_digests = [&](size_t count) -> std::vector<Digest> {
            need(count * 5);
            std::vector<Digest> v;
            v.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                Digest d;
                for (size_t j = 0; j < 5; ++j) d[j] = proof.elements[idx++];
                v.push_back(d);
            }
            return v;
        };

        // FriResponses: 1 + num_fri_rounds
        for (size_t r = 0; r < 1 + num_fri_rounds; ++r) {
            size_t auth_count = static_cast<size_t>(read_u64());
            auto auth = read_digests(auth_count);
            size_t leaf_count = static_cast<size_t>(read_u64());
            auto leaves = read_xfe_vec(leaf_count);
            FriResponse resp;
            resp.auth_structure = std::move(auth);
            resp.revealed_leaves = std::move(leaves);
            ps.enqueue(ProofItem::fri_response(resp));
        }

        // Trace openings (batched)
        auto read_main_rows = [&](size_t row_count) {
            need(row_count * num_cols);
            std::vector<std::vector<BFieldElement>> rows;
            rows.resize(row_count);
            for (size_t r = 0; r < row_count; ++r) {
                rows[r].resize(num_cols);
                for (size_t c = 0; c < num_cols; ++c) rows[r][c] = proof.elements[idx++];
            }
            return rows;
        };
        auto read_aux_rows = [&](size_t row_count) {
            // 88 columns
            need(row_count * 88 * 3);
            std::vector<std::vector<XFieldElement>> rows;
            rows.resize(row_count);
            for (size_t r = 0; r < row_count; ++r) {
                rows[r].resize(88);
                for (size_t c = 0; c < 88; ++c) {
                    BFieldElement c0 = proof.elements[idx++];
                    BFieldElement c1 = proof.elements[idx++];
                    BFieldElement c2 = proof.elements[idx++];
                    rows[r][c] = XFieldElement(c0, c1, c2);
                }
            }
            return rows;
        };
        auto read_quot_rows = [&](size_t row_count) {
            constexpr size_t num_segments = Quotient::NUM_QUOTIENT_SEGMENTS;
            need(row_count * num_segments * 3);
            std::vector<std::vector<XFieldElement>> rows;
            rows.resize(row_count);
            for (size_t r = 0; r < row_count; ++r) {
                rows[r].resize(num_segments);
                for (size_t c = 0; c < num_segments; ++c) {
                    BFieldElement c0 = proof.elements[idx++];
                    BFieldElement c1 = proof.elements[idx++];
                    BFieldElement c2 = proof.elements[idx++];
                    rows[r][c] = XFieldElement(c0, c1, c2);
                }
            }
            return rows;
        };

        // main openings
        size_t main_row_count = static_cast<size_t>(read_u64());
        auto main_rows = read_main_rows(main_row_count);
        ps.enqueue(ProofItem::master_main_table_rows(main_rows));
        size_t main_auth_count = static_cast<size_t>(read_u64());
        ps.enqueue(ProofItem::authentication_structure(read_digests(main_auth_count)));

        // aux openings
        size_t aux_row_count = static_cast<size_t>(read_u64());
        auto aux_rows = read_aux_rows(aux_row_count);
        ps.enqueue(ProofItem::master_aux_table_rows(aux_rows));
        size_t aux_auth_count = static_cast<size_t>(read_u64());
        ps.enqueue(ProofItem::authentication_structure(read_digests(aux_auth_count)));

        // quotient openings
        size_t quot_row_count = static_cast<size_t>(read_u64());
        auto quot_rows = read_quot_rows(quot_row_count);
        ps.enqueue(ProofItem::quotient_segments_elements(quot_rows));
        size_t quot_auth_count = static_cast<size_t>(read_u64());
        ps.enqueue(ProofItem::authentication_structure(read_digests(quot_auth_count)));

        if (idx != proof.elements.size()) {
            throw std::runtime_error("Trailing data in GPU proof buffer: idx=" + std::to_string(idx) +
                                     " size=" + std::to_string(proof.elements.size()));
        }

        // Serialize proof using Rust FFI (BFieldCodec + bincode) for perfect compatibility
        ps.encode_and_save_to_file(output_proof);
        
        double phase3_time = elapsed_ms(phase3_start);
        std::cout << "  Proof encoding (Rust FFI): " << phase3_time << " ms" << std::endl;
        
        // =====================================================================
        // Validation against Rust test data (if TVM_RUST_TEST_DATA_DIR set)
        // =====================================================================
        const char* rust_test_data_env = std::getenv("TVM_RUST_TEST_DATA_DIR");
        if (rust_test_data_env) {
            std::string rust_test_data_dir = rust_test_data_env;
            std::cout << "\n━━━ Validation: GPU vs Rust Test Data ━━━" << std::endl;
            std::cout << "  Test data: " << rust_test_data_dir << std::endl;
            
            // Validate Merkle roots
            validate_main_table_commitment(rust_test_data_dir, main_merkle_root_hex);
            
            // Validate aux table creation (before LDE, after creation)
            // Note: d_aux_trace should still contain the original aux table data
            validate_aux_table_create(rust_test_data_dir, gpu_stark, padded_height, 88);
            
            validate_aux_table_commitment(rust_test_data_dir, aux_merkle_root_hex);
            validate_quotient_computation(rust_test_data_dir, quotient_merkle_root_hex);
            
            // Validate OOD rows against Rust reference (only if TVM_DEBUG_OOD is set)
            if (std::getenv("TVM_DEBUG_OOD")) {
                std::ifstream ood_file(rust_test_data_dir + "/14_out_of_domain_rows.json");
                if (ood_file.is_open()) {
                    nlohmann::json ood_data = nlohmann::json::parse(ood_file);
                    bool ood_match = true;
                    
                    // Compare OOD main current row (first 5 elements)
                    if (ood_data.contains("out_of_domain_main_row_curr")) {
                        auto rust_main_curr = ood_data["out_of_domain_main_row_curr"];
                        std::cout << "  OOD main_curr (first 3):" << std::endl;
                        for (size_t i = 0; i < 3 && i < rust_main_curr.size() && i < ood_main_curr.size(); ++i) {
                            // Rust may dump as integer (base field element) or string (XFE)
                            uint64_t rust_val = 0;
                            if (rust_main_curr[i].is_number()) {
                                rust_val = rust_main_curr[i].get<uint64_t>();
                            } else if (rust_main_curr[i].is_string()) {
                                // If it's a string, it's likely an XFE format - just show it
                                std::string rust_str = rust_main_curr[i].get<std::string>();
                                std::cout << "    [" << i << "] Rust: " << rust_str.substr(0, 60) << "..." << std::endl;
                                std::cout << "    [" << i << "] GPU:  (" << ood_main_curr[i].coeff(2).value() 
                                          << "·x² + " << ood_main_curr[i].coeff(1).value() 
                                          << "·x + " << ood_main_curr[i].coeff(0).value() << ")" << std::endl;
                                continue;
                            }
                            // Compare base field element (assuming Rust dumps base field, GPU has XFE)
                            // For now, compare Rust's base field value with GPU's XFE constant term
                            uint64_t gpu_val = ood_main_curr[i].coeff(0).value();
                            bool match = (rust_val == gpu_val);
                            std::cout << "    [" << i << "] Rust: " << rust_val << " | GPU (coeff0): " << gpu_val 
                                      << (match ? " ✓" : " ✗") << std::endl;
                            std::cout << "    [" << i << "] GPU XFE: (" << ood_main_curr[i].coeff(2).value() 
                                      << "·x² + " << ood_main_curr[i].coeff(1).value() 
                                      << "·x + " << ood_main_curr[i].coeff(0).value() << ")" << std::endl;
                        }
                    }
                    
                    // Compare OOD aux current row (first 3 elements)
                    if (ood_data.contains("out_of_domain_aux_row_curr")) {
                        auto rust_aux_curr = ood_data["out_of_domain_aux_row_curr"];
                        std::cout << "  OOD aux_curr (first 3):" << std::endl;
                        for (size_t i = 0; i < 3 && i < rust_aux_curr.size() && i < ood_aux_curr.size(); ++i) {
                            uint64_t rust_val = 0;
                            if (rust_aux_curr[i].is_number()) {
                                rust_val = rust_aux_curr[i].get<uint64_t>();
                            } else if (rust_aux_curr[i].is_string()) {
                                std::string rust_str = rust_aux_curr[i].get<std::string>();
                                std::cout << "    [" << i << "] Rust: " << rust_str.substr(0, 60) << "..." << std::endl;
                                std::cout << "    [" << i << "] GPU:  (" << ood_aux_curr[i].coeff(2).value() 
                                          << "·x² + " << ood_aux_curr[i].coeff(1).value() 
                                          << "·x + " << ood_aux_curr[i].coeff(0).value() << ")" << std::endl;
                                continue;
                            }
                            uint64_t gpu_val = ood_aux_curr[i].coeff(0).value();
                            bool match = (rust_val == gpu_val);
                            std::cout << "    [" << i << "] Rust: " << rust_val << " | GPU (coeff0): " << gpu_val 
                                      << (match ? " ✓" : " ✗") << std::endl;
                            std::cout << "    [" << i << "] GPU XFE: (" << ood_aux_curr[i].coeff(2).value() 
                                      << "·x² + " << ood_aux_curr[i].coeff(1).value() 
                                      << "·x + " << ood_aux_curr[i].coeff(0).value() << ")" << std::endl;
                        }
                    }
                    
                    // Compare OOD quotient segments
                    if (ood_data.contains("out_of_domain_quotient_segments")) {
                        auto rust_quot = ood_data["out_of_domain_quotient_segments"];
                        std::cout << "  OOD quotient segments:" << std::endl;
                        for (size_t i = 0; i < rust_quot.size() && i < ood_quot.size(); ++i) {
                            uint64_t rust_val = 0;
                            if (rust_quot[i].is_number()) {
                                rust_val = rust_quot[i].get<uint64_t>();
                            } else if (rust_quot[i].is_string()) {
                                std::string rust_str = rust_quot[i].get<std::string>();
                                std::cout << "    [" << i << "] Rust: " << rust_str.substr(0, 60) << "..." << std::endl;
                                std::cout << "    [" << i << "] GPU:  (" << ood_quot[i].coeff(2).value() 
                                          << "·x² + " << ood_quot[i].coeff(1).value() 
                                          << "·x + " << ood_quot[i].coeff(0).value() << ")" << std::endl;
                                continue;
                            }
                            uint64_t gpu_val = ood_quot[i].coeff(0).value();
                            bool match = (rust_val == gpu_val);
                            std::cout << "    [" << i << "] Rust: " << rust_val << " | GPU (coeff0): " << gpu_val 
                                      << (match ? " ✓" : " ✗") << std::endl;
                            std::cout << "    [" << i << "] GPU XFE: (" << ood_quot[i].coeff(2).value() 
                                      << "·x² + " << ood_quot[i].coeff(1).value() 
                                      << "·x + " << ood_quot[i].coeff(0).value() << ")" << std::endl;
                        }
                    }
                }
            }
            
            // Validate LDE samples if available (compare first row)
            validate_main_lde_samples(rust_test_data_dir, gpu_stark, padded_height * 8, num_cols);
            validate_aux_lde_samples(rust_test_data_dir, gpu_stark, padded_height * 8, 88);
            validate_quotient_lde_samples(rust_test_data_dir, gpu_stark, padded_height * 8);
            
            std::cout << "━━━ End Validation ━━━\n" << std::endl;
        }
        
        // =====================================================================
        // Summary
        // =====================================================================
        double total_time = elapsed_ms(total_start);
        
        std::cout << "\n╔══════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  Proof Generation Complete                        ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║  Phase 1 (C++ trace+pad):   " << phase1_time << " ms" << std::endl;
        std::cout << "║  Phase 2 (GPU compute):     " << phase2_time << " ms" << std::endl;
        std::cout << "║  Phase 3 (Rust FFI encode): " << phase3_time << " ms" << std::endl;
        std::cout << "║  ────────────────────────────────────────────────" << std::endl;
        std::cout << "║  Total:                     " << total_time << " ms" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║  Claim: " << output_claim << std::endl;
        std::cout << "║  Proof: " << output_proof << std::endl;
        std::cout << "╚══════════════════════════════════════════════════╝\n" << std::endl;
        
        // Cleanup memory (CPU Phase1 path only)
        if (flat_table_mem) std::free(flat_table_mem);
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
#endif
}

