#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include <sstream>
#include <nlohmann/json.hpp>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// Include stark.hpp first - it defines simplified AlgebraicExecutionTrace struct
#include "stark.hpp"
// Then include VM headers - they define the full AlgebraicExecutionTrace class
// This will cause a redefinition error, so we need to handle it
// Solution: Use a namespace alias for the VM's version
#include "vm/vm.hpp"
#include "vm/program.hpp"
#include "table/master_table.hpp"
#include "bincode_ffi.hpp"

using namespace triton_vm;

static std::vector<uint64_t> parse_u64_list(const std::string& s) {
    std::vector<uint64_t> out;
    std::string normalized = s;
    for (char& ch : normalized) {
        if (ch == ',') ch = ' ';
    }
    std::istringstream iss(normalized);
    std::string tok;
    while (iss >> tok) {
        if (tok.empty()) continue;
        // allow 0x-prefixed hex or decimal
        uint64_t v = 0;
        if (tok.rfind("0x", 0) == 0 || tok.rfind("0X", 0) == 0) {
            v = std::stoull(tok, nullptr, 16);
        } else {
            v = std::stoull(tok, nullptr, 10);
        }
        out.push_back(v);
    }
    return out;
}

/**
 * Load padded main table from Rust JSON export
 */
MasterMainTable load_padded_main_table_from_rust(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open padded main table: " + json_path);
    }

    nlohmann::json json = nlohmann::json::parse(f);
    
    if (!json.contains("padded_table_data")) {
        throw std::runtime_error("JSON missing 'padded_table_data' field");
    }
    
    auto& padded_data = json["padded_table_data"];
    size_t num_rows = json["num_rows"].get<size_t>();
    size_t num_cols = json["num_columns"].get<size_t>();

    std::cout << "Loading Rust padded main table: " << num_rows << " x " << num_cols << std::endl;

    // Create table with correct dimensions
    MasterMainTable table(num_rows, num_cols);

    // Populate with Rust data
    for (size_t r = 0; r < num_rows; r++) {
        auto& row_json = padded_data[r];
        for (size_t c = 0; c < num_cols; c++) {
            uint64_t value = row_json[c].get<uint64_t>();
            table.set(r, c, BFieldElement(value));
        }
    }

    return table;
}

/**
 * Load claim from Rust JSON export
 */
Claim load_claim_from_rust(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open claim: " + json_path);
    }

    nlohmann::json json = nlohmann::json::parse(f);

    Claim claim;

    // Parse program digest from hex string exactly like Rust's `Digest::to_hex()` / `Digest::from_hex()`.
    // The Rust test data's `program_digest` field is already in that canonical hex format.
    std::string digest_hex = json["program_digest"].get<std::string>();
    claim.program_digest = Digest::from_hex(digest_hex);

    claim.version = json["version"].get<uint32_t>();

    // Parse input
    if (json.contains("input")) {
        for (auto& val : json["input"]) {
            claim.input.push_back(BFieldElement(val.get<uint64_t>()));
        }
    }

    // Parse output
    if (json.contains("output")) {
        for (auto& val : json["output"]) {
            claim.output.push_back(BFieldElement(val.get<uint64_t>()));
        }
    }

    return claim;
}

static std::array<uint8_t, 32> random_seed_32() {
    std::array<uint8_t, 32> seed{};
    std::random_device rd;
    for (size_t i = 0; i < seed.size(); ++i) {
        seed[i] = static_cast<uint8_t>(rd() & 0xFF);
    }
    return seed;
}

/**
 * Create simplified AlgebraicExecutionTrace struct (from stark.hpp) from main table data
 * This is only used for the legacy JSON loading path.
 */
triton_vm::AlgebraicExecutionTrace create_aet_from_main_table(const MasterMainTable& main_table) {
    // Use the simplified struct from stark.hpp (not the full class from vm/aet.hpp)
    // We need to forward declare or use a different approach
    // For now, this function is not used in the new pure C++ path
    throw std::runtime_error("create_aet_from_main_table is deprecated - use VM::trace_execution instead");
}

// (Randomizers are generated in C++ now; we do not load them from JSON.)

/**
 * Main function to generate proof from Rust's padded main table
 */
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

    if (argc != 4 && argc != 5) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " <program.tasm> <public_input_list> <output_claim.json> <output_proof.bin>\n";
        std::cerr << "    public_input_list: comma/space-separated u64s (decimal or 0x... hex), e.g. \"1,2,3\" or \"0x2a\"\n";
        std::cerr << "  " << argv[0] << " <rust_test_data_dir> <output_claim.json> <output_proof.bin>  (legacy)\n";
        return 1;
    }

    const bool ffi_mode = (argc == 5);
    std::string output_claim = argv[ffi_mode ? 3 : 2];
    std::string output_proof = argv[ffi_mode ? 4 : 3];

    try {
        MasterMainTable main_table(0, 0);
        Claim claim;

        if (ffi_mode) {
            const std::string program_path = argv[1];
            const std::string public_input_list = argv[2];
            const std::vector<uint64_t> public_input_u64 = parse_u64_list(public_input_list);

            std::cout << "=== Pure C++ Implementation ===" << std::endl;
            std::cout << "Program: " << program_path << std::endl;
            std::cout << "Public input: " << public_input_list << std::endl;
            std::cout << std::endl;

            // Step 1: Load and parse program
            std::cout << "Step 1: Loading program..." << std::endl;
            Program program = Program::from_file(program_path);
            std::cout << "  ✓ Program loaded: " << program.len_bwords() << " instructions" << std::endl;

            // Step 2: Convert public input to BFieldElement vector
            std::vector<BFieldElement> public_input;
            for (uint64_t v : public_input_u64) {
                public_input.push_back(BFieldElement(v));
            }
            std::cout << "  Public input size: " << public_input.size() << std::endl;
            if (!public_input.empty()) {
                std::cout << "  First input value: " << public_input[0].value() << std::endl;
            }

            // Step 3: Execute program and generate trace
            std::cout << "Step 2: Executing program and generating trace..." << std::endl;
            VM::TraceResult trace_result = VM::trace_execution(program, public_input);
            // Use the VM's AlgebraicExecutionTrace class (VMAET alias)
            std::cout << "  ✓ Trace generated: " << trace_result.aet.processor_trace_height() << " rows" << std::endl;

            // Step 4: Create claim
            std::cout << "Step 3: Creating claim..." << std::endl;
            claim.program_digest = program.hash();
            claim.version = 1;
            claim.input = public_input;
            claim.output = trace_result.output;
            std::cout << "  ✓ Claim created" << std::endl;

            // Step 5: Derive parameters and create main table
            std::cout << "Step 4: Deriving parameters and creating main table..." << std::endl;
            // Compute padded_height: use aet.padded_height() which correctly
            // uses height() (max of all tables including Program table)
            // This matches Rust's padded_height() implementation
            size_t padded_height = trace_result.aet.padded_height();
            
            // DEBUG: Print padded_height before ProverDomains::derive()
            std::cout << "DEBUG main.cpp before ProverDomains::derive:" << std::endl;
            std::cout << "  trace_result.aet.processor_trace_height(): " << trace_result.aet.processor_trace_height() << std::endl;
            std::cout << "  trace_result.aet.height(): " << trace_result.aet.height() << std::endl;
            std::cout << "  trace_result.aet.height_of_table(0) [Program]: " << trace_result.aet.height_of_table(0) << std::endl;
            std::cout << "  trace_result.aet.height_of_table(1) [Processor]: " << trace_result.aet.height_of_table(1) << std::endl;
            std::cout << "  trace_result.aet.height_of_table(5) [Hash]: " << trace_result.aet.height_of_table(5) << std::endl;
            std::cout << "  trace_result.aet.height_of_table(7) [Lookup]: " << trace_result.aet.height_of_table(7) << std::endl;
            std::cout << "  trace_result.aet.padded_height(): " << padded_height << std::endl;
            
            // Derive domains exactly like Rust's `Stark::fri(padded_height)` + `ProverDomains::derive`.
            //
            // Rust:
            //   fri_domain_length = fri_expansion_factor * randomized_trace_len(padded_height, num_trace_randomizers)
            //   fri_domain.offset = BFieldElement::generator()
            constexpr size_t num_trace_randomizers = 30;
            Stark stark = Stark::default_stark();
            size_t fri_domain_length =
                stark.fri_expansion_factor() * stark.randomized_trace_len(padded_height);
            ArithmeticDomain fri_domain =
                ArithmeticDomain::of_length(fri_domain_length).with_offset(BFieldElement::generator());
            
            // Derive ProverDomains
            int64_t max_degree = stark.max_degree(padded_height);
            
            ProverDomains domains = ProverDomains::derive(
                padded_height,
                num_trace_randomizers,
                fri_domain,
                max_degree
            );
            
            std::cout << "  ✓ Domains derived:" << std::endl;
            std::cout << "    Trace: " << domains.trace.length << " rows" << std::endl;
            std::cout << "    Randomized trace: " << domains.randomized_trace.length << " rows" << std::endl;
            std::cout << "    Quotient: " << domains.quotient.length << " rows" << std::endl;
            std::cout << "    FRI: " << domains.fri.length << " rows" << std::endl;
            
            // Generate randomizer seed
            std::array<uint8_t, 32> trace_randomizer_seed = random_seed_32();
            
            // Create main table from AET
            main_table = MasterMainTable::from_aet(
                trace_result.aet,
                domains,
                num_trace_randomizers,
                trace_randomizer_seed
            );
            std::cout << "  ✓ Main table created: " << main_table.num_rows() << " x " << main_table.num_columns() << std::endl;

            // Step 6: Pad main table
            std::cout << "Step 5: Padding main table..." << std::endl;
            main_table.pad(padded_height);
            std::cout << "  ✓ Main table padded to: " << main_table.num_rows() << " rows" << std::endl;
            std::cout << std::endl;
        } else {
            // Legacy mode: read Rust-exported JSON test data.
            std::string rust_data_dir = argv[1];

            // Load padded main table data
            std::string padded_table_path = rust_data_dir + "/04_main_tables_pad.json";
            std::cout << "Loading padded main table from: " << padded_table_path << std::endl;
            std::ifstream f(padded_table_path);
            if (!f.is_open()) {
                throw std::runtime_error("Could not open padded main table: " + padded_table_path);
            }
            nlohmann::json json = nlohmann::json::parse(f);
            auto& padded_data = json["padded_table_data"];
            size_t num_rows = json["num_rows"].get<size_t>();
            size_t num_cols = json["num_columns"].get<size_t>();

            std::cout << "  Table dimensions: " << num_rows << " x " << num_cols << std::endl;

            // Randomizer seed: generated locally (ZK randomizers may differ and proofs still verify)
            std::array<uint8_t, 32> trace_randomizer_seed = random_seed_32();

            // Load parameters to get proper domains (matching test_all_steps_verification.cpp)
            std::string params_path = rust_data_dir + "/02_parameters.json";
            size_t padded_height = num_rows; // Default to num_rows if params not available
            
            // Initialize domains with defaults
            ArithmeticDomain trace_domain = ArithmeticDomain::of_length(num_rows).with_offset(BFieldElement(1));
            ArithmeticDomain quotient_domain = ArithmeticDomain::of_length(num_rows * 4);
            ArithmeticDomain fri_domain = ArithmeticDomain::of_length(4096);
            
            try {
                std::ifstream params_file(params_path);
                if (params_file.is_open()) {
                    nlohmann::json params_json = nlohmann::json::parse(params_file);
                    
                    // Load padded_height from parameters (critical for domain consistency)
                    if (params_json.contains("padded_height")) {
                        padded_height = params_json["padded_height"].get<size_t>();
                        std::cout << "  Loaded padded_height from parameters: " << padded_height << std::endl;
                    }
                    
                    // Create domains using padded_height (matching test_all_steps_verification.cpp)
                    trace_domain = ArithmeticDomain::of_length(padded_height).with_offset(BFieldElement(1));
                    
                    // Load quotient domain from parameters (length and offset)
                    size_t quotient_domain_length = padded_height * 4; // Default
                    BFieldElement quotient_offset = BFieldElement::zero(); // Default
                    if (params_json.contains("quotient_domain") && params_json["quotient_domain"].is_object()) {
                        auto& quot_domain = params_json["quotient_domain"];
                        if (quot_domain.contains("length")) {
                            quotient_domain_length = quot_domain["length"].get<size_t>();
                        }
                        if (quot_domain.contains("offset")) {
                            quotient_offset = BFieldElement(quot_domain["offset"].get<uint64_t>());
                        }
                    }
                    quotient_domain = ArithmeticDomain::of_length(quotient_domain_length).with_offset(quotient_offset);
                    std::cout << "  Loaded quotient domain: length=" << quotient_domain_length 
                              << ", offset=" << quotient_offset.value() << std::endl;
                    
                    // Load FRI domain from parameters (length and offset)
                    size_t fri_domain_length = 4096; // Default
                    BFieldElement fri_offset = BFieldElement::zero(); // Default
                    if (params_json.contains("fri_domain") && params_json["fri_domain"].is_object()) {
                        auto& fri_dom_json = params_json["fri_domain"];
                        if (fri_dom_json.contains("length")) {
                            fri_domain_length = fri_dom_json["length"].get<size_t>();
                        }
                        if (fri_dom_json.contains("offset")) {
                            fri_offset = BFieldElement(fri_dom_json["offset"].get<uint64_t>());
                        }
                    }
                    fri_domain = ArithmeticDomain::of_length(fri_domain_length).with_offset(fri_offset);
                    std::cout << "  Loaded FRI domain: length=" << fri_domain_length 
                              << ", offset=" << fri_offset.value() << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not load parameters, using defaults: " << e.what() << std::endl;
            }
            
            // Verify that num_rows matches padded_height (they should be the same)
            if (num_rows != padded_height) {
                std::cerr << "Warning: num_rows (" << num_rows << ") != padded_height (" << padded_height << ")" << std::endl;
                std::cerr << "  Using padded_height for domain creation" << std::endl;
            }

            // Create main table with proper domains
            std::cout << "Creating main table..." << std::endl;
            main_table = MasterMainTable(num_rows, num_cols, trace_domain, quotient_domain, fri_domain, trace_randomizer_seed);

            // Populate main table
            std::cout << "Populating main table..." << std::endl;
            for (size_t r = 0; r < num_rows; r++) {
                auto& row_json = padded_data[r];
                for (size_t c = 0; c < num_cols; c++) {
                    uint64_t value = row_json[c].get<uint64_t>();
                    main_table.set(r, c, BFieldElement(value));
                }
            }

            // Load claim from Rust
            std::string claim_path = rust_data_dir + "/06_claim.json";
            std::cout << "Loading claim from: " << claim_path << std::endl;
            claim = load_claim_from_rust(claim_path);
        }

        std::cout << "Claim loaded:" << std::endl;
        std::cout << "  Version: " << claim.version << std::endl;
        std::cout << "  Program digest: " << claim.program_digest.to_hex() << std::endl;
        std::cout << "  Input size: " << claim.input.size() << std::endl;
        std::cout << "  Output size: " << claim.output.size() << std::endl;

        std::cout << "Setting up STARK prover..." << std::endl;
        auto stark = Stark::default_stark();

        std::cout << "Initializing proof stream..." << std::endl;
        ProofStream proof_stream;
        // prove_with_table expects the claim already absorbed into the Fiat–Shamir transcript.
        // Encode claim in pure C++ matching Rust's derived BFieldCodec for `Claim`.
        std::vector<BFieldElement> claim_encoding;
        auto encode_vec_bfe_field_with_struct_len_prefix =
            [&](const std::vector<BFieldElement>& v) {
                const size_t vec_encoding_len = 1 + v.size();
                claim_encoding.push_back(BFieldElement(static_cast<uint64_t>(vec_encoding_len)));
                claim_encoding.push_back(BFieldElement(static_cast<uint64_t>(v.size())));
                for (const auto& e : v) claim_encoding.push_back(e);
            };
        encode_vec_bfe_field_with_struct_len_prefix(claim.output);
        encode_vec_bfe_field_with_struct_len_prefix(claim.input);
        claim_encoding.push_back(BFieldElement(static_cast<uint64_t>(claim.version)));
        for (size_t i = 0; i < Digest::LEN; ++i) claim_encoding.push_back(claim.program_digest[i]);
        proof_stream.alter_fiat_shamir_state_with(claim_encoding);

        std::cout << "Generating proof with pre-created table..." << std::endl;
        // Pass proof stream to prove_with_table - proof encoding/serialization is handled in Rust FFI
        // Do not pass test_data_dir: we are not matching Rust test vectors anymore.
        Proof proof = stark.prove_with_table(claim, main_table, proof_stream, output_proof, "");

        std::cout << "Saving claim..." << std::endl;
        claim.save_to_file(output_claim);
        // Proof is already saved by Rust FFI in prove_with_table

        std::cout << "\n✓ Success!" << std::endl;
        std::cout << "  Generated proof with " << proof.elements.size() << " BFieldElements" << std::endl;
        std::cout << "  Claim saved to: " << output_claim << std::endl;
        std::cout << "  Proof saved to: " << output_proof << std::endl;
        std::cout << "\nTo verify, run:" << std::endl;
        std::cout << "  ./target/release/triton-cli verify --claim " << output_claim << " --proof " << output_proof << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
