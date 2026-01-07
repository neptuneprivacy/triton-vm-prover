#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <regex>
#include <nlohmann/json.hpp>
#include "table/master_table.hpp"
#include "table/extend_helpers.hpp"
#include "stark/challenges.hpp"
#include "types/x_field_element.hpp"
#include "types/b_field_element.hpp"
#include "test_data_loader.hpp"
#include <iostream>
#include <iomanip>

using namespace triton_vm;
using json = nlohmann::json;

/**
 * Parse XFieldElement from string
 */
static XFieldElement parse_xfield_from_string(const std::string& str) {
    if (str == "0_xfe") {
        return XFieldElement::zero();
    }
    if (str == "1_xfe") {
        return XFieldElement::one();
    }
    
    std::regex single_value_pattern(R"((\d+)_xfe)");
    std::smatch single_match;
    if (std::regex_search(str, single_match, single_value_pattern)) {
        uint64_t value = std::stoull(single_match[1].str());
        return XFieldElement(
            BFieldElement(value),
            BFieldElement::zero(),
            BFieldElement::zero()
        );
    }
    
    std::regex polynomial_pattern(R"(\((\d+)·x² \+ (\d+)·x \+ (\d+)\))");
    std::smatch poly_match;
    
    if (std::regex_search(str, poly_match, polynomial_pattern)) {
        uint64_t coeff2 = std::stoull(poly_match[1].str());
        uint64_t coeff1 = std::stoull(poly_match[2].str());
        uint64_t coeff0 = std::stoull(poly_match[3].str());
        
        return XFieldElement(
            BFieldElement(coeff0),
            BFieldElement(coeff1),
            BFieldElement(coeff2)
        );
    }
    
    throw std::runtime_error("Failed to parse XFieldElement: " + str);
}

/**
 * Debug arithmetic test - trace LookupTable computation step by step
 */
class DebugArithmeticTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data_dir_ = std::string(TEST_DATA_DIR) + "/../test_data_lde";
        
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found.";
        }
    }
    
    std::string test_data_dir_;
    
    json load_json(const std::string& filename) {
        std::string path = test_data_dir_ + "/" + filename;
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + path);
        }
        return json::parse(f);
    }
    
    Challenges load_challenges() {
        auto challenges_json = load_json("07_fiat_shamir_challenges.json");
        std::vector<XFieldElement> sampled;
        
        auto& challenge_strings = challenges_json["challenge_values"];
        for (const auto& str : challenge_strings) {
            XFieldElement xfe = parse_xfield_from_string(str.get<std::string>());
            sampled.push_back(xfe);
        }
        
        Challenges challenges = Challenges::from_sampled(sampled);
        
        // Load claim data to compute derived challenges
        auto claim_json = load_json("06_claim.json");
        Digest program_digest = Digest::from_hex(claim_json["program_digest"].get<std::string>());
        std::vector<BFieldElement> input;
        for (const auto& val : claim_json["input"]) {
            input.push_back(BFieldElement(val.get<uint64_t>()));
        }
        std::vector<BFieldElement> output;
        for (const auto& val : claim_json["output"]) {
            output.push_back(BFieldElement(val.get<uint64_t>()));
        }
        std::vector<BFieldElement> lookup_table;
        
        challenges.compute_derived_challenges(
            program_digest.to_b_field_elements(),
            input,
            output,
            lookup_table
        );
        
        return challenges;
    }
};

// Test: Debug LookupTable row 0 computation
TEST_F(DebugArithmeticTest, LookupTableRow0) {
    std::cout << "\n=== Debug LookupTable Row 0 ===" << std::endl;
    
    // Load test data
    auto pad_json = load_json("04_main_tables_pad.json");
    auto& padded_data = pad_json["padded_table_data"];
    auto aux_json = load_json("07_aux_tables_create.json");
    auto& rust_row0 = aux_json["sample_rows_first"][0];
    
    // Get LookupTable row 0 data
    constexpr size_t LOOKUP_TABLE_START = 135;
    auto& row0 = padded_data[0];
    
    BFieldElement look_in(row0[LOOKUP_TABLE_START + 1].get<uint64_t>());
    BFieldElement look_out(row0[LOOKUP_TABLE_START + 2].get<uint64_t>());
    BFieldElement mult(row0[LOOKUP_TABLE_START + 3].get<uint64_t>());
    
    std::cout << "Main table row 0:" << std::endl;
    std::cout << "  LookIn: " << look_in.value() << std::endl;
    std::cout << "  LookOut: " << look_out.value() << std::endl;
    std::cout << "  Multiplicity: " << mult.value() << std::endl;
    
    // Load challenges
    Challenges challenges = load_challenges();
    using namespace ChallengeId;
    
    XFieldElement cascade_indeterminate = challenges[CascadeLookupIndeterminate];
    XFieldElement lookup_input_weight = challenges[LookupTableInputWeight];
    XFieldElement lookup_output_weight = challenges[LookupTableOutputWeight];
    
    std::cout << "\nChallenges:" << std::endl;
    std::cout << "  CascadeLookupIndeterminate: " << cascade_indeterminate << std::endl;
    std::cout << "  LookupTableInputWeight: " << lookup_input_weight << std::endl;
    std::cout << "  LookupTableOutputWeight: " << lookup_output_weight << std::endl;
    
    // Compute step by step
    std::cout << "\nComputation steps:" << std::endl;
    
    // Step 1: compressed_row = LookIn * input_weight + LookOut * output_weight
    XFieldElement compressed_row = 
        XFieldElement(look_in) * lookup_input_weight +
        XFieldElement(look_out) * lookup_output_weight;
    
    std::cout << "  Step 1 - compressed_row: " << compressed_row << std::endl;
    
    // Step 2: diff = cascade_indeterminate - compressed_row
    XFieldElement diff = cascade_indeterminate - compressed_row;
    std::cout << "  Step 2 - diff: " << diff << std::endl;
    
    // Step 3: diff_inverse = diff.inverse()
    XFieldElement diff_inverse = diff.inverse();
    std::cout << "  Step 3 - diff_inverse: " << diff_inverse << std::endl;
    
    // Step 4: contribution = diff_inverse * mult
    XFieldElement contribution = diff_inverse * mult;
    std::cout << "  Step 4 - contribution: " << contribution << std::endl;
    
    // Step 5: cascade_running_sum = 0 + contribution
    XFieldElement cascade_running_sum = LookupArg::default_initial() + contribution;
    std::cout << "  Step 5 - cascade_running_sum: " << cascade_running_sum << std::endl;
    
    // Verify: LookupArg::default_initial() should be 0
    XFieldElement initial = LookupArg::default_initial();
    std::cout << "  Initial value: " << initial << " (should be 0)" << std::endl;
    std::cout << "  Is initial == 0? " << (initial.is_zero() ? "YES" : "NO") << std::endl;
    
    // Expected value from Rust
    std::string rust_str = rust_row0[27].get<std::string>();  // CascadeRunningSumLogDerivative
    XFieldElement rust_value = parse_xfield_from_string(rust_str);
    std::cout << "\nExpected (Rust): " << rust_str << std::endl;
    std::cout << "Computed (C++): " << cascade_running_sum << std::endl;
    std::cout << "Match: " << (rust_value == cascade_running_sum ? "YES" : "NO") << std::endl;
    
    if (rust_value != cascade_running_sum) {
        std::cout << "\nDetailed comparison:" << std::endl;
        std::cout << "  Rust coeff0: " << rust_value.coeff(0).value() << std::endl;
        std::cout << "  C++  coeff0: " << cascade_running_sum.coeff(0).value() << std::endl;
        std::cout << "  Rust coeff1: " << rust_value.coeff(1).value() << std::endl;
        std::cout << "  C++  coeff1: " << cascade_running_sum.coeff(1).value() << std::endl;
        std::cout << "  Rust coeff2: " << rust_value.coeff(2).value() << std::endl;
        std::cout << "  C++  coeff2: " << cascade_running_sum.coeff(2).value() << std::endl;
    }
}

// Test: Debug PublicRunningEvaluation
TEST_F(DebugArithmeticTest, LookupTablePublicRunningEval) {
    std::cout << "\n=== Debug PublicRunningEvaluation Row 0 ===" << std::endl;
    
    auto pad_json = load_json("04_main_tables_pad.json");
    auto& padded_data = pad_json["padded_table_data"];
    auto aux_json = load_json("07_aux_tables_create.json");
    auto& rust_row0 = aux_json["sample_rows_first"][0];
    
    constexpr size_t LOOKUP_TABLE_START = 135;
    auto& row0 = padded_data[0];
    
    BFieldElement look_out(row0[LOOKUP_TABLE_START + 2].get<uint64_t>());
    
    Challenges challenges = load_challenges();
    using namespace ChallengeId;
    
    XFieldElement lookup_public_indeterminate = challenges[LookupTablePublicIndeterminate];
    
    std::cout << "LookOut: " << look_out.value() << std::endl;
    std::cout << "LookupTablePublicIndeterminate: " << lookup_public_indeterminate << std::endl;
    
    // Compute: public_running_eval = 1 * indeterminate + LookOut
    XFieldElement public_running_eval = 
        EvalArg::default_initial() * lookup_public_indeterminate + 
        XFieldElement(look_out);
    
    std::cout << "Computed: " << public_running_eval << std::endl;
    
    std::string rust_str = rust_row0[28].get<std::string>();
    XFieldElement rust_value = parse_xfield_from_string(rust_str);
    std::cout << "Expected: " << rust_str << std::endl;
    std::cout << "Match: " << (rust_value == public_running_eval ? "YES" : "NO") << std::endl;
}

// Test: Debug LookupTable row 256 (middle row) to see if computation matches
TEST_F(DebugArithmeticTest, LookupTableRow256) {
    std::cout << "\n=== Debug LookupTable Row 256 ===" << std::endl;
    
    auto pad_json = load_json("04_main_tables_pad.json");
    auto& padded_data = pad_json["padded_table_data"];
    auto aux_json = load_json("07_aux_tables_create.json");
    
    // Get middle rows
    if (!aux_json.contains("sample_rows_middle") || aux_json["sample_rows_middle"].size() == 0) {
        GTEST_SKIP() << "No middle rows in test data";
    }
    
    auto& rust_row256 = aux_json["sample_rows_middle"][0];
    size_t row256_idx = aux_json["sample_row_indices_middle"][0].get<size_t>();
    
    constexpr size_t LOOKUP_TABLE_START = 135;
    auto& row256 = padded_data[row256_idx];
    
    BFieldElement look_in(row256[LOOKUP_TABLE_START + 1].get<uint64_t>());
    BFieldElement look_out(row256[LOOKUP_TABLE_START + 2].get<uint64_t>());
    BFieldElement mult(row256[LOOKUP_TABLE_START + 3].get<uint64_t>());
    
    std::cout << "Main table row " << row256_idx << ":" << std::endl;
    std::cout << "  LookIn: " << look_in.value() << std::endl;
    std::cout << "  LookOut: " << look_out.value() << std::endl;
    std::cout << "  Multiplicity: " << mult.value() << std::endl;
    
    Challenges challenges = load_challenges();
    using namespace ChallengeId;
    
    // Compute step by step (same as row 0)
    XFieldElement cascade_indeterminate = challenges[CascadeLookupIndeterminate];
    XFieldElement lookup_input_weight = challenges[LookupTableInputWeight];
    XFieldElement lookup_output_weight = challenges[LookupTableOutputWeight];
    
    XFieldElement compressed_row = 
        XFieldElement(look_in) * lookup_input_weight +
        XFieldElement(look_out) * lookup_output_weight;
    XFieldElement diff = cascade_indeterminate - compressed_row;
    XFieldElement diff_inverse = diff.inverse();
    XFieldElement contribution = diff_inverse * mult;
    
    // For row 256, we need to accumulate from row 0 to row 256
    // This is complex, so let's just check if the extend function produces the right value
    // by loading the computed aux table
    
    std::string rust_str = rust_row256[27].get<std::string>();
    XFieldElement rust_value = parse_xfield_from_string(rust_str);
    
    std::cout << "\nExpected (Rust row " << row256_idx << "): " << rust_str << std::endl;
    std::cout << "Note: This requires accumulating from row 0 to row " << row256_idx << std::endl;
    std::cout << "The computation for a single row is correct, but we need to verify" << std::endl;
    std::cout << "the accumulation matches Rust's." << std::endl;
}

