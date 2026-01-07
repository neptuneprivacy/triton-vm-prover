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
#include "types/digest.hpp"

using namespace triton_vm;
using json = nlohmann::json;

static XFieldElement parse_xfield_from_string(const std::string& str) {
    if (str == "0_xfe") return XFieldElement::zero();
    if (str == "1_xfe") return XFieldElement::one();
    
    std::regex single_value_pattern(R"((\d+)_xfe)");
    std::smatch single_match;
    if (std::regex_search(str, single_match, single_value_pattern)) {
        uint64_t value = std::stoull(single_match[1].str());
        return XFieldElement(BFieldElement(value), BFieldElement::zero(), BFieldElement::zero());
    }
    
    std::regex polynomial_pattern(R"(\((\d+)·x² \+ (\d+)·x \+ (\d+)\))");
    std::smatch poly_match;
    if (std::regex_search(str, poly_match, polynomial_pattern)) {
        uint64_t coeff2 = std::stoull(poly_match[1].str());
        uint64_t coeff1 = std::stoull(poly_match[2].str());
        uint64_t coeff0 = std::stoull(poly_match[3].str());
        return XFieldElement(BFieldElement(coeff0), BFieldElement(coeff1), BFieldElement(coeff2));
    }
    
    throw std::runtime_error("Failed to parse XFieldElement: " + str);
}

class HashTableDebugTest : public ::testing::Test {
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
        
        auto claim_json = load_json("06_claim.json");
        Digest program_digest = Digest::from_hex(claim_json["program_digest"].get<std::string>());
        std::vector<BFieldElement> input, output, lookup_table;
        for (const auto& val : claim_json["input"]) {
            input.push_back(BFieldElement(val.get<uint64_t>()));
        }
        for (const auto& val : claim_json["output"]) {
            output.push_back(BFieldElement(val.get<uint64_t>()));
        }
        
        challenges.compute_derived_challenges(
            program_digest.to_b_field_elements(),
            input,
            output,
            lookup_table
        );
        
        return challenges;
    }
};

// Test: Compare HashTable row 0 and row 256
TEST_F(HashTableDebugTest, HashTableRowComparison) {
    std::cout << "\n=== HashTable Comparison ===" << std::endl;
    
    auto pad_json = load_json("04_main_tables_pad.json");
    auto& padded_data = pad_json["padded_table_data"];
    auto aux_json = load_json("07_aux_tables_create.json");
    
    // Load main table
    size_t num_rows = pad_json["num_rows"].get<size_t>();
    size_t num_cols = pad_json["num_columns"].get<size_t>();
    MasterMainTable main_table(num_rows, num_cols);
    for (size_t r = 0; r < num_rows; r++) {
        auto& row_json = padded_data[r];
        for (size_t c = 0; c < num_cols; c++) {
            uint64_t value = row_json[c].get<uint64_t>();
            main_table.set(r, c, BFieldElement(value));
        }
    }
    
    // Compute extend
    Challenges challenges = load_challenges();
    MasterAuxTable aux_table = main_table.extend(challenges);
    
    using namespace TableColumnOffsets;
    
    // Debug: Check which rows trigger the condition
    std::cout << "\n=== Debugging HashTable Computation ===" << std::endl;
    
    // Check main table values for first few rows
    using namespace TableColumnOffsets;
    const size_t HASH_TABLE_START = 62;
    std::cout << "\nFirst 10 rows - Mode and RoundNumber values:" << std::endl;
    for (size_t r = 0; r < std::min(10UL, num_rows); r++) {
        BFieldElement mode_val = main_table.get(r, HASH_TABLE_START + 0);
        BFieldElement round_val = main_table.get(r, HASH_TABLE_START + 2);
        bool is_program_hashing = (mode_val.value() == 1);
        bool is_round_0 = round_val.is_zero();
        std::cout << "  Row " << r << ": mode=" << mode_val.value() 
                  << ", round=" << round_val.value()
                  << ", trigger=" << (is_program_hashing && is_round_0 ? "YES" : "NO") << std::endl;
    }
    
    // Check AUX_HASH_TABLE_START value
    std::cout << "\nAUX_HASH_TABLE_START = " << AUX_HASH_TABLE_START << std::endl;
    
    // Compare row 0
    std::cout << "\nRow 0 HashTable:" << std::endl;
    auto& rust_row0 = aux_json["sample_rows_first"][0];
    std::string rust_str_0 = rust_row0[24].get<std::string>();
    XFieldElement rust_val_0 = parse_xfield_from_string(rust_str_0);
    XFieldElement cpp_val_0 = aux_table.get(0, AUX_HASH_TABLE_START + 0);
    
    std::cout << "  Rust: " << rust_str_0 << std::endl;
    std::cout << "  C++:  " << cpp_val_0 << std::endl;
    std::cout << "  Match: " << (rust_val_0 == cpp_val_0 ? "YES ✓" : "NO ✗") << std::endl;
    
    // Compare row 256
    std::cout << "\nRow 256 HashTable:" << std::endl;
    auto& middle_indices = aux_json["sample_row_indices_middle"];
    auto idx = std::find(middle_indices.begin(), middle_indices.end(), 256);
    if (idx != middle_indices.end()) {
        size_t array_idx = std::distance(middle_indices.begin(), idx);
        auto& rust_row256 = aux_json["sample_rows_middle"][array_idx];
        std::string rust_str_256 = rust_row256[24].get<std::string>();
        XFieldElement rust_val_256 = parse_xfield_from_string(rust_str_256);
        XFieldElement cpp_val_256 = aux_table.get(256, AUX_HASH_TABLE_START + 0);
        
        std::cout << "  Rust: " << rust_str_256 << std::endl;
        std::cout << "  C++:  " << cpp_val_256 << std::endl;
        std::cout << "  Match: " << (rust_val_256 == cpp_val_256 ? "YES ✓" : "NO ✗") << std::endl;
        
        if (rust_val_256 != cpp_val_256) {
            std::cout << "  Detailed diff:" << std::endl;
            std::cout << "    Rust coeff0: " << rust_val_256.coeff(0).value() << std::endl;
            std::cout << "    C++  coeff0: " << cpp_val_256.coeff(0).value() << std::endl;
            std::cout << "    Rust coeff1: " << rust_val_256.coeff(1).value() << std::endl;
            std::cout << "    C++  coeff1: " << cpp_val_256.coeff(1).value() << std::endl;
            std::cout << "    Rust coeff2: " << rust_val_256.coeff(2).value() << std::endl;
            std::cout << "    C++  coeff2: " << cpp_val_256.coeff(2).value() << std::endl;
        }
    }
}

