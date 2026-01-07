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

/**
 * Parse XFieldElement from string
 */
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

class ExtendRow256Test : public ::testing::Test {
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

// Test: Compare row 256 LookupTable values
TEST_F(ExtendRow256Test, LookupTableRow256) {
    std::cout << "\n=== Row 256 LookupTable Comparison ===" << std::endl;
    
    // Load test data
    auto pad_json = load_json("04_main_tables_pad.json");
    auto& padded_data = pad_json["padded_table_data"];
    auto aux_json = load_json("07_aux_tables_create.json");
    
    // Get row 256
    size_t row256_idx = 256;
    auto& middle_indices = aux_json["sample_row_indices_middle"];
    auto idx = std::find(middle_indices.begin(), middle_indices.end(), row256_idx);
    if (idx == middle_indices.end()) {
        GTEST_SKIP() << "Row 256 not found in test data";
    }
    size_t array_idx = std::distance(middle_indices.begin(), idx);
    auto& rust_row256 = aux_json["sample_rows_middle"][array_idx];
    
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
    
    // Compare LookupTable columns (27-28)
    using namespace TableColumnOffsets;
    
    // Also check row 255 (last non-padding row) - should have same value as row 256
    std::cout << "\nRow 255 (last non-padding row):" << std::endl;
    XFieldElement cpp_val_255 = aux_table.get(255, AUX_LOOKUP_TABLE_START + 0);
    std::cout << "  C++ computed: " << cpp_val_255 << std::endl;
    
    std::cout << "\nRow 256 (first padding row, should match row 255):" << std::endl;
    XFieldElement cpp_val_256 = aux_table.get(row256_idx, AUX_LOOKUP_TABLE_START + 0);
    std::cout << "  C++ computed: " << cpp_val_256 << std::endl;
    std::cout << "  Rows 255 and 256 match: " << (cpp_val_255 == cpp_val_256 ? "YES ✓" : "NO ✗") << std::endl;
    
    std::cout << "\nLookupTable Col 27 (CascadeRunningSumLogDerivative):" << std::endl;
    std::string rust_str_27 = rust_row256[27].get<std::string>();
    XFieldElement rust_val_27 = parse_xfield_from_string(rust_str_27);
    XFieldElement cpp_val_27 = cpp_val_256;  // Use row 256 value
    
    std::cout << "  Rust: " << rust_str_27 << std::endl;
    std::cout << "  C++:  " << cpp_val_27 << std::endl;
    std::cout << "  Match: " << (rust_val_27 == cpp_val_27 ? "YES ✓" : "NO ✗") << std::endl;
    
    if (rust_val_27 != cpp_val_27) {
        std::cout << "  Detailed diff:" << std::endl;
        std::cout << "    Rust coeff0: " << rust_val_27.coeff(0).value() << std::endl;
        std::cout << "    C++  coeff0: " << cpp_val_27.coeff(0).value() << std::endl;
        std::cout << "    Rust coeff1: " << rust_val_27.coeff(1).value() << std::endl;
        std::cout << "    C++  coeff1: " << cpp_val_27.coeff(1).value() << std::endl;
        std::cout << "    Rust coeff2: " << rust_val_27.coeff(2).value() << std::endl;
        std::cout << "    C++  coeff2: " << cpp_val_27.coeff(2).value() << std::endl;
    }
    
    std::cout << "\nLookupTable Col 28 (PublicRunningEvaluation):" << std::endl;
    std::string rust_str_28 = rust_row256[28].get<std::string>();
    XFieldElement rust_val_28 = parse_xfield_from_string(rust_str_28);
    XFieldElement cpp_val_28 = aux_table.get(row256_idx, AUX_LOOKUP_TABLE_START + 1);
    
    std::cout << "  Rust: " << rust_str_28 << std::endl;
    std::cout << "  C++:  " << cpp_val_28 << std::endl;
    std::cout << "  Match: " << (rust_val_28 == cpp_val_28 ? "YES ✓" : "NO ✗") << std::endl;
    
    if (rust_val_28 != cpp_val_28) {
        std::cout << "  Detailed diff:" << std::endl;
        std::cout << "    Rust coeff0: " << rust_val_28.coeff(0).value() << std::endl;
        std::cout << "    C++  coeff0: " << cpp_val_28.coeff(0).value() << std::endl;
        std::cout << "    Rust coeff1: " << rust_val_28.coeff(1).value() << std::endl;
        std::cout << "    C++  coeff1: " << cpp_val_28.coeff(1).value() << std::endl;
        std::cout << "    Rust coeff2: " << rust_val_28.coeff(2).value() << std::endl;
        std::cout << "    C++  coeff2: " << cpp_val_28.coeff(2).value() << std::endl;
    }
}

