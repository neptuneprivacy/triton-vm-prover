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
#include "stark/cross_table_arg.hpp"

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

class TraceLookupAccumulationTest : public ::testing::Test {
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

// Test: Manually trace accumulation for first few rows
TEST_F(TraceLookupAccumulationTest, TraceFirstFewRows) {
    std::cout << "\n=== Tracing LookupTable Accumulation (First 10 Rows) ===" << std::endl;
    
    auto pad_json = load_json("04_main_tables_pad.json");
    auto& padded_data = pad_json["padded_table_data"];
    auto aux_json = load_json("07_aux_tables_create.json");
    
    Challenges challenges = load_challenges();
    using namespace ChallengeId;
    using namespace TableColumnOffsets;
    
    constexpr size_t LOOKUP_TABLE_START = 135;
    
    XFieldElement cascade_running_sum = LookupArg::default_initial();
    XFieldElement cascade_lookup_indeterminate = challenges[CascadeLookupIndeterminate];
    XFieldElement lookup_input_weight = challenges[LookupTableInputWeight];
    XFieldElement lookup_output_weight = challenges[LookupTableOutputWeight];
    
    std::cout << "\nInitial values:" << std::endl;
    std::cout << "  cascade_running_sum: " << cascade_running_sum << std::endl;
    std::cout << "  cascade_lookup_indeterminate: " << cascade_lookup_indeterminate << std::endl;
    std::cout << "  lookup_input_weight: " << lookup_input_weight << std::endl;
    std::cout << "  lookup_output_weight: " << lookup_output_weight << std::endl;
    
    // Trace first 10 rows
    // LookupTable columns: IsPadding=0, LookIn=1, LookOut=2, LookupMultiplicity=3
    for (size_t idx = 0; idx < 10 && idx < padded_data.size(); idx++) {
        auto& row_json = padded_data[idx];
        
        BFieldElement look_in(row_json[LOOKUP_TABLE_START + 1].get<uint64_t>());
        BFieldElement look_out(row_json[LOOKUP_TABLE_START + 2].get<uint64_t>());
        BFieldElement mult(row_json[LOOKUP_TABLE_START + 3].get<uint64_t>());
        
        std::cout << "\nRow " << idx << ":" << std::endl;
        std::cout << "  LookIn: " << look_in.value() << std::endl;
        std::cout << "  LookOut: " << look_out.value() << std::endl;
        std::cout << "  Multiplicity: " << mult.value() << std::endl;
        
        // Compute compressed_row
        XFieldElement compressed_row = 
            XFieldElement(look_in) * lookup_input_weight +
            XFieldElement(look_out) * lookup_output_weight;
        std::cout << "  compressed_row: " << compressed_row << std::endl;
        
        // Compute diff
        XFieldElement diff = cascade_lookup_indeterminate - compressed_row;
        std::cout << "  diff: " << diff << std::endl;
        
        // Compute inverse
        XFieldElement diff_inverse = diff.inverse();
        std::cout << "  diff_inverse: " << diff_inverse << std::endl;
        
        // Compute contribution
        XFieldElement contribution = diff_inverse * mult;
        std::cout << "  contribution: " << contribution << std::endl;
        
        // Accumulate
        XFieldElement old_sum = cascade_running_sum;
        cascade_running_sum = cascade_running_sum + contribution;
        std::cout << "  old_sum: " << old_sum << std::endl;
        std::cout << "  new_sum: " << cascade_running_sum << std::endl;
        
        // Get Rust expected value
        if (idx < aux_json["sample_rows_first"].size()) {
            auto& rust_row = aux_json["sample_rows_first"][idx];
            std::string rust_str = rust_row[27].get<std::string>();
            XFieldElement rust_val = parse_xfield_from_string(rust_str);
            std::cout << "  Rust expected: " << rust_str << std::endl;
            std::cout << "  Match: " << (rust_val == cascade_running_sum ? "YES ✓" : "NO ✗") << std::endl;
            
            if (rust_val != cascade_running_sum) {
                std::cout << "  Mismatch details:" << std::endl;
                std::cout << "    Rust coeff0: " << rust_val.coeff(0).value() << std::endl;
                std::cout << "    C++  coeff0: " << cascade_running_sum.coeff(0).value() << std::endl;
                std::cout << "    Diff coeff0: " << (rust_val.coeff(0).value() > cascade_running_sum.coeff(0).value() ? 
                    rust_val.coeff(0).value() - cascade_running_sum.coeff(0).value() :
                    cascade_running_sum.coeff(0).value() - rust_val.coeff(0).value()) << std::endl;
            }
        }
    }
}

