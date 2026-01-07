#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "table/master_table.hpp"
#include "table/extend_helpers.hpp"
#include "stark/challenges.hpp"
#include "types/x_field_element.hpp"
#include "types/b_field_element.hpp"
#include "types/digest.hpp"

using namespace triton_vm;
using json = nlohmann::json;

class LookupRowCountTest : public ::testing::Test {
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
            // Simple parsing - just extract numbers
            std::string s = str.get<std::string>();
            if (s == "0_xfe") {
                sampled.push_back(XFieldElement::zero());
            } else if (s == "1_xfe") {
                sampled.push_back(XFieldElement::one());
            } else {
                // Parse polynomial format
                size_t pos1 = s.find("(");
                size_t pos2 = s.find("·x²");
                size_t pos3 = s.find("·x");
                size_t pos4 = s.find(")");
                
                if (pos1 != std::string::npos && pos2 != std::string::npos && 
                    pos3 != std::string::npos && pos4 != std::string::npos) {
                    uint64_t coeff2 = std::stoull(s.substr(pos1 + 1, pos2 - pos1 - 1));
                    size_t pos_plus = s.find("+ ", pos2);
                    uint64_t coeff1 = std::stoull(s.substr(pos_plus + 2, pos3 - pos_plus - 2));
                    size_t pos_plus2 = s.find("+ ", pos3);
                    uint64_t coeff0 = std::stoull(s.substr(pos_plus2 + 2, pos4 - pos_plus2 - 2));
                    
                    sampled.push_back(XFieldElement(
                        BFieldElement(coeff0),
                        BFieldElement(coeff1),
                        BFieldElement(coeff2)
                    ));
                } else {
                    sampled.push_back(XFieldElement::zero());
                }
            }
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

// Test: Count how many rows contribute to LookupTable accumulation
TEST_F(LookupRowCountTest, CountContributingRows) {
    std::cout << "\n=== Counting LookupTable Contributing Rows ===" << std::endl;
    
    auto pad_json = load_json("04_main_tables_pad.json");
    auto& padded_data = pad_json["padded_table_data"];
    
    constexpr size_t LOOKUP_TABLE_START = 135;
    
    size_t total_rows = 0;
    size_t non_padding_rows = 0;
    size_t rows_with_mult = 0;
    size_t total_multiplicity = 0;
    
    for (size_t idx = 0; idx < padded_data.size(); idx++) {
        auto& row_json = padded_data[idx];
        
        uint64_t is_padding = row_json[LOOKUP_TABLE_START + 0].get<uint64_t>();
        uint64_t mult = row_json[LOOKUP_TABLE_START + 3].get<uint64_t>();
        
        total_rows++;
        if (is_padding == 0) {
            non_padding_rows++;
        }
        if (mult > 0) {
            rows_with_mult++;
            total_multiplicity += mult;
        }
        
        if (is_padding == 1) {
            std::cout << "First padding row found at index: " << idx << std::endl;
            break;
        }
    }
    
    std::cout << "\nRow statistics:" << std::endl;
    std::cout << "  Total rows checked: " << total_rows << std::endl;
    std::cout << "  Non-padding rows: " << non_padding_rows << std::endl;
    std::cout << "  Rows with non-zero multiplicity: " << rows_with_mult << std::endl;
    std::cout << "  Total multiplicity sum: " << total_multiplicity << std::endl;
    
    EXPECT_EQ(non_padding_rows, 256) << "Should have exactly 256 non-padding rows";
}

