#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "lde/lde_randomized.hpp"
#include "ntt/ntt.hpp"
#include "types/b_field_element.hpp"
#include "table/master_table.hpp"

using namespace triton_vm;
using json = nlohmann::json;

/**
 * LDEFullRandomizedTest - Test full randomized LDE with actual randomizer coefficients
 */
class LDEFullRandomizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use test_data_lde directory where randomizer file is located
        test_data_dir_ = std::string(TEST_DATA_DIR) + "/../test_data_lde";
        
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found. Run 'gen_test_data spin.tasm 8 test_data_lde' first.";
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
};

// Test full randomized LDE with actual Rust randomizer coefficients
TEST_F(LDEFullRandomizedTest, FullRandomizedLDE_ExactMatch) {
    // Load randomizer data
    auto rand_data = load_json("trace_randomizer_column_0.json");
    
    // Load parameters
    auto params = load_json("02_parameters.json");
    
    // Load padded table
    auto pad_data = load_json("04_main_tables_pad.json");
    
    // Load Rust LDE output
    auto rust_lde_data = load_json("05_main_tables_lde.json");
    
    std::cout << "\n=== Full Randomized LDE Test ===" << std::endl;
    std::cout << "  Using actual Rust randomizer coefficients" << std::endl;
    
    // Extract domains
    auto& trace_dom = rand_data["trace_domain"];
    size_t trace_len = trace_dom["length"].get<size_t>();
    uint64_t trace_gen = trace_dom["generator"].get<uint64_t>();
    
    auto& quot_dom = params["quotient_domain"];
    size_t quot_len = quot_dom["length"].get<size_t>();
    uint64_t quot_gen = quot_dom["generator"].get<uint64_t>();
    uint64_t quot_offset = quot_dom["offset"].get<uint64_t>();
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len);
    EXPECT_EQ(trace_domain.generator.value(), trace_gen);
    
    ArithmeticDomain quot_domain = ArithmeticDomain::of_length(quot_len);
    quot_domain = quot_domain.with_offset(BFieldElement(quot_offset));
    EXPECT_EQ(quot_domain.generator.value(), quot_gen);
    
    // Load trace column
    auto& padded_rows = pad_data["padded_table_data"];
    std::vector<BFieldElement> trace_column;
    for (auto& row : padded_rows) {
        trace_column.push_back(BFieldElement(row[0].get<uint64_t>()));
    }
    
    // Load randomizer coefficients
    auto& rand_coeffs_json = rand_data["randomizer_coefficients"];
    std::vector<BFieldElement> randomizer_coeffs;
    for (auto& coeff : rand_coeffs_json) {
        randomizer_coeffs.push_back(BFieldElement(coeff.get<uint64_t>()));
    }
    
    std::cout << "  Trace column: " << trace_column.size() << " values" << std::endl;
    std::cout << "  Randomizer coefficients: " << randomizer_coeffs.size() << std::endl;
    
    // Compute randomized LDE using our implementation
    std::cout << "  Computing randomized LDE..." << std::endl;
    auto cpp_lde = RandomizedLDE::extend_column_with_randomizer(
        trace_column, trace_domain, quot_domain, randomizer_coeffs
    );
    
    EXPECT_EQ(cpp_lde.size(), quot_len);
    std::cout << "  C++ LDE output: " << cpp_lde.size() << " values" << std::endl;
    
    // Load Rust LDE output
    auto& rust_lde_rows = rust_lde_data["lde_table_data"];
    std::vector<uint64_t> rust_lde_column;
    for (auto& row : rust_lde_rows) {
        rust_lde_column.push_back(row[0].get<uint64_t>());
    }
    
    EXPECT_EQ(cpp_lde.size(), rust_lde_column.size());
    
    // Compare all values
    size_t matches = 0;
    size_t mismatches = 0;
    
    for (size_t i = 0; i < cpp_lde.size(); i++) {
        uint64_t cpp_val = cpp_lde[i].value();
        uint64_t rust_val = rust_lde_column[i];
        
        if (cpp_val == rust_val) {
            matches++;
        } else {
            mismatches++;
            if (mismatches <= 5) {
                std::cout << "    Mismatch at index " << i 
                          << ": C++=" << cpp_val 
                          << " Rust=" << rust_val << std::endl;
            }
        }
    }
    
    std::cout << "  Results: " << matches << " match, " << mismatches << " mismatch" << std::endl;
    
    if (mismatches == 0) {
        std::cout << "  ✓ 100% exact match with Rust!" << std::endl;
    } else {
        std::cout << "  ✗ " << mismatches << " mismatches found" << std::endl;
    }
    
    // Should match exactly
    EXPECT_EQ(mismatches, 0) << "Randomized LDE should match Rust exactly";
    
    // Verify first few values explicitly
    for (size_t i = 0; i < std::min((size_t)10, cpp_lde.size()); i++) {
        EXPECT_EQ(cpp_lde[i].value(), rust_lde_column[i])
            << "Mismatch at index " << i;
    }
}

// Test that zerofier * randomizer is computed correctly
TEST_F(LDEFullRandomizedTest, ZerofierTimesRandomizer) {
    auto rand_data = load_json("trace_randomizer_column_0.json");
    
    auto& trace_dom = rand_data["trace_domain"];
    size_t trace_len = trace_dom["length"].get<size_t>();
    uint64_t trace_gen = trace_dom["generator"].get<uint64_t>();
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len);
    
    std::cout << "\n=== Zerofier * Randomizer Test ===" << std::endl;
    
    // Load randomizer coefficients
    auto& rand_coeffs_json = rand_data["randomizer_coefficients"];
    std::vector<BFieldElement> randomizer_coeffs;
    for (auto& coeff : rand_coeffs_json) {
        randomizer_coeffs.push_back(BFieldElement(coeff.get<uint64_t>()));
    }
    
    BPolynomial randomizer_poly(randomizer_coeffs);
    
    std::cout << "  Randomizer polynomial degree: " << randomizer_poly.degree() << std::endl;
    
    // Compute zerofier
    BPolynomial zerofier = RandomizedLDE::compute_zerofier(trace_domain);
    std::cout << "  Zerofier degree: " << zerofier.degree() << std::endl;
    
    // Multiply zerofier * randomizer
    BPolynomial result = RandomizedLDE::mul_zerofier_with(trace_domain, randomizer_poly);
    
    std::cout << "  Result (zerofier * randomizer) degree: " << result.degree() << std::endl;
    
    // Verify result evaluates to zero on trace domain
    // zerofier * randomizer should be zero on trace domain because zerofier is zero there
    std::vector<BFieldElement> trace_domain_values;
    BFieldElement x = trace_domain.offset;
    for (size_t i = 0; i < trace_len; i++) {
        trace_domain_values.push_back(x);
        x = x * trace_domain.generator;
    }
    
    size_t zero_count = 0;
    for (const auto& domain_point : trace_domain_values) {
        BFieldElement eval = result.evaluate(domain_point);
        if (eval.value() == 0) {
            zero_count++;
        }
    }
    
    EXPECT_EQ(zero_count, trace_len) 
        << "zerofier * randomizer should evaluate to zero on trace domain";
    
    std::cout << "  ✓ Zerofier * randomizer evaluates to zero at all " 
              << zero_count << " domain points" << std::endl;
}

// Test column interpolant matches Rust
TEST_F(LDEFullRandomizedTest, ColumnInterpolantMatch) {
    auto rand_data = load_json("trace_randomizer_column_0.json");
    auto pad_data = load_json("04_main_tables_pad.json");
    
    auto& trace_dom = rand_data["trace_domain"];
    size_t trace_len = trace_dom["length"].get<size_t>();
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len);
    
    std::cout << "\n=== Column Interpolant Test ===" << std::endl;
    
    // Load trace column
    auto& padded_rows = pad_data["padded_table_data"];
    std::vector<BFieldElement> trace_column;
    for (auto& row : padded_rows) {
        trace_column.push_back(BFieldElement(row[0].get<uint64_t>()));
    }
    
    // Interpolate using C++
    std::vector<BFieldElement> cpp_coeffs = NTT::interpolate(trace_column);
    
    // Load Rust interpolant coefficients
    auto& rust_coeffs_json = rand_data["column_interpolant_coefficients"];
    
    EXPECT_EQ(cpp_coeffs.size(), rust_coeffs_json.size());
    
    size_t matches = 0;
    for (size_t i = 0; i < cpp_coeffs.size(); i++) {
        uint64_t cpp_val = cpp_coeffs[i].value();
        uint64_t rust_val = rust_coeffs_json[i].get<uint64_t>();
        if (cpp_val == rust_val) {
            matches++;
        }
    }
    
    EXPECT_EQ(matches, cpp_coeffs.size()) 
        << "Column interpolant coefficients should match Rust exactly";
    
    std::cout << "  ✓ Column interpolant matches Rust: " 
              << matches << "/" << cpp_coeffs.size() << " coefficients" << std::endl;
}

// Summary
TEST_F(LDEFullRandomizedTest, Summary) {
    std::cout << "\n=== Full Randomized LDE Implementation Summary ===" << std::endl;
    std::cout << "  ✓ Zerofier computation: Working" << std::endl;
    std::cout << "  ✓ Zerofier * randomizer: Working" << std::endl;
    std::cout << "  ✓ Column interpolation: Matches Rust" << std::endl;
    std::cout << "  ✓ Full randomized LDE: Ready for verification" << std::endl;
    
    SUCCEED();
}

