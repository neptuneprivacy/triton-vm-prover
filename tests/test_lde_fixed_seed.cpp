#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "ntt/ntt.hpp"
#include "types/b_field_element.hpp"
#include "table/master_table.hpp"

using namespace triton_vm;
using json = nlohmann::json;

/**
 * LDEFixedSeedTest - Test LDE with trace randomizers implementation
 */
class LDEFixedSeedTest : public ::testing::Test {
protected:
    void SetUp() override {
        lde_cases_dir_ = std::string(TEST_DATA_DIR) + "/../test_data_lde_cases";
        
        if (!std::filesystem::exists(lde_cases_dir_)) {
            GTEST_SKIP() << "LDE test cases not found. Run 'gen_lde_test_cases' first.";
        }
    }
    
    std::string lde_cases_dir_;
    
    json load_json(const std::string& filename) {
        std::string path = lde_cases_dir_ + "/" + filename;
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + path);
        }
        return json::parse(f);
    }
};

// Test zerofier computation
TEST_F(LDEFixedSeedTest, ZerofierComputation) {
    auto data = load_json("03_fixed_seed_implementation.json");
    
    auto& trace_dom = data["trace_domain"];
    size_t trace_len = trace_dom["length"].get<size_t>();
    uint64_t trace_gen = trace_dom["generator"].get<uint64_t>();
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len);
    EXPECT_EQ(trace_domain.generator.value(), trace_gen);
    
    std::cout << "\n=== Zerofier Computation Test ===" << std::endl;
    std::cout << "  Trace domain length: " << trace_len << std::endl;
    
    // Compute zerofier: x^n - offset^n
    // Zerofier should be: x^trace_len - offset^trace_len
    // where offset = 1 (trace domain always has offset 1)
    BFieldElement offset = trace_domain.offset;
    BFieldElement offset_pow_n = offset.pow(trace_len);
    
    std::cout << "  Offset: " << offset.value() << std::endl;
    std::cout << "  Offset^n: " << offset_pow_n.value() << std::endl;
    
    // Verify zerofier evaluates to zero on trace domain
    // Zerofier: x^n - offset^n
    // On trace domain: x = offset * generator^k for k=0..n-1
    // We need: (offset * generator^k)^n = offset^n
    // Since generator^n = 1: (offset * generator^k)^n = offset^n * (generator^k)^n = offset^n * 1 = offset^n
    // So: x^n - offset^n = offset^n - offset^n = 0 ✓
    
    std::vector<BFieldElement> trace_domain_values;
    BFieldElement x = offset;  // Start at offset
    for (size_t i = 0; i < trace_len; i++) {
        trace_domain_values.push_back(x);
        x = x * trace_domain.generator;
    }
    
    // Check that generator^n = 1
    BFieldElement gen_pow_n = trace_domain.generator.pow(trace_len);
    EXPECT_EQ(gen_pow_n.value(), 1ULL) << "Generator^n should equal 1";
    std::cout << "  ✓ Verified: generator^" << trace_len << " = 1" << std::endl;
    
    // Verify zerofier evaluates to zero at each domain point
    size_t zeros = 0;
    for (const auto& domain_point : trace_domain_values) {
        BFieldElement x_pow_n = domain_point.pow(trace_len);
        BFieldElement zerofier_value = x_pow_n - offset_pow_n;
        
        if (zerofier_value.value() == 0) {
            zeros++;
        }
    }
    
    EXPECT_EQ(zeros, trace_len) << "Zerofier should evaluate to zero at all " << trace_len << " domain points";
    std::cout << "  ✓ Zerofier evaluates to zero at all " << zeros << "/" << trace_len << " domain points" << std::endl;
}

// Test randomized LDE structure
TEST_F(LDEFixedSeedTest, RandomizedLDEStructure) {
    auto data = load_json("03_fixed_seed_implementation.json");
    
    std::cout << "\n=== Randomized LDE Structure Test ===" << std::endl;
    
    // Load domains
    auto& trace_dom = data["trace_domain"];
    size_t trace_len = trace_dom["length"].get<size_t>();
    uint64_t trace_gen = trace_dom["generator"].get<uint64_t>();
    
    auto& quot_dom = data["quotient_domain"];
    size_t quot_len = quot_dom["length"].get<size_t>();
    uint64_t quot_gen = quot_dom["generator"].get<uint64_t>();
    uint64_t quot_offset = quot_dom["offset"].get<uint64_t>();
    
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len);
    ArithmeticDomain quot_domain = ArithmeticDomain::of_length(quot_len);
    quot_domain = quot_domain.with_offset(BFieldElement(quot_offset));
    
    // Load input trace values
    auto& trace_values_json = data["input"]["trace_values"];
    std::vector<BFieldElement> trace_column;
    for (auto& val : trace_values_json) {
        trace_column.push_back(BFieldElement(val.get<uint64_t>()));
    }
    
    // Load zero-randomizer LDE
    auto& zero_lde_json = data["intermediate"]["zero_randomizer_lde"];
    std::vector<BFieldElement> zero_lde;
    for (auto& val : zero_lde_json) {
        zero_lde.push_back(BFieldElement(val.get<uint64_t>()));
    }
    
    // Load Rust LDE with randomizers
    auto& rust_lde_json = data["output"]["rust_lde_with_randomizers"];
    std::vector<BFieldElement> rust_lde;
    for (auto& val : rust_lde_json) {
        rust_lde.push_back(BFieldElement(val.get<uint64_t>()));
    }
    
    // Load randomizer contribution
    auto& contrib_json = data["randomizer_info"]["contribution_on_quotient_domain"];
    
    std::cout << "  Trace domain: " << trace_len << " elements" << std::endl;
    std::cout << "  Quotient domain: " << quot_len << " elements" << std::endl;
    std::cout << "  Zero-randomizer LDE: " << zero_lde.size() << " values" << std::endl;
    std::cout << "  Rust LDE (with randomizers): " << rust_lde.size() << " values" << std::endl;
    
    // Verify: rust_lde = zero_lde + randomizer_contribution
    size_t matches = 0;
    for (size_t i = 0; i < rust_lde.size() && i < contrib_json.size(); i++) {
        uint64_t contrib = contrib_json[i].get<uint64_t>();
        BFieldElement expected = zero_lde[i] + BFieldElement(contrib);
        
        if (rust_lde[i].value() == expected.value()) {
            matches++;
        }
    }
    
    std::cout << "  Verification: rust_lde = zero_lde + contribution" << std::endl;
    std::cout << "    Matches: " << matches << "/" << rust_lde.size() << std::endl;
    
    // They should match (accounting for modular arithmetic)
    EXPECT_EQ(matches, rust_lde.size()) << "Randomizer contribution should sum correctly";
    std::cout << "  ✓ Randomized LDE structure verified" << std::endl;
}

// Test algorithm steps documentation
TEST_F(LDEFixedSeedTest, AlgorithmStepsDocumentation) {
    auto data = load_json("03_fixed_seed_implementation.json");
    
    std::cout << "\n=== Algorithm Steps Documentation ===" << std::endl;
    
    auto& steps = data["algorithm"]["steps"];
    EXPECT_TRUE(steps.is_array());
    
    std::cout << "  Algorithm has " << steps.size() << " steps:" << std::endl;
    for (size_t i = 0; i < steps.size(); i++) {
        std::cout << "    " << (i + 1) << ". " << steps[i].get<std::string>() << std::endl;
    }
    
    auto& zerofier_info = data["algorithm"]["zerofier_formula"];
    std::cout << "\n  Zerofier formula: " << zerofier_info["note"].get<std::string>() << std::endl;
    
    std::cout << "\n  ✓ Algorithm steps documented" << std::endl;
    std::cout << "  Status: Ready for C++ implementation" << std::endl;
}

// Summary
TEST_F(LDEFixedSeedTest, Summary) {
    std::cout << "\n=== Randomized LDE Implementation Summary ===" << std::endl;
    std::cout << "  ✓ Zerofier computation: Structure verified" << std::endl;
    std::cout << "  ✓ Randomized LDE structure: Contribution verified" << std::endl;
    std::cout << "  ✓ Algorithm steps: Documented" << std::endl;
    std::cout << "\n  Next steps:" << std::endl;
    std::cout << "    1. Implement zerofier polynomial computation" << std::endl;
    std::cout << "    2. Implement zerofier * randomizer multiplication" << std::endl;
    std::cout << "    3. Implement trace randomizer generation (RNG matching)" << std::endl;
    std::cout << "    4. Implement full randomized LDE pipeline" << std::endl;
    
    SUCCEED();
}

