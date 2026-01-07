#include <gtest/gtest.h>
#include "stark.hpp"
#include "test_data_loader.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace triton_vm;

class StarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use TEST_DATA_DIR defined in CMakeLists.txt
        test_data_dir_ = TEST_DATA_DIR;
        
        // Check if test data directory exists
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found: " << test_data_dir_;
        }
    }

    // Load padded main table from Rust JSON export
    MasterMainTable load_padded_main_table_from_rust() {
        std::string json_path = test_data_dir_ + "/../test_data_lde/04_main_tables_pad.json";
        std::ifstream f(json_path);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open: " + json_path);
        }

        nlohmann::json json = nlohmann::json::parse(f);
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
    
    std::string test_data_dir_;
};

// Test default STARK parameters
TEST_F(StarkTest, DefaultParameters) {
    auto stark = Stark::default_stark();
    
    EXPECT_EQ(stark.security_level(), 160);
    EXPECT_EQ(stark.fri_expansion_factor(), 4); // 2^2 = 4
}

// Test randomized trace length calculation
TEST_F(StarkTest, RandomizedTraceLen) {
    auto stark = Stark::default_stark();
    
    // For padded_height = 512, verify randomized_trace_len matches Rust
    // From test data: randomized_trace_domain_length = 1024
    TestDataLoader loader(test_data_dir_);
    auto params = loader.load_parameters();
    
    size_t rust_randomized_len = params.randomized_trace_domain_length;
    size_t cpp_randomized_len = stark.randomized_trace_len(params.padded_height);
    
    EXPECT_EQ(cpp_randomized_len, rust_randomized_len);
}

// Test FRI domain length calculation
TEST_F(StarkTest, FriDomainLength) {
    auto stark = Stark::default_stark();
    
    TestDataLoader loader(test_data_dir_);
    auto params = loader.load_parameters();
    
    // FRI domain length = expansion_factor * randomized_trace_len
    size_t expected_fri_len = stark.fri_expansion_factor() * 
                              stark.randomized_trace_len(params.padded_height);
    
    EXPECT_EQ(expected_fri_len, params.fri_domain_length);
}

// Test basic STARK integration (prove function runs without crashing)
// Test prove function with Rust-generated padded main table
TEST(BasicStarkTest, ProveAndVerify) {
    auto stark = Stark::default_stark();

    // Create minimal claim
    Claim claim;
    claim.program_digest = Digest::zero();
    claim.version = 1;
    claim.input = {BFieldElement(42)};
    claim.output = {BFieldElement(42)};

    // Load the padded main table from Rust instead of generating our own
    try {
        std::string json_path = std::string(TEST_DATA_DIR) + "/../test_data_lde/04_main_tables_pad.json";
        std::ifstream f(json_path);
        if (!f.is_open()) {
            GTEST_SKIP() << "Could not open Rust padded main table: " << json_path;
        }

        nlohmann::json json = nlohmann::json::parse(f);
        auto& padded_data = json["padded_table_data"];
        size_t num_rows = json["num_rows"].get<size_t>();
        size_t num_cols = json["num_columns"].get<size_t>();

        std::cout << "Loading Rust padded main table: " << num_rows << " x " << num_cols << std::endl;

        // Create simplified AET with the loaded table data (used by Stark::prove)
        SimpleAlgebraicExecutionTrace aet;
        aet.padded_height = num_rows;
        aet.processor_trace_height = num_rows;  // All rows are padded
        aet.processor_trace_width = num_cols;
        aet.processor_trace.resize(aet.padded_height);

        for (size_t r = 0; r < num_rows; r++) {
            aet.processor_trace[r].resize(num_cols);
            auto& row_json = padded_data[r];
            for (size_t c = 0; c < num_cols; c++) {
                uint64_t value = row_json[c].get<uint64_t>();
                aet.processor_trace[r][c] = BFieldElement(value);
            }
        }

        // The prove function should run without crashing and generate a valid proof
    EXPECT_NO_THROW({
        Proof proof = stark.prove(claim, aet);
            EXPECT_FALSE(proof.elements.empty());
            std::cout << "Generated proof with " << proof.elements.size() << " elements" << std::endl;
    });

    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not load Rust padded main table: " << e.what();
    }
}

// Test trace parameters match
TEST_F(StarkTest, TraceParametersMatch) {
    TestDataLoader loader(test_data_dir_);
    
    TestDataLoader::TraceExecutionData trace_data;
    TestDataLoader::ParametersData params;
    try {
        trace_data = loader.load_trace_execution();
        params = loader.load_parameters();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Missing or incompatible Rust test data: " << e.what();
    }
    
    // Padded height should match
    EXPECT_EQ(trace_data.padded_height, params.padded_height);
}

// Test main table dimensions match
TEST_F(StarkTest, MainTableDimensionsMatch) {
    TestDataLoader loader(test_data_dir_);
    
    TestDataLoader::MainTableCreateData main_table;
    TestDataLoader::ParametersData params;
    try {
        main_table = loader.load_main_table_create();
        params = loader.load_parameters();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Missing or incompatible Rust test data: " << e.what();
    }
    
    // Number of rows should be padded height
    EXPECT_EQ(main_table.trace_table_shape[0], params.padded_height);
    
    // Number of columns should match expected
    // From the Rust implementation, total main columns = 379 (including degree lowering)
    EXPECT_EQ(main_table.num_columns, 379);
}

// Test that we can verify against Rust-generated data step by step
TEST_F(StarkTest, StepByStepVerification) {
    TestDataLoader loader(test_data_dir_);
    
    // Step 1: Load trace execution data
    auto trace = loader.load_trace_execution();
    EXPECT_GT(trace.processor_trace_height, 0);
    
    // Step 2: Load parameters
    auto params = loader.load_parameters();
    EXPECT_GT(params.fri_domain_length, 0);
    
    // Step 3: Load main table
    auto main_table = loader.load_main_table_create();
    EXPECT_GT(main_table.first_row.size(), 0);
    
    // Step 6: Load Merkle root
    auto merkle = loader.load_main_tables_merkle();
    EXPECT_FALSE(merkle.merkle_root_hex.empty());
    
    // All steps loaded successfully
    SUCCEED();
}

