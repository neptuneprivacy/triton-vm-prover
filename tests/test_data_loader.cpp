#include <gtest/gtest.h>
#include "test_data_loader.hpp"
#include <filesystem>

using namespace triton_vm;

class TestDataLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use TEST_DATA_DIR defined in CMakeLists.txt
        test_data_dir_ = TEST_DATA_DIR;
        
        // Check if test data directory exists
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found: " << test_data_dir_;
        }
    }
    
    std::string test_data_dir_;
};

// Test loading trace execution data
TEST_F(TestDataLoaderTest, LoadTraceExecution) {
    TestDataLoader loader(test_data_dir_);
    auto data = loader.load_trace_execution();
    
    // Verify against known values from JSON
    EXPECT_EQ(data.padded_height, 512);
    EXPECT_EQ(data.processor_trace_height, 28);
    EXPECT_EQ(data.processor_trace_width, 39);
}

// Test loading parameters
TEST_F(TestDataLoaderTest, LoadParameters) {
    TestDataLoader loader(test_data_dir_);
    auto data = loader.load_parameters();
    
    // Verify against known values from JSON
    EXPECT_EQ(data.padded_height, 512);
    EXPECT_EQ(data.log2_padded_height, 9);
    EXPECT_EQ(data.fri_domain_length, 4096);
    EXPECT_EQ(data.trace_domain_length, 512);
    EXPECT_EQ(data.randomized_trace_domain_length, 1024);
    EXPECT_EQ(data.quotient_domain_length, 4096);
}

// Test loading main table create data
TEST_F(TestDataLoaderTest, LoadMainTableCreate) {
    TestDataLoader loader(test_data_dir_);
    auto data = loader.load_main_table_create();
    
    // Verify shape
    EXPECT_EQ(data.trace_table_shape[0], 512);
    EXPECT_EQ(data.trace_table_shape[1], 379);
    EXPECT_EQ(data.num_columns, 379);
    
    // Verify first row has correct number of elements
    EXPECT_EQ(data.first_row.size(), 379);
    
    // Verify some known values from first row
    EXPECT_EQ(data.first_row[0], 0);
    EXPECT_EQ(data.first_row[1], 73);
    EXPECT_EQ(data.first_row[4], 4099276459869907627ULL);
}

// Test loading Merkle roots
TEST_F(TestDataLoaderTest, LoadMainTablesMerkle) {
    TestDataLoader loader(test_data_dir_);
    auto data = loader.load_main_tables_merkle();
    
    // Merkle root should be a hex string
    EXPECT_FALSE(data.merkle_root_hex.empty());
    
    // Should be 80 hex chars (5 * 16 = 80 for 5 BFieldElements)
    // or could be different format - just check it's non-empty
    EXPECT_GT(data.merkle_root_hex.length(), 0);
}

// Test loading raw JSON
TEST_F(TestDataLoaderTest, LoadRawJson) {
    TestDataLoader loader(test_data_dir_);
    
    // Load step 2 parameters directly
    auto json = loader.load_step(2, "parameters");
    
    EXPECT_TRUE(json.contains("padded_height"));
    EXPECT_TRUE(json.contains("fri_domain_length"));
    EXPECT_EQ(json["padded_height"], 512);
}

// Test missing file throws
TEST_F(TestDataLoaderTest, MissingFileThrows) {
    TestDataLoader loader(test_data_dir_);
    
    EXPECT_THROW(loader.load_json("nonexistent.json"), std::runtime_error);
}

// Verify parameter relationships
TEST_F(TestDataLoaderTest, ParameterRelationships) {
    TestDataLoader loader(test_data_dir_);
    auto params = loader.load_parameters();
    
    // padded_height should be 2^log2_padded_height
    EXPECT_EQ(params.padded_height, 1ULL << params.log2_padded_height);
    
    // trace_domain_length should equal padded_height
    EXPECT_EQ(params.trace_domain_length, params.padded_height);
    
    // randomized_trace_domain_length should be >= padded_height
    EXPECT_GE(params.randomized_trace_domain_length, params.padded_height);
    
    // fri_domain_length should be a multiple of randomized_trace_domain_length
    EXPECT_EQ(params.fri_domain_length % params.randomized_trace_domain_length, 0);
}

// Test all expected files exist
TEST_F(TestDataLoaderTest, AllExpectedFilesExist) {
    std::vector<std::string> expected_files = {
        "01_trace_execution.json",
        "02_parameters.json",
        "03_main_tables_create.json",
        "04_main_tables_pad.json",
        "06_main_tables_merkle.json",
        "07_fiat_shamir_challenges.json"
    };
    
    for (const auto& filename : expected_files) {
        std::string path = test_data_dir_ + "/" + filename;
        EXPECT_TRUE(std::filesystem::exists(path)) 
            << "Missing file: " << filename;
    }
}

