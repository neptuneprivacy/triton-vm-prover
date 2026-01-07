#include <gtest/gtest.h>
#include "table/master_table.hpp"
#include "test_data_loader.hpp"
#include <filesystem>
#include <cmath>

using namespace triton_vm;

class TablesTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data_dir_ = TEST_DATA_DIR;
        
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found: " << test_data_dir_;
        }
    }
    
    std::string test_data_dir_;
};

// Test ArithmeticDomain creation
TEST_F(TablesTest, ArithmeticDomainOfLength) {
    // Test power-of-2 lengths
    EXPECT_NO_THROW({
        auto domain = ArithmeticDomain::of_length(512);
        EXPECT_EQ(domain.length, 512);
        EXPECT_FALSE(domain.generator.is_zero());
    });
    
    EXPECT_NO_THROW({
        auto domain = ArithmeticDomain::of_length(1024);
        EXPECT_EQ(domain.length, 1024);
    });
    
    // Test non-power-of-2 lengths should throw
    EXPECT_THROW(ArithmeticDomain::of_length(513), std::invalid_argument);
    EXPECT_THROW(ArithmeticDomain::of_length(100), std::invalid_argument);
}

// Test ArithmeticDomain halve
TEST_F(TablesTest, ArithmeticDomainHalve) {
    auto domain = ArithmeticDomain::of_length(1024);
    auto halved = domain.halve();
    
    EXPECT_EQ(halved.length, 512);
    
    // Test that halving an already-halved domain works
    auto halved_again = halved.halve();
    EXPECT_EQ(halved_again.length, 256);
    
    // Test that halving an odd-length domain throws
    ArithmeticDomain odd_domain;
    odd_domain.length = 513;
    odd_domain.offset = BFieldElement::one();
    odd_domain.generator = BFieldElement::generator();
    EXPECT_THROW(odd_domain.halve(), std::invalid_argument);
}

// Test ProverDomains derivation matches Rust
TEST_F(TablesTest, ProverDomainsDerive) {
    TestDataLoader loader(test_data_dir_);
    auto params = loader.load_parameters();
    
    // Create FRI domain
    ArithmeticDomain fri_domain = ArithmeticDomain::of_length(params.fri_domain_length)
        .with_offset(BFieldElement::generator());
    
    // Derive domains
    constexpr size_t num_trace_randomizers = 30; // Default from Rust
    int64_t max_degree = 2047; // Approximate for now
    
    auto domains = ProverDomains::derive(
        params.padded_height,
        num_trace_randomizers,
        fri_domain,
        max_degree
    );
    
    // Verify dimensions match Rust test data
    EXPECT_EQ(domains.trace.length, params.trace_domain_length);
    EXPECT_EQ(domains.randomized_trace.length, params.randomized_trace_domain_length);
    EXPECT_EQ(domains.fri.length, params.fri_domain_length);
}

// Test MasterMainTable creation
TEST_F(TablesTest, MasterMainTableCreation) {
    TestDataLoader loader(test_data_dir_);
    auto create_data = loader.load_main_table_create();
    
    size_t num_rows = create_data.trace_table_shape[0];
    size_t num_columns = create_data.trace_table_shape[1];
    
    MasterMainTable table(num_rows, num_columns);
    
    EXPECT_EQ(table.num_rows(), num_rows);
    EXPECT_EQ(table.num_columns(), num_columns);
}

// Test MasterMainTable padding
TEST_F(TablesTest, MasterMainTablePadding) {
    TestDataLoader loader(test_data_dir_);
    auto create_data = loader.load_main_table_create();
    auto pad_data = loader.load_step(4, "main_tables_pad");
    
    size_t initial_rows = create_data.trace_table_shape[0];
    size_t padded_rows = pad_data["trace_table_shape_after_pad"][0];
    size_t num_columns = create_data.trace_table_shape[1];
    
    MasterMainTable table(initial_rows, num_columns);
    
    // Pad to target height
    table.pad(padded_rows);
    
    EXPECT_EQ(table.num_rows(), padded_rows);
    EXPECT_EQ(table.num_columns(), num_columns);
    
    // Verify padded rows are initialized
    for (size_t i = initial_rows; i < padded_rows; ++i) {
        for (size_t j = 0; j < num_columns; ++j) {
            EXPECT_TRUE(table.get(i, j).is_zero());
        }
    }
}

// Test MasterMainTable data access
TEST_F(TablesTest, MasterMainTableDataAccess) {
    TestDataLoader loader(test_data_dir_);
    auto create_data = loader.load_main_table_create();
    
    size_t num_rows = create_data.trace_table_shape[0];
    size_t num_columns = create_data.num_columns;
    
    MasterMainTable table(num_rows, num_columns);
    
    // Set and get values
    BFieldElement value(12345);
    table.set(0, 0, value);
    EXPECT_EQ(table.get(0, 0), value);
    
    // Test bounds checking
    EXPECT_THROW(table.get(num_rows, 0), std::out_of_range);
    EXPECT_THROW(table.get(0, num_columns), std::out_of_range);
}

// Test MasterMainTable row access
TEST_F(TablesTest, MasterMainTableRowAccess) {
    TestDataLoader loader(test_data_dir_);
    auto create_data = loader.load_main_table_create();
    
    size_t num_rows = create_data.trace_table_shape[0];
    size_t num_columns = create_data.num_columns;
    
    MasterMainTable table(num_rows, num_columns);
    
    // Fill first row with test data
    for (size_t i = 0; i < num_columns && i < create_data.first_row.size(); ++i) {
        BFieldElement value(create_data.first_row[i]);
        table.set(0, i, value);
    }
    
    // Access row
    const auto& row = table.row(0);
    EXPECT_EQ(row.size(), num_columns);
    
    // Verify values match
    if (!create_data.first_row.empty() && num_columns > 0) {
        EXPECT_EQ(row[0], BFieldElement(create_data.first_row[0]));
    }
}

// Test MasterAuxTable creation
TEST_F(TablesTest, MasterAuxTableCreation) {
    size_t num_rows = 512;
    size_t num_columns = 50; // Approximate aux columns
    
    MasterAuxTable table(num_rows, num_columns);
    
    EXPECT_EQ(table.num_rows(), num_rows);
    EXPECT_EQ(table.num_columns(), num_columns);
}

// Test MasterAuxTable data access
TEST_F(TablesTest, MasterAuxTableDataAccess) {
    size_t num_rows = 512;
    size_t num_columns = 50;
    
    MasterAuxTable table(num_rows, num_columns);
    
    XFieldElement value(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    table.set(0, 0, value);
    
    EXPECT_EQ(table.get(0, 0), value);
}

// Verify table dimensions match Rust
TEST_F(TablesTest, TableDimensionsMatchRust) {
    EXPECT_EQ(TableDimensions::PROCESSOR_TABLE_WIDTH, 39);
    EXPECT_EQ(TableDimensions::OP_STACK_TABLE_WIDTH, 4);
    EXPECT_EQ(TableDimensions::RAM_TABLE_WIDTH, 7);
    EXPECT_EQ(TableDimensions::HASH_TABLE_WIDTH, 67);
    
    // Total should match test data: 379 columns
    TestDataLoader loader(test_data_dir_);
    auto create_data = loader.load_main_table_create();
    
    // Note: 379 includes degree-lowering columns, so it's more than just the sum
    // The sum of main columns is less than 379
    size_t sum_main_columns = 
        TableDimensions::PROCESSOR_TABLE_WIDTH +
        TableDimensions::OP_STACK_TABLE_WIDTH +
        TableDimensions::RAM_TABLE_WIDTH +
        TableDimensions::JUMP_STACK_TABLE_WIDTH +
        TableDimensions::HASH_TABLE_WIDTH +
        TableDimensions::CASCADE_TABLE_WIDTH +
        TableDimensions::LOOKUP_TABLE_WIDTH +
        TableDimensions::U32_TABLE_WIDTH +
        TableDimensions::PROGRAM_TABLE_WIDTH;
    
    EXPECT_LE(sum_main_columns, create_data.num_columns);
}

// Test MasterMainTable low-degree extension
TEST_F(TablesTest, MasterMainTableLowDegreeExtend) {
    // Create a small table for testing
    const size_t trace_length = 8;  // 2^3
    const size_t num_columns = 2;
    const size_t quotient_length = 32;  // 2^5 = 8 * 4

    // Create domains (quotient domain is 4x larger than trace domain)
    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_length);
    ArithmeticDomain quotient_domain = ArithmeticDomain::of_length(quotient_length);

    // Create table with domains
    MasterMainTable table(trace_length, num_columns, trace_domain, quotient_domain);

    // Fill with test data (simple polynomial: f(x) = x for first column, f(x) = x^2 for second)
    for (size_t row = 0; row < trace_length; row++) {
        BFieldElement x = trace_domain.generator.pow(row);
        table.set(row, 0, x);                    // First column: x
        table.set(row, 1, x * x);               // Second column: x^2
    }

    // Perform LDE to quotient domain
    table.low_degree_extend(quotient_domain);

    // Check that LDE was computed
    ASSERT_TRUE(table.has_lde());
    EXPECT_EQ(table.lde_table().size(), quotient_length);
    EXPECT_EQ(table.lde_table()[0].size(), num_columns);

    // Verify LDE preserves trace values at subgroup points
    for (size_t row = 0; row < trace_length; row++) {
        BFieldElement x = trace_domain.generator.pow(row);
        EXPECT_EQ(table.lde_table()[row * (quotient_length / trace_length)][0], x);
        EXPECT_EQ(table.lde_table()[row * (quotient_length / trace_length)][1], x * x);
    }
}

