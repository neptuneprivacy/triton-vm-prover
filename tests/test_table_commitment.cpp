#include <gtest/gtest.h>
#include "table/table_commitment.hpp"
#include "test_data_loader.hpp"
#include <filesystem>

using namespace triton_vm;

class TableCommitmentTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data_dir_ = TEST_DATA_DIR;
        
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found: " << test_data_dir_;
        }
    }
    
    std::string test_data_dir_;
};

// Test committing to a small main table
TEST_F(TableCommitmentTest, CommitMainTable) {
    // Create a small test table
    MasterMainTable table(8, 10);  // 8 rows (power of 2), 10 columns
    
    // Fill with some data
    for (size_t r = 0; r < table.num_rows(); ++r) {
        for (size_t c = 0; c < table.num_columns(); ++c) {
            table.set(r, c, BFieldElement(r * 100 + c));
        }
    }
    
    // Commit
    auto commitment = TableCommitment::commit(table);
    
    // Root should be non-zero
    Digest root = commitment.root();
    bool all_zero = true;
    for (size_t i = 0; i < Digest::LEN; ++i) {
        if (!root[i].is_zero()) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
    
    // Number of rows should match
    EXPECT_EQ(commitment.num_rows(), 8);
}

// Test committing to aux table
TEST_F(TableCommitmentTest, CommitAuxTable) {
    // Create a small aux table
    MasterAuxTable table(4, 5);  // 4 rows, 5 columns
    
    // Fill with some data
    for (size_t r = 0; r < table.num_rows(); ++r) {
        for (size_t c = 0; c < table.num_columns(); ++c) {
            table.set(r, c, XFieldElement(
                BFieldElement(r * 100 + c),
                BFieldElement(r + c),
                BFieldElement(r * c)
            ));
        }
    }
    
    // Commit
    auto commitment = TableCommitment::commit(table);
    
    // Root should be non-zero
    Digest root = commitment.root();
    bool all_zero = true;
    for (size_t i = 0; i < Digest::LEN; ++i) {
        if (!root[i].is_zero()) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
    
    EXPECT_EQ(commitment.num_rows(), 4);
}

// Test commitment is deterministic
TEST_F(TableCommitmentTest, CommitmentDeterministic) {
    MasterMainTable table(4, 3);
    for (size_t r = 0; r < table.num_rows(); ++r) {
        for (size_t c = 0; c < table.num_columns(); ++c) {
            table.set(r, c, BFieldElement(r * 10 + c + 42));
        }
    }
    
    auto commitment1 = TableCommitment::commit(table);
    auto commitment2 = TableCommitment::commit(table);
    
    EXPECT_EQ(commitment1.root(), commitment2.root());
}

// Test authentication path and verification
TEST_F(TableCommitmentTest, AuthenticationPathVerification) {
    MasterMainTable table(8, 5);
    for (size_t r = 0; r < table.num_rows(); ++r) {
        for (size_t c = 0; c < table.num_columns(); ++c) {
            table.set(r, c, BFieldElement(r * 100 + c));
        }
    }
    
    auto commitment = TableCommitment::commit(table);
    Digest root = commitment.root();
    
    // Verify each row
    for (size_t r = 0; r < table.num_rows(); ++r) {
        Digest row_hash = hash_bfield_row(table.row(r));
        auto auth_path = commitment.authentication_path(r);
        
        bool valid = TableCommitment::verify(root, r, row_hash, auth_path);
        EXPECT_TRUE(valid) << "Verification failed for row " << r;
    }
}

// Test verification fails with wrong data
TEST_F(TableCommitmentTest, VerificationFailsWithWrongData) {
    MasterMainTable table(4, 3);
    for (size_t r = 0; r < table.num_rows(); ++r) {
        for (size_t c = 0; c < table.num_columns(); ++c) {
            table.set(r, c, BFieldElement(r * 10 + c));
        }
    }
    
    auto commitment = TableCommitment::commit(table);
    Digest root = commitment.root();
    auto auth_path = commitment.authentication_path(0);
    
    // Create a wrong row hash
    Digest wrong_hash(BFieldElement(999), BFieldElement(999), BFieldElement(999),
                     BFieldElement(999), BFieldElement(999));
    
    bool valid = TableCommitment::verify(root, 0, wrong_hash, auth_path);
    EXPECT_FALSE(valid);
}

// Test from_digests
TEST_F(TableCommitmentTest, FromDigests) {
    std::vector<Digest> digests;
    for (size_t i = 0; i < 8; ++i) {
        digests.push_back(Digest(
            BFieldElement(i * 5), BFieldElement(i * 5 + 1), BFieldElement(i * 5 + 2),
            BFieldElement(i * 5 + 3), BFieldElement(i * 5 + 4)
        ));
    }
    
    auto commitment = TableCommitment::from_digests(digests);
    
    EXPECT_EQ(commitment.num_rows(), 8);
    
    // Verify each digest
    Digest root = commitment.root();
    for (size_t i = 0; i < digests.size(); ++i) {
        auto auth_path = commitment.authentication_path(i);
        bool valid = TableCommitment::verify(root, i, digests[i], auth_path);
        EXPECT_TRUE(valid);
    }
}

// Test hash_bfield_row
TEST_F(TableCommitmentTest, HashBFieldRow) {
    std::vector<BFieldElement> row1 = {BFieldElement(1), BFieldElement(2), BFieldElement(3)};
    std::vector<BFieldElement> row2 = {BFieldElement(1), BFieldElement(2), BFieldElement(3)};
    std::vector<BFieldElement> row3 = {BFieldElement(1), BFieldElement(2), BFieldElement(4)};
    
    Digest hash1 = hash_bfield_row(row1);
    Digest hash2 = hash_bfield_row(row2);
    Digest hash3 = hash_bfield_row(row3);
    
    // Same row should give same hash
    EXPECT_EQ(hash1, hash2);
    
    // Different row should give different hash
    EXPECT_NE(hash1, hash3);
}

// Test hash_xfield_row
TEST_F(TableCommitmentTest, HashXFieldRow) {
    std::vector<XFieldElement> row1 = {
        XFieldElement(BFieldElement(1), BFieldElement(2), BFieldElement(3)),
        XFieldElement(BFieldElement(4), BFieldElement(5), BFieldElement(6))
    };
    std::vector<XFieldElement> row2 = {
        XFieldElement(BFieldElement(1), BFieldElement(2), BFieldElement(3)),
        XFieldElement(BFieldElement(4), BFieldElement(5), BFieldElement(6))
    };
    
    Digest hash1 = hash_xfield_row(row1);
    Digest hash2 = hash_xfield_row(row2);
    
    // Same row should give same hash
    EXPECT_EQ(hash1, hash2);
}

// Test with larger table matching test data dimensions
TEST_F(TableCommitmentTest, LargeTableCommitment) {
    // Create table with 512 rows (from test data)
    MasterMainTable table(512, 10);
    
    for (size_t r = 0; r < table.num_rows(); ++r) {
        for (size_t c = 0; c < table.num_columns(); ++c) {
            table.set(r, c, BFieldElement(r * 1000 + c));
        }
    }
    
    auto commitment = TableCommitment::commit(table);
    
    EXPECT_EQ(commitment.num_rows(), 512);
    
    // Verify a few random rows
    Digest root = commitment.root();
    for (size_t r : {0, 100, 255, 400, 511}) {
        Digest row_hash = hash_bfield_row(table.row(r));
        auto auth_path = commitment.authentication_path(r);
        
        bool valid = TableCommitment::verify(root, r, row_hash, auth_path);
        EXPECT_TRUE(valid) << "Verification failed for row " << r;
    }
}

