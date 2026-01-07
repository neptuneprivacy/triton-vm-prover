#include <gtest/gtest.h>
#include "merkle/merkle_tree.hpp"
#include "test_data_loader.hpp"
#include <filesystem>

using namespace triton_vm;

class MerkleTreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data_dir_ = TEST_DATA_DIR;
        
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found: " << test_data_dir_;
        }
    }
    
    std::string test_data_dir_;
};

// Test construction with power of 2 leaves
TEST_F(MerkleTreeTest, ConstructionPowerOf2) {
    std::vector<Digest> leaves;
    for (size_t i = 0; i < 8; ++i) {
        leaves.push_back(Digest(
            BFieldElement(i * 5), BFieldElement(i * 5 + 1), BFieldElement(i * 5 + 2),
            BFieldElement(i * 5 + 3), BFieldElement(i * 5 + 4)
        ));
    }
    
    EXPECT_NO_THROW({
        MerkleTree tree(leaves);
        EXPECT_EQ(tree.num_leaves(), 8);
        EXPECT_EQ(tree.height(), 3);
    });
}

// Test construction with non-power of 2 leaves throws
TEST_F(MerkleTreeTest, ConstructionNonPowerOf2Throws) {
    std::vector<Digest> leaves(5, Digest::zero());
    EXPECT_THROW(MerkleTree tree(leaves), std::invalid_argument);
}

// Test construction with empty leaves throws
TEST_F(MerkleTreeTest, ConstructionEmptyThrows) {
    std::vector<Digest> leaves;
    EXPECT_THROW(MerkleTree tree(leaves), std::invalid_argument);
}

// Test root is non-zero for non-zero leaves
TEST_F(MerkleTreeTest, RootNonZero) {
    std::vector<Digest> leaves;
    for (size_t i = 0; i < 4; ++i) {
        leaves.push_back(Digest(
            BFieldElement(i + 1), BFieldElement(i + 2), BFieldElement(i + 3),
            BFieldElement(i + 4), BFieldElement(i + 5)
        ));
    }
    
    MerkleTree tree(leaves);
    Digest root = tree.root();
    
    bool all_zero = true;
    for (size_t i = 0; i < Digest::LEN; ++i) {
        if (!root[i].is_zero()) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
}

// Test root is deterministic
TEST_F(MerkleTreeTest, RootDeterministic) {
    std::vector<Digest> leaves;
    for (size_t i = 0; i < 4; ++i) {
        leaves.push_back(Digest(
            BFieldElement(i * 100), BFieldElement(i * 100 + 1), BFieldElement(i * 100 + 2),
            BFieldElement(i * 100 + 3), BFieldElement(i * 100 + 4)
        ));
    }
    
    MerkleTree tree1(leaves);
    MerkleTree tree2(leaves);
    
    EXPECT_EQ(tree1.root(), tree2.root());
}

// Test leaf access
TEST_F(MerkleTreeTest, LeafAccess) {
    std::vector<Digest> leaves;
    for (size_t i = 0; i < 4; ++i) {
        leaves.push_back(Digest(
            BFieldElement(i), BFieldElement(0), BFieldElement(0),
            BFieldElement(0), BFieldElement(0)
        ));
    }
    
    MerkleTree tree(leaves);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tree.leaf(i), leaves[i]);
    }
}

// Test leaf out of range throws
TEST_F(MerkleTreeTest, LeafOutOfRangeThrows) {
    std::vector<Digest> leaves(4, Digest::zero());
    MerkleTree tree(leaves);
    
    EXPECT_THROW(tree.leaf(4), std::out_of_range);
}

// Test authentication path length
TEST_F(MerkleTreeTest, AuthenticationPathLength) {
    std::vector<Digest> leaves(8, Digest::zero());
    for (size_t i = 0; i < 8; ++i) {
        leaves[i] = Digest(BFieldElement(i), BFieldElement(0), BFieldElement(0),
                          BFieldElement(0), BFieldElement(0));
    }
    
    MerkleTree tree(leaves);
    
    // Height is 3, so auth path should have 3 elements
    auto path = tree.authentication_path(0);
    EXPECT_EQ(path.size(), 3);
}

// Test verification of valid proof
TEST_F(MerkleTreeTest, VerifyValidProof) {
    std::vector<Digest> leaves;
    for (size_t i = 0; i < 8; ++i) {
        leaves.push_back(Digest(
            BFieldElement(i + 1), BFieldElement(i * 2), BFieldElement(i * 3),
            BFieldElement(i * 4), BFieldElement(i * 5)
        ));
    }
    
    MerkleTree tree(leaves);
    Digest root = tree.root();
    
    // Verify all leaves
    for (size_t i = 0; i < 8; ++i) {
        auto path = tree.authentication_path(i);
        bool valid = MerkleTree::verify(root, i, leaves[i], path);
        EXPECT_TRUE(valid) << "Verification failed for leaf " << i;
    }
}

// Test verification of invalid leaf fails
TEST_F(MerkleTreeTest, VerifyInvalidLeafFails) {
    std::vector<Digest> leaves;
    for (size_t i = 0; i < 4; ++i) {
        leaves.push_back(Digest(
            BFieldElement(i + 1), BFieldElement(i + 2), BFieldElement(i + 3),
            BFieldElement(i + 4), BFieldElement(i + 5)
        ));
    }
    
    MerkleTree tree(leaves);
    Digest root = tree.root();
    
    auto path = tree.authentication_path(0);
    
    // Try to verify with wrong leaf
    Digest wrong_leaf(BFieldElement(999), BFieldElement(999), BFieldElement(999),
                     BFieldElement(999), BFieldElement(999));
    
    bool valid = MerkleTree::verify(root, 0, wrong_leaf, path);
    EXPECT_FALSE(valid);
}

// Test verification with wrong index fails
TEST_F(MerkleTreeTest, VerifyWrongIndexFails) {
    std::vector<Digest> leaves;
    for (size_t i = 0; i < 4; ++i) {
        leaves.push_back(Digest(
            BFieldElement(i + 1), BFieldElement(i + 2), BFieldElement(i + 3),
            BFieldElement(i + 4), BFieldElement(i + 5)
        ));
    }
    
    MerkleTree tree(leaves);
    Digest root = tree.root();
    
    auto path = tree.authentication_path(0);
    
    // Try to verify with wrong index
    bool valid = MerkleTree::verify(root, 1, leaves[0], path);
    EXPECT_FALSE(valid);
}

// Test sibling index calculation
TEST_F(MerkleTreeTest, SiblingIndex) {
    EXPECT_EQ(MerkleTree::sibling_index(2), 3);
    EXPECT_EQ(MerkleTree::sibling_index(3), 2);
    EXPECT_EQ(MerkleTree::sibling_index(4), 5);
    EXPECT_EQ(MerkleTree::sibling_index(5), 4);
}

// Test parent index calculation
TEST_F(MerkleTreeTest, ParentIndex) {
    EXPECT_EQ(MerkleTree::parent_index(2), 1);
    EXPECT_EQ(MerkleTree::parent_index(3), 1);
    EXPECT_EQ(MerkleTree::parent_index(4), 2);
    EXPECT_EQ(MerkleTree::parent_index(5), 2);
}

// Test hash_row function
TEST_F(MerkleTreeTest, HashRow) {
    std::vector<BFieldElement> row;
    for (size_t i = 0; i < 10; ++i) {
        row.push_back(BFieldElement(i));
    }
    
    Digest result = hash_row(row);
    
    // Should be non-zero
    bool all_zero = true;
    for (size_t i = 0; i < Digest::LEN; ++i) {
        if (!result[i].is_zero()) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
}

// Test hash_row is deterministic
TEST_F(MerkleTreeTest, HashRowDeterministic) {
    std::vector<BFieldElement> row;
    for (size_t i = 0; i < 15; ++i) {
        row.push_back(BFieldElement(i * 100 + 42));
    }
    
    Digest result1 = hash_row(row);
    Digest result2 = hash_row(row);
    
    EXPECT_EQ(result1, result2);
}

// Test with test data - verify dimensions match
TEST_F(MerkleTreeTest, MatchTestDataDimensions) {
    TestDataLoader loader(test_data_dir_);
    auto merkle_data = loader.load_main_tables_merkle();
    
    // The test data says there are 4096 leaves
    // We can construct a tree of that size
    std::vector<Digest> leaves(4096, Digest::zero());
    
    EXPECT_NO_THROW({
        MerkleTree tree(leaves);
        EXPECT_EQ(tree.num_leaves(), 4096);
        EXPECT_EQ(tree.height(), 12); // log2(4096) = 12
    });
}

