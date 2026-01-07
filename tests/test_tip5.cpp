#include <gtest/gtest.h>
#include "hash/tip5.hpp"
#include "test_data_loader.hpp"
#include <filesystem>

using namespace triton_vm;

class Tip5Test : public ::testing::Test {
protected:
    void SetUp() override {
        test_data_dir_ = TEST_DATA_DIR;
    }
    
    std::string test_data_dir_;
};

// Test Tip5 constants
TEST_F(Tip5Test, Constants) {
    EXPECT_EQ(Tip5::STATE_SIZE, 16);
    EXPECT_EQ(Tip5::RATE, 10);
    EXPECT_EQ(Tip5::CAPACITY, 6);
    EXPECT_EQ(Tip5::NUM_ROUNDS, 5);
    EXPECT_EQ(Tip5::NUM_SPLIT_AND_LOOKUP, 4);
}

// Test initialization
TEST_F(Tip5Test, Init) {
    Tip5 tip5;
    
    // Default state is all zeros
    for (size_t i = 0; i < Tip5::STATE_SIZE; ++i) {
        EXPECT_TRUE(tip5.state[i].is_zero());
    }
    
    // Fixed-length domain has capacity set to ones
    Tip5 fixed = Tip5::init();
    for (size_t i = 0; i < Tip5::RATE; ++i) {
        EXPECT_TRUE(fixed.state[i].is_zero());
    }
    for (size_t i = Tip5::RATE; i < Tip5::STATE_SIZE; ++i) {
        EXPECT_TRUE(fixed.state[i].is_one());
    }
}

// Test lookup table size
TEST_F(Tip5Test, LookupTableSize) {
    EXPECT_EQ(Tip5::LOOKUP_TABLE.size(), 256);
}

// Test round constants size
TEST_F(Tip5Test, RoundConstantsSize) {
    EXPECT_EQ(Tip5::ROUND_CONSTANTS.size(), Tip5::NUM_ROUNDS * Tip5::STATE_SIZE);
}

// Test MDS matrix first column
TEST_F(Tip5Test, MDSMatrixFirstColumn) {
    EXPECT_EQ(Tip5::MDS_MATRIX_FIRST_COLUMN.size(), Tip5::STATE_SIZE);
    
    // First element should be 61402
    EXPECT_EQ(Tip5::MDS_MATRIX_FIRST_COLUMN[0], 61402);
}

// Test hash_10 produces non-zero digest
TEST_F(Tip5Test, Hash10NonZero) {
    std::array<BFieldElement, Tip5::RATE> input;
    for (size_t i = 0; i < Tip5::RATE; ++i) {
        input[i] = BFieldElement(i + 1);
    }
    
    Digest result = Tip5::hash_10(input);
    
    // Should not be all zeros
    bool all_zero = true;
    for (size_t i = 0; i < Digest::LEN; ++i) {
        if (!result[i].is_zero()) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
}

// Test hash_10 is deterministic
TEST_F(Tip5Test, Hash10Deterministic) {
    std::array<BFieldElement, Tip5::RATE> input;
    for (size_t i = 0; i < Tip5::RATE; ++i) {
        input[i] = BFieldElement(i * 100 + 42);
    }
    
    Digest result1 = Tip5::hash_10(input);
    Digest result2 = Tip5::hash_10(input);
    
    EXPECT_EQ(result1, result2);
}

// Test hash_10 changes with different input
TEST_F(Tip5Test, Hash10DifferentInputs) {
    std::array<BFieldElement, Tip5::RATE> input1;
    std::array<BFieldElement, Tip5::RATE> input2;
    
    for (size_t i = 0; i < Tip5::RATE; ++i) {
        input1[i] = BFieldElement(i);
        input2[i] = BFieldElement(i + 1);
    }
    
    Digest result1 = Tip5::hash_10(input1);
    Digest result2 = Tip5::hash_10(input2);
    
    EXPECT_NE(result1, result2);
}

// Test hash_pair
TEST_F(Tip5Test, HashPair) {
    Digest left(BFieldElement(1), BFieldElement(2), BFieldElement(3),
                BFieldElement(4), BFieldElement(5));
    Digest right(BFieldElement(6), BFieldElement(7), BFieldElement(8),
                 BFieldElement(9), BFieldElement(10));
    
    Digest result = Tip5::hash_pair(left, right);
    
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

// Test hash_pair is deterministic
TEST_F(Tip5Test, HashPairDeterministic) {
    Digest left(BFieldElement(123), BFieldElement(456), BFieldElement(789),
                BFieldElement(101112), BFieldElement(131415));
    Digest right(BFieldElement(161718), BFieldElement(192021), BFieldElement(222324),
                 BFieldElement(252627), BFieldElement(282930));
    
    Digest result1 = Tip5::hash_pair(left, right);
    Digest result2 = Tip5::hash_pair(left, right);
    
    EXPECT_EQ(result1, result2);
}

// Test hash_pair order matters
TEST_F(Tip5Test, HashPairOrderMatters) {
    Digest a(BFieldElement(1), BFieldElement(2), BFieldElement(3),
             BFieldElement(4), BFieldElement(5));
    Digest b(BFieldElement(6), BFieldElement(7), BFieldElement(8),
             BFieldElement(9), BFieldElement(10));
    
    Digest result1 = Tip5::hash_pair(a, b);
    Digest result2 = Tip5::hash_pair(b, a);
    
    EXPECT_NE(result1, result2);
}

// Test variable-length hash
TEST_F(Tip5Test, HashVarlen) {
    std::vector<BFieldElement> input;
    for (size_t i = 0; i < 25; ++i) {
        input.push_back(BFieldElement(i));
    }
    
    Digest result = Tip5::hash_varlen(input);
    
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

// Test permutation changes state
TEST_F(Tip5Test, PermutationChangesState) {
    Tip5 tip5;
    
    // Set non-zero state
    for (size_t i = 0; i < Tip5::STATE_SIZE; ++i) {
        tip5.state[i] = BFieldElement(i + 1);
    }
    
    auto original_state = tip5.state;
    tip5.permutation();
    
    // State should have changed
    bool changed = false;
    for (size_t i = 0; i < Tip5::STATE_SIZE; ++i) {
        if (tip5.state[i] != original_state[i]) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

// Test single round
TEST_F(Tip5Test, SingleRound) {
    Tip5 tip5;
    
    // Set non-zero state
    for (size_t i = 0; i < Tip5::STATE_SIZE; ++i) {
        tip5.state[i] = BFieldElement(i + 1);
    }
    
    auto original_state = tip5.state;
    tip5.round(0);
    
    // State should have changed
    bool changed = false;
    for (size_t i = 0; i < Tip5::STATE_SIZE; ++i) {
        if (tip5.state[i] != original_state[i]) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

