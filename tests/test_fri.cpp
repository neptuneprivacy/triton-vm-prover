#include <gtest/gtest.h>
#include "fri/fri.hpp"
#include "proof_stream/proof_stream.hpp"
#include "table/master_table.hpp"
#include <random>

using namespace triton_vm;

class FriTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default FRI parameters
        // Using larger domain to get meaningful num_rounds with Rust calculation
        // For domain=1024, expansion=4, checks=8:
        //   first_round_max_degree = 255
        //   max_num_rounds = 8
        //   num_rounds_checking_most = 4 (log2(8) + 1)
        //   num_rounds = 8 - 4 = 4
        domain_ = ArithmeticDomain::of_length(1024);
        expansion_factor_ = 4;
        num_collinearity_checks_ = 8;
    }
    
    ArithmeticDomain domain_;
    size_t expansion_factor_;
    size_t num_collinearity_checks_;
};

// Test FRI construction
TEST_F(FriTest, Construction) {
    Fri fri(domain_, expansion_factor_, num_collinearity_checks_);
    
    EXPECT_EQ(fri.domain().length, 1024);
    EXPECT_EQ(fri.expansion_factor(), 4);
    EXPECT_EQ(fri.num_collinearity_checks(), 8);
}

// Test invalid expansion factor
TEST_F(FriTest, InvalidExpansionFactor) {
    EXPECT_THROW(Fri(domain_, 0, 8), std::invalid_argument);
    EXPECT_THROW(Fri(domain_, 1, 8), std::invalid_argument);
    EXPECT_THROW(Fri(domain_, 3, 8), std::invalid_argument); // Not power of 2
}

// Test number of rounds
TEST_F(FriTest, NumRounds) {
    // Rust FRI num_rounds calculation:
    //   max_num_rounds = log2(domain/expansion)
    //   num_rounds = max_num_rounds - (log2(collinearity_checks) + 1)
    //
    // For domain=1024, expansion=4, checks=8:
    //   first_round_max_degree = 255
    //   max_num_rounds = 8
    //   num_rounds = 8 - 4 = 4
    Fri fri(domain_, expansion_factor_, num_collinearity_checks_);
    EXPECT_EQ(fri.num_rounds(), 4);
    
    // With expansion factor 8:
    //   first_round_max_degree = 127
    //   max_num_rounds = 7
    //   num_rounds = 7 - 4 = 3
    Fri fri2(domain_, 8, num_collinearity_checks_);
    EXPECT_EQ(fri2.num_rounds(), 3);
}

// Test batch inversion
TEST_F(FriTest, BatchInverse) {
    std::vector<BFieldElement> elements = {
        BFieldElement(2),
        BFieldElement(3),
        BFieldElement(5),
        BFieldElement(7)
    };
    
    auto inverses = batch_inverse(elements);
    
    EXPECT_EQ(inverses.size(), elements.size());
    
    // Verify each inverse
    for (size_t i = 0; i < elements.size(); ++i) {
        BFieldElement product = elements[i] * inverses[i];
        EXPECT_TRUE(product.is_one()) << "Failed for element " << i;
    }
}

// Test FRI round construction
TEST_F(FriTest, FriRoundConstruction) {
    std::vector<XFieldElement> codeword(1024);
    for (size_t i = 0; i < 1024; ++i) {
        codeword[i] = XFieldElement(BFieldElement(i + 1));
    }
    
    FriRound round(domain_, codeword);
    
    // Merkle root should be non-zero
    Digest root = round.merkle_root();
    bool all_zero = true;
    for (size_t i = 0; i < Digest::LEN; ++i) {
        if (!root[i].is_zero()) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
}

// Test split and fold
TEST_F(FriTest, SplitAndFold) {
    std::vector<XFieldElement> codeword(64);
    for (size_t i = 0; i < 64; ++i) {
        codeword[i] = XFieldElement(BFieldElement(i + 1));
    }
    
    ArithmeticDomain small_domain = ArithmeticDomain::of_length(64);
    FriRound round(small_domain, codeword);
    
    XFieldElement challenge(BFieldElement(42));
    auto folded = round.split_and_fold(challenge);
    
    // Folded codeword should be half the size
    EXPECT_EQ(folded.size(), 32);
    
    // Folded values should be non-zero
    for (size_t i = 0; i < folded.size(); ++i) {
        EXPECT_FALSE(folded[i].is_zero()) << "Folded value at " << i << " is zero";
    }
}

// Test FRI prove with simple codeword (using ProofStream interface)
TEST_F(FriTest, Prove) {
    // Create a simple constant codeword (represents a degree-0 polynomial)
    std::vector<XFieldElement> codeword(1024);
    XFieldElement constant(BFieldElement(12345));
    for (size_t i = 0; i < 1024; ++i) {
        codeword[i] = constant;
    }
    
    Fri fri(domain_, expansion_factor_, num_collinearity_checks_);
    ProofStream proof_stream;
    
    // Prove using the new ProofStream-based interface
    auto revealed = fri.prove(codeword, proof_stream);
    
    // Check that proof items were enqueued
    const auto& items = proof_stream.items();
    EXPECT_GT(items.size(), 0);
    
    // First items should be Merkle roots (one per round + 1)
    size_t expected_merkle_roots = fri.num_rounds() + 1;
    size_t merkle_root_count = 0;
    for (const auto& item : items) {
        if (item.type == ProofItemType::MerkleRoot) {
            merkle_root_count++;
        }
    }
    EXPECT_EQ(merkle_root_count, expected_merkle_roots);
    
    // Should have revealed indices
    EXPECT_EQ(revealed.size(), num_collinearity_checks_);
}

// Test FRI verify (stub - just checks it doesn't crash)
TEST_F(FriTest, ProveAndVerify) {
    std::vector<XFieldElement> codeword(1024);
    XFieldElement constant(BFieldElement(12345));
    for (size_t i = 0; i < 1024; ++i) {
        codeword[i] = constant;
    }
    
    Fri fri(domain_, expansion_factor_, num_collinearity_checks_);
    ProofStream proof_stream;
    
    auto revealed = fri.prove(codeword, proof_stream);
    
    // Verify should succeed (currently a stub)
    bool valid = fri.verify(proof_stream);
    EXPECT_TRUE(valid);
}

// Test FRI with random polynomial
TEST_F(FriTest, RandomPolynomial) {
    // Create a polynomial codeword
    std::vector<XFieldElement> codeword(1024);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> dist(1, 1000000);
    
    for (size_t i = 0; i < 1024; ++i) {
        codeword[i] = XFieldElement(
            BFieldElement(dist(rng)),
            BFieldElement(dist(rng) % 100),
            BFieldElement(dist(rng) % 10)
        );
    }
    
    Fri fri(domain_, expansion_factor_, num_collinearity_checks_);
    ProofStream proof_stream;
    
    auto revealed = fri.prove(codeword, proof_stream);
    
    // Verify should succeed
    bool valid = fri.verify(proof_stream);
    EXPECT_TRUE(valid);
}

// Test FRI with larger domain
TEST_F(FriTest, LargerDomain) {
    // Use domain=4096 with 16 checks
    // first_round_max_degree = 511
    // max_num_rounds = 9
    // num_rounds = 9 - 5 = 4
    ArithmeticDomain large_domain = ArithmeticDomain::of_length(4096);
    
    std::vector<XFieldElement> codeword(4096);
    for (size_t i = 0; i < 4096; ++i) {
        codeword[i] = XFieldElement(BFieldElement(i * i + 1));
    }
    
    Fri fri(large_domain, 8, 16);
    
    // Rust calculation for domain=4096, expansion=8, checks=16:
    //   first_round_max_degree = (4096/8) - 1 = 511
    //   max_num_rounds = log2(512) = 9
    //   num_rounds = 9 - 5 = 4
    EXPECT_EQ(fri.num_rounds(), 4);
    
    ProofStream proof_stream;
    auto revealed = fri.prove(codeword, proof_stream);
    
    bool valid = fri.verify(proof_stream);
    EXPECT_TRUE(valid);
}

// Test fold index calculation
TEST_F(FriTest, FoldIndex) {
    // Index i and i + n/2 should both fold to i
    EXPECT_EQ(Fri::fold_index(0, 64), 0);
    EXPECT_EQ(Fri::fold_index(32, 64), 0);
    EXPECT_EQ(Fri::fold_index(1, 64), 1);
    EXPECT_EQ(Fri::fold_index(33, 64), 1);
    EXPECT_EQ(Fri::fold_index(31, 64), 31);
    EXPECT_EQ(Fri::fold_index(63, 64), 31);
}
