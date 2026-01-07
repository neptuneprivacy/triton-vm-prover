#include <gtest/gtest.h>
#include "proof_stream/proof_stream.hpp"
#include "types/x_field_element.hpp"
#include <unordered_set>

using namespace triton_vm;

class ProofStreamTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

// Test construction
TEST_F(ProofStreamTest, Construction) {
    ProofStream ps;
    EXPECT_EQ(ps.items().size(), 0);
}

// Test enqueue and dequeue
TEST_F(ProofStreamTest, EnqueueDequeue) {
    ProofStream ps;
    
    // Enqueue a Merkle root
    Digest root(BFieldElement(1), BFieldElement(2), BFieldElement(3),
               BFieldElement(4), BFieldElement(5));
    ps.enqueue(ProofItem::merkle_root(root));
    
    EXPECT_EQ(ps.items().size(), 1);
    
    // Dequeue
    ProofItem item = ps.dequeue();
    EXPECT_EQ(item.type, ProofItemType::MerkleRoot);
    EXPECT_EQ(item.digest, root);
}

// Test dequeue from empty throws
TEST_F(ProofStreamTest, DequeueEmptyThrows) {
    ProofStream ps;
    EXPECT_THROW(ps.dequeue(), std::runtime_error);
}

// Test Fiat-Shamir heuristic flags
TEST_F(ProofStreamTest, FiatShamirHeuristicFlags) {
    // Merkle root should be included
    auto mr = ProofItem::merkle_root(Digest::zero());
    EXPECT_TRUE(mr.include_in_fiat_shamir_heuristic());
    
    // Log2PaddedHeight should NOT be included
    auto lph = ProofItem::make_log2_padded_height(10);
    EXPECT_FALSE(lph.include_in_fiat_shamir_heuristic());
    
    // FRI codeword should NOT be included
    auto fc = ProofItem::fri_codeword({});
    EXPECT_FALSE(fc.include_in_fiat_shamir_heuristic());
}

// Test sample_scalars returns correct count
TEST_F(ProofStreamTest, SampleScalarsCount) {
    ProofStream ps;
    
    auto scalars = ps.sample_scalars(5);
    EXPECT_EQ(scalars.size(), 5);
    
    // Sample more
    auto more = ps.sample_scalars(10);
    EXPECT_EQ(more.size(), 10);
}

// Test sample_scalars returns non-zero elements
TEST_F(ProofStreamTest, SampleScalarsNonZero) {
    ProofStream ps;
    
    // Alter state first to get different values
    ps.alter_fiat_shamir_state_with({BFieldElement(42)});
    
    auto scalars = ps.sample_scalars(10);
    
    bool any_nonzero = false;
    for (const auto& s : scalars) {
        if (!s.is_zero()) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

// Test sample_indices returns correct count
TEST_F(ProofStreamTest, SampleIndicesCount) {
    ProofStream ps;
    
    auto indices = ps.sample_indices(64, 10);
    EXPECT_EQ(indices.size(), 10);
}

// Test sample_indices respects upper bound
TEST_F(ProofStreamTest, SampleIndicesBound) {
    ProofStream ps;
    
    size_t upper_bound = 128;
    auto indices = ps.sample_indices(upper_bound, 50);
    
    for (size_t idx : indices) {
        EXPECT_LT(idx, upper_bound);
    }
}

// Test sample_indices non-power-of-2 throws
TEST_F(ProofStreamTest, SampleIndicesNonPowerOf2Throws) {
    ProofStream ps;
    EXPECT_THROW(ps.sample_indices(100, 10), std::invalid_argument);
}

// Test deterministic behavior
TEST_F(ProofStreamTest, Deterministic) {
    ProofStream ps1, ps2;
    
    // Same operations should produce same results
    ps1.alter_fiat_shamir_state_with({BFieldElement(123), BFieldElement(456)});
    ps2.alter_fiat_shamir_state_with({BFieldElement(123), BFieldElement(456)});
    
    auto scalars1 = ps1.sample_scalars(5);
    auto scalars2 = ps2.sample_scalars(5);
    
    EXPECT_EQ(scalars1, scalars2);
}

// Test different states produce different results
TEST_F(ProofStreamTest, DifferentStates) {
    ProofStream ps1, ps2;
    
    ps1.alter_fiat_shamir_state_with({BFieldElement(111)});
    ps2.alter_fiat_shamir_state_with({BFieldElement(222)});
    
    auto scalars1 = ps1.sample_scalars(5);
    auto scalars2 = ps2.sample_scalars(5);
    
    EXPECT_NE(scalars1, scalars2);
}

// Test ProofItem encoding
TEST_F(ProofStreamTest, ProofItemEncode) {
    // Test MerkleRoot encoding
    Digest root(BFieldElement(1), BFieldElement(2), BFieldElement(3),
               BFieldElement(4), BFieldElement(5));
    auto mr = ProofItem::merkle_root(root);
    auto encoded = mr.encode();

    EXPECT_EQ(encoded.size(), 6);
    EXPECT_EQ(encoded[0], BFieldElement(0));
    EXPECT_EQ(encoded[1], BFieldElement(1));
    EXPECT_EQ(encoded[5], BFieldElement(5));

    // Test Log2PaddedHeight encoding
    auto lph = ProofItem::make_log2_padded_height(16);
    encoded = lph.encode();

    EXPECT_EQ(encoded.size(), 2);
    EXPECT_EQ(encoded[0], BFieldElement(7));  // Rust discriminant order
    EXPECT_EQ(encoded[1].value(), 16);

    // Test OutOfDomainMainRow encoding (length-prefixed x-field vector)
    ProofItem ood;
    ood.type = ProofItemType::OutOfDomainMainRow;
    ood.xfield_vec = {XFieldElement(BFieldElement(9), BFieldElement::zero(), BFieldElement::one())};
    encoded = ood.encode();
    EXPECT_EQ(encoded[0], BFieldElement(1));  // discriminant
    EXPECT_EQ(encoded[1], BFieldElement(1));  // length (one element)
    EXPECT_EQ(encoded[2], BFieldElement(9));  // coeff 0
    EXPECT_EQ(encoded[4], BFieldElement::one());  // coeff 2
}

TEST_F(ProofStreamTest, ProofStreamEncodeAddsLengths) {
    ProofStream ps;
    ps.enqueue(ProofItem::make_log2_padded_height(3));
    auto encoding = ps.encode();
    ASSERT_EQ(encoding.size(), 4);
    EXPECT_EQ(encoding[0], BFieldElement(1));   // number of items
    EXPECT_EQ(encoding[1], BFieldElement(2));   // item length
    EXPECT_EQ(encoding[2], BFieldElement(7));   // discriminant
    EXPECT_EQ(encoding[3], BFieldElement(3));   // payload value
}

// Test interaction: enqueue affects sampling
TEST_F(ProofStreamTest, EnqueueAffectsSampling) {
    ProofStream ps1, ps2;
    
    // Add different items
    Digest root1(BFieldElement(111), BFieldElement(0), BFieldElement(0),
                BFieldElement(0), BFieldElement(0));
    Digest root2(BFieldElement(222), BFieldElement(0), BFieldElement(0),
                BFieldElement(0), BFieldElement(0));
    
    ps1.enqueue(ProofItem::merkle_root(root1));
    ps2.enqueue(ProofItem::merkle_root(root2));
    
    auto scalars1 = ps1.sample_scalars(3);
    auto scalars2 = ps2.sample_scalars(3);
    
    // Different Merkle roots should lead to different challenges
    EXPECT_NE(scalars1, scalars2);
}

// Test multiple enqueue/sample cycles
TEST_F(ProofStreamTest, MultipleCycles) {
    ProofStream ps;
    
    // First round
    Digest root1(BFieldElement(1), BFieldElement(0), BFieldElement(0),
                BFieldElement(0), BFieldElement(0));
    ps.enqueue(ProofItem::merkle_root(root1));
    auto challenge1 = ps.sample_scalars(1)[0];
    
    // Second round
    Digest root2(BFieldElement(2), BFieldElement(0), BFieldElement(0),
                BFieldElement(0), BFieldElement(0));
    ps.enqueue(ProofItem::merkle_root(root2));
    auto challenge2 = ps.sample_scalars(1)[0];
    
    // Challenges should be different
    EXPECT_NE(challenge1, challenge2);
}

// Test sample many scalars
TEST_F(ProofStreamTest, SampleManyScalars) {
    ProofStream ps;
    ps.alter_fiat_shamir_state_with({BFieldElement(12345)});
    
    auto scalars = ps.sample_scalars(100);
    EXPECT_EQ(scalars.size(), 100);
    
    // Check some variety (not all the same)
    std::unordered_set<uint64_t> unique_values;
    for (const auto& s : scalars) {
        unique_values.insert(s.coeff(0).value());
    }
    EXPECT_GT(unique_values.size(), 10);  // Should have reasonable variety
}

