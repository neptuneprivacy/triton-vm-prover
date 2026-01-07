#include <gtest/gtest.h>
#include "types/b_field_element.hpp"

using namespace triton_vm;

class BFieldElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
    }
};

// Basic construction tests
TEST_F(BFieldElementTest, DefaultConstruction) {
    BFieldElement zero;
    EXPECT_EQ(zero.value(), 0ULL);
}

TEST_F(BFieldElementTest, ValueConstruction) {
    BFieldElement elem(42);
    EXPECT_EQ(elem.value(), 42ULL);
}

TEST_F(BFieldElementTest, ModularReduction) {
    // Value larger than modulus should be reduced
    BFieldElement elem(BFieldElement::MODULUS + 5);
    EXPECT_EQ(elem.value(), 5ULL);
}

// Factory methods
TEST_F(BFieldElementTest, Zero) {
    auto zero = BFieldElement::zero();
    EXPECT_EQ(zero.value(), 0ULL);
    EXPECT_TRUE(zero.is_zero());
}

TEST_F(BFieldElementTest, One) {
    auto one = BFieldElement::one();
    EXPECT_EQ(one.value(), 1ULL);
    EXPECT_TRUE(one.is_one());
}

TEST_F(BFieldElementTest, Generator) {
    auto gen = BFieldElement::generator();
    EXPECT_EQ(gen.value(), BFieldElement::GENERATOR);
}

// Arithmetic tests
TEST_F(BFieldElementTest, Addition) {
    BFieldElement a(100);
    BFieldElement b(200);
    auto result = a + b;
    EXPECT_EQ(result.value(), 300ULL);
}

TEST_F(BFieldElementTest, AdditionWithOverflow) {
    BFieldElement a(BFieldElement::MODULUS - 5);
    BFieldElement b(10);
    auto result = a + b;
    EXPECT_EQ(result.value(), 5ULL);
}

TEST_F(BFieldElementTest, Subtraction) {
    BFieldElement a(200);
    BFieldElement b(100);
    auto result = a - b;
    EXPECT_EQ(result.value(), 100ULL);
}

TEST_F(BFieldElementTest, SubtractionWithUnderflow) {
    BFieldElement a(5);
    BFieldElement b(10);
    auto result = a - b;
    EXPECT_EQ(result.value(), BFieldElement::MODULUS - 5);
}

TEST_F(BFieldElementTest, Multiplication) {
    BFieldElement a(100);
    BFieldElement b(200);
    auto result = a * b;
    EXPECT_EQ(result.value(), 20000ULL);
}

TEST_F(BFieldElementTest, MultiplicationWithReduction) {
    // Test that large products are reduced correctly
    BFieldElement a(1ULL << 32);
    BFieldElement b(1ULL << 32);
    auto result = a * b;
    // (2^32)^2 mod p = 2^64 mod p
    // Since p = 2^64 - 2^32 + 1, we have 2^64 = 2^32 - 1 mod p
    EXPECT_EQ(result.value(), (1ULL << 32) - 1);
}

TEST_F(BFieldElementTest, Negation) {
    BFieldElement a(100);
    auto neg_a = -a;
    EXPECT_EQ((a + neg_a).value(), 0ULL);
}

TEST_F(BFieldElementTest, NegationOfZero) {
    auto zero = BFieldElement::zero();
    auto neg_zero = -zero;
    EXPECT_EQ(neg_zero.value(), 0ULL);
}

// Power tests
TEST_F(BFieldElementTest, PowerOfZero) {
    BFieldElement a(5);
    auto result = a.pow(0);
    EXPECT_EQ(result.value(), 1ULL);
}

TEST_F(BFieldElementTest, PowerOfOne) {
    BFieldElement a(5);
    auto result = a.pow(1);
    EXPECT_EQ(result.value(), 5ULL);
}

TEST_F(BFieldElementTest, PowerOfTwo) {
    BFieldElement a(5);
    auto result = a.pow(2);
    EXPECT_EQ(result.value(), 25ULL);
}

TEST_F(BFieldElementTest, PowerOfTen) {
    BFieldElement a(2);
    auto result = a.pow(10);
    EXPECT_EQ(result.value(), 1024ULL);
}

// Inverse tests
TEST_F(BFieldElementTest, InverseOfOne) {
    auto one = BFieldElement::one();
    auto inv = one.inverse();
    EXPECT_EQ(inv.value(), 1ULL);
}

TEST_F(BFieldElementTest, InverseProperty) {
    BFieldElement a(12345);
    auto inv = a.inverse();
    auto product = a * inv;
    EXPECT_EQ(product.value(), 1ULL);
}

TEST_F(BFieldElementTest, InverseOfGenerator) {
    auto gen = BFieldElement::generator();
    auto inv = gen.inverse();
    auto product = gen * inv;
    EXPECT_EQ(product.value(), 1ULL);
}

TEST_F(BFieldElementTest, InverseOfZeroThrows) {
    auto zero = BFieldElement::zero();
    EXPECT_THROW(zero.inverse(), std::domain_error);
}

// Division tests
TEST_F(BFieldElementTest, Division) {
    BFieldElement a(100);
    BFieldElement b(5);
    auto result = a / b;
    EXPECT_EQ(result.value(), 20ULL);
}

TEST_F(BFieldElementTest, DivisionProperty) {
    BFieldElement a(12345);
    BFieldElement b(67890);
    auto quotient = a / b;
    auto product = quotient * b;
    EXPECT_EQ(product.value(), a.value());
}

// Primitive root of unity tests
TEST_F(BFieldElementTest, PrimitiveRootOfUnityOrder1) {
    auto root = BFieldElement::primitive_root_of_unity(1);
    auto squared = root.pow(2);
    EXPECT_EQ(squared.value(), 1ULL);
}

TEST_F(BFieldElementTest, PrimitiveRootOfUnityOrder32) {
    auto root = BFieldElement::primitive_root_of_unity(32);
    auto power = root.pow(1ULL << 32);
    EXPECT_EQ(power.value(), 1ULL);
    
    // Should not be 1 at half the order
    auto half_power = root.pow(1ULL << 31);
    EXPECT_NE(half_power.value(), 1ULL);
}

// Test against known Rust values
TEST_F(BFieldElementTest, KnownValueFromTestData) {
    // From test_data: first row value 4099276459869907627
    BFieldElement elem(4099276459869907627ULL);
    EXPECT_EQ(elem.value(), 4099276459869907627ULL);
    
    // Verify it's less than modulus
    EXPECT_LT(elem.value(), BFieldElement::MODULUS);
}

TEST_F(BFieldElementTest, ModulusValue) {
    // Goldilocks prime: 2^64 - 2^32 + 1
    uint64_t expected = (1ULL << 64) - (1ULL << 32) + 1;
    // Note: 2^64 overflows, so compute differently
    expected = 0xFFFFFFFF00000001ULL;
    EXPECT_EQ(BFieldElement::MODULUS, expected);
}

