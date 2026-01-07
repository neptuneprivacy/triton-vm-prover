#include <gtest/gtest.h>
#include "types/x_field_element.hpp"

using namespace triton_vm;

class XFieldElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
    }
};

// Basic construction tests
TEST_F(XFieldElementTest, DefaultConstruction) {
    XFieldElement zero;
    EXPECT_TRUE(zero.is_zero());
}

TEST_F(XFieldElementTest, CoefficientConstruction) {
    BFieldElement a(1), b(2), c(3);
    XFieldElement elem(a, b, c);
    
    EXPECT_EQ(elem.coeff(0).value(), 1ULL);
    EXPECT_EQ(elem.coeff(1).value(), 2ULL);
    EXPECT_EQ(elem.coeff(2).value(), 3ULL);
}

TEST_F(XFieldElementTest, FromBFieldElement) {
    BFieldElement base(42);
    XFieldElement elem(base);
    
    EXPECT_EQ(elem.coeff(0).value(), 42ULL);
    EXPECT_EQ(elem.coeff(1).value(), 0ULL);
    EXPECT_EQ(elem.coeff(2).value(), 0ULL);
}

// Factory methods
TEST_F(XFieldElementTest, Zero) {
    auto zero = XFieldElement::zero();
    EXPECT_TRUE(zero.is_zero());
}

TEST_F(XFieldElementTest, One) {
    auto one = XFieldElement::one();
    EXPECT_TRUE(one.is_one());
    EXPECT_EQ(one.coeff(0).value(), 1ULL);
    EXPECT_EQ(one.coeff(1).value(), 0ULL);
    EXPECT_EQ(one.coeff(2).value(), 0ULL);
}

// Arithmetic tests
TEST_F(XFieldElementTest, Addition) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    XFieldElement b(BFieldElement(4), BFieldElement(5), BFieldElement(6));
    auto result = a + b;
    
    EXPECT_EQ(result.coeff(0).value(), 5ULL);
    EXPECT_EQ(result.coeff(1).value(), 7ULL);
    EXPECT_EQ(result.coeff(2).value(), 9ULL);
}

TEST_F(XFieldElementTest, Subtraction) {
    XFieldElement a(BFieldElement(10), BFieldElement(20), BFieldElement(30));
    XFieldElement b(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    auto result = a - b;
    
    EXPECT_EQ(result.coeff(0).value(), 9ULL);
    EXPECT_EQ(result.coeff(1).value(), 18ULL);
    EXPECT_EQ(result.coeff(2).value(), 27ULL);
}

TEST_F(XFieldElementTest, MultiplicationByOne) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    auto one = XFieldElement::one();
    auto result = a * one;
    
    EXPECT_EQ(result, a);
}

TEST_F(XFieldElementTest, MultiplicationByZero) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    auto zero = XFieldElement::zero();
    auto result = a * zero;
    
    EXPECT_TRUE(result.is_zero());
}

TEST_F(XFieldElementTest, Multiplication) {
    // Test (1 + X) * (1 + X) = 1 + 2X + X^2
    XFieldElement a(BFieldElement(1), BFieldElement(1), BFieldElement(0));
    auto result = a * a;
    
    EXPECT_EQ(result.coeff(0).value(), 1ULL);
    EXPECT_EQ(result.coeff(1).value(), 2ULL);
    EXPECT_EQ(result.coeff(2).value(), 1ULL);
}

TEST_F(XFieldElementTest, MultiplicationWithReduction) {
    // Test X^2 * X = X^3 = X - 1 (by the Shah polynomial X^3 - X + 1 = 0)
    XFieldElement x2(BFieldElement(0), BFieldElement(0), BFieldElement(1)); // X^2
    XFieldElement x(BFieldElement(0), BFieldElement(1), BFieldElement(0));  // X
    auto result = x2 * x;
    
    // X^3 = X - 1
    // In the field, -1 is represented as (p - 1)
    static constexpr uint64_t MODULUS = 0xFFFFFFFF00000001ULL;
    EXPECT_EQ(result.coeff(0).value(), MODULUS - 1);  // constant term = -1
    EXPECT_EQ(result.coeff(1).value(), 1ULL);         // X coefficient = 1
    EXPECT_EQ(result.coeff(2).value(), 0ULL);         // X^2 coefficient = 0
}

TEST_F(XFieldElementTest, Negation) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    auto neg_a = -a;
    auto sum = a + neg_a;
    
    EXPECT_TRUE(sum.is_zero());
}

// Mixed arithmetic with BFieldElement
TEST_F(XFieldElementTest, AddBFieldElement) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    BFieldElement b(10);
    auto result = a + b;
    
    EXPECT_EQ(result.coeff(0).value(), 11ULL);
    EXPECT_EQ(result.coeff(1).value(), 2ULL);
    EXPECT_EQ(result.coeff(2).value(), 3ULL);
}

TEST_F(XFieldElementTest, MultiplyByBFieldElement) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    BFieldElement b(10);
    auto result = a * b;
    
    EXPECT_EQ(result.coeff(0).value(), 10ULL);
    EXPECT_EQ(result.coeff(1).value(), 20ULL);
    EXPECT_EQ(result.coeff(2).value(), 30ULL);
}

// Power tests
TEST_F(XFieldElementTest, PowerOfZero) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    auto result = a.pow(0);
    EXPECT_TRUE(result.is_one());
}

TEST_F(XFieldElementTest, PowerOfOne) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    auto result = a.pow(1);
    EXPECT_EQ(result, a);
}

TEST_F(XFieldElementTest, PowerOfTwo) {
    XFieldElement a(BFieldElement(1), BFieldElement(1), BFieldElement(0)); // 1 + X
    auto result = a.pow(2);
    
    // (1 + X)^2 = 1 + 2X + X^2
    EXPECT_EQ(result.coeff(0).value(), 1ULL);
    EXPECT_EQ(result.coeff(1).value(), 2ULL);
    EXPECT_EQ(result.coeff(2).value(), 1ULL);
}

// Inverse tests
TEST_F(XFieldElementTest, InverseOfOne) {
    auto one = XFieldElement::one();
    auto inv = one.inverse();
    EXPECT_TRUE(inv.is_one());
}

TEST_F(XFieldElementTest, InverseProperty) {
    XFieldElement a(BFieldElement(123), BFieldElement(456), BFieldElement(789));
    auto inv = a.inverse();
    auto product = a * inv;
    
    EXPECT_TRUE(product.is_one());
}

TEST_F(XFieldElementTest, InverseOfZeroThrows) {
    auto zero = XFieldElement::zero();
    EXPECT_THROW(zero.inverse(), std::domain_error);
}

// Division tests
TEST_F(XFieldElementTest, DivisionProperty) {
    XFieldElement a(BFieldElement(123), BFieldElement(456), BFieldElement(789));
    XFieldElement b(BFieldElement(111), BFieldElement(222), BFieldElement(333));
    
    auto quotient = a / b;
    auto product = quotient * b;
    
    EXPECT_EQ(product, a);
}

// Comparison tests
TEST_F(XFieldElementTest, Equality) {
    XFieldElement a(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    XFieldElement b(BFieldElement(1), BFieldElement(2), BFieldElement(3));
    XFieldElement c(BFieldElement(1), BFieldElement(2), BFieldElement(4));
    
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

