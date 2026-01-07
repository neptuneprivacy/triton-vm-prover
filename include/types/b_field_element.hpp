#pragma once

#include <cstdint>
#include <string>
#include <ostream>
#include <array>
#include <vector>

// Check for AVX2 support
#if defined(__AVX2__)
#include <immintrin.h>
#define BFIELD_AVX2_ENABLED 1
#else
#define BFIELD_AVX2_ENABLED 0
#endif

namespace triton_vm {

// 128-bit unsigned integer type for intermediate calculations
using uint128_t = __uint128_t;

/**
 * BFieldElement - Base Field Element
 * 
 * Element of the Goldilocks prime field with modulus p = 2^64 - 2^32 + 1
 * This is the same field used in Triton VM (Rust implementation).
 */
class BFieldElement {
public:
    // The Goldilocks prime: 2^64 - 2^32 + 1
    static constexpr uint64_t MODULUS = 18446744069414584321ULL;
    
    // Montgomery constant R = 2^64 mod p
    static constexpr uint64_t MONTGOMERY_R = 4294967295ULL; // 2^32 - 1
    
    // Generator of the multiplicative group
    static constexpr uint64_t GENERATOR = 7ULL;

    // Constructors
    constexpr BFieldElement() : value_(0) {}
    constexpr explicit BFieldElement(uint64_t value) : value_(value % MODULUS) {}
    // from_raw_u64: In Rust, this assumes the value is already in Montgomery representation.
    // Since C++ uses canonical representation, we need to convert from Montgomery to canonical.
    static BFieldElement from_raw_u64(uint64_t montgomery_value);
    
    // Factory methods
    static constexpr BFieldElement zero() { return BFieldElement(0); }
    static constexpr BFieldElement one() { return BFieldElement(1); }
    static constexpr BFieldElement generator() { return BFieldElement(GENERATOR); }
    
    // Accessors
    constexpr uint64_t value() const { return value_; }
    
    // Arithmetic operations
    BFieldElement operator+(const BFieldElement& rhs) const;
    BFieldElement operator-(const BFieldElement& rhs) const;
    BFieldElement operator*(const BFieldElement& rhs) const;
    BFieldElement operator/(const BFieldElement& rhs) const;
    BFieldElement operator-() const;
    
    BFieldElement& operator+=(const BFieldElement& rhs);
    BFieldElement& operator-=(const BFieldElement& rhs);
    BFieldElement& operator*=(const BFieldElement& rhs);
    BFieldElement& operator/=(const BFieldElement& rhs);
    
    // Comparison
    bool operator==(const BFieldElement& rhs) const;
    bool operator!=(const BFieldElement& rhs) const;
    bool operator<(const BFieldElement& rhs) const;
    bool operator>(const BFieldElement& rhs) const;
    bool operator<=(const BFieldElement& rhs) const;
    bool operator>=(const BFieldElement& rhs) const;
    
    // Field operations
    BFieldElement inverse() const;
    BFieldElement pow(uint64_t exp) const;
    bool is_zero() const { return value_ == 0; }
    bool is_one() const { return value_ == 1; }

    static std::vector<BFieldElement> batch_inversion(const std::vector<BFieldElement>& elements);
    
    // SIMD batch operations (AVX2 accelerated when available)
    // These operate on arrays of 4 elements at a time
    static void batch_add(BFieldElement* dst, const BFieldElement* a, const BFieldElement* b, size_t count);
    static void batch_mul(BFieldElement* dst, const BFieldElement* a, const BFieldElement* b, size_t count);
    static void batch_sub(BFieldElement* dst, const BFieldElement* a, const BFieldElement* b, size_t count);
    
    // Primitive root of unity for NTT
    static BFieldElement primitive_root_of_unity(uint32_t log2_order);
    
    // String representation
    std::string to_string() const;
    
    friend std::ostream& operator<<(std::ostream& os, const BFieldElement& elem);

private:
    uint64_t value_;
    
    // Internal helper for modular reduction
    static uint64_t reduce(uint128_t value);
    
    // Montgomery reduction (for converting Montgomery representation to canonical)
    static uint64_t montyred(uint128_t value);
};

// Type alias for convenience
using BFE = BFieldElement;

} // namespace triton_vm

