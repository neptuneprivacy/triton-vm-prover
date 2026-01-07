#include "types/b_field_element.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace triton_vm {

// Fast Goldilocks reduction for p = 2^64 - 2^32 + 1
// The 128-bit modulo is actually well-optimized by GCC/Clang for this specific modulus
// We use the simple approach which the compiler can optimize
uint64_t BFieldElement::reduce(uint128_t value) {
    return static_cast<uint64_t>(value % MODULUS);
}

uint64_t BFieldElement::montyred(uint128_t value) {
    // Montgomery reduction - converts from Montgomery representation to canonical
    // Implementation matches Rust's montyred function and tip5xx library
    // Extract the low and high 64 bits
    uint64_t xl = static_cast<uint64_t>(value);
    uint64_t xh = static_cast<uint64_t>(value >> 64);
    
    // Manual overflow detection for xl + (xl << 32)
    uint64_t shifted = xl << 32;
    uint64_t a = xl + shifted;
    bool e = (a < xl) || (a < shifted);  // Overflow occurred if result is less than either operand
    
    // Wrapping subtraction operations
    uint64_t b = a - (a >> 32);
    if (e) {
        b--;  // Subtract 1 if there was overflow
    }
    
    // Manual underflow detection for xh - b
    uint64_t r = xh - b;
    bool c = (xh < b);  // Underflow occurred if xh < b
    
    // Final correction: r.wrapping_sub((1 + !P) * c)
    return r - ((1ULL + ~MODULUS) * (c ? 1ULL : 0ULL));
}

BFieldElement BFieldElement::from_raw_u64(uint64_t montgomery_value) {
    // Rust's from_raw_u64 assumes the value is in Montgomery representation.
    // Since C++ uses canonical representation, we need to convert from Montgomery to canonical.
    // This is done using montyred: canonical = montyred(montgomery_value as u128)
    uint64_t canonical_value = montyred(static_cast<uint128_t>(montgomery_value));
    return BFieldElement(canonical_value);
}

BFieldElement BFieldElement::operator+(const BFieldElement& rhs) const {
    // Branchless addition: compute sum, then subtract MODULUS if overflow
    uint64_t sum = value_ + rhs.value_;
    // If sum overflowed or sum >= MODULUS, subtract MODULUS
    // Overflow happens when sum < value_ (wrapping)
    uint64_t overflow = static_cast<uint64_t>(sum < value_);
    uint64_t too_large = static_cast<uint64_t>(sum >= MODULUS);
    // Use conditional subtraction without branch
    sum -= MODULUS & (-(overflow | too_large));
    return BFieldElement(sum);
}

BFieldElement BFieldElement::operator-(const BFieldElement& rhs) const {
    // Branchless subtraction: compute diff, add MODULUS if underflow
    uint64_t diff = value_ - rhs.value_;
    // If value_ < rhs.value_, diff wrapped around (underflow)
    uint64_t underflow = static_cast<uint64_t>(value_ < rhs.value_);
    // Add MODULUS if underflow occurred
    diff += MODULUS & (-underflow);
    return BFieldElement(diff);
}

BFieldElement BFieldElement::operator*(const BFieldElement& rhs) const {
    uint128_t product = static_cast<uint128_t>(value_) * static_cast<uint128_t>(rhs.value_);
    return BFieldElement(reduce(product));
}

BFieldElement BFieldElement::operator-() const {
    if (value_ == 0) return *this;
    return BFieldElement(MODULUS - value_);
}

BFieldElement& BFieldElement::operator+=(const BFieldElement& rhs) {
    *this = *this + rhs;
    return *this;
}

BFieldElement& BFieldElement::operator-=(const BFieldElement& rhs) {
    *this = *this - rhs;
    return *this;
}

BFieldElement& BFieldElement::operator*=(const BFieldElement& rhs) {
    *this = *this * rhs;
    return *this;
}

BFieldElement& BFieldElement::operator/=(const BFieldElement& rhs) {
    *this = *this / rhs;
    return *this;
}

bool BFieldElement::operator==(const BFieldElement& rhs) const {
    return value_ == rhs.value_;
}

bool BFieldElement::operator!=(const BFieldElement& rhs) const {
    return value_ != rhs.value_;
}

bool BFieldElement::operator<(const BFieldElement& rhs) const {
    return value_ < rhs.value_;
}

bool BFieldElement::operator>(const BFieldElement& rhs) const {
    return value_ > rhs.value_;
}

bool BFieldElement::operator<=(const BFieldElement& rhs) const {
    return value_ <= rhs.value_;
}

bool BFieldElement::operator>=(const BFieldElement& rhs) const {
    return value_ >= rhs.value_;
}

BFieldElement BFieldElement::pow(uint64_t exp) const {
    BFieldElement result = BFieldElement::one();
    BFieldElement base = *this;
    
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    return result;
}

BFieldElement BFieldElement::inverse() const {
    if (value_ == 0) {
        throw std::domain_error("Cannot invert zero");
    }
    
    // Use Fermat's little theorem: a^(-1) = a^(p-2) mod p
    return pow(MODULUS - 2);
}

BFieldElement BFieldElement::operator/(const BFieldElement& rhs) const {
    return *this * rhs.inverse();
}

std::vector<BFieldElement> BFieldElement::batch_inversion(const std::vector<BFieldElement>& elements) {
    const size_t n = elements.size();
    std::vector<BFieldElement> inverses(n, BFieldElement::zero());
    if (n == 0) {
        return inverses;
    }

    std::vector<BFieldElement> prefix_products(n, BFieldElement::one());
    BFieldElement accumulator = BFieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        if (elements[i].is_zero()) {
            throw std::domain_error("batch_inversion encountered zero element");
        }
        prefix_products[i] = accumulator;
        accumulator *= elements[i];
    }

    BFieldElement inverse_acc = accumulator.inverse();
    for (size_t i = n; i-- > 0;) {
        inverses[i] = prefix_products[i] * inverse_acc;
        inverse_acc *= elements[i];
    }

    return inverses;
}

BFieldElement BFieldElement::primitive_root_of_unity(uint32_t log2_order) {
    // The multiplicative group has order p - 1 = 2^64 - 2^32
    // For NTT, we need roots of unity of order 2^k where k <= 32
    
    if (log2_order > 32) {
        throw std::invalid_argument("log2_order must be <= 32");
    }
    
    // Generator of the multiplicative group is 7
    // Root of unity of order 2^32 is g^((p-1) / 2^32) = g^(2^32 - 1)
    BFieldElement g(GENERATOR);
    uint64_t exp = (MODULUS - 1) >> log2_order;
    return g.pow(exp);
}

// SIMD batch operations - process 4 elements at a time using AVX2 when available
void BFieldElement::batch_add(BFieldElement* dst, const BFieldElement* a, const BFieldElement* b, size_t count) {
#if BFIELD_AVX2_ENABLED
    // Process 4 elements at a time with AVX2 (256-bit = 4 x 64-bit)
    size_t i = 0;
    const __m256i modulus = _mm256_set1_epi64x(MODULUS);
    
    for (; i + 4 <= count; i += 4) {
        // Load 4 elements each
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
        
        // Add (may overflow, but that's fine for uint64)
        __m256i sum = _mm256_add_epi64(va, vb);
        
        // Check if sum >= MODULUS or if overflow occurred (sum < va)
        // For Goldilocks, sum - MODULUS is correct when sum >= MODULUS
        __m256i diff = _mm256_sub_epi64(sum, modulus);
        
        // Use comparison to select: if sum >= modulus, use diff; else use sum
        // Since we can't directly compare unsigned in AVX2, use a workaround:
        // If sum overflowed or sum >= modulus, then diff is the correct result
        __m256i overflow = _mm256_cmpgt_epi64(va, sum);  // Check overflow
        __m256i too_large = _mm256_cmpgt_epi64(sum, _mm256_sub_epi64(modulus, _mm256_set1_epi64x(1)));  // sum > MODULUS-1
        __m256i need_reduce = _mm256_or_si256(overflow, too_large);
        
        // Branchless select: result = need_reduce ? diff : sum
        __m256i result = _mm256_blendv_epi8(sum, diff, need_reduce);
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), result);
    }
    
    // Handle remaining elements
    for (; i < count; ++i) {
        dst[i] = a[i] + b[i];
    }
#else
    // Fallback: scalar
    for (size_t i = 0; i < count; ++i) {
        dst[i] = a[i] + b[i];
    }
#endif
}

void BFieldElement::batch_sub(BFieldElement* dst, const BFieldElement* a, const BFieldElement* b, size_t count) {
#if BFIELD_AVX2_ENABLED
    size_t i = 0;
    const __m256i modulus = _mm256_set1_epi64x(MODULUS);
    
    for (; i + 4 <= count; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
        
        // Subtract
        __m256i diff = _mm256_sub_epi64(va, vb);
        
        // Check if underflow (a < b means diff > a after wrapping)
        __m256i underflow = _mm256_cmpgt_epi64(vb, va);
        
        // Add MODULUS if underflow
        __m256i result = _mm256_add_epi64(diff, _mm256_and_si256(modulus, underflow));
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), result);
    }
    
    for (; i < count; ++i) {
        dst[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = a[i] - b[i];
    }
#endif
}

void BFieldElement::batch_mul(BFieldElement* dst, const BFieldElement* a, const BFieldElement* b, size_t count) {
    // Multiplication is harder to vectorize due to the 128-bit intermediate
    // Use OpenMP for parallel scalar multiplication
    #pragma omp parallel for schedule(static) if(count > 256)
    for (size_t i = 0; i < count; ++i) {
        dst[i] = a[i] * b[i];
    }
}

std::string BFieldElement::to_string() const {
    return std::to_string(value_);
}

std::ostream& operator<<(std::ostream& os, const BFieldElement& elem) {
    return os << elem.value_;
}

} // namespace triton_vm

