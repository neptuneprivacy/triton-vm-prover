#ifndef FIELD_ARITHMETIC_CUH
#define FIELD_ARITHMETIC_CUH

#include <cstdint>

// Type aliases matching Rust
typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;  // GPU doesn't have native u128, we'll handle this

// ============================================================================
// BFieldElement - Base field arithmetic over F_p where p = 2^64 - 2^32 + 1
// ============================================================================

// The base field's prime: 2^64 - 2^32 + 1
#define BFIELD_PRIME 0xFFFFFFFF00000001ULL
#define BFIELD_R2    0xFFFFFFFE00000001ULL  // 2^128 mod P for Montgomery

// BFieldElement in Montgomery representation
struct BFieldElement {
    u64 value;

    __device__ __host__ BFieldElement() : value(0) {}
    __device__ __host__ BFieldElement(u64 v) : value(v) {}
};

// Montgomery reduction: reduce x (128-bit) mod P to 64-bit
__device__ __forceinline__ u64 bfield_montyred(u64 xl, u64 xh) {
    // Montgomery reduction algorithm
    // See: BFieldElement::montyred in Rust implementation

    u64 a, b, r;
    bool e, c;

    // a = xl + (xl << 32), with overflow flag e
    a = xl + (xl << 32);
    e = (a < xl);  // overflow if result < input

    // b = a - (a >> 32) - e
    b = a - (a >> 32) - (e ? 1 : 0);

    // r = xh - b, with carry flag c
    r = xh - b;
    c = (xh < b);  // carry if xh < b

    // Final correction
    return r - ((1 + ~BFIELD_PRIME) * (c ? 1 : 0));
}

// Convert raw u64 to Montgomery representation
__device__ __forceinline__ BFieldElement bfield_new(u64 value) {
    // Multiply by R2 and reduce
    // This is equivalent to Rust's BFieldElement::new()
    u64 hi, lo;

    // 64x64 -> 128 bit multiplication
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(value), "l"(BFIELD_R2));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(value), "l"(BFIELD_R2));

    BFieldElement result;
    result.value = bfield_montyred(lo, hi);
    return result;
}

// Create from raw Montgomery value (already in Montgomery form)
__device__ __forceinline__ BFieldElement bfield_from_raw(u64 raw_value) {
    return BFieldElement(raw_value);
}

// BField addition: a + b = a - (P - b)
__device__ __forceinline__ BFieldElement bfield_add(BFieldElement a, BFieldElement b) {
    // Compute a + b = a - (P - b)
    // This is the trick used in Rust implementation
    // See: https://github.com/Neptune-Crypto/twenty-first/pull/70

    u64 x1 = a.value - (BFIELD_PRIME - b.value);
    bool c1 = (a.value < (BFIELD_PRIME - b.value));  // overflow occurred

    BFieldElement result;
    if (c1) {
        // Overflow: add P back (wrapping_add in Rust)
        result.value = x1 + BFIELD_PRIME;
    } else {
        result.value = x1;
    }

    return result;
}

// BField subtraction
__device__ __forceinline__ BFieldElement bfield_sub(BFieldElement a, BFieldElement b) {
    // Match Rust implementation exactly:
    // let (x1, c1) = self.0.overflowing_sub(rhs.0);
    // Self(x1.wrapping_sub((1 + !Self::P) * c1 as u64))

    u64 x1 = a.value - b.value;  // This is wrapping subtraction in CUDA
    bool c1 = (a.value < b.value);  // Underflow flag

    BFieldElement result;
    // wrapping_sub in Rust is just regular subtraction with wrapping overflow (default in CUDA)
    result.value = x1 - ((1 + ~BFIELD_PRIME) * (c1 ? 1 : 0));
    return result;
}

// BField multiplication
__device__ __forceinline__ BFieldElement bfield_mul(BFieldElement a, BFieldElement b) {
    u64 hi, lo;

    // 64x64 -> 128 bit multiplication
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a.value), "l"(b.value));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a.value), "l"(b.value));

    BFieldElement result;
    result.value = bfield_montyred(lo, hi);
    return result;
}

// BField negation
__device__ __forceinline__ BFieldElement bfield_neg(BFieldElement a) {
    BFieldElement result;
    result.value = (a.value == 0) ? 0 : (BFIELD_PRIME - a.value);
    return result;
}

// BField constants
__device__ __forceinline__ BFieldElement bfield_zero() {
    return BFieldElement(0);
}

__device__ __forceinline__ BFieldElement bfield_one() {
    // ONE in Montgomery form
    return bfield_new(1);
}

// BField modular exponentiation: base^exp mod P
// Uses square-and-multiply algorithm
__device__ __forceinline__ BFieldElement bfield_pow_u32(BFieldElement base, u32 exp) {
    BFieldElement result = bfield_one();
    BFieldElement b = base;

    while (exp > 0) {
        if (exp & 1) {
            result = bfield_mul(result, b);
        }
        b = bfield_mul(b, b);  // Square
        exp >>= 1;
    }

    return result;
}

// BField modular inverse: 1/a mod P
// Uses Fermat's little theorem: a^(p-1) = 1 mod p  =>  a^(p-2) = a^(-1) mod p
// Since P = 2^64 - 2^32 + 1, we compute a^(P-2)
__device__ __forceinline__ BFieldElement bfield_inverse(BFieldElement a) {
    // P - 2 = 0xFFFFFFFF00000001 - 2 = 0xFFFFFFFEFFFFFFFF
    // We need to compute a^(P-2) using square-and-multiply
    // This is expensive but only done once per thread for zerofiers

    BFieldElement result = bfield_one();
    BFieldElement base = a;

    // P - 2 = 0xFFFFFFFEFFFFFFFF in binary
    // Process from LSB to MSB
    u64 exp_lo = 0xFFFFFFFFULL;  // Lower 32 bits: all 1s
    u64 exp_hi = 0xFFFFFFFEULL;  // Upper 32 bits: 0xFFFFFFFE

    // Process lower 32 bits
    for (int i = 0; i < 32; i++) {
        if (exp_lo & (1ULL << i)) {
            result = bfield_mul(result, base);
        }
        base = bfield_mul(base, base);
    }

    // Process upper 32 bits
    for (int i = 0; i < 32; i++) {
        if (exp_hi & (1ULL << i)) {
            result = bfield_mul(result, base);
        }
        base = bfield_mul(base, base);
    }

    return result;
}

// ============================================================================
// XFieldElement - Extension field F_{p^3} using irreducible polynomial x^3 - x + 1
// ============================================================================

struct XFieldElement {
    BFieldElement c0, c1, c2;  // coefficients [c0, c1, c2]

    __device__ __host__ XFieldElement() : c0(0), c1(0), c2(0) {}
    __device__ __host__ XFieldElement(BFieldElement _c0, BFieldElement _c1, BFieldElement _c2)
        : c0(_c0), c1(_c1), c2(_c2) {}
};

// XField addition
__device__ __forceinline__ XFieldElement xfield_add(XFieldElement a, XFieldElement b) {
    XFieldElement result;
    result.c0 = bfield_add(a.c0, b.c0);
    result.c1 = bfield_add(a.c1, b.c1);
    result.c2 = bfield_add(a.c2, b.c2);
    return result;
}

// XField subtraction
__device__ __forceinline__ XFieldElement xfield_sub(XFieldElement a, XFieldElement b) {
    XFieldElement result;
    result.c0 = bfield_sub(a.c0, b.c0);
    result.c1 = bfield_sub(a.c1, b.c1);
    result.c2 = bfield_sub(a.c2, b.c2);
    return result;
}

// XField multiplication: (ax^2 + bx + c) * (dx^2 + ex + f) mod (x^3 - x + 1)
__device__ __forceinline__ XFieldElement xfield_mul(XFieldElement lhs, XFieldElement rhs) {
    // Using the reduction formula from Rust implementation:
    // r0 = c * f - a * e - b * d
    // r1 = b * f + c * e - a * d + a * e + b * d
    // r2 = a * f + b * e + c * d + a * d

    BFieldElement c = lhs.c0;
    BFieldElement b = lhs.c1;
    BFieldElement a = lhs.c2;

    BFieldElement f = rhs.c0;
    BFieldElement e = rhs.c1;
    BFieldElement d = rhs.c2;

    // Compute products
    BFieldElement cf = bfield_mul(c, f);
    BFieldElement ae = bfield_mul(a, e);
    BFieldElement bd = bfield_mul(b, d);
    BFieldElement bf = bfield_mul(b, f);
    BFieldElement ce = bfield_mul(c, e);
    BFieldElement ad = bfield_mul(a, d);
    BFieldElement af = bfield_mul(a, f);
    BFieldElement be = bfield_mul(b, e);
    BFieldElement cd = bfield_mul(c, d);

    // r0 = c * f - a * e - b * d
    BFieldElement r0 = bfield_sub(bfield_sub(cf, ae), bd);

    // r1 = b * f + c * e - a * d + a * e + b * d
    BFieldElement r1 = bfield_add(bfield_add(bf, ce), bfield_sub(bfield_add(ae, bd), ad));

    // r2 = a * f + b * e + c * d + a * d
    BFieldElement r2 = bfield_add(bfield_add(bfield_add(af, be), cd), ad);

    return XFieldElement(r0, r1, r2);
}

// Scalar multiplication: BField * XField
__device__ __forceinline__ XFieldElement xfield_scalar_mul(BFieldElement scalar, XFieldElement x) {
    XFieldElement result;
    result.c0 = bfield_mul(scalar, x.c0);
    result.c1 = bfield_mul(scalar, x.c1);
    result.c2 = bfield_mul(scalar, x.c2);
    return result;
}

// XField * BField (same as scalar mul)
__device__ __forceinline__ XFieldElement xfield_mul_bfield(XFieldElement x, BFieldElement scalar) {
    return xfield_scalar_mul(scalar, x);
}

// XField + BField (add to constant term)
__device__ __forceinline__ XFieldElement xfield_add_bfield(XFieldElement x, BFieldElement b) {
    XFieldElement result = x;
    result.c0 = bfield_add(result.c0, b);
    return result;
}

// BField + XField
__device__ __forceinline__ XFieldElement bfield_add_xfield(BFieldElement b, XFieldElement x) {
    return xfield_add_bfield(x, b);
}

// XField constants
__device__ __forceinline__ XFieldElement xfield_zero() {
    return XFieldElement(bfield_zero(), bfield_zero(), bfield_zero());
}

__device__ __forceinline__ XFieldElement xfield_one() {
    return XFieldElement(bfield_one(), bfield_zero(), bfield_zero());
}

// XField negation
__device__ __forceinline__ XFieldElement xfield_neg(XFieldElement a) {
    return XFieldElement(bfield_neg(a.c0), bfield_neg(a.c1), bfield_neg(a.c2));
}






// Field characteristic
#define P 0xFFFFFFFF00000001ULL
// R^2 mod P for Montgomery conversion
#define R2 0xFFFFFFFE00000001ULL



__device__ __forceinline__ u64 montyred(u128 x) {
    u64 xl = (u64)x;
    u64 xh = (u64)(x >> 64);
    u64 a = xl + (xl << 32);
    u64 b = a - (a >> 32) - (a < xl);
    u64 r = xh - b;
    r += (xh < b) ? P : 0;
    return r;
}

__device__ __forceinline__ u64 bfe_add(u64 a, u64 b) {
    u64 sum = a + b;
    u64 cond = (u64)((sum < a) || (sum >= P));
    sum -= cond * P;
    return sum;
}

__device__ __forceinline__ u64 bfe_sub(u64 a, u64 b) {
    u64 res = a - b;
    u64 cond = (u64)(a < b);
    res += cond * P;
    return res;
}

__device__ __forceinline__ u64 bfe_neg(u64 a) {
    if (a == 0) return 0;
    return P - a;
}

__device__ __forceinline__ u64 bfe_mul(u64 a, u64 b) {
    u128 prod = (u128)a * b;
    return montyred(prod);
}

__device__ __forceinline__ u64 to_montgomery(u64 a) {
    return bfe_mul(a, R2);
}

__device__ __forceinline__ u64 new_bfe(u64 value) {
    return to_montgomery(value);
}

#endif // FIELD_ARITHMETIC_CUH
