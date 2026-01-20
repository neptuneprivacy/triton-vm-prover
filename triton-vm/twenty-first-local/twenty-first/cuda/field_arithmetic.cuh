#ifndef FIELD_ARITHMETIC_CUH
#define FIELD_ARITHMETIC_CUH

#include <cstdint>

// Type aliases matching Rust
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned __int128 u128;  // GPU doesn't have native u128, we'll handle this
typedef unsigned char u8; 

// ============================================================================
// BFieldElement - Base field arithmetic over F_p where p = 2^64 - 2^32 + 1
// ============================================================================

// The base field's prime: 2^64 - 2^32 + 1
#define BFIELD_PRIME 0xFFFFFFFF00000001ULL
#define BFIELD_R2    0xFFFFFFFE00000001ULL  // 2^128 mod BFIELD_PRIME for Montgomery

// BFieldElement in Montgomery representation
struct BFieldElement {
    u64 value;

    __device__ __host__ BFieldElement() : value(0) {}
    __device__ __host__ BFieldElement(u64 v) : value(v) {}
};

// Montgomery reduction: reduce x (128-bit) mod BFIELD_PRIME to 64-bit
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
    // Multiply by BFIELD_R2 and reduce
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

// BField addition: a + b = a - (BFIELD_PRIME - b)
__device__ __forceinline__ BFieldElement bfield_add(BFieldElement a, BFieldElement b) {
    // Compute a + b = a - (BFIELD_PRIME - b)
    // This is the trick used in Rust implementation
    // See: https://github.com/Neptune-Crypto/twenty-first/pull/70

    u64 x1 = a.value - (BFIELD_PRIME - b.value);
    bool c1 = (a.value < (BFIELD_PRIME - b.value));  // overflow occurred

    BFieldElement result;
    if (c1) {
        // Overflow: add BFIELD_PRIME back (wrapping_add in Rust)
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
    // Self(x1.wrapping_sub((1 + !Self::BFIELD_PRIME) * c1 as u64))

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

// BField modular exponentiation: base^exp mod BFIELD_PRIME
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

// BField modular inverse: 1/a mod BFIELD_PRIME
// Uses Fermat's little theorem: a^(p-1) = 1 mod p  =>  a^(p-2) = a^(-1) mod p
// Since BFIELD_PRIME = 2^64 - 2^32 + 1, we compute a^(BFIELD_PRIME-2)
__device__ __forceinline__ BFieldElement bfield_inverse(BFieldElement a) {
    // BFIELD_PRIME - 2 = 0xFFFFFFFF00000001 - 2 = 0xFFFFFFFEFFFFFFFF
    // We need to compute a^(BFIELD_PRIME-2) using square-and-multiply
    // This is expensive but only done once per thread for zerofiers

    BFieldElement result = bfield_one();
    BFieldElement base = a;

    // BFIELD_PRIME - 2 = 0xFFFFFFFEFFFFFFFF in binary
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






















__device__ __forceinline__ u64 montyred(u128 x)
{
    unsigned long long xl = (unsigned long long)x;
    unsigned long long xh = (unsigned long long)(x >> 64);

    unsigned long long result;
    asm (
        "{\n\t"
        ".reg .u64 a, b, r, mask;\n\t"
        ".reg .pred p;\n\t"
        "shl.b64 a, %1, 32;\n\t"
        "add.u64 a, a, %1;\n\t"
        "shr.u64 b, a, 32;\n\t"
        "sub.u64 b, a, b;\n\t"
        "setp.lo.u64 p, a, %1;\n\t"
        "selp.u64 mask, 1, 0, p;\n\t"
        "sub.u64 b, b, mask;\n\t"
        "sub.u64 r, %2, b;\n\t"
        "setp.lo.u64 p, %2, b;\n\t"
        "mov.u64 mask, 0xffffffff00000001;\n\t"
        "selp.u64 mask, mask, 0, p;\n\t"
        "add.u64 %0, r, mask;\n\t"
        "}"
        : "=l"(result) 
        : "l"(xl), "l"(xh)
    );

    return result;
}

/// Addition in the field
__device__ __forceinline__ u64 bfe_add(u64 a, u64 b) {
    u64 sum = a + b;
    // If an overflow occurs (sum < a) or sum >= BFIELD_PRIME, subtract BFIELD_PRIME
    u64 cond = (u64)((sum < a) || (sum >= BFIELD_PRIME));
    sum -= cond * BFIELD_PRIME;
    return sum;
}

/// Subtraction in the field
__device__ __forceinline__ u64 bfe_sub(u64 a, u64 b) {
    u64 res = a - b;
    // If a < b then there was an underflow, so add BFIELD_PRIME
    u64 cond = (u64)(a < b);
    res += cond * BFIELD_PRIME;
    return res;
}

/// Multiplication in the field (Montgomery)
__device__ __forceinline__ u64 bfe_mul(u64 a, u64 b) {
    u128 prod = (u128)a * b;
    return montyred(prod);
}

/// Convert from canonical representation to Montgomery form
__device__ __forceinline__ u64 to_montgomery(u64 a) {
    return bfe_mul(a, BFIELD_R2);
}

/// Create a new BFieldElement (convert to Montgomery form)
__device__ __forceinline__ u64 new_bfe(u64 value) {
    return to_montgomery(value);
}

//============================================================================
// Bit Reversal
//============================================================================

/// Bit-reverse a 32-bit index
/// Uses CUDA's built-in __brev for hardware acceleration
__device__ __forceinline__ u32 bitreverse(u32 n, u32 log2_len) {
    return __brev(n) >> (32 - log2_len);
}


/// Fast exponentiation: compute base^exp using binary exponentiation
__device__ u64 bfe_pow(u64 base, u64 exp) {
    u64 result = to_montgomery(1);
    u64 b = base;

    while (exp > 0) {
        if (exp & 1) {
            result = bfe_mul(result, b);
        }
        b = bfe_mul(b, b);
        exp >>= 1;
    }

    return result;
}


/// Convert from Montgomery to canonical form
__device__ __forceinline__ u64 from_montgomery(u64 a) {
    return montyred((u128)a);
}

__device__ __forceinline__ u64 bfe_square(u64 a) {
    u128 prod = (u128)a * a;
    return montyred(prod);
}







struct XFieldElementArr {
    u64 coefficients[3];
};


/// Multiply XFieldElementArr by a BFieldElement scalar
/// xfe *= bfe (scalar multiplication)
__device__ __forceinline__ void xfe_mul_scalar(XFieldElementArr& a, u64 b) {
    u128 prod0 = (u128)a.coefficients[0] * b;
    u128 prod1 = (u128)a.coefficients[1] * b;
    u128 prod2 = (u128)a.coefficients[2] * b;
    a.coefficients[0] = montyred(prod0);
    a.coefficients[1] = montyred(prod1);
    a.coefficients[2] = montyred(prod2);
}

__device__ __forceinline__ XFieldElementArr xfe_mul_scalar_ret(XFieldElementArr a, u64 b) {
    XFieldElementArr c;
    c.coefficients[0] = bfe_mul(a.coefficients[0], b);
    c.coefficients[1] = bfe_mul(a.coefficients[1], b);
    c.coefficients[2] = bfe_mul(a.coefficients[2], b);
    return c;
}


/// Add two XFieldElements
__device__ __forceinline__ XFieldElementArr xfe_add(XFieldElementArr a, XFieldElementArr b) {
    XFieldElementArr c;
    c.coefficients[0] = bfe_add(a.coefficients[0], b.coefficients[0]);
    c.coefficients[1] = bfe_add(a.coefficients[1], b.coefficients[1]);
    c.coefficients[2] = bfe_add(a.coefficients[2], b.coefficients[2]);
    return c;
}

/// Subtract two XFieldElements
__device__ __forceinline__ XFieldElementArr xfe_sub(XFieldElementArr a, XFieldElementArr b) {
    XFieldElementArr c;
    c.coefficients[0] = bfe_sub(a.coefficients[0], b.coefficients[0]);
    c.coefficients[1] = bfe_sub(a.coefficients[1], b.coefficients[1]);
    c.coefficients[2] = bfe_sub(a.coefficients[2], b.coefficients[2]);
    return c;
}







__device__ __forceinline__ XFieldElementArr xfe_zero() {
    XFieldElementArr zero;
    zero.coefficients[0] = 0;
    zero.coefficients[1] = 0;
    zero.coefficients[2] = 0;
    return zero;
}

__device__ __forceinline__ XFieldElementArr xfe_one() {
    XFieldElementArr one;
    one.coefficients[0] = to_montgomery(1);
    one.coefficients[1] = 0;
    one.coefficients[2] = 0;
    return one;
}


/// Full XFieldElementArr multiplication using irreducible polynomial x^3 - x + 1
/// Matches the Rust implementation exactly
__device__ __forceinline__ XFieldElementArr xfe_mul(XFieldElementArr a, XFieldElementArr b) {
    // Coefficients are stored as [c, b, a] for polynomial ax^2 + bx + c
    u64 c = a.coefficients[0];
    u64 b_coef = a.coefficients[1];
    u64 a_coef = a.coefficients[2];
    u64 f = b.coefficients[0];
    u64 e = b.coefficients[1];
    u64 d = b.coefficients[2];

    // Rust implementation formula (from twenty-first/src/math/x_field_element.rs:511-513):
    // r0 = c * f - a * e - b * d
    // r1 = b * f + c * e - a * d + a * e + b * d
    // r2 = a * f + b * e + c * d + a * d

    u64 cf = bfe_mul(c, f);
    u64 ae = bfe_mul(a_coef, e);
    u64 bd = bfe_mul(b_coef, d);

    u64 r0 = bfe_sub(bfe_sub(cf, ae), bd);

    u64 bf = bfe_mul(b_coef, f);
    u64 ce = bfe_mul(c, e);
    u64 ad = bfe_mul(a_coef, d);

    u64 r1 = bfe_add(bfe_add(bf, ce), bfe_add(ae, bd));
    r1 = bfe_sub(r1, ad);

    u64 af = bfe_mul(a_coef, f);
    u64 be = bfe_mul(b_coef, e);
    u64 cd = bfe_mul(c, d);

    u64 r2 = bfe_add(bfe_add(bfe_add(af, be), cd), ad);

    XFieldElementArr result;
    result.coefficients[0] = r0;
    result.coefficients[1] = r1;
    result.coefficients[2] = r2;
    return result;
}


#endif // FIELD_ARITHMETIC_CUH
