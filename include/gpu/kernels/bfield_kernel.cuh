#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Goldilocks prime: p = 2^64 - 2^32 + 1
constexpr uint64_t GOLDILOCKS_P = 18446744069414584321ULL;
constexpr uint64_t GOLDILOCKS_EPSILON = 0xFFFFFFFFULL;  // 2^32 - 1

// ============================================================================
// Device Functions (inline implementations for use in other kernels)
// ============================================================================

/**
 * Modular addition: (a + b) mod p
 */
__device__ __forceinline__ uint64_t bfield_add_impl(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    // Check for overflow: if sum < a, we wrapped around 2^64
    // Also need to reduce if sum >= p
    uint64_t carry = (sum < a) ? 1 : 0;
    
    if (carry || sum >= GOLDILOCKS_P) {
        // sum mod p = sum - p + carry * (2^64 mod p)
        // 2^64 mod p = 2^32 - 1 = GOLDILOCKS_EPSILON
        // But simpler: just subtract p if sum >= p, or add epsilon if we wrapped
        if (carry) {
            // We wrapped around 2^64, so we need to add 2^64 mod p = epsilon
            // sum = (a + b) mod 2^64, actual value = a + b = sum + 2^64
            // (sum + 2^64) mod p = sum + epsilon (mod p)
            sum += GOLDILOCKS_EPSILON;
            if (sum >= GOLDILOCKS_P) sum -= GOLDILOCKS_P;
        } else {
            sum -= GOLDILOCKS_P;
        }
    }
    return sum;
}

/**
 * Modular subtraction: (a - b) mod p
 */
__device__ __forceinline__ uint64_t bfield_sub_impl(uint64_t a, uint64_t b) {
    if (a >= b) {
        uint64_t diff = a - b;
        return (diff >= GOLDILOCKS_P) ? diff - GOLDILOCKS_P : diff;
    } else {
        // a < b, so we need to add p to get positive result
        // (a - b + p) mod p
        // Since a < b, a - b + p = p - (b - a)
        return GOLDILOCKS_P - (b - a);
    }
}

/**
 * Modular negation: -a mod p = p - a (if a != 0), 0 (if a == 0)
 */
__device__ __forceinline__ uint64_t bfield_neg_impl(uint64_t a) {
    return (a == 0) ? 0 : GOLDILOCKS_P - a;
}

/**
 * Goldilocks reduction: reduce 128-bit value to 64-bit
 * Uses the identity: 2^64 ≡ 2^32 - 1 (mod p)
 */
__device__ __forceinline__ uint64_t goldilocks_reduce(uint64_t lo, uint64_t hi) {
    // result = lo + hi * (2^64 mod p) = lo + hi * epsilon
    // epsilon = 2^32 - 1
    
    // hi * epsilon = hi * (2^32 - 1) = (hi << 32) - hi
    uint64_t hi_shifted = hi << 32;
    uint64_t carry_from_shift = hi >> 32;
    
    uint64_t prod_lo = hi_shifted - hi;
    bool borrow = (hi_shifted < hi);
    uint64_t prod_hi = carry_from_shift - (borrow ? 1 : 0);
    
    // Now add lo + (prod_hi, prod_lo)
    uint64_t sum_lo = lo + prod_lo;
    bool carry1 = (sum_lo < lo);
    uint64_t sum_hi = prod_hi + (carry1 ? 1 : 0);
    
    // Reduce sum_hi * 2^64 again
    uint64_t final_add = (sum_hi << 32) - sum_hi;
    uint64_t result = sum_lo + final_add;
    bool final_carry = (result < sum_lo);
    
    // One more reduction if needed
    if (final_carry) {
        result += GOLDILOCKS_EPSILON;
    }
    
    // Final modular reduction
    if (result >= GOLDILOCKS_P) {
        result -= GOLDILOCKS_P;
    }
    
    return result;
}

/**
 * Modular multiplication: (a * b) mod p using 128-bit intermediate
 */
__device__ __forceinline__ uint64_t bfield_mul_impl(uint64_t a, uint64_t b) {
    // Use PTX for 128-bit multiplication
    uint64_t lo, hi;
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b));
    
    return goldilocks_reduce(lo, hi);
}

/**
 * Modular exponentiation: a^exp mod p using square-and-multiply
 */
__device__ __forceinline__ uint64_t bfield_pow_impl(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = bfield_mul_impl(result, base);
        }
        base = bfield_mul_impl(base, base);
        exp >>= 1;
    }
    return result;
}

/**
 * Modular inverse: a^{-1} mod p using Fermat's little theorem
 * a^{-1} = a^{p-2} mod p
 */
__device__ __forceinline__ uint64_t bfield_inv_impl(uint64_t a) {
    return bfield_pow_impl(a, GOLDILOCKS_P - 2);
}

// ============================================================================
// Host Interface Functions
// ============================================================================

/**
 * Batch addition: out[i] = a[i] + b[i]
 */
void bfield_add_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Batch subtraction: out[i] = a[i] - b[i]
 */
void bfield_sub_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Batch multiplication: out[i] = a[i] * b[i]
 */
void bfield_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Batch negation: out[i] = -a[i]
 */
void bfield_neg_gpu(
    const uint64_t* d_a,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Batch inverse: out[i] = a[i]^{-1}
 */
void bfield_inv_gpu(
    const uint64_t* d_a,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Batch power: out[i] = a[i]^exp
 */
void bfield_pow_gpu(
    const uint64_t* d_a,
    uint64_t exp,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * Scalar multiplication: out[i] = a[i] * scalar
 */
void bfield_scalar_mul_gpu(
    const uint64_t* d_a,
    uint64_t scalar,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
