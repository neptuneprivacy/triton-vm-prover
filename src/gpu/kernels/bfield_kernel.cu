/**
 * BFieldElement CUDA Kernel Implementation
 * 
 * Goldilocks field arithmetic on GPU.
 * Prime: p = 2^64 - 2^32 + 1 = 18446744069414584321
 * 
 * This is the foundation for all GPU proof generation operations.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Constants
// ============================================================================

// Goldilocks prime: p = 2^64 - 2^32 + 1
__constant__ uint64_t GOLDILOCKS_P = 18446744069414584321ULL;

// Epsilon = 2^32 - 1 (since p = 2^64 - epsilon where epsilon = 2^32 - 1)
__constant__ uint64_t EPSILON = 4294967295ULL;

// ============================================================================
// Core Field Arithmetic (Device Functions)
// ============================================================================

/**
 * Modular addition: (a + b) mod p
 * 
 * Since a, b < p and p > 2^63, we need careful overflow handling.
 */
__device__ __forceinline__ uint64_t bfield_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    // Check for overflow or >= p
    // If sum < a, overflow occurred (sum wrapped around)
    // If sum >= p, need to subtract p
    if (sum < a || sum >= GOLDILOCKS_P) {
        sum -= GOLDILOCKS_P;
    }
    return sum;
}

/**
 * Modular subtraction: (a - b) mod p
 */
__device__ __forceinline__ uint64_t bfield_sub(uint64_t a, uint64_t b) {
    uint64_t diff = a - b;
    // If a < b, underflow occurred, add p
    if (a < b) {
        diff += GOLDILOCKS_P;
    }
    return diff;
}

/**
 * Modular negation: -a mod p
 */
__device__ __forceinline__ uint64_t bfield_neg(uint64_t a) {
    return (a == 0) ? 0 : (GOLDILOCKS_P - a);
}

/**
 * Modular multiplication: (a * b) mod p
 * 
 * Uses 128-bit intermediate with Goldilocks-specific reduction.
 * For p = 2^64 - 2^32 + 1, we have: 2^64 ≡ 2^32 - 1 (mod p)
 */
__device__ __forceinline__ uint64_t bfield_mul(uint64_t a, uint64_t b) {
    // Compute 128-bit product using PTX
    uint64_t lo, hi;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    
    // Reduce: result = lo + hi * 2^64
    // Since 2^64 ≡ 2^32 - 1 (mod p), we have:
    // result ≡ lo + hi * (2^32 - 1) (mod p)
    //        ≡ lo + hi * 2^32 - hi (mod p)
    
    // Step 1: Compute hi * 2^32 = (hi << 32)
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;  // Lower 32 bits of hi
    uint64_t hi_hi = hi >> 32;             // Upper 32 bits of hi
    
    // hi * 2^32 = (hi_hi * 2^64) + (hi_lo * 2^32)
    // We need to reduce hi_hi * 2^64 first
    // hi_hi * 2^64 ≡ hi_hi * (2^32 - 1) (mod p)
    
    uint64_t t1 = hi_lo << 32;  // hi_lo * 2^32 (fits in 64 bits)
    
    // Now compute: lo + t1 - hi + hi_hi * (2^32 - 1)
    // = lo + t1 - hi + hi_hi * 2^32 - hi_hi
    // = lo + t1 - hi + (hi_hi << 32) - hi_hi
    
    uint64_t t2 = hi_hi << 32;  // hi_hi * 2^32
    
    // Combine: result = lo + t1 + t2 - hi - hi_hi
    // This can overflow, so we do it carefully
    
    // First: r = lo - hi (may underflow)
    uint64_t r = lo - hi;
    bool underflow1 = (lo < hi);
    
    // Add t1
    uint64_t r2 = r + t1;
    bool overflow1 = (r2 < r);
    r = r2;
    
    // Add t2
    r2 = r + t2;
    bool overflow2 = (r2 < r);
    r = r2;
    
    // Subtract hi_hi
    uint64_t r3 = r - hi_hi;
    bool underflow2 = (r < hi_hi);
    r = r3;
    
    // Apply corrections
    // Each overflow means we exceeded 2^64, add epsilon (since 2^64 = p + epsilon)
    // Wait, 2^64 = p + (2^32 - 1), so overflowing by 2^64 means subtracting p and adding epsilon
    // Actually: if we overflow, result wrapped around, so we need to add (2^64 mod p) = 2^32 - 1
    
    // Each underflow means we went negative, add p
    
    int64_t correction = 0;
    if (underflow1) correction -= 1;  // Need to add p
    if (overflow1) correction += 1;   // Need to subtract p (add epsilon, sub p = sub (p - epsilon) = sub 1... complex)
    if (overflow2) correction += 1;
    if (underflow2) correction -= 1;
    
    // Apply correction: each +1 means we overflowed (sub p), each -1 means underflowed (add p)
    // Simpler approach: just do final reduction
    
    // For Goldilocks, a cleaner reduction:
    // If we have overflows/underflows, the result is off by multiples of 2^64
    // 2^64 mod p = epsilon = 2^32 - 1
    // So add/subtract epsilon for each overflow/underflow
    
    // Reset and use cleaner algorithm
    // Actually, let me use the standard Goldilocks reduction algorithm
    
    // -------------------------------------------------------------------------
    // Cleaner Goldilocks reduction using the identity:
    // (lo, hi) where value = lo + hi * 2^64
    // 2^64 ≡ -epsilon ≡ -(2^32 - 1) ≡ -2^32 + 1 (mod p)
    // Wait, p = 2^64 - 2^32 + 1, so 2^64 = p + 2^32 - 1 = p + epsilon
    // Thus 2^64 ≡ epsilon (mod p) where epsilon = 2^32 - 1
    // -------------------------------------------------------------------------
    
    // result = lo + hi * epsilon (mod p)
    // hi * epsilon = hi * (2^32 - 1) = (hi << 32) - hi
    
    // Compute hi * epsilon where epsilon = 2^32 - 1
    // = hi * 2^32 - hi
    uint64_t hi_shifted = hi << 32;
    uint64_t carry_from_shift = hi >> 32;  // This is the overflow from hi << 32
    
    // hi * epsilon = hi_shifted - hi, but hi_shifted may have lost upper bits
    // Full computation:
    // hi * (2^32 - 1) = hi * 2^32 - hi
    // hi * 2^32 as 128-bit: (hi >> 32, hi << 32)
    // So hi * epsilon as 128-bit: 
    //   lo part: (hi << 32) - hi
    //   hi part: (hi >> 32) - borrow
    
    uint64_t prod_lo = hi_shifted - hi;
    bool borrow = (hi_shifted < hi);
    uint64_t prod_hi = carry_from_shift - (borrow ? 1 : 0);
    
    // Now add lo + (prod_hi, prod_lo)
    // This is: lo + prod_lo + prod_hi * 2^64
    // = lo + prod_lo + prod_hi * epsilon (applying reduction again)
    
    uint64_t sum_lo = lo + prod_lo;
    bool carry1 = (sum_lo < lo);
    uint64_t sum_hi = prod_hi + (carry1 ? 1 : 0);
    
    // Now reduce sum_hi * 2^64 again
    // sum_hi * epsilon = sum_hi * (2^32 - 1) = (sum_hi << 32) - sum_hi
    // Since sum_hi is small (at most a few bits), this fits in 64 bits
    
    uint64_t final_add = (sum_hi << 32) - sum_hi;
    uint64_t result = sum_lo + final_add;
    
    // Check for overflow in final addition
    if (result < sum_lo || result >= GOLDILOCKS_P) {
        result -= GOLDILOCKS_P;
    }
    
    // Final reduction if still >= p
    if (result >= GOLDILOCKS_P) {
        result -= GOLDILOCKS_P;
    }
    
    return result;
}

/**
 * Modular exponentiation: a^exp mod p
 * Uses binary exponentiation (square-and-multiply)
 */
__device__ __forceinline__ uint64_t bfield_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    
    while (exp > 0) {
        if (exp & 1) {
            result = bfield_mul(result, base);
        }
        base = bfield_mul(base, base);
        exp >>= 1;
    }
    
    return result;
}

/**
 * Modular inverse: a^{-1} mod p
 * Uses Fermat's little theorem: a^{-1} = a^{p-2} mod p
 */
__device__ __forceinline__ uint64_t bfield_inv(uint64_t a) {
    return bfield_pow(a, GOLDILOCKS_P - 2);
}

// ============================================================================
// Batch Operation Kernels
// ============================================================================

/**
 * Batch addition: out[i] = a[i] + b[i]
 */
__global__ void bfield_add_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bfield_add(a[idx], b[idx]);
    }
}

/**
 * Batch subtraction: out[i] = a[i] - b[i]
 */
__global__ void bfield_sub_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bfield_sub(a[idx], b[idx]);
    }
}

/**
 * Batch multiplication: out[i] = a[i] * b[i]
 */
__global__ void bfield_mul_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bfield_mul(a[idx], b[idx]);
    }
}

/**
 * Batch negation: out[i] = -a[i]
 */
__global__ void bfield_neg_kernel(
    const uint64_t* a,
    uint64_t* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bfield_neg(a[idx]);
    }
}

/**
 * Batch inverse: out[i] = a[i]^{-1}
 * Note: This is slow for large batches. Consider batch inversion algorithm.
 */
__global__ void bfield_inv_kernel(
    const uint64_t* a,
    uint64_t* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bfield_inv(a[idx]);
    }
}

/**
 * Batch power: out[i] = a[i]^exp
 */
__global__ void bfield_pow_kernel(
    const uint64_t* a,
    uint64_t exp,
    uint64_t* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bfield_pow(a[idx], exp);
    }
}

/**
 * Scalar multiplication: out[i] = a[i] * scalar
 */
__global__ void bfield_scalar_mul_kernel(
    const uint64_t* a,
    uint64_t scalar,
    uint64_t* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bfield_mul(a[idx], scalar);
    }
}

// ============================================================================
// Host Interface Functions
// ============================================================================

void bfield_add_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    bfield_add_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_out, n);
}

void bfield_sub_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    bfield_sub_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_out, n);
}

void bfield_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    bfield_mul_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_out, n);
}

void bfield_neg_gpu(
    const uint64_t* d_a,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    bfield_neg_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_out, n);
}

void bfield_inv_gpu(
    const uint64_t* d_a,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    bfield_inv_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_out, n);
}

void bfield_pow_gpu(
    const uint64_t* d_a,
    uint64_t exp,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    bfield_pow_kernel<<<grid_size, block_size, 0, stream>>>(d_a, exp, d_out, n);
}

void bfield_scalar_mul_gpu(
    const uint64_t* d_a,
    uint64_t scalar,
    uint64_t* d_out,
    size_t n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    bfield_scalar_mul_kernel<<<grid_size, block_size, 0, stream>>>(d_a, scalar, d_out, n);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

