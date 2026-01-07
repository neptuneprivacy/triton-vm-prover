#pragma once

/**
 * XFieldElement CUDA Kernel Declarations
 * 
 * Extension field F_p³ = F_p[x] / (x³ - x + 1)
 * Each XFieldElement is stored as 3 consecutive uint64_t values (c0, c1, c2)
 * representing the polynomial c0 + c1*X + c2*X²
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include "gpu/kernels/bfield_kernel.cuh"

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Device Functions (inline with definitions)
// ============================================================================

/**
 * XField addition: (a0,a1,a2) + (b0,b1,b2)
 */
__device__ __forceinline__ void xfield_add_impl(
    uint64_t a0, uint64_t a1, uint64_t a2,
    uint64_t b0, uint64_t b1, uint64_t b2,
    uint64_t& r0, uint64_t& r1, uint64_t& r2
) {
    r0 = bfield_add_impl(a0, b0);
    r1 = bfield_add_impl(a1, b1);
    r2 = bfield_add_impl(a2, b2);
}

/**
 * XField subtraction: (a0,a1,a2) - (b0,b1,b2)
 */
__device__ __forceinline__ void xfield_sub_impl(
    uint64_t a0, uint64_t a1, uint64_t a2,
    uint64_t b0, uint64_t b1, uint64_t b2,
    uint64_t& r0, uint64_t& r1, uint64_t& r2
) {
    r0 = bfield_sub_impl(a0, b0);
    r1 = bfield_sub_impl(a1, b1);
    r2 = bfield_sub_impl(a2, b2);
}

/**
 * XField negation: -(a0,a1,a2)
 */
__device__ __forceinline__ void xfield_neg_impl(
    uint64_t a0, uint64_t a1, uint64_t a2,
    uint64_t& r0, uint64_t& r1, uint64_t& r2
) {
    r0 = bfield_neg_impl(a0);
    r1 = bfield_neg_impl(a1);
    r2 = bfield_neg_impl(a2);
}

/**
 * XField multiplication: (a0,a1,a2) * (b0,b1,b2)
 * Coefficient order is (c0, c1, c2) for c0 + c1·x + c2·x^2.
 * Modulus is `x^3 - x + 1 = 0`, i.e. `x^3 = x - 1`.
 */
__device__ __forceinline__ void xfield_mul_impl(
    uint64_t a0, uint64_t a1, uint64_t a2,
    uint64_t b0, uint64_t b1, uint64_t b2,
    uint64_t& r0, uint64_t& r1, uint64_t& r2
) {
    // Karatsuba-style polynomial multiplication (degree-2 × degree-2) using 6 base muls,
    // then reduction using x^3 = x - 1 (since x^3 - x + 1 = 0).
    //
    // v0 = a0*b0
    // v1 = a1*b1
    // v2 = a2*b2
    // v3 = (a0+a1)(b0+b1) - v0 - v1 = a0*b1 + a1*b0
    // v4 = (a0+a2)(b0+b2) - v0 - v2 = a0*b2 + a2*b0
    // v5 = (a1+a2)(b1+b2) - v1 - v2 = a1*b2 + a2*b1
    //
    // c0 = v0
    // c1 = v3
    // c2 = v4 + v1
    // c3 = v5
    // c4 = v2
    const uint64_t v0 = bfield_mul_impl(a0, b0);
    const uint64_t v1 = bfield_mul_impl(a1, b1);
    const uint64_t v2 = bfield_mul_impl(a2, b2);

    const uint64_t a01 = bfield_add_impl(a0, a1);
    const uint64_t b01 = bfield_add_impl(b0, b1);
    const uint64_t v3 = bfield_sub_impl(bfield_sub_impl(bfield_mul_impl(a01, b01), v0), v1);

    const uint64_t a02 = bfield_add_impl(a0, a2);
    const uint64_t b02 = bfield_add_impl(b0, b2);
    const uint64_t v4 = bfield_sub_impl(bfield_sub_impl(bfield_mul_impl(a02, b02), v0), v2);

    const uint64_t a12 = bfield_add_impl(a1, a2);
    const uint64_t b12 = bfield_add_impl(b1, b2);
    const uint64_t v5 = bfield_sub_impl(bfield_sub_impl(bfield_mul_impl(a12, b12), v1), v2);

    const uint64_t c0 = v0;
    const uint64_t c1 = v3;
    const uint64_t c2 = bfield_add_impl(v4, v1);
    const uint64_t c3 = v5;
    const uint64_t c4 = v2;

    // Reduce:
    // c3*x^3 -> -c3 + c3*x
    // c4*x^4 -> -c4*x + c4*x^2
    r0 = bfield_sub_impl(c0, c3);
    r1 = bfield_sub_impl(bfield_add_impl(c1, c3), c4);
    r2 = bfield_add_impl(c2, c4);
}

/**
 * XField scalar multiplication: (a0,a1,a2) * s
 */
__device__ __forceinline__ void xfield_scalar_mul_impl(
    uint64_t a0, uint64_t a1, uint64_t a2,
    uint64_t s,
    uint64_t& r0, uint64_t& r1, uint64_t& r2
) {
    r0 = bfield_mul_impl(a0, s);
    r1 = bfield_mul_impl(a1, s);
    r2 = bfield_mul_impl(a2, s);
}

/**
 * XField inverse: (a,b,c)^{-1} for element a + b·x + c·x^2.
 * Uses an explicit adjugate/determinant formula specialized to x^3 = x - 1.
 */
__device__ __forceinline__ void xfield_inv_impl(
    uint64_t a, uint64_t b, uint64_t c,
    uint64_t& r0, uint64_t& r1, uint64_t& r2
) {
    // Precompute squares and cubes
    uint64_t a2 = bfield_mul_impl(a, a);
    uint64_t b2 = bfield_mul_impl(b, b);
    uint64_t c2 = bfield_mul_impl(c, c);
    uint64_t a3 = bfield_mul_impl(a2, a);
    uint64_t b3 = bfield_mul_impl(b2, b);
    uint64_t c3 = bfield_mul_impl(c2, c);

    // Products
    uint64_t ab = bfield_mul_impl(a, b);
    uint64_t ac = bfield_mul_impl(a, c);
    uint64_t bc = bfield_mul_impl(b, c);
    uint64_t abc = bfield_mul_impl(ab, c);

    // Determinant: det = a^3 - b^3 + c^3 + 3abc + 2a^2 c + a c^2 + b c^2 - a b^2
    uint64_t det = a3;
    det = bfield_sub_impl(det, b3);
    det = bfield_add_impl(det, c3);
    // + 3abc
    uint64_t _3abc = bfield_add_impl(abc, bfield_add_impl(abc, abc));
    det = bfield_add_impl(det, _3abc);
    // + 2a^2 c
    uint64_t a2c = bfield_mul_impl(a2, c);
    det = bfield_add_impl(det, bfield_add_impl(a2c, a2c));
    // + a c^2
    det = bfield_add_impl(det, bfield_mul_impl(ac, c));
    // + b c^2
    det = bfield_add_impl(det, bfield_mul_impl(bc, c));
    // - a b^2
    det = bfield_sub_impl(det, bfield_mul_impl(ab, b));

    uint64_t det_inv = bfield_inv_impl(det);

    // First column of adjugate matrix:
    // adj0 = (a+c)^2 - b^2 + b c
    // adj1 = -(a b + c^2)
    // adj2 = b^2 - a c - c^2
    uint64_t a_plus_c = bfield_add_impl(a, c);
    uint64_t adj0 = bfield_add_impl(
        bfield_sub_impl(bfield_mul_impl(a_plus_c, a_plus_c), b2),
        bc
    );
    uint64_t adj1 = bfield_neg_impl(bfield_add_impl(ab, c2));
    uint64_t adj2 = bfield_sub_impl(bfield_sub_impl(b2, ac), c2);

    r0 = bfield_mul_impl(adj0, det_inv);
    r1 = bfield_mul_impl(adj1, det_inv);
    r2 = bfield_mul_impl(adj2, det_inv);
}

// ============================================================================
// Host Interface Functions
// ============================================================================

/**
 * XField element-wise addition on GPU
 * @param d_a Input array (3*n uint64_t values)
 * @param d_b Input array (3*n uint64_t values)
 * @param d_c Output array (3*n uint64_t values)
 * @param n Number of XFieldElements
 * @param stream CUDA stream
 */
void xfield_add_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XField element-wise subtraction on GPU
 */
void xfield_sub_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XField element-wise multiplication on GPU
 */
void xfield_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XField element-wise negation on GPU
 */
void xfield_neg_gpu(
    const uint64_t* d_a,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XField scalar multiplication on GPU
 * Multiplies each XFieldElement by corresponding BFieldElement scalar
 */
void xfield_scalar_mul_gpu(
    const uint64_t* d_a,
    const uint64_t* d_scalars,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XField element-wise inversion on GPU
 */
void xfield_inv_gpu(
    const uint64_t* d_a,
    uint64_t* d_c,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XField batch inversion using Montgomery's trick
 * Converts n inversions into 1 inversion + ~3n multiplications
 * Requires d_scratch buffer of size n * 3 uint64_t values
 * 
 * @param d_values Input values to invert (3*n uint64_t), modified in-place with inverses
 * @param d_scratch Scratch buffer (3*n uint64_t) for intermediate products
 * @param n Number of XFieldElements to invert
 * @param stream CUDA stream
 */
void xfield_batch_inv_gpu(
    uint64_t* d_values,
    uint64_t* d_scratch,
    size_t n,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
