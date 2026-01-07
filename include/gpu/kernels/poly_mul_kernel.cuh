/**
 * GPU-accelerated polynomial multiplication using NTT
 * 
 * Provides a C-linkage wrapper that can be called from .cpp files
 */

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU polynomial multiplication using NTT
 * 
 * Computes result = a * b where a and b are polynomials.
 * 
 * @param a_coeffs Polynomial a coefficients (ascending order)
 * @param a_size Number of coefficients in a
 * @param b_coeffs Polynomial b coefficients (ascending order)
 * @param b_size Number of coefficients in b
 * @param result_coeffs Output buffer for result (must be at least a_size + b_size - 1)
 * @param result_size Output: actual size of result
 * @return 0 on success, non-zero on failure
 */
int gpu_poly_mul_ntt(
    const uint64_t* a_coeffs, size_t a_size,
    const uint64_t* b_coeffs, size_t b_size,
    uint64_t* result_coeffs, size_t* result_size
);

/**
 * Check if GPU poly_mul is available
 * @return 1 if available, 0 otherwise
 */
int gpu_poly_mul_available(void);

#ifdef __cplusplus
}
#endif

