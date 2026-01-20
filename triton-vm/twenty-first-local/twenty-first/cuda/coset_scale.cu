// CUDA kernel for BFieldElement coset scaling
// Computes: output[i] = input[i] * offset^i for coset evaluation preprocessing
// Prime field: P = 2^64 - 2^32 + 1 (0xFFFFFFFF00000001)
// Uses Montgomery representation for efficient field arithmetic

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "field_arithmetic.cuh"

//============================================================================
// Coset Scaling Kernel
//============================================================================

/// GPU coset scaling kernel for BFieldElement arrays
/// Computes coefficients[i] = coefficients[i] * offset^i for each polynomial
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of polynomials)
///   - blockDim.x = threads per block (typically 256-1024)
///
/// Each block processes one polynomial, with threads cooperating to compute
/// offset powers and scale coefficients
///
/// Parameters:
///   coefficients: Input/output array of polynomials (each poly_length elements)
///   offset: Coset offset in Montgomery form (same for all polynomials)
///   poly_length: Length of each polynomial (must be power of 2)
///   batch_size: Number of polynomials to process
extern "C" __global__ void coset_scale_bfield(
    u64* coefficients,    // Input/output: batch_size * poly_length elements
    u64 offset,           // Coset offset (in Montgomery form)
    u64 poly_length,      // Length of each polynomial
    u64 batch_size        // Number of polynomials
) {
    // Each block processes one polynomial
    u64 poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;

    // Pointer to this polynomial's coefficients
    u64* poly = coefficients + poly_idx * poly_length;

    // Compute offset^i for each coefficient index this thread handles
    // Strategy: Each thread computes its own offset^i using iterative multiplication
    // This is memory-efficient and avoids synchronization overhead

    // Thread stride for processing elements
    u64 stride = blockDim.x;

    for (u64 i = threadIdx.x; i < poly_length; i += stride) {
        // Compute offset^i using fast exponentiation
        // Start with offset^0 = 1 (in Montgomery form)
        u64 offset_power = 0xFFFFFFFE00000001ULL; // Montgomery form of 1 (R mod P)

        // Compute offset^i by repeated multiplication
        // This is more efficient than storing all powers in shared memory
        // for large polynomials (2^21 elements)
        for (u64 j = 0; j < i; j++) {
            offset_power = bfe_mul(offset_power, offset);
        }

        // Scale coefficient: coeff[i] *= offset^i
        poly[i] = bfe_mul(poly[i], offset_power);
    }
}

/// Optimized GPU coset scaling kernel using shared memory for offset powers
/// This version precomputes offset powers in shared memory to reduce redundant computation
///
/// This kernel is more efficient for small to medium polynomial sizes where
/// shared memory can hold all offset powers. For very large polynomials,
/// the basic version above may be preferred.
extern "C" __global__ void coset_scale_bfield_shared(
    u64* coefficients,
    u64 offset,
    u64 poly_length,
    u64 batch_size
) {
    u64 poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;

    u64* poly = coefficients + poly_idx * poly_length;

    // Shared memory for offset powers (limited size)
    // Max 8192 elements (64 KB / 8 bytes per u64)
    extern __shared__ u64 offset_powers[];

    // Precompute offset powers cooperatively
    // Each thread computes a subset of powers
    u64 stride = blockDim.x;

    // Only compute powers up to what fits in shared memory
    u64 max_cached_powers = min(poly_length, (u64)8192);

    for (u64 i = threadIdx.x; i < max_cached_powers; i += stride) {
        if (i == 0) {
            offset_powers[0] = 0xFFFFFFFE00000001ULL; // Montgomery form of 1
        } else {
            // Compute offset^i = offset^(i-1) * offset
            // Wait for previous power to be computed
            __syncthreads();
            offset_powers[i] = bfe_mul(offset_powers[i-1], offset);
        }
    }

    __syncthreads();

    // Scale coefficients using precomputed powers
    for (u64 i = threadIdx.x; i < poly_length; i += stride) {
        u64 offset_power;

        if (i < max_cached_powers) {
            offset_power = offset_powers[i];
        } else {
            // For indices beyond shared memory, compute on the fly
            offset_power = offset_powers[max_cached_powers - 1];
            for (u64 j = max_cached_powers; j <= i; j++) {
                offset_power = bfe_mul(offset_power, offset);
            }
        }

        poly[i] = bfe_mul(poly[i], offset_power);
    }
}


/// Fast GPU coset scaling using optimized power computation
/// Uses binary exponentiation for computing initial offset^threadIdx,
/// then incremental multiplication for the stride pattern
extern "C" __global__ void coset_scale_bfield_fast(
    u64* coefficients,
    u64 offset,
    u64 poly_length,
    u64 batch_size
) {
    u64 poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;

    u64* poly = coefficients + poly_idx * poly_length;

    // Shared memory for precomputed powers: offset^0, offset^1, ..., offset^(blockDim.x-1)
    __shared__ u64 offset_powers[1024];  // Max 1024 threads per block

    // Cooperatively compute first blockDim.x powers
    // Each thread computes one power using fast exponentiation
    if (threadIdx.x < blockDim.x) {
        offset_powers[threadIdx.x] = bfe_pow(offset, threadIdx.x);
    }
    __syncthreads();

    // Compute offset^blockDim.x for stride jumps
    u64 offset_stride = bfe_pow(offset, blockDim.x);

    // Get initial power for this thread
    u64 offset_power = offset_powers[threadIdx.x];

    // Process elements with stride, incrementing offset_power each iteration
    for (u64 i = threadIdx.x; i < poly_length; i += blockDim.x) {
        poly[i] = bfe_mul(poly[i], offset_power);

        // Jump to next power: offset^(i+blockDim.x) = offset^i * offset^blockDim.x
        offset_power = bfe_mul(offset_power, offset_stride);
    }
}
