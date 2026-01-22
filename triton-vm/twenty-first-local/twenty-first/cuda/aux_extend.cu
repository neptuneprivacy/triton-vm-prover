// CUDA kernel for GPU-accelerated auxiliary table extension
// Implements parallel prefix scan for running evaluations and log derivatives
// Optimized for B200 and Ampere+ architectures

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "field_arithmetic.cuh"

//============================================================================
// Parallel Prefix Scan for Running Evaluations
//============================================================================

/// Block-level inclusive scan for XFieldElementArr running evaluations
/// Computes: out[i] = eval[i] where eval[i] = eval[i-1] * challenge + values[i]
/// Uses shared memory for efficient block-level reduction
template<int BLOCK_SIZE>
__device__ void block_scan_running_eval(
    XFieldElementArr* values,
    XFieldElementArr* output,
    XFieldElementArr challenge,
    int n,
    XFieldElementArr* block_results
) {
    __shared__ XFieldElementArr sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Load data into shared memory
    if (global_id < n) {
        sdata[tid] = values[global_id];
    } else {
        sdata[tid] = xfe_zero();
    }
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            // sdata[index] = sdata[index - stride] * challenge + sdata[index]
            XFieldElementArr prev = xfe_mul(sdata[index - stride], challenge);
            sdata[index] = xfe_add(prev, sdata[index]);
        }
        __syncthreads();
    }

    // Down-sweep phase
    if (tid == 0) {
        // Save block result for inter-block propagation
        if (block_results != nullptr) {
            block_results[blockIdx.x] = sdata[BLOCK_SIZE - 1];
        }
    }

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE) {
            // Right child = left * challenge + right
            XFieldElementArr left = xfe_mul(sdata[index], challenge);
            sdata[index + stride] = xfe_add(left, sdata[index + stride]);
        }
    }
    __syncthreads();

    // Write result
    if (global_id < n) {
        output[global_id] = sdata[tid];
    }
}

/// Kernel for running evaluation: out[i] = out[i-1] * challenge + values[i]
/// Handles evaluation arguments like in Lookup table
/// Sequential implementation for correctness validation
extern "C" __global__ void running_evaluation_scan(
    const u64* __restrict__ values,      // Input values (BFieldElement, already in Montgomery form)
    u64* __restrict__ output,             // Output running evaluation (3 u64s per XFieldElementArr)
    const u64* __restrict__ challenge,    // Challenge value (3 u64s for XFieldElementArr, in Montgomery form)
    u32 n,                                // Number of elements
    u32 stride                            // Stride for reading values (1 for contiguous, >1 for column)
) {
    // Single-threaded sequential implementation
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Load challenge (already in Montgomery form)
        XFieldElementArr chal;
        chal.coefficients[0] = challenge[0];
        chal.coefficients[1] = challenge[1];
        chal.coefficients[2] = challenge[2];

        // Initialize running evaluation to ONE (EvalArg::default_initial() = 1)
        XFieldElementArr eval = xfe_one();

        for (u32 i = 0; i < n; i++) {
            // Multiply current eval by challenge
            eval = xfe_mul(eval, chal);

            // Add current value as embedded BFieldElement (val, 0, 0)
            // values[i] is already in Montgomery form from Rust
            u64 val_monty = values[i * stride];
            eval.coefficients[0] = bfe_add(eval.coefficients[0], val_monty);
            // coefficients[1] and [2] unchanged (adding 0)

            // Write result
            output[i * 3 + 0] = eval.coefficients[0];
            output[i * 3 + 1] = eval.coefficients[1];
            output[i * 3 + 2] = eval.coefficients[2];
        }
    }
}

//============================================================================
// Batch Field Inversion using Montgomery's Trick
//============================================================================

/// BFieldElement inversion using Fermat's Little Theorem
/// For prime P, a^(-1) = a^(P-2) mod P
/// Uses binary exponentiation in Montgomery form directly
__device__ u64 bfe_inverse(u64 a) {
    if (a == 0) {
        return 0; // Division by zero
    }

    // Fermat's Little Theorem: a^(-1) = a^(P-2) mod P
    // P - 2 = 0xFFFFFFFE00000000 + 0xFFFFFFFF
    // We work directly in Montgomery form throughout

    u64 result = to_montgomery(1);  // Result = 1 in Montgomery form
    u64 base = a;  // Base is already in Montgomery form

    // Exponent P - 2 = 0xFFFFFFFEFFFFFFFF
    // Process as two 32-bit halves for clarity
    // Low 64 bits: 0xFFFFFFFFFFFFFFFF (all 1s)
    u64 exp_low = 0xFFFFFFFFFFFFFFFFULL;
    // High part: 0xFFFFFFFE00000000, but we only need 32 bits: 0xFFFFFFFE
    u32 exp_high = 0xFFFFFFFEU;

    // Process low 64 bits
    for (int i = 0; i < 64; i++) {
        if (exp_low & 1) {
            result = bfe_mul(result, base);
        }
        base = bfe_mul(base, base);
        exp_low >>= 1;
    }

    // Process high 32 bits
    for (int i = 0; i < 32; i++) {
        if (exp_high & 1) {
            result = bfe_mul(result, base);
        }
        base = bfe_mul(base, base);
        exp_high >>= 1;
    }

    return result;
}

/// XFieldElementArr helper: check if zero
__device__ __forceinline__ bool xfe_is_zero(XFieldElementArr a) {
    return (a.coefficients[0] == 0) && (a.coefficients[1] == 0) && (a.coefficients[2] == 0);
}

/// XFieldElementArr helper: get multiplicative identity (1, 0, 0)
/// XFieldElementArr inversion using Fermat's Little Theorem
/// For F_p^3 where p = 0xFFFFFFFF00000001, we have α^(p^3 - 1) = 1
/// Therefore α^(-1) = α^(p^3 - 2)
///
/// Uses binary exponentiation to compute α^(p^3 - 2) efficiently
__device__ XFieldElementArr xfe_inverse(XFieldElementArr a) {
    // Check for zero (cannot invert)
    if (xfe_is_zero(a)) {
        return xfe_zero(); // Division by zero - return zero
    }

    // Fast path: if this is an embedded BFieldElement (c1 = c2 = 0)
    if (a.coefficients[1] == 0 && a.coefficients[2] == 0) {
        XFieldElementArr result;
        result.coefficients[0] = bfe_inverse(a.coefficients[0]);
        result.coefficients[1] = 0;
        result.coefficients[2] = 0;
        return result;
    }

    // Full XFieldElementArr inversion using Fermat's Little Theorem
    // Exponent = p^3 - 2 where p = 0xFFFFFFFF00000001
    // p^3 - 2 = 0xFFFFFFFD00000005FFFFFFF900000005FFFFFFFCFFFFFFFF (192 bits)

    // Binary exponentiation
    XFieldElementArr result = xfe_one();
    XFieldElementArr base = a;

    // Process exponent in 64-bit chunks from LSB to MSB
    // Chunk 0 (bits 0-63):   0xFFFFFFFCFFFFFFFF
    // Chunk 1 (bits 64-127): 0xFFFFFFF900000005
    // Chunk 2 (bits 128-191): 0xFFFFFFFD00000005

    u64 exp0 = 0xFFFFFFFCFFFFFFFFULL;
    u64 exp1 = 0xFFFFFFF900000005ULL;
    u64 exp2 = 0xFFFFFFFD00000005ULL;

    // Process chunk 0
    for (int i = 0; i < 64; i++) {
        if (exp0 & 1ULL) {
            result = xfe_mul(result, base);
        }
        base = xfe_mul(base, base);
        exp0 >>= 1;
    }

    // Process chunk 1
    for (int i = 0; i < 64; i++) {
        if (exp1 & 1ULL) {
            result = xfe_mul(result, base);
        }
        base = xfe_mul(base, base);
        exp1 >>= 1;
    }

    // Process chunk 2
    for (int i = 0; i < 64; i++) {
        if (exp2 & 1ULL) {
            result = xfe_mul(result, base);
        }
        base = xfe_mul(base, base);
        exp2 >>= 1;
    }

    return result;
}

/// Batch inversion kernel using Montgomery's trick
/// Computes inverses[i] = values[i]^(-1) for all i
extern "C" __global__ void batch_inversion(
    const u64* __restrict__ values,
    u64* __restrict__ inverses,
    u32 n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // For now, compute inverses independently
    // TODO: Implement Montgomery's trick for batching
    if (tid < n) {
        inverses[tid] = bfe_inverse(values[tid]);
    }
}

//============================================================================
// Running Sum with Inversions (for Log Derivatives)
//============================================================================

/// Kernel for log derivative accumulation with inversions
/// out[i] = out[i-1] + (challenge - compressed_row[i])^(-1) * multiplicity[i]
extern "C" __global__ void log_derivative_scan(
    const u64* __restrict__ compressed_rows,  // Compressed row values (3 u64s per XFieldElementArr)
    const u64* __restrict__ multiplicities,    // Multiplicity values (BFieldElement)
    u64* __restrict__ output,                  // Output (3 u64s per XFieldElementArr)
    const u64* __restrict__ challenge,         // Challenge (3 u64s for XFieldElementArr)
    u32 n,
    u32 element_stride  // Stride between elements (should be 3 for XFieldElementArr)
) {
    // For now, compute sequentially in first thread
    // TODO: Implement parallel scan with batched inversions
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        XFieldElementArr chal;
        chal.coefficients[0] = challenge[0];
        chal.coefficients[1] = challenge[1];
        chal.coefficients[2] = challenge[2];

        XFieldElementArr sum = xfe_zero();

        for (u32 i = 0; i < n; i++) {
            // Load compressed row (XFieldElementArr)
            XFieldElementArr compressed;
            compressed.coefficients[0] = compressed_rows[i * 3 + 0];
            compressed.coefficients[1] = compressed_rows[i * 3 + 1];
            compressed.coefficients[2] = compressed_rows[i * 3 + 2];

            // Load multiplicity (BFieldElement)
            u64 mult = multiplicities[i];

            // diff = challenge - compressed_row
            XFieldElementArr diff = xfe_sub(chal, compressed);

            // Compute inverse of XFieldElementArr
            XFieldElementArr inv = xfe_inverse(diff);

            // Multiply inverse by multiplicity (scalar multiplication)
            XFieldElementArr term = xfe_mul_scalar_ret(inv, mult);

            // sum += term
            sum = xfe_add(sum, term);

            // Write result
            output[i * 3 + 0] = sum.coefficients[0];
            output[i * 3 + 1] = sum.coefficients[1];
            output[i * 3 + 2] = sum.coefficients[2];
        }
    }
}

//============================================================================
// Parallel Prefix Scan (Blelloch Algorithm) for Running Evaluation
//============================================================================

/// Affine transformation struct for running evaluation parallelization
/// Represents the transformation: f(x) = multiplier * x + addend
/// Used to parallelize the linear recurrence: eval[i] = challenge * eval[i-1] + value[i]
struct AffineTransform {
    XFieldElementArr multiplier;  // The "challenge" part
    XFieldElementArr addend;       // The accumulated value
};

/// Compose two affine transformations
/// (a, b) ⊗ (c, d) = (a * c, a * d + b)
/// This represents: f1(f2(x)) = f1(c*x + d) = a*(c*x + d) + b = (a*c)*x + (a*d + b)
__device__ __forceinline__ AffineTransform affine_compose(AffineTransform left, AffineTransform right) {
    AffineTransform result;
    // multiplier = left.mult * right.mult
    result.multiplier = xfe_mul(left.multiplier, right.multiplier);
    // addend = left.mult * right.addend + left.addend
    result.addend = xfe_add(xfe_mul(left.multiplier, right.addend), left.addend);
    return result;
}

/// Apply affine transformation to a value
__device__ __forceinline__ XFieldElementArr affine_apply(AffineTransform transform, XFieldElementArr x) {
    return xfe_add(xfe_mul(transform.multiplier, x), transform.addend);
}

/// Parallel running evaluation using Blelloch scan
/// Optimized for 256 elements (Lookup table size) with 256 threads
extern "C" __global__ void running_evaluation_scan_parallel(
    const u64* __restrict__ values,
    u64* __restrict__ output,
    const u64* __restrict__ challenge,
    u32 n,
    u32 stride
) {
    __shared__ AffineTransform shared_transforms[256];

    u32 tid = threadIdx.x;
    u32 bid = blockIdx.x;
    u32 global_idx = bid * blockDim.x + tid;

    // Load challenge
    XFieldElementArr chal;
    chal.coefficients[0] = challenge[0];
    chal.coefficients[1] = challenge[1];
    chal.coefficients[2] = challenge[2];

    // Initialize transform for this thread
    // Each element represents: f_i(x) = challenge * x + value[i]
    AffineTransform my_transform;
    if (global_idx < n) {
        my_transform.multiplier = chal;
        my_transform.addend = xfe_zero();
        u64 val_monty = values[global_idx * stride];
        my_transform.addend.coefficients[0] = val_monty;
    } else {
        // Identity transform for out-of-bounds threads
        my_transform.multiplier = xfe_one();
        my_transform.addend = xfe_zero();
    }

    shared_transforms[tid] = my_transform;
    __syncthreads();

    // Up-sweep (reduce) phase
    for (u32 stride_val = 1; stride_val < blockDim.x; stride_val *= 2) {
        u32 index = (tid + 1) * stride_val * 2 - 1;
        if (index < blockDim.x) {
            // Compose right-to-left: apply earlier transforms first
            shared_transforms[index] = affine_compose(
                shared_transforms[index],
                shared_transforms[index - stride_val]
            );
        }
        __syncthreads();
    }

    // Down-sweep phase
    if (tid == 0) {
        shared_transforms[blockDim.x - 1].multiplier = xfe_one();
        shared_transforms[blockDim.x - 1].addend = xfe_zero();
    }
    __syncthreads();

    for (u32 stride_val = blockDim.x / 2; stride_val > 0; stride_val /= 2) {
        u32 index = (tid + 1) * stride_val * 2 - 1;
        if (index < blockDim.x) {
            AffineTransform temp = shared_transforms[index - stride_val];
            shared_transforms[index - stride_val] = shared_transforms[index];
            shared_transforms[index] = affine_compose(temp, shared_transforms[index]);
        }
        __syncthreads();
    }

    // Compute final result by applying accumulated transform to identity (ONE)
    if (global_idx < n) {
        XFieldElementArr init = xfe_one();

        // Compose with the original value to get inclusive scan
        AffineTransform final_transform = affine_compose(my_transform, shared_transforms[tid]);
        XFieldElementArr result = affine_apply(final_transform, init);

        output[global_idx * 3 + 0] = result.coefficients[0];
        output[global_idx * 3 + 1] = result.coefficients[1];
        output[global_idx * 3 + 2] = result.coefficients[2];
    }
}

//============================================================================
// Parallel Prefix Scan for Log Derivative (Addition Scan)
//============================================================================

/// Parallel log derivative using Blelloch scan for addition
/// Simpler than running evaluation - just parallel sum
extern "C" __global__ void log_derivative_scan_parallel(
    const u64* __restrict__ compressed_rows,
    const u64* __restrict__ multiplicities,
    u64* __restrict__ output,
    const u64* __restrict__ challenge,
    u32 n,
    u32 element_stride
) {
    __shared__ XFieldElementArr shared_terms[256];

    u32 tid = threadIdx.x;
    u32 bid = blockIdx.x;
    u32 global_idx = bid * blockDim.x + tid;

    // Load challenge
    XFieldElementArr chal;
    chal.coefficients[0] = challenge[0];
    chal.coefficients[1] = challenge[1];
    chal.coefficients[2] = challenge[2];

    // Compute term for this thread: (challenge - compressed)^(-1) * multiplicity
    XFieldElementArr my_term;
    if (global_idx < n) {
        // Load compressed row
        XFieldElementArr compressed;
        compressed.coefficients[0] = compressed_rows[global_idx * element_stride + 0];
        compressed.coefficients[1] = compressed_rows[global_idx * element_stride + 1];
        compressed.coefficients[2] = compressed_rows[global_idx * element_stride + 2];

        // Compute (challenge - compressed)
        XFieldElementArr diff = xfe_sub(chal, compressed);

        // Invert
        XFieldElementArr inv = xfe_inverse(diff);

        // Multiply by multiplicity
        u64 mult = multiplicities[global_idx];
        my_term = xfe_mul_scalar_ret(inv, mult);
    } else {
        my_term = xfe_zero();
    }

    shared_terms[tid] = my_term;
    __syncthreads();

    // Blelloch scan for addition
    // Up-sweep
    for (u32 stride_val = 1; stride_val < blockDim.x; stride_val *= 2) {
        u32 index = (tid + 1) * stride_val * 2 - 1;
        if (index < blockDim.x) {
            shared_terms[index] = xfe_add(shared_terms[index - stride_val], shared_terms[index]);
        }
        __syncthreads();
    }

    // Down-sweep
    if (tid == 0) {
        shared_terms[blockDim.x - 1] = xfe_zero();
    }
    __syncthreads();

    for (u32 stride_val = blockDim.x / 2; stride_val > 0; stride_val /= 2) {
        u32 index = (tid + 1) * stride_val * 2 - 1;
        if (index < blockDim.x) {
            XFieldElementArr temp = shared_terms[index - stride_val];
            shared_terms[index - stride_val] = shared_terms[index];
            shared_terms[index] = xfe_add(shared_terms[index], temp);
        }
        __syncthreads();
    }

    // Add original value for inclusive scan
    if (global_idx < n) {
        XFieldElementArr result = xfe_add(shared_terms[tid], my_term);

        output[global_idx * 3 + 0] = result.coefficients[0];
        output[global_idx * 3 + 1] = result.coefficients[1];
        output[global_idx * 3 + 2] = result.coefficients[2];
    }
}
