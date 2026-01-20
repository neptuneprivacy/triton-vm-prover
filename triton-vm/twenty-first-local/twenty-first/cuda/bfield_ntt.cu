// CUDA kernel for BFieldElement NTT (Number Theoretic Transform)
// Prime field: P = 2^64 - 2^32 + 1 (0xFFFFFFFF00000001)
// Uses Montgomery representation for efficient field arithmetic

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "field_arithmetic.cuh"

using namespace cooperative_groups;



//============================================================================
// BFieldElement Arithmetic (Montgomery Representation)
//============================================================================

/// Montgomery reduction: convert from u128 to u64 in Montgomery form
/// This is the core operation for multiplication
/*
__device__ __forceinline__ u64 montyred(u128 x) {
    // Split 128-bit number into low and high 64-bit parts
    u64 xl = (u64)x;
    u64 xh = (u64)(x >> 64);

    // Compute a = xl + (xl << 32). Detect overflow via (a < xl)
    u64 a = xl + (xl << 32);

    // b = a - (a >> 32) adjusted by the carry from the addition
    u64 b = a - (a >> 32) - (a < xl);

    // Compute r = xh - b
    u64 r = xh - b;

    // If xh < b, then add P back
    r += (xh < b) ? P : 0;

    return r;
}
*/

//============================================================================
// NTT Kernel
//============================================================================

/// Core NTT implementation (device function)
/// Can be called from both ntt_bfield and intt_bfield kernels
__device__ void ntt_core(
    u64* poly,
    u64 slice_len,
    u64* omegas,
    u32 log2_slice_len
) {
    // Phase 1: Bit-reversal permutation
    // Each thread handles multiple elements
    for (u64 k = threadIdx.x; k < slice_len; k += blockDim.x) {
        u32 rk = bitreverse(k, log2_slice_len);
        if (k < rk) {
            u64 temp = poly[k];
            poly[k] = poly[rk];
            poly[rk] = temp;
        }
    }

    __syncthreads();

    // Calculate log2 of block dimension for phased NTT
    u32 log2_blockDim = 0;
    u32 temp = blockDim.x;
    if (temp == 1) {
        log2_blockDim = 0;
    } else {
        while (temp > 0) {
            log2_blockDim += 1;
            temp /= 2;
        }
    }

    // Shared memory for twiddle factors (1024 elements max for 1024 threads/block)
    __shared__ u64 w_precalc[1024];
    __shared__ u64 w_precalc_temp[1024];

    u32 m = 1;  // Current butterfly stride
    u64 bfe_1 = to_montgomery(1);  // Montgomery form of 1

    // NOTE: omegas array is already in Montgomery form from Rust!
    // Do NOT convert them again

    // Phase 2: Butterfly operations (thread-cooperative within block)
    // This phase handles stages where m < blockDim.x
    for (u32 i = 0; i < log2_blockDim; i++) {
        u64 w_m = omegas[i];  // Already in Montgomery form!

        // Thread 0 precomputes twiddle factors for this stage
        if (threadIdx.x == 0) {
            u64 w = bfe_1;
            for (u32 j = 0; j < m; j++) {
                w_precalc[j] = w;
                w = bfe_mul(w, w_m);
            }
        }
        __syncthreads();

        // Each thread processes its assigned butterfly groups
        u32 k = 2 * m * threadIdx.x;
        while (k < slice_len) {
            for (u32 j = 0; j < m; j++) {
                u64 u = poly[k + j];
                u64 v = poly[k + j + m];
                v = bfe_mul(v, w_precalc[j]);
                poly[k + j] = bfe_add(u, v);
                poly[k + j + m] = bfe_sub(u, v);
            }
            k += 2 * m * blockDim.x;
        }
        __syncthreads();

        m *= 2;
    }

    // Phase 3: Butterfly operations (independent threads)
    // This phase handles stages where m >= blockDim.x
    for (u32 i = log2_blockDim; i < log2_slice_len; i++) {
        u64 w_m = omegas[i];  // Already in Montgomery form!

        // Compute local power increment for this thread
        u64 local_power_w = bfe_1;
        for (u32 j = 0; j < blockDim.x; j++) {
            local_power_w = bfe_mul(local_power_w, w_m);
        }
        __syncthreads();

        // Each thread precomputes its starting twiddle factor
        for (u32 j = threadIdx.x; j < blockDim.x; j += blockDim.x) {
            u64 temp_w = bfe_1;
            for (u32 k = 0; k < j; k++) {
                temp_w = bfe_mul(temp_w, w_m);
            }
            w_precalc[j] = temp_w;
        }
        __syncthreads();

        // Process butterfly groups
        u32 k = 0;
        while (k < slice_len) {
            w_precalc_temp[threadIdx.x] = w_precalc[threadIdx.x];

            for (u32 j = threadIdx.x; j < m; j += blockDim.x) {
                u64 u = poly[k + j];
                u64 v = poly[k + j + m];
                v = bfe_mul(v, w_precalc_temp[threadIdx.x]);
                poly[k + j] = bfe_add(u, v);
                poly[k + j + m] = bfe_sub(u, v);

                w_precalc_temp[threadIdx.x] = bfe_mul(w_precalc_temp[threadIdx.x], local_power_w);
            }
            __syncthreads();  // Moved OUTSIDE the j-loop to avoid deadlock

            k += 2 * m;
        }
        __syncthreads();

        m *= 2;
    }
}

/// GPU NTT kernel wrapper - calls ntt_core for each block's polynomial
extern "C" __global__ void ntt_bfield(
    u64* poly_global,
    u64 slice_len,
    u64* omegas,
    u32 log2_slice_len
) {
    u64* poly = poly_global + blockIdx.x * slice_len;
    ntt_core(poly, slice_len, omegas, log2_slice_len);
}

//============================================================================
// Inverse NTT Kernel
//============================================================================

/// GPU inverse NTT kernel for BFieldElement arrays
/// Same algorithm as forward NTT but uses inverse twiddle factors
/// Note: Caller must divide by array length after calling this
extern "C" __global__ void intt_bfield(
    u64* poly_global,
    u64 slice_len,
    u64* omegas_inv,
    u32 log2_slice_len
) {
    u64* poly = poly_global + blockIdx.x * slice_len;
    ntt_core(poly, slice_len, omegas_inv, log2_slice_len);
}

//============================================================================
// Fused Coset Scaling + NTT Kernel
//============================================================================


/// Fused coset scaling + NTT kernel for BFieldElement arrays
/// This kernel combines coset scaling (coefficients[i] *= offset^i) with NTT
/// to eliminate PCIe transfer overhead from separate operations
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of polynomials)
///   - blockDim.x = threads per block (typically 256-1024)
///
/// Parameters:
///   poly_global: Input/output array of polynomials (each slice_len elements)
///   offset: Coset offset in Montgomery form
///   slice_len: Length of each polynomial (must be power of 2)
///   omegas: NTT twiddle factors (in Montgomery form)
///   log2_slice_len: log2 of slice_len
extern "C" __global__ void ntt_bfield_fused_coset(
    u64* poly_global,
    u64 offset,
    u64 slice_len,
    u64* omegas,
    u32 log2_slice_len
) {
    u64* poly = poly_global + blockIdx.x * slice_len;

    // ========================================================================
    // Phase 0: Coset Scaling (coefficients[i] *= offset^i)
    // ========================================================================
    // Skip coset scaling if offset == 1 (standard domain)
    u64 bfe_1 = to_montgomery(1);
    if (offset != bfe_1) {
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
        for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
            poly[i] = bfe_mul(poly[i], offset_power);

            // Jump to next power: offset^(i+blockDim.x) = offset^i * offset^blockDim.x
            offset_power = bfe_mul(offset_power, offset_stride);
        }

        __syncthreads();  // Ensure coset scaling is complete before NTT
    }

    // ========================================================================
    // Phase 1+: NTT (existing algorithm)
    // ========================================================================
    ntt_core(poly, slice_len, omegas, log2_slice_len);
}

//============================================================================
// Strided Fused Coset Scaling + NTT Kernel (for row-major table data)
//============================================================================

/// Strided fused coset scaling + NTT kernel for row-major table data
/// This kernel operates on data stored in row-major format where columns are
/// accessed with a stride.
///
/// Memory layout (row-major):
///   [Row0_Col0, Row0_Col1, ..., Row0_ColN,
///    Row1_Col0, Row1_Col1, ..., Row1_ColN,
///    ...]
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of columns to process)
///   - blockDim.x = threads per block (typically 256)
///
/// Parameters:
///   data_global: Input/output array in row-major format
///   offset: Coset offset in Montgomery form
///   slice_len: Length of each column (number of rows)
///   stride: Distance between consecutive elements in a column (= num_columns)
///   omegas: NTT twiddle factors (in Montgomery form)
///   log2_slice_len: log2 of slice_len
/// Strided in-place NTT core (operates directly on global memory with stride)
/// This version works without shared memory by operating directly on strided global memory
__device__ void ntt_core_strided(
    u64* data_base,
    u64 stride,
    u64 slice_len,
    u64* omegas,
    u32 log2_slice_len
) {
    // Phase 1: Bit-reversal permutation (with stride)
    for (u64 k = threadIdx.x; k < slice_len; k += blockDim.x) {
        u32 rk = bitreverse(k, log2_slice_len);
        if (k < rk) {
            u64 temp = data_base[k * stride];
            data_base[k * stride] = data_base[rk * stride];
            data_base[rk * stride] = temp;
        }
    }
    __syncthreads();

    // Calculate log2 of block dimension
    u32 log2_blockDim = 0;
    u32 temp = blockDim.x;
    if (temp > 1) {
        while (temp > 0) {
            log2_blockDim += 1;
            temp /= 2;
        }
    }

    __shared__ u64 w_precalc[1024];
    __shared__ u64 w_precalc_temp[1024];

    u32 m = 1;
    u64 bfe_1 = to_montgomery(1);

    // Phase 2: Butterfly operations (thread-cooperative within block)
    for (u32 i = 0; i < log2_blockDim; i++) {
        u64 w_m = omegas[i];

        if (threadIdx.x == 0) {
            u64 w = bfe_1;
            for (u32 j = 0; j < m; j++) {
                w_precalc[j] = w;
                w = bfe_mul(w, w_m);
            }
        }
        __syncthreads();

        u32 k = 2 * m * threadIdx.x;
        while (k < slice_len) {
            for (u32 j = 0; j < m; j++) {
                u64 u = data_base[(k + j) * stride];
                u64 v = data_base[(k + j + m) * stride];
                v = bfe_mul(v, w_precalc[j]);
                data_base[(k + j) * stride] = bfe_add(u, v);
                data_base[(k + j + m) * stride] = bfe_sub(u, v);
            }
            k += 2 * m * blockDim.x;
        }
        __syncthreads();
        m *= 2;
    }

    // Phase 3: Butterfly operations (independent threads)
    for (u32 i = log2_blockDim; i < log2_slice_len; i++) {
        u64 w_m = omegas[i];

        u64 local_power_w = bfe_pow(w_m,blockDim.x);

        for (u32 j = threadIdx.x; j < blockDim.x; j += blockDim.x) {
            u64 temp_w = bfe_1;
            temp_w = bfe_pow(w_m,j);
            w_precalc[j] = temp_w;
        }
        __syncthreads();

        u32 k = 0;
        while (k < slice_len) {
            w_precalc_temp[threadIdx.x] = w_precalc[threadIdx.x];

            for (u32 j = threadIdx.x; j < m; j += blockDim.x) {
                u64 u = data_base[(k + j) * stride];
                u64 v = data_base[(k + j + m) * stride];
                v = bfe_mul(v, w_precalc_temp[threadIdx.x]);
                data_base[(k + j) * stride] = bfe_add(u, v);
                data_base[(k + j + m) * stride] = bfe_sub(u, v);
                w_precalc_temp[threadIdx.x] = bfe_mul(w_precalc_temp[threadIdx.x], local_power_w);
            }
            __syncthreads();
            k += 2 * m;
        }
        __syncthreads();
        m *= 2;
    }
    
}

extern "C" __global__ void ntt_bfield_fused_coset_strided_old(
    u64* data_global,
    u64 offset,
    u64 slice_len,
    u64 stride,
    u64* omegas,
    u32 log2_slice_len
) {
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("in_kernel_mod ntt_bfield_fused_coset_strided %d %d\n",gridDim.x,blockDim.x);
    }
    // Calculate column base index
    u64 col_idx = blockIdx.x;
    u64 col_base = col_idx;  // BFieldElement is 1 u64
    u64* data_base = data_global + col_base;

    // ========================================================================
    // Phase 0: Coset Scaling (in-place on global memory with stride)
    // ========================================================================
    u64 bfe_1 = to_montgomery(1);
    if (offset != bfe_1) {
        /*
        __shared__ u64 offset_powers[1024];

        if (threadIdx.x < blockDim.x) {
            offset_powers[threadIdx.x] = bfe_pow(offset, threadIdx.x);
        }
        __syncthreads();
        */

        u64 offset_stride = bfe_pow(offset, blockDim.x);
        u64 offset_power =   bfe_pow(offset, threadIdx.x); //offset_powers[threadIdx.x];

        for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
            data_base[i * stride] = bfe_mul(data_base[i * stride], offset_power);
            offset_power = bfe_mul(offset_power, offset_stride);
        }
        __syncthreads();
    }

    // ========================================================================
    // Phase 1: NTT (in-place on global memory with stride)
    // ========================================================================

    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("start_ntt_core %llu stride=%llu\n",slice_len,stride);
    }
    ntt_core_strided(data_base, stride, slice_len, omegas, log2_slice_len);

    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("in_kernel_mod FINISHED %d %d\n",gridDim.x,blockDim.x);
    }
}

extern "C" __global__ void ntt_bfield_extract(
    u64* data_global,
    u64 ntt_idx,
    u64* buffer,
    u64 ntt_count,
    u64 slice_len
){
    u64 globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    u64 kernelSize = gridDim.x * blockDim.x;

    for(u64 i = globalIdx; i < slice_len;i += kernelSize){
        buffer[i] = data_global[ntt_count * i + ntt_idx];
    }
}

extern "C" __global__ void ntt_bfield_restore(
    u64* data_global,
    u64 ntt_idx,
    u64* buffer,
    u64 ntt_count,
    u64 slice_len
){
    u64 globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    u64 kernelSize = gridDim.x * blockDim.x;

    for(u64 i = globalIdx; i < slice_len;i += kernelSize){
        data_global[ntt_count * i + ntt_idx] = buffer[i];
    }
}

extern "C" __global__ void poly_fill_table_bfield(
    u64 * table,
    u64 table_rows,
    u64 table_cols,
    u64 * all_poly,
    u64 * poly_lengths,
    u64 * poly_starts,
    u64 col_start,
    u64 col_end
){
    // Support column-wise splitting for dual-GPU parallelization
    // Each block processes one column in the range [col_start, col_end)
    u64 poly_idx = col_start + blockIdx.x;
    if (poly_idx >= col_end) return;

    u64 * poly = all_poly + poly_starts[poly_idx];
    u64 poly_len = poly_lengths[poly_idx];

    for(u64 i = threadIdx.x;i < poly_len; i += blockDim.x){
        table[i * table_cols + poly_idx] = poly[i];
    }

    __syncthreads();

    for(u64 i = poly_len + threadIdx.x;i < table_rows; i += blockDim.x){
        table[i * table_cols + poly_idx] = 0;
    }
}

extern "C" __global__ void ntt_bfield_init_omegas(
    u64 slice_len,
    u64 * omegas,
    u64 * omegas_store
){
    u64 ntt_stage = blockIdx.x;
    u64 start_idx = 0;
    u64 iter=1;
    for(u64 i = 0; i<ntt_stage;i++){
        start_idx += iter;
        iter *= 2;
    }

    u64 m = 1ULL << ntt_stage;
    u64 w_m_init = omegas[ntt_stage];
    u64 w_m;
    if(threadIdx.x == 0){
        w_m = to_montgomery(1);
    } else {
        w_m = bfe_pow(w_m_init,threadIdx.x);
    }

    u64 w_m_stride = bfe_pow(w_m_init,blockDim.x);

    for (u64 pair = threadIdx.x; pair < m; pair += blockDim.x) {
        u64 j = pair;
        omegas_store[start_idx + j] = w_m;
        w_m = bfe_mul(w_m, w_m_stride);
    }
}

extern "C" __global__ void ntt_bfield_fused_coset_single(
    u64* data,
    u64 slice_len,
    u64 ntt_stage,
    u64 * omegas,
    u64 * omegas_store,
    u64 offset,
    u64 log2_slice_len
){
    grid_group grid = this_grid();
    u64 globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    u64 kernelSize = gridDim.x * blockDim.x;

    if(ntt_stage == 0){
        u64 bfe_1 = to_montgomery(1);
        if (offset != bfe_1) {
            u64 offset_stride = bfe_pow(offset, kernelSize);
            u64 offset_power =   bfe_pow(offset, globalIdx); //offset_powers[threadIdx.x];

            for (u64 i = globalIdx; i < slice_len; i += kernelSize) {
                data[i] = bfe_mul(data[i], offset_power);
                offset_power = bfe_mul(offset_power, offset_stride);
            }
        
        }
    }

    grid.sync();

    if(ntt_stage == 0){
        for (u64 k = globalIdx; k < slice_len; k += kernelSize) {
            u64 rk = bitreverse(k, log2_slice_len);
            if (k < rk) {
                u64 tmp = data[k];
                data[k] = data[rk];
                data[rk] = tmp;
            }
        }
    }
    grid.sync();


    
        u64 m = 1ULL << ntt_stage;
        u64 group_size = m << 1;
        u64 omega_base = (1ULL << (ntt_stage)) - 1;  // stride between twiddles
        //u64 w_m = omegas[ntt_stage];

        for (u64 pair = globalIdx; pair < (slice_len >> 1); pair += kernelSize) {
            u64 j = pair & (m - 1);
            u64 k = (pair >> ntt_stage) * group_size;

            u64 omega = omegas_store[omega_base + j]; //bfe_pow(w_m, j);

            u64 u = data[k + j];
            u64 v = data[k + j + m];
            v = bfe_mul(v, omega);

            data[k + j]     = bfe_add(u, v);
            data[k + j + m] = bfe_sub(u, v);
        }
        grid.sync();
}

extern "C" __global__ void ntt_bfield_fused_coset_strided(
    u64* data_global,
    u64 offset,
    u64 slice_len,
    u64 stride,
    u64* omegas,
    u32 log2_slice_len
) {
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("in_kernel_mod ntt_bfield_fused_coset_strided %d %d\n",gridDim.x,blockDim.x);
    }
    // Calculate column base index
    u64 col_idx = blockIdx.x;
    u64 col_base = col_idx;  // BFieldElement is 1 u64
    u64* data_base = data_global + col_base;

    // ========================================================================
    // Phase 0: Coset Scaling (in-place on global memory with stride)
    // ========================================================================
    u64 bfe_1 = to_montgomery(1);
    if (offset != bfe_1) {
        /*
        __shared__ u64 offset_powers[1024];

        if (threadIdx.x < blockDim.x) {
            offset_powers[threadIdx.x] = bfe_pow(offset, threadIdx.x);
        }
        __syncthreads();
        */

        u64 offset_stride = bfe_pow(offset, blockDim.x);
        u64 offset_power =   bfe_pow(offset, threadIdx.x); //offset_powers[threadIdx.x];

        for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
            data_base[i * stride] = bfe_mul(data_base[i * stride], offset_power);
            offset_power = bfe_mul(offset_power, offset_stride);
        }
        __syncthreads();
    }

    // ========================================================================
    // Phase 1: NTT (in-place on global memory with stride)
    // ========================================================================

    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("start_ntt_core %llu stride=%llu\n",slice_len,stride);
    }
    ntt_core_strided(data_base, stride, slice_len, omegas, log2_slice_len);

    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("in_kernel_mod FINISHED %d %d\n",gridDim.x,blockDim.x);
    }
}


//============================================================================
// Strided INTT Kernel (for row-major table data)
//============================================================================

/// Strided INTT kernel for row-major table data
/// This kernel operates on data stored in row-major format where columns are
/// accessed with a stride.
///
/// Grid configuration:
///   - gridDim.x = num_columns (number of columns to process)
///   - blockDim.x = threads per block (typically 256)
///
/// Parameters:
///   data_global: Input/output array in row-major format
///   slice_len: Length of each column (number of rows)
///   stride: Distance between consecutive elements in a column (= num_columns)
///   omegas_inv: Inverse NTT twiddle factors (in Montgomery form)
///   log2_slice_len: log2 of slice_len
///
/// Note: Caller must divide by array length after calling this
extern "C" __global__ void intt_bfield_strided(
    u64* data_global,
    u64 slice_len,
    u64 stride,
    u64* omegas_inv,
    u32 log2_slice_len
) {
    // Calculate column base index
    u64 col_idx = blockIdx.x;
    u64 col_base = col_idx;  // BFieldElement is 1 u64
    u64* data_base = data_global + col_base;

    // Perform INTT using inverse twiddle factors
    ntt_core_strided(data_base, stride, slice_len, omegas_inv, log2_slice_len);
}

//============================================================================
// Fused INTT + Unscaling Kernel
//============================================================================

/// Fused INTT + unscaling kernel for BFieldElement arrays
/// This kernel combines inverse NTT with the unscaling step (multiply by n_inv)
/// to eliminate CPU postprocessing overhead
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of polynomials)
///   - blockDim.x = threads per block (typically 256-1024)
///
/// Parameters:
///   poly_global: Input/output array of polynomials (each slice_len elements)
///   slice_len: Length of each polynomial (must be power of 2)
///   omegas_inv: Inverse NTT twiddle factors (in Montgomery form)
///   n_inv: Inverse of the domain length (in Montgomery form)
///   log2_slice_len: log2 of slice_len
extern "C" __global__ void intt_bfield_fused_unscale(
    u64* poly_global,
    u64 slice_len,
    u64* omegas_inv,
    u64 n_inv,
    u32 log2_slice_len
) {
    u64* poly = poly_global + blockIdx.x * slice_len;

    // ========================================================================
    // Phase 1: INTT (existing algorithm)
    // ========================================================================
    ntt_core(poly, slice_len, omegas_inv, log2_slice_len);

    __syncthreads();  // Ensure INTT is complete before unscaling

    // ========================================================================
    // Phase 2: Unscaling (multiply all elements by n_inv)
    // ========================================================================
    // Each thread processes multiple elements
    for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
        poly[i] = bfe_mul(poly[i], n_inv);
    }
}

//============================================================================
// Fused INTT + Unscaling + Randomizer Kernel
//============================================================================

/// Fused INTT + unscaling + randomizer addition kernel for BFieldElement arrays
/// This kernel performs:
///   1. Inverse NTT
///   2. Unscaling (multiply by n_inv)
///   3. Add zerofier-multiplied randomizer for zero-knowledge
///
/// The zerofier multiplication is: poly(x) + randomizer(x) * (x^slice_len - offset^slice_len)
/// Which simplifies to:
///   - Add randomizer coefficients at positions [slice_len, slice_len+num_randomizers)
///   - Subtract randomizer * offset_power_n from positions [0, num_randomizers)
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of polynomials)
///   - blockDim.x = threads per block (typically 256-1024)
///
/// Parameters:
///   poly_global: Input/output array of polynomials (size: batch * (slice_len + num_randomizers))
///   slice_len: Length of each polynomial (must be power of 2)
///   num_randomizers: Number of randomizer coefficients
///   omegas_inv: Inverse NTT twiddle factors (in Montgomery form)
///   n_inv: Inverse of the domain length (in Montgomery form)
///   randomizers: Randomizer coefficients (batch * num_randomizers)
///   offset_power_n: offset^slice_len (in Montgomery form)
///   log2_slice_len: log2 of slice_len
extern "C" __global__ void intt_bfield_fused_unscale_randomize(
    u64* poly_global,
    u64 slice_len,
    u32 num_randomizers,
    u64* omegas_inv,
    u64 n_inv,
    u64* randomizers,  // [batch_size * num_randomizers] array
    u64 offset_power_n,
    u32 log2_slice_len
) {
    u64 total_len = slice_len + num_randomizers;
    u64* poly = poly_global + blockIdx.x * total_len;
    u64* batch_randomizers = randomizers + blockIdx.x * num_randomizers;

    // ========================================================================
    // Phase 1: INTT on first slice_len elements
    // ========================================================================
    ntt_core(poly, slice_len, omegas_inv, log2_slice_len);

    __syncthreads();

    // ========================================================================
    // Phase 2: Unscaling (multiply first slice_len elements by n_inv)
    // ========================================================================
    for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
        poly[i] = bfe_mul(poly[i], n_inv);
    }

    __syncthreads();

    // ========================================================================
    // Phase 3: Add randomizer (zerofier multiplication)
    // ========================================================================
    // Add randomizer coefficients at high-degree positions: result[slice_len..] = randomizer
    for (u32 i = threadIdx.x; i < num_randomizers; i += blockDim.x) {
        poly[slice_len + i] = batch_randomizers[i];
    }

    // Subtract randomizer * offset_power_n from low-degree positions
    for (u32 i = threadIdx.x; i < num_randomizers; i += blockDim.x) {
        u64 scaled_rand = bfe_mul(batch_randomizers[i], offset_power_n);
        poly[i] = bfe_sub(poly[i], scaled_rand);
    }
}
