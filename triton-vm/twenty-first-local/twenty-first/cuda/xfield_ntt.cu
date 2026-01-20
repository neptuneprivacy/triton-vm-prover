// CUDA kernel for XFieldElementArr NTT (Number Theoretic Transform)
// XFieldElementArr is an extension field with 3 BFieldElement coefficients
// Irreducible polynomial: x^3 - x + 1

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "field_arithmetic.cuh"

using namespace cooperative_groups;

//============================================================================
// NTT Kernel for XFieldElementArr
//============================================================================

/// Core NTT implementation for XFieldElementArr (device function)
/// Can be called from both ntt_xfield and intt_xfield kernels
__device__ void ntt_core_xfield(
    XFieldElementArr* poly,
    u64 slice_len,
    u64* omegas,
    u32 log2_slice_len
) {

    // Phase 1: Bit-reversal permutation
    for (u64 k = threadIdx.x; k < slice_len; k += blockDim.x) {
        u32 rk = bitreverse(k, log2_slice_len);
        if (k < rk) {
            XFieldElementArr temp = poly[k];
            poly[k] = poly[rk];
            poly[rk] = temp;
        }
    }

    __syncthreads();

    // Calculate log2 of block dimension
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

    u32 m = 1;
    u64 bfe_1 = to_montgomery(1);

    // NOTE: omegas array is already in Montgomery form from Rust!
    // Do NOT convert them again

    // Phase 2: Thread-cooperative butterfly operations
    for (u32 i = 0; i < log2_blockDim; i++) {
        u64 w_m = omegas[i];  // Already in Montgomery form!

        // Thread 0 precomputes twiddle factors
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
                XFieldElementArr u = poly[k + j];
                XFieldElementArr v = poly[k + j + m];

                // Multiply v by twiddle factor (scalar multiplication)
                xfe_mul_scalar(v, w_precalc[j]);

                poly[k + j] = xfe_add(u, v);
                poly[k + j + m] = xfe_sub(u, v);
            }
            k += 2 * m * blockDim.x;
        }
        __syncthreads();

        m *= 2;
    }

    // Phase 3: Independent thread butterfly operations
    for (u32 i = log2_blockDim; i < log2_slice_len; i++) {
        u64 w_m = omegas[i];  // Already in Montgomery form!

        // Compute local power increment
        u64 local_power_w = bfe_1;
        for (u32 j = 0; j < blockDim.x; j++) {
            local_power_w = bfe_mul(local_power_w, w_m);
        }
        __syncthreads();

        // Precompute starting twiddle factors
        for (u32 j = threadIdx.x; j < blockDim.x; j += blockDim.x) {
            u64 temp_w = bfe_1;
            for (u32 k = 0; k < j; k++) {
                temp_w = bfe_mul(temp_w, w_m);
            }
            w_precalc[j] = temp_w;
        }
        __syncthreads();

        u32 k = 0;
        while (k < slice_len) {
            w_precalc_temp[threadIdx.x] = w_precalc[threadIdx.x];

            for (u32 j = threadIdx.x; j < m; j += blockDim.x) {
                XFieldElementArr u = poly[k + j];
                XFieldElementArr v = poly[k + j + m];

                xfe_mul_scalar(v, w_precalc_temp[threadIdx.x]);

                poly[k + j] = xfe_add(u, v);
                poly[k + j + m] = xfe_sub(u, v);

                w_precalc_temp[threadIdx.x] = bfe_mul(w_precalc_temp[threadIdx.x], local_power_w);
            }
            __syncthreads();  // Moved OUTSIDE the j-loop to avoid deadlock

            k += 2 * m;
        }
        __syncthreads();

        m *= 2;
    }
}

/// GPU NTT kernel wrapper for XFieldElementArr - calls ntt_core_xfield
extern "C" __global__ void ntt_xfield(
    u64* poly_global,
    u64 slice_len,
    u64* omegas,
    u32 log2_slice_len
) {
    XFieldElementArr* poly = ((XFieldElementArr*)poly_global) + blockIdx.x * slice_len;
    ntt_core_xfield(poly, slice_len, omegas, log2_slice_len);
}

//============================================================================
// Inverse NTT Kernel for XFieldElementArr
//============================================================================

extern "C" __global__ void intt_xfield(
    u64* poly_global,
    u64 slice_len,
    u64* omegas_inv,
    u32 log2_slice_len
) {
    XFieldElementArr* poly = ((XFieldElementArr*)poly_global) + blockIdx.x * slice_len;
    ntt_core_xfield(poly, slice_len, omegas_inv, log2_slice_len);
}


/// Fused coset scaling + NTT kernel for XFieldElementArr arrays
/// This kernel combines coset scaling (coefficients[i] *= offset^i) with NTT
/// to eliminate PCIe transfer overhead from separate operations
///
/// For XFieldElementArr, coset scaling means: xfe_coefficients[i] *= bfe_offset^i
/// (scalar multiplication by a BFieldElement power)
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of polynomials)
///   - blockDim.x = threads per block (typically 256-1024)
///
/// Parameters:
///   poly_global: Input/output array of XFieldElementArr polynomials (each slice_len elements)
///   offset: Coset offset in Montgomery form (BFieldElement)
///   slice_len: Length of each polynomial (must be power of 2)
///   omegas: NTT twiddle factors (in Montgomery form)
///   log2_slice_len: log2 of slice_len
extern "C" __global__ void ntt_xfield_fused_coset(
    u64* poly_global,
    u64 offset,
    u64 slice_len,
    u64* omegas,
    u32 log2_slice_len
) {
    XFieldElementArr* poly = ((XFieldElementArr*)poly_global) + blockIdx.x * slice_len;

    // ========================================================================
    // Phase 0: Coset Scaling (xfe_coefficients[i] *= offset^i)
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
            // Scale XFieldElementArr by BFieldElement power (scalar multiplication)
            xfe_mul_scalar(poly[i], offset_power);

            // Jump to next power: offset^(i+blockDim.x) = offset^i * offset^blockDim.x
            offset_power = bfe_mul(offset_power, offset_stride);
        }

        __syncthreads();  // Ensure coset scaling is complete before NTT
    }

    // ========================================================================
    // Phase 1+: NTT (existing algorithm)
    // ========================================================================
    ntt_core_xfield(poly, slice_len, omegas, log2_slice_len);
}

//============================================================================
// Strided Fused Coset Scaling + NTT Kernel (for row-major table data)
//============================================================================

/// Strided fused coset scaling + NTT kernel for row-major table data
/// This kernel operates on XFieldElementArr data stored in row-major format where
/// columns are accessed with a stride.
///
/// Memory layout (row-major):
///   [Row0_Col0_c0, Row0_Col0_c1, Row0_Col0_c2,  // XField has 3 coefficients
///    Row0_Col1_c0, Row0_Col1_c1, Row0_Col1_c2,
///    ...,
///    Row1_Col0_c0, Row1_Col0_c1, Row1_Col0_c2,
///    ...]
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of columns to process)
///   - blockDim.x = threads per block (typically 256)
///
/// Parameters:
///   data_global: Input/output array in row-major format (u64 array)
///   offset: Coset offset in Montgomery form (BFieldElement)
///   slice_len: Length of each column (number of rows)
///   stride: Distance between consecutive elements in a column (= num_columns * 3 for XField)
///   omegas: NTT twiddle factors (in Montgomery form)
///   log2_slice_len: log2 of slice_len
/// Strided in-place NTT core for XFieldElementArr (operates directly on global memory with stride)
__device__ void ntt_core_xfield_strided(
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
            u64 k_offset = k * stride;
            u64 rk_offset = rk * stride;
            // Swap XFieldElements (3 u64s each)
            u64 temp0 = data_base[k_offset];
            u64 temp1 = data_base[k_offset + 1];
            u64 temp2 = data_base[k_offset + 2];
            data_base[k_offset] = data_base[rk_offset];
            data_base[k_offset + 1] = data_base[rk_offset + 1];
            data_base[k_offset + 2] = data_base[rk_offset + 2];
            data_base[rk_offset] = temp0;
            data_base[rk_offset + 1] = temp1;
            data_base[rk_offset + 2] = temp2;
        }
    }
    __syncthreads();

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

    // Phase 2: Butterfly operations (thread-cooperative)
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
                u64 u_off = (k + j) * stride;
                u64 v_off = (k + j + m) * stride;
                
                XFieldElementArr u, v;
                u.coefficients[0] = data_base[u_off];
                u.coefficients[1] = data_base[u_off + 1];
                u.coefficients[2] = data_base[u_off + 2];
                v.coefficients[0] = data_base[v_off];
                v.coefficients[1] = data_base[v_off + 1];
                v.coefficients[2] = data_base[v_off + 2];

                xfe_mul_scalar(v, w_precalc[j]);
                XFieldElementArr u_plus_v = xfe_add(u, v);
                XFieldElementArr u_minus_v = xfe_sub(u, v);

                data_base[u_off] = u_plus_v.coefficients[0];
                data_base[u_off + 1] = u_plus_v.coefficients[1];
                data_base[u_off + 2] = u_plus_v.coefficients[2];
                data_base[v_off] = u_minus_v.coefficients[0];
                data_base[v_off + 1] = u_minus_v.coefficients[1];
                data_base[v_off + 2] = u_minus_v.coefficients[2];
            }
            k += 2 * m * blockDim.x;
        }
        __syncthreads();
        m *= 2;
    }

    // Phase 3: Butterfly operations (independent threads)
    for (u32 i = log2_blockDim; i < log2_slice_len; i++) {
        u64 w_m = omegas[i];
        u64 local_power_w = bfe_1;
        for (u32 j = 0; j < blockDim.x; j++) {
            local_power_w = bfe_mul(local_power_w, w_m);
        }
        __syncthreads();

        for (u32 j = threadIdx.x; j < blockDim.x; j += blockDim.x) {
            u64 temp_w = bfe_1;
            for (u32 k = 0; k < j; k++) {
                temp_w = bfe_mul(temp_w, w_m);
            }
            w_precalc[j] = temp_w;
        }
        __syncthreads();

        u32 k = 0;
        while (k < slice_len) {
            w_precalc_temp[threadIdx.x] = w_precalc[threadIdx.x];

            for (u32 j = threadIdx.x; j < m; j += blockDim.x) {
                u64 u_off = (k + j) * stride;
                u64 v_off = (k + j + m) * stride;
                
                XFieldElementArr u, v;
                u.coefficients[0] = data_base[u_off];
                u.coefficients[1] = data_base[u_off + 1];
                u.coefficients[2] = data_base[u_off + 2];
                v.coefficients[0] = data_base[v_off];
                v.coefficients[1] = data_base[v_off + 1];
                v.coefficients[2] = data_base[v_off + 2];

                xfe_mul_scalar(v, w_precalc_temp[threadIdx.x]);
                XFieldElementArr u_plus_v = xfe_add(u, v);
                XFieldElementArr u_minus_v = xfe_sub(u, v);

                data_base[u_off] = u_plus_v.coefficients[0];
                data_base[u_off + 1] = u_plus_v.coefficients[1];
                data_base[u_off + 2] = u_plus_v.coefficients[2];
                data_base[v_off] = u_minus_v.coefficients[0];
                data_base[v_off + 1] = u_minus_v.coefficients[1];
                data_base[v_off + 2] = u_minus_v.coefficients[2];

                w_precalc_temp[threadIdx.x] = bfe_mul(w_precalc_temp[threadIdx.x], local_power_w);
            }
            __syncthreads();
            k += 2 * m;
        }
        __syncthreads();
        m *= 2;
    }
}


extern "C" __global__ void ntt_xfield_extract(
    u64* data_global,
    u64 ntt_idx,
    u64* buffer,
    u64 ntt_count,
    u64 slice_len
){
    u64 globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    u64 kernelSize = gridDim.x * blockDim.x;

    for(u64 i = globalIdx; i < slice_len;i += kernelSize){
        buffer[i*3] = data_global[ntt_count * (i * 3) + ntt_idx];
        buffer[i*3+1] = data_global[ntt_count * (i * 3 + 1) + ntt_idx];
        buffer[i*3+2] = data_global[ntt_count * (i * 3 + 2) + ntt_idx];
    }
}

extern "C" __global__ void ntt_xfield_restore(
    u64* data_global,
    u64 ntt_idx,
    u64* buffer,
    u64 ntt_count,
    u64 slice_len
){
    u64 globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    u64 kernelSize = gridDim.x * blockDim.x;

    for(u64 i = globalIdx; i < slice_len;i += kernelSize){
        data_global[ntt_count * (i * 3) + ntt_idx] = buffer[i*3];
        data_global[ntt_count * (i * 3 + 1) + ntt_idx] = buffer[i*3+1];
        data_global[ntt_count * (i * 3 + 2) + ntt_idx] =  buffer[i*3+2];
    }
}

extern "C" __global__ void poly_fill_table_xfield(
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

    u64 * poly = all_poly + poly_starts[poly_idx] * 3ULL;
    u64 poly_len = poly_lengths[poly_idx];

    for(u64 i = threadIdx.x;i < poly_len; i += blockDim.x){
        table[3ULL * (i * table_cols + poly_idx) + 0] = poly[i * 3 + 0];
        table[3ULL * (i * table_cols + poly_idx) + 1] = poly[i * 3 + 1];
        table[3ULL * (i * table_cols + poly_idx) + 2] = poly[i * 3 + 2];
    }

    __syncthreads();

    for(u64 i = poly_len + threadIdx.x;i < table_rows; i += blockDim.x){
        table[3ULL * (i * table_cols + poly_idx) + 0] = 0;
        table[3ULL * (i * table_cols + poly_idx) + 1] = 0;
        table[3ULL * (i * table_cols + poly_idx) + 2] = 0;
    }
}

extern "C" __global__ void ntt_xfield_init_omegas(
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

extern "C" __global__ void ntt_xfield_fused_coset_single(
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
                u64 elem_offset = i * 3;
                u64 c0 = data[elem_offset];
                u64 c1 = data[elem_offset + 1];
                u64 c2 = data[elem_offset + 2];
                data[elem_offset] = bfe_mul(c0, offset_power);
                data[elem_offset + 1] = bfe_mul(c1, offset_power);
                data[elem_offset + 2] = bfe_mul(c2, offset_power);
                offset_power = bfe_mul(offset_power, offset_stride);
            }
        
        }
    }

    grid.sync();

    if(ntt_stage == 0){
        for (u64 k = globalIdx; k < slice_len; k += kernelSize) {
            u64 rk = bitreverse(k, log2_slice_len);
            if (k < rk) {
                u64 k_offset = k * 3;
                u64 rk_offset = rk * 3;
                u64 temp0 = data[k_offset];
                u64 temp1 = data[k_offset + 1];
                u64 temp2 = data[k_offset + 2];
                data[k_offset] = data[rk_offset];
                data[k_offset + 1] = data[rk_offset + 1];
                data[k_offset + 2] = data[rk_offset + 2];
                data[rk_offset] = temp0;
                data[rk_offset + 1] = temp1;
                data[rk_offset + 2] = temp2;
            }
        }
    }
    grid.sync();


    u64 m = 1ULL << ntt_stage;
    u64 group_size = m << 1;
    u64 omega_base = (1ULL << (ntt_stage)) - 1;
    //u64 omega_step = slice_len >> (ntt_stage + 1);  // stride between twiddles
    u64 w_m = omegas[ntt_stage];

    for (u64 pair = globalIdx; pair < (slice_len >> 1); pair += kernelSize) {
        u64 j = pair & (m - 1);
        u64 k = (pair >> ntt_stage) * group_size;

        u64 omega = omegas_store[omega_base + j];

        u64 u_off = (k + j) * 3;
        u64 v_off = (k + j + m) * 3;
                
        XFieldElementArr u, v;
        u.coefficients[0] = data[u_off];
        u.coefficients[1] = data[u_off + 1];
        u.coefficients[2] = data[u_off + 2];
        v.coefficients[0] = data[v_off];
        v.coefficients[1] = data[v_off + 1];
        v.coefficients[2] = data[v_off + 2];

        xfe_mul_scalar(v, omega);
        XFieldElementArr u_plus_v = xfe_add(u, v);
        XFieldElementArr u_minus_v = xfe_sub(u, v);

        data[u_off] = u_plus_v.coefficients[0];
        data[u_off + 1] = u_plus_v.coefficients[1];
        data[u_off + 2] = u_plus_v.coefficients[2];
        data[v_off] = u_minus_v.coefficients[0];
        data[v_off + 1] = u_minus_v.coefficients[1];
        data[v_off + 2] = u_minus_v.coefficients[2];
    }
}

extern "C" __global__ void ntt_xfield_fused_coset_single_interpolate(
    u64* data,
    u64 slice_len,
    u64 ntt_stage,
    u64 * omegas,
    u64 * omegas_store,
    u64 offset,
    u64 log2_slice_len,
    u64 unscale_param
){
    grid_group grid = this_grid();
    u64 globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    u64 kernelSize = gridDim.x * blockDim.x;


    if(ntt_stage == 0){
        for (u64 k = globalIdx; k < slice_len; k += kernelSize) {
            u64 rk = bitreverse(k, log2_slice_len);
            if (k < rk) {
                u64 k_offset = k * 3;
                u64 rk_offset = rk * 3;
                u64 temp0 = data[k_offset];
                u64 temp1 = data[k_offset + 1];
                u64 temp2 = data[k_offset + 2];
                data[k_offset] = data[rk_offset];
                data[k_offset + 1] = data[rk_offset + 1];
                data[k_offset + 2] = data[rk_offset + 2];
                data[rk_offset] = temp0;
                data[rk_offset + 1] = temp1;
                data[rk_offset + 2] = temp2;
            }
        }
    }
    grid.sync();

    u64 m = 1ULL << ntt_stage;
    u64 group_size = m << 1;
    u64 omega_base = (1ULL << (ntt_stage)) - 1;
    //u64 omega_step = slice_len >> (ntt_stage + 1);  // stride between twiddles
    u64 w_m = omegas[ntt_stage];

    for (u64 pair = globalIdx; pair < (slice_len >> 1); pair += kernelSize) {
        u64 j = pair & (m - 1);
        u64 k = (pair >> ntt_stage) * group_size;

        u64 omega = omegas_store[omega_base + j];

        u64 u_off = (k + j) * 3;
        u64 v_off = (k + j + m) * 3;

        XFieldElementArr u, v;
        u.coefficients[0] = data[u_off];
        u.coefficients[1] = data[u_off + 1];
        u.coefficients[2] = data[u_off + 2];
        v.coefficients[0] = data[v_off];
        v.coefficients[1] = data[v_off + 1];
        v.coefficients[2] = data[v_off + 2];

        xfe_mul_scalar(v, omega);
        XFieldElementArr u_plus_v = xfe_add(u, v);
        XFieldElementArr u_minus_v = xfe_sub(u, v);

        data[u_off] = u_plus_v.coefficients[0];
        data[u_off + 1] = u_plus_v.coefficients[1];
        data[u_off + 2] = u_plus_v.coefficients[2];
        data[v_off] = u_minus_v.coefficients[0];
        data[v_off + 1] = u_minus_v.coefficients[1];
        data[v_off + 2] = u_minus_v.coefficients[2];
    }

    grid.sync();

    if(ntt_stage == log2_slice_len - 1){

        for (u64 i = globalIdx; i < slice_len; i += kernelSize) {
            XFieldElementArr u;
            u.coefficients[0] = data[i * 3 + 0];
            u.coefficients[1] = data[i * 3 + 1];
            u.coefficients[2] = data[i * 3 + 2];
            xfe_mul_scalar(u,unscale_param);

            data[i * 3 + 0] = u.coefficients[0];
            data[i * 3 + 1] = u.coefficients[1];
            data[i * 3 + 2] = u.coefficients[2];
        }

        u64 bfe_1 = to_montgomery(1);
        if (offset != bfe_1) {
            u64 offset_stride = bfe_pow(offset, kernelSize);
            u64 offset_power =   bfe_pow(offset, globalIdx); //offset_powers[threadIdx.x];

            for (u64 i = globalIdx; i < slice_len; i += kernelSize) {
                u64 elem_offset = i * 3;
                u64 c0 = data[elem_offset];
                u64 c1 = data[elem_offset + 1];
                u64 c2 = data[elem_offset + 2];
                data[elem_offset] = bfe_mul(c0, offset_power);
                data[elem_offset + 1] = bfe_mul(c1, offset_power);
                data[elem_offset + 2] = bfe_mul(c2, offset_power);
                offset_power = bfe_mul(offset_power, offset_stride);
            }

        }
    }

}

extern "C" __global__ void ntt_xfield_fused_coset_strided(
    u64* data_global,
    u64 offset,
    u64 slice_len,
    u64 stride,
    u64* omegas,
    u32 log2_slice_len
) {
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("in_kernel_mod ntt_xfield_fused_coset_strided %d %d\n",gridDim.x,blockDim.x);
    }

    u64 col_idx = blockIdx.x;
    u64 col_base = col_idx * 3;  // XFieldElementArr is 3 u64s
    u64* data_base = data_global + col_base;

    // Coset Scaling (in-place on global memory with stride)
    u64 bfe_1 = to_montgomery(1);
    if (offset != bfe_1) {
        __shared__ u64 offset_powers[1024];
        if (threadIdx.x < blockDim.x) {
            offset_powers[threadIdx.x] = bfe_pow(offset, threadIdx.x);
        }
        __syncthreads();

        u64 offset_stride_pow = bfe_pow(offset, blockDim.x);
        u64 offset_power = offset_powers[threadIdx.x];

        for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
            u64 elem_offset = i * stride;
            u64 c0 = data_base[elem_offset];
            u64 c1 = data_base[elem_offset + 1];
            u64 c2 = data_base[elem_offset + 2];
            data_base[elem_offset] = bfe_mul(c0, offset_power);
            data_base[elem_offset + 1] = bfe_mul(c1, offset_power);
            data_base[elem_offset + 2] = bfe_mul(c2, offset_power);
            offset_power = bfe_mul(offset_power, offset_stride_pow);
        }
        __syncthreads();
    }

    // NTT (in-place on global memory with stride)
    ntt_core_xfield_strided(data_base, stride, slice_len, omegas, log2_slice_len);

    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("END in_kernel_mod ntt_xfield_fused_coset_strided %d %d\n",gridDim.x,blockDim.x);
    }
}

//============================================================================
// Strided INTT Kernel (for row-major table data)
//============================================================================

/// Strided INTT kernel for row-major table data with XFieldElements
/// This kernel operates on data stored in row-major format where columns are
/// accessed with a stride. Each XFieldElementArr is 3 u64s.
///
/// Grid configuration:
///   - gridDim.x = num_columns (number of columns to process)
///   - blockDim.x = threads per block (typically 256)
///
/// Parameters:
///   data_global: Input/output array in row-major format (XFieldElements as 3 contiguous u64s)
///   slice_len: Length of each column (number of rows)
///   stride: Distance between consecutive XFieldElements in a column (= num_columns * 3)
///   omegas_inv: Inverse NTT twiddle factors (in Montgomery form)
///   log2_slice_len: log2 of slice_len
///
/// Note: Caller must divide by array length after calling this
extern "C" __global__ void intt_xfield_strided(
    u64* data_global,
    u64 slice_len,
    u64 stride,
    u64* omegas_inv,
    u32 log2_slice_len
) {
    // Calculate column base index
    u64 col_idx = blockIdx.x;
    u64 col_base = col_idx * 3;  // XFieldElementArr is 3 u64s
    u64* data_base = data_global + col_base;

    // Perform INTT using inverse twiddle factors
    ntt_core_xfield_strided(data_base, stride, slice_len, omegas_inv, log2_slice_len);
}

//============================================================================
// Fused INTT + Unscaling Kernel for XFieldElementArr
//============================================================================

/// Fused INTT + unscaling kernel for XFieldElementArr arrays
/// This kernel combines inverse NTT with the unscaling step (multiply by n_inv)
/// to eliminate CPU postprocessing overhead
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of polynomials)
///   - blockDim.x = threads per block (typically 256-1024)
///
/// Parameters:
///   poly_global: Input/output array of polynomials (each slice_len XFieldElements)
///   slice_len: Length of each polynomial (must be power of 2)
///   omegas_inv: Inverse NTT twiddle factors (in Montgomery form)
///   n_inv: Inverse of the domain length (in Montgomery form)
///   log2_slice_len: log2 of slice_len
extern "C" __global__ void intt_xfield_fused_unscale(
    u64* poly_global,
    u64 slice_len,
    u64* omegas_inv,
    u64 n_inv,
    u32 log2_slice_len
) {
    XFieldElementArr* poly = ((XFieldElementArr*)poly_global) + blockIdx.x * slice_len;

    // ========================================================================
    // Phase 1: INTT (existing algorithm)
    // ========================================================================
    ntt_core_xfield(poly, slice_len, omegas_inv, log2_slice_len);

    __syncthreads();  // Ensure INTT is complete before unscaling

    // ========================================================================
    // Phase 2: Unscaling (multiply all elements by n_inv)
    // ========================================================================
    // Each thread processes multiple elements
    for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
        xfe_mul_scalar(poly[i], n_inv);
    }
}

//============================================================================
// Fused INTT + Unscaling + Randomizer Kernel for XFieldElementArr
//============================================================================

/// Fused INTT + unscaling + randomizer addition kernel for XFieldElementArr arrays
/// This kernel performs:
///   1. Inverse NTT
///   2. Unscaling (multiply by n_inv)
///   3. Add zerofier-multiplied randomizer for zero-knowledge
///
/// Grid configuration:
///   - gridDim.x = batch_size (number of polynomials)
///   - blockDim.x = threads per block (typically 256-1024)
///
/// Parameters:
///   poly_global: Input/output array of polynomials (size: batch * (slice_len + num_randomizers) * 3)
///   slice_len: Length of each polynomial (must be power of 2)
///   num_randomizers: Number of randomizer coefficients
///   omegas_inv: Inverse NTT twiddle factors (in Montgomery form)
///   n_inv: Inverse of the domain length (in Montgomery form, BFieldElement)
///   randomizers: Randomizer coefficients (batch * num_randomizers * 3 for XField)
///   offset_power_n: offset^slice_len (in Montgomery form, BFieldElement)
///   log2_slice_len: log2 of slice_len
extern "C" __global__ void intt_xfield_fused_unscale_randomize(
    u64* poly_global,
    u64 slice_len,
    u32 num_randomizers,
    u64* omegas_inv,
    u64 n_inv,
    u64* randomizers,  // [batch_size * num_randomizers * 3] array for XField
    u64 offset_power_n,
    u32 log2_slice_len
) {
    u64 total_len = slice_len + num_randomizers;
    XFieldElementArr* poly = ((XFieldElementArr*)poly_global) + blockIdx.x * total_len;
    XFieldElementArr* batch_randomizers = ((XFieldElementArr*)randomizers) + blockIdx.x * num_randomizers;

    // ========================================================================
    // Phase 1: INTT on first slice_len elements
    // ========================================================================
    ntt_core_xfield(poly, slice_len, omegas_inv, log2_slice_len);

    __syncthreads();

    // ========================================================================
    // Phase 2: Unscaling (multiply first slice_len elements by n_inv)
    // ========================================================================
    for (u64 i = threadIdx.x; i < slice_len; i += blockDim.x) {
        xfe_mul_scalar(poly[i], n_inv);
    }

    __syncthreads();

    // ========================================================================
    // Phase 3: Add randomizer (zerofier multiplication)
    // ========================================================================
    // Add randomizer coefficients at high-degree positions
    for (u32 i = threadIdx.x; i < num_randomizers; i += blockDim.x) {
        poly[slice_len + i] = batch_randomizers[i];
    }

    // Subtract randomizer * offset_power_n from low-degree positions
    for (u32 i = threadIdx.x; i < num_randomizers; i += blockDim.x) {
        XFieldElementArr scaled_rand = batch_randomizers[i];
        xfe_mul_scalar(scaled_rand, offset_power_n);
        poly[i] = xfe_sub(poly[i], scaled_rand);
    }
}
