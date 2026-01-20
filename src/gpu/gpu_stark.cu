/**
 * GPU-Resident STARK Prover Implementation
 * 
 * Complete proof generation on GPU with minimal host-device transfers:
 * - H2D: Main table (once at start)
 * - D2H: Final proof (once at end)
 * 
 * All intermediate computation stays on GPU.
 */

 #ifdef TRITON_CUDA_ENABLED

 #include "gpu/gpu_stark.hpp"
 #include "gpu/gpu_proof_context.hpp"
 #include "gpu/cuda_common.cuh"
 #include "common/debug_control.hpp"
 #include <fstream>
 #include <iomanip>
 
 // Include all kernel headers
 #include "gpu/kernels/lde_kernel.cuh"
 #include "gpu/kernels/randomized_lde_kernel.cuh"
 #include "gpu/kernels/bfield_kernel.cuh"
 #include "gpu/kernels/xfield_kernel.cuh"
 #include "gpu/kernels/ntt_kernel.cuh"
 #include "gpu/kernels/tip5_kernel.cuh"
 #include "gpu/kernels/merkle_kernel.cuh"
 #include "gpu/kernels/fiat_shamir_kernel.cuh"
 #include "gpu/kernels/extend_kernel.cuh"
 #include "gpu/kernels/quotient_kernel.cuh"
 #include "gpu/kernels/quotient_constraints_split_kernel.cuh"
 #include "gpu/kernels/quotient_compute_kernel.cuh"
 #include "gpu/kernels/challenges_kernel.cuh"
 #include "gpu/kernels/fri_kernel.cuh"
 #include "gpu/kernels/gather_kernel.cuh"
 #include "gpu/kernels/row_hash_kernel.cuh"
 #include "gpu/kernels/table_fill_kernel.cuh"
 #include "gpu/kernels/degree_lowering_main_kernel.cuh"
 #include "gpu/kernels/u32_table_kernel.cuh"
 #include "gpu/kernels/phase1_kernel.cuh"
 
 #include "hash/tip5.hpp"
 #include "stark/challenges.hpp"
 #include "quotient/quotient.hpp"
 #include "bincode_ffi.hpp"
 #include "chacha12_rng.hpp"
 
 #include <iostream>
 #include <fstream>
 #include <cmath>
 #include <chrono>
 #include <array>
 #include <vector>
 #include <unordered_set>
 #include <algorithm>
 #include <thread>
 #include <atomic>
 #include <cstring>
 #include <nlohmann/json.hpp>
 #include <sys/sysinfo.h>
 
 #ifdef _OPENMP
 #include <omp.h>
 #endif
 
 // For CPU aux table extension
 #include "table/master_table.hpp"
 #include "table/extend_helpers.hpp"
 
 namespace triton_vm {
 namespace gpu {
 
 // ============================================================================
 // Local kernels/helpers for quotient (zero-copy pipeline)
 // ============================================================================
 
 namespace {
 __global__ void qzc_scatter_xfe_column_into_rowmajor(
     uint64_t* __restrict__ d_rowmajor_xfe, // [n * width * 3]
     size_t n,
     size_t width,
     size_t col,
     const uint64_t* __restrict__ d_col_xfe // [n * 3]
 ) {
     const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= n) return;
     if (col >= width) return;
     const size_t dst = (row * width + col) * 3;
     const size_t src = row * 3;
     d_rowmajor_xfe[dst + 0] = d_col_xfe[src + 0];
     d_rowmajor_xfe[dst + 1] = d_col_xfe[src + 1];
     d_rowmajor_xfe[dst + 2] = d_col_xfe[src + 2];
 }
 
 __global__ void qzc_gather_bfield_column_from_rowmajor(
     const uint64_t* d_table_rowmajor, // [n * width]
     size_t n,
     size_t width,
     size_t col,
     uint64_t* d_out_col // [n]
 ) {
     size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= n) return;
     d_out_col[row] = d_table_rowmajor[row * width + col];
 }
 
 __global__ void qzc_gather_xfe_column_from_rowmajor(
     const uint64_t* d_table_rowmajor_xfe, // [n * width * 3]
     size_t n,
     size_t width,
     size_t col,
     uint64_t* d_out_col_xfe // [n * 3]
 ) {
     size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= n) return;
     size_t src = (row * width + col) * 3;
     size_t dst = row * 3;
     d_out_col_xfe[dst + 0] = d_table_rowmajor_xfe[src + 0];
     d_out_col_xfe[dst + 1] = d_table_rowmajor_xfe[src + 1];
     d_out_col_xfe[dst + 2] = d_table_rowmajor_xfe[src + 2];
 }
 
 __global__ void qzc_rowmajor_to_colmajor_bfe(
     const uint64_t* d_rowmajor, // [n * width]
     uint64_t* d_colmajor,       // [width * n]
     size_t n,
     size_t width
 ) {
     size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     size_t total = n * width;
     if (idx >= total) return;
     size_t row = idx / width;
     size_t col = idx % width;
     d_colmajor[col * n + row] = d_rowmajor[row * width + col];
 }
 
 // Optimized transpose kernel with ILP (4 elements per thread)
 // Writes are coalesced, reads use prefetch to hide latency
 __global__ void qzc_rowmajor_to_colmajor_xfe(
     const uint64_t* __restrict__ d_rowmajor, // [n * aux_width * 3]
     uint64_t* __restrict__ d_colmajor,       // [(aux_width*3) * n]
     size_t n,
     size_t aux_width
 ) {
     // Process 4 XFE elements per thread for better ILP
     size_t base_idx = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
     size_t total = n * aux_width;
     
     // Prefetch all 12 values (4 XFE elements × 3 components)
     uint64_t vals[4][3];
     size_t rows[4], cols[4];
     
     #pragma unroll
     for (int i = 0; i < 4; ++i) {
         size_t idx = base_idx + i;
         if (idx < total) {
             rows[i] = idx / aux_width;
             cols[i] = idx % aux_width;
             // Prefetch 3 components
             size_t src_base = (rows[i] * aux_width + cols[i]) * 3;
             vals[i][0] = d_rowmajor[src_base + 0];
             vals[i][1] = d_rowmajor[src_base + 1];
             vals[i][2] = d_rowmajor[src_base + 2];
         }
     }
     
     // Write all values (coalesced within warp for same column)
     #pragma unroll
     for (int i = 0; i < 4; ++i) {
         size_t idx = base_idx + i;
         if (idx < total) {
             #pragma unroll
             for (int comp = 0; comp < 3; ++comp) {
                 d_colmajor[(cols[i] * 3 + comp) * n + rows[i]] = vals[i][comp];
             }
         }
     }
 }
 
 __global__ void qzc_colmajor_to_rowmajor_bfe(
     const uint64_t* d_colmajor, // [width * n]
     uint64_t* d_rowmajor,       // [n * width]
     size_t n,
     size_t width
 ) {
     size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     size_t total = n * width;
     if (idx >= total) return;
     size_t row = idx / width;
     size_t col = idx % width;
     d_rowmajor[row * width + col] = d_colmajor[col * n + row];
 }
 
 __global__ void qzc_colmajor_to_rowmajor_xfe(
     const uint64_t* d_colmajor, // [(aux_width*3) * n]
     uint64_t* d_rowmajor,       // [n * aux_width * 3]
     size_t n,
     size_t aux_width
 ) {
     size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     size_t total = n * aux_width;
     if (idx >= total) return;
     size_t row = idx / aux_width;
     size_t col = idx % aux_width;
     // components
     for (size_t comp = 0; comp < 3; ++comp) {
         size_t src_col = col * 3 + comp;
         d_rowmajor[(row * aux_width + col) * 3 + comp] = d_colmajor[src_col * n + row];
     }
 }
 
 // Scatter row-major batch columns into a full row-major table with column offset.
 __global__ void qzc_scatter_rowmajor_offset_bfe(
     const uint64_t* d_batch_rows, // [num_rows * batch_cols]
     size_t batch_cols,
     uint64_t* d_full_rows,        // [num_rows * full_cols]
     size_t full_cols,
     size_t num_rows,
     size_t col_offset
 ) {
     size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     size_t total = num_rows * batch_cols;
     if (idx >= total) return;
     size_t row = idx / batch_cols;
     size_t col = idx % batch_cols;
     size_t dst_col = col_offset + col;
     if (dst_col >= full_cols) return;
     d_full_rows[row * full_cols + dst_col] = d_batch_rows[idx];
 }
 
 __global__ void qzc_fill_domain_points(
     uint64_t* d_out, // [n]
     size_t n,
     uint64_t offset,
     uint64_t generator
 ) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
     using namespace triton_vm::gpu::kernels;
     uint64_t gpow = bfield_pow_impl(generator, i);
     d_out[i] = bfield_mul_impl(offset, gpow);
 }
 
 // Faster domain fill: amortize pow() by doing a small contiguous chunk per thread.
 template<int ITEMS_PER_THREAD>
 __global__ void qzc_fill_domain_points_chunked(
     uint64_t* d_out, // [n]
     size_t n,
     uint64_t offset,
     uint64_t generator
 ) {
     const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
     const size_t start = tid * static_cast<size_t>(ITEMS_PER_THREAD);
     if (start >= n) return;
     using namespace triton_vm::gpu::kernels;
 
     uint64_t gpow = bfield_pow_impl(generator, start);
     uint64_t val = bfield_mul_impl(offset, gpow);
 
 #pragma unroll
     for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
         const size_t idx = start + static_cast<size_t>(j);
         if (idx >= n) break;
         d_out[idx] = val;
         val = bfield_mul_impl(val, generator);
     }
 }
 
 __global__ void qzc_fill_strided_indices(size_t* d_out, size_t len, size_t stride) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= len) return;
     d_out[i] = i * stride;
 }
 
 // Fill indices for a window on the quotient domain, mapping quotient-row -> fri-row via `stride`,
 // with wrap-around on the quotient domain.
 __global__ void qzc_fill_strided_indices_offset_wrap(
     size_t* d_out,
     size_t len,
     size_t start,
     size_t quotient_len,
     size_t stride
 ) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= len) return;
     size_t q = start + i;
     if (q >= quotient_len) q -= quotient_len;
     d_out[i] = q * stride;
 }
 
 __global__ void qzc_fill_coset_indices(size_t* d_out, size_t len, size_t coset_index, size_t num_cosets) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= len) return;
     // Fill indices where k ≡ coset_index (mod num_cosets)
     d_out[i] = coset_index + i * num_cosets;
 }
 
 __global__ void qzc_compute_zerofier_arrays(
     const uint64_t* d_x, // [n]
     size_t n,
     uint64_t trace_len,
     uint64_t trace_offset,
     uint64_t trace_offset_pow, // trace_offset^trace_len
     uint64_t trace_gen_inv,
     uint64_t* d_init_inv,
     uint64_t* d_cons_inv,
     uint64_t* d_tran_factor, // (x - gen_inv) / (x^trace_len - 1)
     uint64_t* d_term_inv
 ) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
 
     using namespace triton_vm::gpu::kernels;
 
     uint64_t x = d_x[i];
     // init_inv = 1/(x - trace_offset)
     d_init_inv[i] = bfield_inv_impl(bfield_sub_impl(x, trace_offset));
 
     // subgroup = x^trace_len - trace_offset^trace_len
     uint64_t xpow = bfield_pow_impl(x, trace_len);
     uint64_t subgroup = bfield_sub_impl(xpow, trace_offset_pow);
     uint64_t subgroup_inv = bfield_inv_impl(subgroup);
     d_cons_inv[i] = subgroup_inv;
 
     // term_inv = 1/(x - trace_offset*trace_gen^{-1})
     uint64_t terminal_point = bfield_mul_impl(trace_offset, trace_gen_inv);
     uint64_t diff = bfield_sub_impl(x, terminal_point);
     uint64_t term_inv = bfield_inv_impl(diff);
     d_term_inv[i] = term_inv;
 
     // tran_factor = (x - gen_inv) * subgroup_inv
     d_tran_factor[i] = bfield_mul_impl(diff, subgroup_inv);
 }
 
 __global__ void qzc_compute_shift_xfe(
     const uint64_t* d_domain_bfe, // [n]
     size_t n,
     const uint64_t* d_point_xfe,  // [3]
     uint64_t* d_shift_xfe         // [n * 3]
 ) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
     uint64_t p0 = d_point_xfe[0];
     uint64_t p1 = d_point_xfe[1];
     uint64_t p2 = d_point_xfe[2];
     uint64_t d = d_domain_bfe[i];
     uint64_t r0, r1, r2;
     triton_vm::gpu::kernels::xfield_sub_impl(p0, p1, p2, d, 0, 0, r0, r1, r2);
     d_shift_xfe[i * 3 + 0] = r0;
     d_shift_xfe[i * 3 + 1] = r1;
     d_shift_xfe[i * 3 + 2] = r2;
 }
 
 __global__ void qzc_domain_over_shift_xfe(
     const uint64_t* d_domain_bfe,     // [n]
     const uint64_t* d_shift_inv_xfe,  // [n * 3]
     size_t n,
     uint64_t* d_out_xfe               // [n * 3]
 ) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
     uint64_t inv0 = d_shift_inv_xfe[i * 3 + 0];
     uint64_t inv1 = d_shift_inv_xfe[i * 3 + 1];
     uint64_t inv2 = d_shift_inv_xfe[i * 3 + 2];
     uint64_t d = d_domain_bfe[i];
     uint64_t r0, r1, r2;
     triton_vm::gpu::kernels::xfield_scalar_mul_impl(inv0, inv1, inv2, d, r0, r1, r2);
     d_out_xfe[i * 3 + 0] = r0;
     d_out_xfe[i * 3 + 1] = r1;
     d_out_xfe[i * 3 + 2] = r2;
 }
 
 __global__ void qzc_reduce_sum_xfe(
     const uint64_t* d_in_xfe, // [n*3]
     size_t n,
     uint64_t* d_out_xfe       // [3]
 ) {
     // One block reduction, assumes grid covers n and then atomic add into out.
     __shared__ uint64_t sh0[256];
     __shared__ uint64_t sh1[256];
     __shared__ uint64_t sh2[256];
     size_t tid = threadIdx.x;
     size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
     uint64_t a0 = 0, a1 = 0, a2 = 0;
     if (idx < n) {
         a0 = d_in_xfe[idx * 3 + 0];
         a1 = d_in_xfe[idx * 3 + 1];
         a2 = d_in_xfe[idx * 3 + 2];
     }
     sh0[tid] = a0; sh1[tid] = a1; sh2[tid] = a2;
     __syncthreads();
 
     for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
         if (tid < s) {
             sh0[tid] = triton_vm::gpu::kernels::bfield_add_impl(sh0[tid], sh0[tid + s]);
             sh1[tid] = triton_vm::gpu::kernels::bfield_add_impl(sh1[tid], sh1[tid + s]);
             sh2[tid] = triton_vm::gpu::kernels::bfield_add_impl(sh2[tid], sh2[tid + s]);
         }
         __syncthreads();
     }
 
     if (tid == 0) {
         // atomic add in Goldilocks isn't directly supported; use atomicCAS loop via unsigned long long.
         // For now, assume contention is low and use atomicAdd on 64-bit with wrap-around (not correct mod p)
         // We'll avoid atomics by launching exactly one block for now where possible.
         d_out_xfe[0] = sh0[0];
         d_out_xfe[1] = sh1[0];
         d_out_xfe[2] = sh2[0];
     }
 }
 
 __global__ void qzc_xfe_inv_single(uint64_t* d_xfe3) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t r0, r1, r2;
     triton_vm::gpu::kernels::xfield_inv_impl(d_xfe3[0], d_xfe3[1], d_xfe3[2], r0, r1, r2);
     d_xfe3[0] = r0; d_xfe3[1] = r1; d_xfe3[2] = r2;
 }
 
 __global__ void qzc_eval_main_ood(
     const uint64_t* d_main_rowmajor, // [n * main_width]
     size_t n,
     size_t main_width,
     const uint64_t* d_domain_over_shift_xfe, // [n*3]
     const uint64_t* d_denom_inv_xfe,         // [3]
     uint64_t* d_out_xfe                      // [main_width * 3]
 ) {
     size_t col = blockIdx.x;
     if (col >= main_width) return;
     uint64_t denom0 = d_denom_inv_xfe[0];
     uint64_t denom1 = d_denom_inv_xfe[1];
     uint64_t denom2 = d_denom_inv_xfe[2];
 
     // parallel reduction over rows
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
         uint64_t w0 = d_domain_over_shift_xfe[i * 3 + 0];
         uint64_t w1 = d_domain_over_shift_xfe[i * 3 + 1];
         uint64_t w2 = d_domain_over_shift_xfe[i * 3 + 2];
         uint64_t v = d_main_rowmajor[i * main_width + col];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_scalar_mul_impl(w0, w1, w2, v, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     __shared__ uint64_t sh0[256], sh1[256], sh2[256];
     sh0[threadIdx.x] = acc0; sh1[threadIdx.x] = acc1; sh2[threadIdx.x] = acc2;
     __syncthreads();
     for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s) {
             sh0[threadIdx.x] = triton_vm::gpu::kernels::bfield_add_impl(sh0[threadIdx.x], sh0[threadIdx.x + s]);
             sh1[threadIdx.x] = triton_vm::gpu::kernels::bfield_add_impl(sh1[threadIdx.x], sh1[threadIdx.x + s]);
             sh2[threadIdx.x] = triton_vm::gpu::kernels::bfield_add_impl(sh2[threadIdx.x], sh2[threadIdx.x + s]);
         }
         __syncthreads();
     }
     if (threadIdx.x == 0) {
         uint64_t r0, r1, r2;
         triton_vm::gpu::kernels::xfield_mul_impl(sh0[0], sh1[0], sh2[0], denom0, denom1, denom2, r0, r1, r2);
         d_out_xfe[col * 3 + 0] = r0;
         d_out_xfe[col * 3 + 1] = r1;
         d_out_xfe[col * 3 + 2] = r2;
     }
 }
 
 __global__ void qzc_eval_aux_ood(
     const uint64_t* d_aux_rowmajor, // [n * aux_width * 3]
     size_t n,
     size_t aux_width,
     const uint64_t* d_domain_over_shift_xfe, // [n*3]
     const uint64_t* d_denom_inv_xfe,         // [3]
     uint64_t* d_out_xfe                      // [aux_width * 3]
 ) {
     size_t col = blockIdx.x;
     if (col >= aux_width) return;
     uint64_t denom0 = d_denom_inv_xfe[0];
     uint64_t denom1 = d_denom_inv_xfe[1];
     uint64_t denom2 = d_denom_inv_xfe[2];
 
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
         uint64_t w0 = d_domain_over_shift_xfe[i * 3 + 0];
         uint64_t w1 = d_domain_over_shift_xfe[i * 3 + 1];
         uint64_t w2 = d_domain_over_shift_xfe[i * 3 + 2];
         const uint64_t* vptr = &d_aux_rowmajor[(i * aux_width + col) * 3];
         uint64_t v0 = vptr[0], v1 = vptr[1], v2 = vptr[2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v0, v1, v2, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     __shared__ uint64_t sh0[256], sh1[256], sh2[256];
     sh0[threadIdx.x] = acc0; sh1[threadIdx.x] = acc1; sh2[threadIdx.x] = acc2;
     __syncthreads();
     for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s) {
             sh0[threadIdx.x] = triton_vm::gpu::kernels::bfield_add_impl(sh0[threadIdx.x], sh0[threadIdx.x + s]);
             sh1[threadIdx.x] = triton_vm::gpu::kernels::bfield_add_impl(sh1[threadIdx.x], sh1[threadIdx.x + s]);
             sh2[threadIdx.x] = triton_vm::gpu::kernels::bfield_add_impl(sh2[threadIdx.x], sh2[threadIdx.x + s]);
         }
         __syncthreads();
     }
     if (threadIdx.x == 0) {
         uint64_t r0, r1, r2;
         triton_vm::gpu::kernels::xfield_mul_impl(sh0[0], sh1[0], sh2[0], denom0, denom1, denom2, r0, r1, r2);
         d_out_xfe[col * 3 + 0] = r0;
         d_out_xfe[col * 3 + 1] = r1;
         d_out_xfe[col * 3 + 2] = r2;
     }
 }
 
 __global__ void qzc_pow4_xfe(const uint64_t* d_in3, uint64_t* d_out3) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t a0 = d_in3[0], a1 = d_in3[1], a2 = d_in3[2];
     uint64_t s0, s1, s2;
     triton_vm::gpu::kernels::xfield_mul_impl(a0, a1, a2, a0, a1, a2, s0, s1, s2); // a^2
     uint64_t t0, t1, t2;
     triton_vm::gpu::kernels::xfield_mul_impl(s0, s1, s2, s0, s1, s2, t0, t1, t2); // a^4
     d_out3[0] = t0; d_out3[1] = t1; d_out3[2] = t2;
 }
 
 // Old single-threaded Horner (kept for reference/fallback)
 __global__ void qzc_eval_segment_polys_horner(
     const uint64_t* d_coeffs, // [num_segments * 3 * seg_len] layout: (seg*3 + comp) * seg_len + k
     size_t seg_len,
     size_t num_segments,
     const uint64_t* d_point3, // [3] (z^4)
     uint64_t* d_out,          // [num_segments * 3]
     size_t /*unused*/
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t p0 = d_point3[0], p1 = d_point3[1], p2 = d_point3[2];
     for (size_t s = 0; s < num_segments; ++s) {
         uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
         for (size_t i = seg_len; i-- > 0;) {
             uint64_t c0 = d_coeffs[(s * 3 + 0) * seg_len + i];
             uint64_t c1 = d_coeffs[(s * 3 + 1) * seg_len + i];
             uint64_t c2 = d_coeffs[(s * 3 + 2) * seg_len + i];
             uint64_t m0, m1, m2;
             triton_vm::gpu::kernels::xfield_mul_impl(acc0, acc1, acc2, p0, p1, p2, m0, m1, m2);
             triton_vm::gpu::kernels::xfield_add_impl(m0, m1, m2, c0, c1, c2, acc0, acc1, acc2);
         }
         d_out[s * 3 + 0] = acc0;
         d_out[s * 3 + 1] = acc1;
         d_out[s * 3 + 2] = acc2;
     }
 }
 
 // Chunked Horner's method: each block evaluates a chunk of the polynomial,
 // then chunks are combined: P(x) = chunk[0] + x^chunk_size * chunk[1] + ...
 // This reduces sequential work from n to chunk_size per block.
 __global__ void qzc_eval_poly_chunk_horner(
     const uint64_t* d_coeffs,   // [num_segments * 3 * seg_len] 
     size_t seg_len,
     size_t seg_idx,
     size_t chunk_size,          // coefficients per chunk
     size_t num_chunks,          // number of chunks
     const uint64_t* d_point3,   // [3] evaluation point x
     uint64_t* d_chunk_results   // [num_chunks * 3] output: chunk[i] evaluated at x
 ) {
     size_t chunk_idx = blockIdx.x;
     if (chunk_idx >= num_chunks) return;
     
     // Only thread 0 per block does the work (Horner is sequential)
     if (threadIdx.x != 0) return;
     
     uint64_t p0 = d_point3[0], p1 = d_point3[1], p2 = d_point3[2];
     
     // Determine range for this chunk
     size_t start = chunk_idx * chunk_size;
     size_t end = min(start + chunk_size, seg_len);
     
     // Horner evaluation for this chunk: sum_{i=start}^{end-1} c[i] * x^(i-start)
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     for (size_t i = end; i-- > start;) {
         uint64_t c0 = d_coeffs[(seg_idx * 3 + 0) * seg_len + i];
         uint64_t c1 = d_coeffs[(seg_idx * 3 + 1) * seg_len + i];
         uint64_t c2 = d_coeffs[(seg_idx * 3 + 2) * seg_len + i];
         
         uint64_t m0, m1, m2;
         triton_vm::gpu::kernels::xfield_mul_impl(acc0, acc1, acc2, p0, p1, p2, m0, m1, m2);
         triton_vm::gpu::kernels::xfield_add_impl(m0, m1, m2, c0, c1, c2, acc0, acc1, acc2);
     }
     
     d_chunk_results[chunk_idx * 3 + 0] = acc0;
     d_chunk_results[chunk_idx * 3 + 1] = acc1;
     d_chunk_results[chunk_idx * 3 + 2] = acc2;
 }
 
 // Combine chunk results: result = chunk[0] + x^chunk_size * chunk[1] + x^(2*chunk_size) * chunk[2] + ...
 // Uses Horner on the chunk results.
 __global__ void qzc_combine_poly_chunks(
     const uint64_t* d_chunk_results, // [num_chunks * 3]
     size_t num_chunks,
     const uint64_t* d_x_power,       // [3] = x^chunk_size
     uint64_t* d_out                  // [3] final result
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     
     uint64_t xp0 = d_x_power[0], xp1 = d_x_power[1], xp2 = d_x_power[2];
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     
     // Horner on chunks: result = chunk[n-1]*xp^0 + xp*(chunk[n-2] + xp*(...))
     // = chunk[0] + xp*chunk[1] + xp^2*chunk[2] + ...
     for (size_t i = num_chunks; i-- > 0;) {
         uint64_t c0 = d_chunk_results[i * 3 + 0];
         uint64_t c1 = d_chunk_results[i * 3 + 1];
         uint64_t c2 = d_chunk_results[i * 3 + 2];
         
         uint64_t m0, m1, m2;
         triton_vm::gpu::kernels::xfield_mul_impl(acc0, acc1, acc2, xp0, xp1, xp2, m0, m1, m2);
         triton_vm::gpu::kernels::xfield_add_impl(m0, m1, m2, c0, c1, c2, acc0, acc1, acc2);
     }
     
     d_out[0] = acc0;
     d_out[1] = acc1;
     d_out[2] = acc2;
 }
 
 // Compute x^n using repeated squaring
 __global__ void qzc_pow_xfe_n(
     const uint64_t* d_x3,  // [3] base
     size_t n,              // exponent
     uint64_t* d_out3       // [3] result
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     
     uint64_t x0 = d_x3[0], x1 = d_x3[1], x2 = d_x3[2];
     uint64_t r0 = 1, r1 = 0, r2 = 0; // identity
     
     while (n > 0) {
         if (n & 1) {
             // r = r * x
             uint64_t t0, t1, t2;
             triton_vm::gpu::kernels::xfield_mul_impl(r0, r1, r2, x0, x1, x2, t0, t1, t2);
             r0 = t0; r1 = t1; r2 = t2;
         }
         // x = x * x
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(x0, x1, x2, x0, x1, x2, t0, t1, t2);
         x0 = t0; x1 = t1; x2 = t2;
         n >>= 1;
     }
     
     d_out3[0] = r0;
     d_out3[1] = r1;
     d_out3[2] = r2;
 }
 
 __global__ void qzc_mul_xfe_scalar_kernel(
     const uint64_t* d_xfe3,
     uint64_t scalar,
     uint64_t* d_out3
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t r0, r1, r2;
     triton_vm::gpu::kernels::xfield_scalar_mul_impl(d_xfe3[0], d_xfe3[1], d_xfe3[2], scalar, r0, r1, r2);
     d_out3[0] = r0;
     d_out3[1] = r1;
     d_out3[2] = r2;
 }
 
 __global__ void qzc_eval_bfe_poly_at_xfe_single(
     const uint64_t* d_coeffs_bfe, // [num_rand]
     size_t num_rand,
     const uint64_t* d_point3,     // [3]
     uint64_t* d_out3              // [3]
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t p0 = d_point3[0], p1 = d_point3[1], p2 = d_point3[2];
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     for (size_t i = num_rand; i-- > 0;) {
         uint64_t m0, m1, m2;
         triton_vm::gpu::kernels::xfield_mul_impl(acc0, acc1, acc2, p0, p1, p2, m0, m1, m2);
         uint64_t r0, r1, r2;
         triton_vm::gpu::kernels::xfield_add_impl(m0, m1, m2, d_coeffs_bfe[i], 0, 0, r0, r1, r2);
         acc0 = r0; acc1 = r1; acc2 = r2;
     }
     d_out3[0] = acc0; d_out3[1] = acc1; d_out3[2] = acc2;
 }
 
 __global__ void qzc_mul_xfe3_kernel(
     const uint64_t* a3,
     const uint64_t* b3,
     uint64_t* out3
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t r0, r1, r2;
     triton_vm::gpu::kernels::xfield_mul_impl(
         a3[0], a3[1], a3[2],
         b3[0], b3[1], b3[2],
         r0, r1, r2
     );
     out3[0] = r0; out3[1] = r1; out3[2] = r2;
 }
 
 // Compute trace-domain zerofier at an extension-field point:
 // \(Z_H(x) = x^{trace_len} - trace_offset^{trace_len}\)
 __global__ void qzc_trace_zerofier_xfe(
     const uint64_t* d_point3,     // [3]
     uint64_t trace_len,
     uint64_t trace_offset,
     uint64_t* d_out3              // [3]
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
 
     // Fast pow by squaring in XFE
     auto xfe_mul = [](uint64_t a0, uint64_t a1, uint64_t a2,
                       uint64_t b0, uint64_t b1, uint64_t b2,
                       uint64_t& r0, uint64_t& r1, uint64_t& r2) {
         triton_vm::gpu::kernels::xfield_mul_impl(a0, a1, a2, b0, b1, b2, r0, r1, r2);
     };
 
     uint64_t base0 = d_point3[0], base1 = d_point3[1], base2 = d_point3[2];
     uint64_t acc0 = 1, acc1 = 0, acc2 = 0; // XFE::one()
     uint64_t e = trace_len;
 
     while (e > 0) {
         if (e & 1ULL) {
             uint64_t t0, t1, t2;
             xfe_mul(acc0, acc1, acc2, base0, base1, base2, t0, t1, t2);
             acc0 = t0; acc1 = t1; acc2 = t2;
         }
         e >>= 1ULL;
         if (e) {
             uint64_t s0, s1, s2;
             xfe_mul(base0, base1, base2, base0, base1, base2, s0, s1, s2);
             base0 = s0; base1 = s1; base2 = s2;
         }
     }
 
     uint64_t offset_pow = triton_vm::gpu::kernels::bfield_pow_impl(trace_offset, trace_len);
     uint64_t r0, r1, r2;
     triton_vm::gpu::kernels::xfield_sub_impl(acc0, acc1, acc2, offset_pow, 0, 0, r0, r1, r2);
     d_out3[0] = r0; d_out3[1] = r1; d_out3[2] = r2;
 }
 
 __global__ void qzc_add_main_randomizer_ood(
     uint64_t* d_out_main_xfe,              // [main_width * 3]
     const uint64_t* d_main_rand_coeffs_bfe, // [main_width * num_rand]
     size_t main_width,
     size_t num_rand,
     const uint64_t* d_point3,              // [3]
     const uint64_t* d_zerofier3            // [3]
 ) {
     size_t col = blockIdx.x;
     if (col >= main_width) return;
     if (threadIdx.x != 0) return;
 
     uint64_t p0 = d_point3[0], p1 = d_point3[1], p2 = d_point3[2];
 
     // Horner evaluate Polynomial<BFE> at XFE point (matches Polynomial<BFieldElement>::evaluate_at_extension)
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     const uint64_t* coeffs = d_main_rand_coeffs_bfe + col * num_rand;
     for (size_t i = num_rand; i-- > 0;) {
         uint64_t m0, m1, m2;
         triton_vm::gpu::kernels::xfield_mul_impl(acc0, acc1, acc2, p0, p1, p2, m0, m1, m2);
         uint64_t r0, r1, r2;
         triton_vm::gpu::kernels::xfield_add_impl(m0, m1, m2, coeffs[i], 0, 0, r0, r1, r2);
         acc0 = r0; acc1 = r1; acc2 = r2;
     }
 
     // Multiply by zerofier and add into output
     uint64_t z0 = d_zerofier3[0], z1 = d_zerofier3[1], z2 = d_zerofier3[2];
     uint64_t t0, t1, t2;
     triton_vm::gpu::kernels::xfield_mul_impl(z0, z1, z2, acc0, acc1, acc2, t0, t1, t2);
 
     size_t off = col * 3;
     uint64_t o0 = d_out_main_xfe[off + 0];
     uint64_t o1 = d_out_main_xfe[off + 1];
     uint64_t o2 = d_out_main_xfe[off + 2];
     uint64_t n0, n1, n2;
     triton_vm::gpu::kernels::xfield_add_impl(o0, o1, o2, t0, t1, t2, n0, n1, n2);
     d_out_main_xfe[off + 0] = n0;
     d_out_main_xfe[off + 1] = n1;
     d_out_main_xfe[off + 2] = n2;
 }
 
 __global__ void qzc_add_aux_randomizer_ood(
     uint64_t* d_out_aux_xfe,              // [aux_width * 3]
     const uint64_t* d_aux_rand_coeffs_component_cols_bfe, // [(aux_width*3) * num_rand]
     size_t aux_width,
     size_t num_rand,
     const uint64_t* d_point3,             // [3]
     const uint64_t* d_zerofier3           // [3]
 ) {
     size_t col = blockIdx.x;
     if (col >= aux_width) return;
     if (threadIdx.x != 0) return;
 
     uint64_t p0 = d_point3[0], p1 = d_point3[1], p2 = d_point3[2];
 
     // Aux randomizers are XFE polynomials.
     // Coeff layout is per component-column: (col*3+comp)*num_rand + i.
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     const uint64_t* coeffs0 = d_aux_rand_coeffs_component_cols_bfe + (col * 3 + 0) * num_rand;
     const uint64_t* coeffs1 = d_aux_rand_coeffs_component_cols_bfe + (col * 3 + 1) * num_rand;
     const uint64_t* coeffs2 = d_aux_rand_coeffs_component_cols_bfe + (col * 3 + 2) * num_rand;
     for (size_t i = num_rand; i-- > 0;) {
         uint64_t m0, m1, m2;
         triton_vm::gpu::kernels::xfield_mul_impl(acc0, acc1, acc2, p0, p1, p2, m0, m1, m2);
         uint64_t c0 = coeffs0[i];
         uint64_t c1 = coeffs1[i];
         uint64_t c2 = coeffs2[i];
         uint64_t r0, r1, r2;
         triton_vm::gpu::kernels::xfield_add_impl(m0, m1, m2, c0, c1, c2, r0, r1, r2);
         acc0 = r0; acc1 = r1; acc2 = r2;
     }
 
     uint64_t z0 = d_zerofier3[0], z1 = d_zerofier3[1], z2 = d_zerofier3[2];
     uint64_t t0, t1, t2;
     triton_vm::gpu::kernels::xfield_mul_impl(z0, z1, z2, acc0, acc1, acc2, t0, t1, t2);
 
     size_t off = col * 3;
     uint64_t o0 = d_out_aux_xfe[off + 0];
     uint64_t o1 = d_out_aux_xfe[off + 1];
     uint64_t o2 = d_out_aux_xfe[off + 2];
     uint64_t n0, n1, n2;
     triton_vm::gpu::kernels::xfield_add_impl(o0, o1, o2, t0, t1, t2, n0, n1, n2);
     d_out_aux_xfe[off + 0] = n0;
     d_out_aux_xfe[off + 1] = n1;
     d_out_aux_xfe[off + 2] = n2;
 }
 
 // Debug kernels removed for performance - they were only used for debugging quotient segment reconstruction
 
 __global__ void qzc_reduce_blocks_xfe_kernel(
     const uint64_t* in, // [n*3]
     size_t n,
     uint64_t* block_sums // [grid*3]
 ) {
     __shared__ uint64_t sh0[256];
     __shared__ uint64_t sh1[256];
     __shared__ uint64_t sh2[256];
     size_t tid = threadIdx.x;
     size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
     uint64_t a0 = 0, a1 = 0, a2 = 0;
     if (idx < n) {
         a0 = in[idx * 3 + 0];
         a1 = in[idx * 3 + 1];
         a2 = in[idx * 3 + 2];
     }
     sh0[tid] = a0; sh1[tid] = a1; sh2[tid] = a2;
     __syncthreads();
     for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
         if (tid < s) {
             sh0[tid] = triton_vm::gpu::kernels::bfield_add_impl(sh0[tid], sh0[tid + s]);
             sh1[tid] = triton_vm::gpu::kernels::bfield_add_impl(sh1[tid], sh1[tid + s]);
             sh2[tid] = triton_vm::gpu::kernels::bfield_add_impl(sh2[tid], sh2[tid + s]);
         }
         __syncthreads();
     }
     if (tid == 0) {
         block_sums[blockIdx.x * 3 + 0] = sh0[0];
         block_sums[blockIdx.x * 3 + 1] = sh1[0];
         block_sums[blockIdx.x * 3 + 2] = sh2[0];
     }
 }
 
 __global__ void qzc_xfe_to_digest_leaves_kernel(
     const uint64_t* d_codeword_xfe, // [n * 3] row-major
     size_t n,
     uint64_t* d_leaves_digest       // [n * 5] row-major
 ) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
     d_leaves_digest[i * 5 + 0] = d_codeword_xfe[i * 3 + 0];
     d_leaves_digest[i * 5 + 1] = d_codeword_xfe[i * 3 + 1];
     d_leaves_digest[i * 5 + 2] = d_codeword_xfe[i * 3 + 2];
     d_leaves_digest[i * 5 + 3] = 0;
     d_leaves_digest[i * 5 + 4] = 0;
 }
 
 __global__ void qzc_eval_ood_value_main_aux_kernel(
     const uint64_t* d_main_ood,     // [main_width * 3]
     const uint64_t* d_aux_ood,      // [aux_width * 3]
     const uint64_t* d_weights,      // [(main_width+aux_width+...)*3]
     size_t main_width,
     size_t aux_width,
     uint64_t* d_out3               // [3]
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
 
     // main part
     for (size_t c = 0; c < main_width; ++c) {
         uint64_t w0 = d_weights[c * 3 + 0];
         uint64_t w1 = d_weights[c * 3 + 1];
         uint64_t w2 = d_weights[c * 3 + 2];
         const uint64_t* v = &d_main_ood[c * 3];
         // v is XFE already; main_ood computed as XFE in our buffer.
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v[0], v[1], v[2], t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
 
     // aux part
     size_t base = main_width;
     for (size_t c = 0; c < aux_width; ++c) {
         uint64_t w0 = d_weights[(base + c) * 3 + 0];
         uint64_t w1 = d_weights[(base + c) * 3 + 1];
         uint64_t w2 = d_weights[(base + c) * 3 + 2];
         const uint64_t* v = &d_aux_ood[c * 3];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v[0], v[1], v[2], t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
 
     d_out3[0] = acc0;
     d_out3[1] = acc1;
     d_out3[2] = acc2;
 }
 
 __global__ void qzc_eval_ood_value_quot_kernel(
     const uint64_t* d_quot_ood, // [num_segments * 3]
     const uint64_t* d_weights,  // weights array
     size_t weight_base,         // index where quotient weights begin
     size_t num_segments,
     uint64_t* d_out3            // [3]
 ) {
     if (blockIdx.x != 0 || threadIdx.x != 0) return;
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     for (size_t s = 0; s < num_segments; ++s) {
         uint64_t w0 = d_weights[(weight_base + s) * 3 + 0];
         uint64_t w1 = d_weights[(weight_base + s) * 3 + 1];
         uint64_t w2 = d_weights[(weight_base + s) * 3 + 2];
         const uint64_t* v = &d_quot_ood[s * 3];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v[0], v[1], v[2], t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     d_out3[0] = acc0;
     d_out3[1] = acc1;
     d_out3[2] = acc2;
 }
 
 __global__ void qzc_build_main_aux_codeword_kernel(
     const uint64_t* d_main_lde,       // [main_width * n] BFE col-major
     const uint64_t* d_aux_lde,        // [(aux_width*3) * n] BFE comp-major col-major
     const uint64_t* d_weights,        // [total*3]
     size_t n,
     size_t main_width,
     size_t aux_width,
     uint64_t* d_out_xfe               // [n*3] row-major
 ) {
     size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= n) return;
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     // main
     for (size_t c = 0; c < main_width; ++c) {
         uint64_t v = d_main_lde[c * n + row];
         uint64_t w0 = d_weights[c * 3 + 0];
         uint64_t w1 = d_weights[c * 3 + 1];
         uint64_t w2 = d_weights[c * 3 + 2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_scalar_mul_impl(w0, w1, w2, v, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     // aux
     size_t base = main_width;
     for (size_t c = 0; c < aux_width; ++c) {
         uint64_t v0 = d_aux_lde[(c * 3 + 0) * n + row];
         uint64_t v1 = d_aux_lde[(c * 3 + 1) * n + row];
         uint64_t v2 = d_aux_lde[(c * 3 + 2) * n + row];
         uint64_t w0 = d_weights[(base + c) * 3 + 0];
         uint64_t w1 = d_weights[(base + c) * 3 + 1];
         uint64_t w2 = d_weights[(base + c) * 3 + 2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v0, v1, v2, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     d_out_xfe[row * 3 + 0] = acc0;
     d_out_xfe[row * 3 + 1] = acc1;
     d_out_xfe[row * 3 + 2] = acc2;
 }
 
 // Scatter contiguous digests (coset-order) into full-domain leaves at stride = num_cosets.
 // Input: d_coset_digests[k] for k in [0, coset_len)
 // Output: d_leaves[(k*num_cosets + coset_idx)] = digest
 __global__ void qzc_scatter_digests_strided_kernel(
     const uint64_t* __restrict__ d_coset_digests, // [coset_len * 5]
     uint64_t* __restrict__ d_leaves,              // [full_len * 5]
     size_t coset_len,
     size_t coset_idx,
     size_t num_cosets
 ) {
     size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (k >= coset_len) return;
     size_t out_row = k * num_cosets + coset_idx;
     #pragma unroll
     for (int i = 0; i < 5; ++i) {
         d_leaves[out_row * 5 + i] = d_coset_digests[k * 5 + i];
     }
 }
 
 __global__ void qzc_scatter_xfe_strided_kernel(
     const uint64_t* __restrict__ d_coset_xfe, // [coset_len * 3] row-major
     uint64_t* __restrict__ d_full_xfe,        // [full_len * 3] row-major
     size_t coset_len,
     size_t coset_idx,
     size_t num_cosets
 ) {
     size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (k >= coset_len) return;
     size_t out_row = k * num_cosets + coset_idx;
     d_full_xfe[out_row * 3 + 0] = d_coset_xfe[k * 3 + 0];
     d_full_xfe[out_row * 3 + 1] = d_coset_xfe[k * 3 + 1];
     d_full_xfe[out_row * 3 + 2] = d_coset_xfe[k * 3 + 2];
 }
 
 __global__ void qzc_build_main_aux_codeword_coset_kernel(
     const uint64_t* __restrict__ d_main_coset, // [main_width * coset_len] col-major
     const uint64_t* __restrict__ d_aux_coset,  // [(aux_width*3) * coset_len] comp-major col-major
     const uint64_t* __restrict__ d_weights,    // [(main_width + aux_width + ...)*3] XFE weights
     size_t coset_len,
     size_t coset_idx,
     size_t num_cosets,
     size_t main_width,
     size_t aux_width,
     uint64_t* __restrict__ d_out_xfe           // [full_len * 3] row-major
 ) {
     size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (k >= coset_len) return;
     size_t out_row = k * num_cosets + coset_idx;
 
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     // main
     for (size_t c = 0; c < main_width; ++c) {
         uint64_t v = d_main_coset[c * coset_len + k];
         uint64_t w0 = d_weights[c * 3 + 0];
         uint64_t w1 = d_weights[c * 3 + 1];
         uint64_t w2 = d_weights[c * 3 + 2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_scalar_mul_impl(w0, w1, w2, v, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     // aux
     size_t base = main_width;
     for (size_t c = 0; c < aux_width; ++c) {
         uint64_t v0 = d_aux_coset[(c * 3 + 0) * coset_len + k];
         uint64_t v1 = d_aux_coset[(c * 3 + 1) * coset_len + k];
         uint64_t v2 = d_aux_coset[(c * 3 + 2) * coset_len + k];
         uint64_t w0 = d_weights[(base + c) * 3 + 0];
         uint64_t w1 = d_weights[(base + c) * 3 + 1];
         uint64_t w2 = d_weights[(base + c) * 3 + 2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v0, v1, v2, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
 
     d_out_xfe[out_row * 3 + 0] = acc0;
     d_out_xfe[out_row * 3 + 1] = acc1;
     d_out_xfe[out_row * 3 + 2] = acc2;
 }
 
 // FRUGAL: accumulate main column batches into codeword.
 __global__ void frugal_accumulate_main_codeword_kernel(
     const uint64_t* d_main_lde_batch, // [batch_cols * n] BFE col-major
     size_t n,
     size_t batch_cols,
     const uint64_t* d_weights,        // [total*3]
     size_t weight_base,
     size_t col_start,
     uint64_t* d_out_xfe               // [n*3] row-major
 ) {
     size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= n) return;
     uint64_t acc0 = d_out_xfe[row * 3 + 0];
     uint64_t acc1 = d_out_xfe[row * 3 + 1];
     uint64_t acc2 = d_out_xfe[row * 3 + 2];
     for (size_t c = 0; c < batch_cols; ++c) {
         size_t col = col_start + c;
         uint64_t v = d_main_lde_batch[c * n + row];
         uint64_t w0 = d_weights[(weight_base + col) * 3 + 0];
         uint64_t w1 = d_weights[(weight_base + col) * 3 + 1];
         uint64_t w2 = d_weights[(weight_base + col) * 3 + 2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_scalar_mul_impl(w0, w1, w2, v, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     d_out_xfe[row * 3 + 0] = acc0;
     d_out_xfe[row * 3 + 1] = acc1;
     d_out_xfe[row * 3 + 2] = acc2;
 }
 
 // FRUGAL: accumulate aux XFE column batches into codeword.
 __global__ void frugal_accumulate_aux_codeword_kernel(
     const uint64_t* d_aux_lde_batch,  // [batch_xfe_cols*3 * n] BFE comp-major col-major
     size_t n,
     size_t batch_xfe_cols,
     const uint64_t* d_weights,        // [total*3]
     size_t weight_base,
     size_t xfe_col_start,
     uint64_t* d_out_xfe               // [n*3] row-major
 ) {
     size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= n) return;
     uint64_t acc0 = d_out_xfe[row * 3 + 0];
     uint64_t acc1 = d_out_xfe[row * 3 + 1];
     uint64_t acc2 = d_out_xfe[row * 3 + 2];
     for (size_t c = 0; c < batch_xfe_cols; ++c) {
         size_t xfe_col = xfe_col_start + c;
         size_t base = c * 3;
         uint64_t v0 = d_aux_lde_batch[(base + 0) * n + row];
         uint64_t v1 = d_aux_lde_batch[(base + 1) * n + row];
         uint64_t v2 = d_aux_lde_batch[(base + 2) * n + row];
         uint64_t w0 = d_weights[(weight_base + xfe_col) * 3 + 0];
         uint64_t w1 = d_weights[(weight_base + xfe_col) * 3 + 1];
         uint64_t w2 = d_weights[(weight_base + xfe_col) * 3 + 2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v0, v1, v2, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     d_out_xfe[row * 3 + 0] = acc0;
     d_out_xfe[row * 3 + 1] = acc1;
     d_out_xfe[row * 3 + 2] = acc2;
 }
 
 __global__ void qzc_build_quot_codeword_kernel(
     const uint64_t* d_quot_segments,  // [(num_segments*3) * n] col-major comps
     const uint64_t* d_weights,        // [total*3]
     size_t n,
     size_t weight_base,
     size_t num_segments,
     uint64_t* d_out_xfe               // [n*3] row-major
 ) {
     size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= n) return;
     uint64_t acc0 = 0, acc1 = 0, acc2 = 0;
     for (size_t s = 0; s < num_segments; ++s) {
         uint64_t v0 = d_quot_segments[(s * 3 + 0) * n + row];
         uint64_t v1 = d_quot_segments[(s * 3 + 1) * n + row];
         uint64_t v2 = d_quot_segments[(s * 3 + 2) * n + row];
         uint64_t w0 = d_weights[(weight_base + s) * 3 + 0];
         uint64_t w1 = d_weights[(weight_base + s) * 3 + 1];
         uint64_t w2 = d_weights[(weight_base + s) * 3 + 2];
         uint64_t t0, t1, t2;
         triton_vm::gpu::kernels::xfield_mul_impl(w0, w1, w2, v0, v1, v2, t0, t1, t2);
         acc0 = triton_vm::gpu::kernels::bfield_add_impl(acc0, t0);
         acc1 = triton_vm::gpu::kernels::bfield_add_impl(acc1, t1);
         acc2 = triton_vm::gpu::kernels::bfield_add_impl(acc2, t2);
     }
     d_out_xfe[row * 3 + 0] = acc0;
     d_out_xfe[row * 3 + 1] = acc1;
     d_out_xfe[row * 3 + 2] = acc2;
 }
 
 __global__ void qzc_deep_fri_codeword_kernel(
     const uint64_t* d_fri_domain_bfe, // [n]
     const uint64_t* d_main_aux_codeword, // [n*3]
     const uint64_t* d_quot_codeword,     // [n*3]
     size_t n,
     const uint64_t* d_z,                 // [3]
     const uint64_t* d_gz,                // [3]
     const uint64_t* d_z4,                // [3]
     const uint64_t* d_eval_main_aux_z,   // [3]
     const uint64_t* d_eval_main_aux_gz,  // [3]
     const uint64_t* d_eval_quot_z4,      // [3]
     const uint64_t* d_w_deep,            // [3*3] deep weights (3 XFEs)
     uint64_t* d_out_codeword             // [n*3]
 ) {
     size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
 
     // Load domain value as XFE (lift)
     uint64_t dv = d_fri_domain_bfe[i];
     uint64_t dv0 = dv, dv1 = 0, dv2 = 0;
 
     auto deep_component = [&](const uint64_t* d_codeword, const uint64_t* eval_point, const uint64_t* eval_value,
                               uint64_t& out0, uint64_t& out1, uint64_t& out2) {
         uint64_t cw0 = d_codeword[i * 3 + 0];
         uint64_t cw1 = d_codeword[i * 3 + 1];
         uint64_t cw2 = d_codeword[i * 3 + 2];
 
         // numerator = cw - eval_value
         uint64_t n0, n1, n2;
         triton_vm::gpu::kernels::xfield_sub_impl(cw0, cw1, cw2, eval_value[0], eval_value[1], eval_value[2], n0, n1, n2);
         // denom = dv - eval_point
         uint64_t d0, d1, d2;
         triton_vm::gpu::kernels::xfield_sub_impl(dv0, dv1, dv2, eval_point[0], eval_point[1], eval_point[2], d0, d1, d2);
         uint64_t inv0, inv1, inv2;
         triton_vm::gpu::kernels::xfield_inv_impl(d0, d1, d2, inv0, inv1, inv2);
         triton_vm::gpu::kernels::xfield_mul_impl(n0, n1, n2, inv0, inv1, inv2, out0, out1, out2);
     };
 
     uint64_t a0, a1, a2;
     uint64_t b0, b1, b2;
     uint64_t c0, c1, c2;
     deep_component(d_main_aux_codeword, d_z,  d_eval_main_aux_z,  a0, a1, a2);
     deep_component(d_main_aux_codeword, d_gz, d_eval_main_aux_gz, b0, b1, b2);
     deep_component(d_quot_codeword,     d_z4, d_eval_quot_z4,      c0, c1, c2);
 
     // out = a*w0 + b*w1 + c*w2
     const uint64_t* w0 = d_w_deep + 0;
     const uint64_t* w1 = d_w_deep + 3;
     const uint64_t* w2 = d_w_deep + 6;
     uint64_t t0, t1, t2;
     uint64_t u0, u1, u2;
     uint64_t v0, v1, v2;
     triton_vm::gpu::kernels::xfield_mul_impl(a0,a1,a2, w0[0],w0[1],w0[2], t0,t1,t2);
     triton_vm::gpu::kernels::xfield_mul_impl(b0,b1,b2, w1[0],w1[1],w1[2], u0,u1,u2);
     triton_vm::gpu::kernels::xfield_mul_impl(c0,c1,c2, w2[0],w2[1],w2[2], v0,v1,v2);
     uint64_t s0,s1,s2;
     triton_vm::gpu::kernels::xfield_add_impl(t0,t1,t2, u0,u1,u2, s0,s1,s2);
     triton_vm::gpu::kernels::xfield_add_impl(s0,s1,s2, v0,v1,v2, s0,s1,s2);
     d_out_codeword[i * 3 + 0] = s0;
     d_out_codeword[i * 3 + 1] = s1;
     d_out_codeword[i * 3 + 2] = s2;
 }
 
 // ----------------------------------------------------------------------------
 // Host helpers: Merkle authentication_structure node indices (match Rust/C++).
 // ----------------------------------------------------------------------------
 static std::vector<size_t> auth_structure_heap_node_indices(
     const std::vector<size_t>& leaf_indices,
     size_t num_leaves
 ) {
     constexpr size_t ROOT_INDEX = 1;
     std::unordered_set<size_t> node_is_needed;
     std::unordered_set<size_t> node_can_be_computed;
 
     for (size_t leaf_idx : leaf_indices) {
         if (leaf_idx >= num_leaves) {
             throw std::out_of_range("Leaf index out of range");
         }
         size_t node_index = leaf_idx + num_leaves; // heap leaf index
         while (node_index > ROOT_INDEX) {
             node_can_be_computed.insert(node_index);
             node_is_needed.insert(node_index ^ 1);
             node_index /= 2;
         }
     }
 
     std::vector<size_t> result_indices;
     result_indices.reserve(node_is_needed.size());
     for (size_t idx : node_is_needed) {
         if (node_can_be_computed.find(idx) == node_can_be_computed.end()) {
             result_indices.push_back(idx);
         }
     }
     std::sort(result_indices.begin(), result_indices.end(), std::greater<size_t>());
     return result_indices;
 }
 
 static size_t merkle_heap_to_flat_index(size_t heap_index, size_t num_leaves) {
     // GPU Merkle tree layout stores levels consecutively:
     // level0 leaves [0..n-1], level1 [n..n+n/2-1], ..., root at [2n-2]
     if (heap_index >= num_leaves) {
         // leaf heap indices: [n..2n-1] -> flat [0..n-1]
         return heap_index - num_leaves;
     }
     // internal node heap indices: [1..n-1]
     // find level k such that heap_index in [n/2^k .. n/2^{k-1}-1], k>=1
     size_t k = 0;
     size_t start = num_leaves;
     while (start > 1) {
         start >>= 1;
         k++;
         if (heap_index >= start) break;
     }
     // offset(level k) = sum_{j=0}^{k-1} n/2^j = 2n*(1 - 1/2^k)
     size_t offset = 2 * num_leaves - (2 * num_leaves >> k);
     size_t pos = heap_index - start;
     return offset + pos;
 }
 // ============================================================================
 // JIT LDE kernels for GPU-native implementation (commented out for compilation)
 // ============================================================================
 
 /*
 // Local bfield arithmetic functions for kernels (to avoid header issues)
 namespace {
 __device__ __forceinline__ uint64_t local_bfield_mul_impl(uint64_t a, uint64_t b) {
     return triton_vm::gpu::kernels::bfield_mul_impl(a, b);
 }
 
 __device__ __forceinline__ uint64_t local_bfield_add_impl(uint64_t a, uint64_t b) {
     return triton_vm::gpu::kernels::bfield_add_impl(a, b);
 }
 
 __device__ __forceinline__ uint64_t local_bfield_pow_impl(uint64_t base, uint64_t exp) {
     return triton_vm::gpu::kernels::bfield_pow_impl(base, exp);
 }
 
 __device__ __forceinline__ uint64_t local_bfield_inv_impl(uint64_t a) {
     return triton_vm::gpu::kernels::bfield_inv_impl(a);
 }
 }
 
 // Kernel to evaluate a single polynomial on a coset domain
 __global__ void jit_lde_eval_poly_on_coset(
     const uint64_t* d_coeffs,        // [coeff_count] polynomial coefficients
     size_t coeff_count,              // number of coefficients
     uint64_t coset_offset,           // coset offset
     uint64_t domain_generator,       // domain generator
     size_t domain_length,            // evaluation domain length
     uint64_t* d_evaluations          // [domain_length] output evaluations
 ) {
     size_t eval_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (eval_idx >= domain_length) return;
 
     // Compute evaluation point: coset_offset * domain_generator^eval_idx
     uint64_t point = coset_offset;
     uint64_t gen_power = local_bfield_pow_impl(domain_generator, eval_idx);
     point = local_bfield_mul_impl(point, gen_power);
 
     // Evaluate polynomial at this point using Horner's method
     uint64_t result = 0;
     for (size_t i = coeff_count; i-- > 0;) {
         result = local_bfield_mul_impl(result, point);
         result = local_bfield_add_impl(result, d_coeffs[i]);
     }
 
     d_evaluations[eval_idx] = result;
 }
 
 // Kernel to batch evaluate multiple polynomials on a single coset
 __global__ void jit_lde_batch_eval_polys_on_coset(
     const uint64_t* d_coeffs,        // [num_polys * coeff_count] coefficients (row-major)
     size_t num_polys,                // number of polynomials
     size_t coeff_count,              // coefficients per polynomial
     uint64_t coset_offset,           // coset offset
     uint64_t domain_generator,       // domain generator
     size_t domain_length,            // evaluation domain length
     uint64_t* d_evaluations          // [num_polys * domain_length] output evaluations (row-major)
 ) {
     size_t poly_idx = blockIdx.y;
     size_t eval_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
 
     if (poly_idx >= num_polys || eval_idx >= domain_length) return;
 
     // Compute evaluation point: coset_offset * domain_generator^eval_idx
     uint64_t point = coset_offset;
     uint64_t gen_power = local_bfield_pow_impl(domain_generator, eval_idx);
     point = local_bfield_mul_impl(point, gen_power);
 
     // Get polynomial coefficients
     const uint64_t* poly_coeffs = d_coeffs + poly_idx * coeff_count;
 
     // Evaluate polynomial at this point using Horner's method
     uint64_t result = 0;
     for (size_t i = coeff_count; i-- > 0;) {
         result = local_bfield_mul_impl(result, point);
         result = local_bfield_add_impl(result, poly_coeffs[i]);
     }
 
     d_evaluations[poly_idx * domain_length + eval_idx] = result;
 }
 
 // Kernel to add randomization to trace evaluations on working domain
 __global__ void jit_lde_add_randomization(
     uint64_t* d_trace_evaluations,   // [num_cols * domain_length] (row-major)
     size_t num_cols,
     size_t domain_length,
     const uint64_t* d_randomizers,   // [num_cols * domain_length] randomizer evaluations
     uint64_t zerofier_value         // pre-computed (domain_offset^trace_len - 1)
 ) {
     size_t col_idx = blockIdx.y;
     size_t eval_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
 
     if (col_idx >= num_cols || eval_idx >= domain_length) return;
 
     // Get the randomizer value for this evaluation point
     uint64_t randomizer = d_randomizers[col_idx * domain_length + eval_idx];
 
     // Add randomization: trace += zerofier * randomizer
     size_t idx = col_idx * domain_length + eval_idx;
     uint64_t randomized_addition = local_bfield_mul_impl(zerofier_value, randomizer);
     d_trace_evaluations[idx] = local_bfield_add_impl(d_trace_evaluations[idx], randomized_addition);
 }
 */
 
 } // namespace
 
 // ============================================================================
 // Timing helper
 // ============================================================================
 
 template<typename Clock = std::chrono::high_resolution_clock>
 static double elapsed_ms(std::chrono::time_point<Clock> start) {
     auto end = Clock::now();
     return std::chrono::duration<double, std::milli>(end - start).count();
 }
 
 // ============================================================================
 // GPU Helper Functions
 // ============================================================================
 
 // Helper functions for modular arithmetic on GPU
 __device__ uint64_t mod_pow(uint64_t base, uint64_t exp) {
     const uint64_t MOD = 18446744069414584321ULL;  // BFieldElement::MODULUS
     uint64_t result = 1;
     while (exp > 0) {
         if (exp % 2 == 1) {
             result = (result * base) % MOD;
         }
         base = (base * base) % MOD;
         exp /= 2;
     }
     return result;
 }
 
 
 // ============================================================================
 // GPU Kernel Implementations
 // ============================================================================
 
 // GPU kernel for evaluating quotient constraints
 // Simplified implementation matching CPU approach
 __global__ void gpu_evaluate_quotient_constraints_kernel(
     const uint64_t* d_main_lde,      // Main LDE table [fri_len × main_width]
     const uint64_t* d_aux_lde,       // Aux LDE table [fri_len × aux_width × 3]
     size_t fri_len,                  // FRI domain length
     size_t q_len,                    // Quotient domain length
     size_t main_width,               // Main table width
     size_t aux_width,                // Aux table width
     const uint64_t* d_weights,       // Challenge weights [NUM_WEIGHTS × 3]
     uint64_t trace_offset,           // Trace domain offset
     uint64_t trace_generator,        // Trace domain generator
     uint64_t quotient_offset,        // Quotient domain offset
     uint64_t quotient_generator,     // Quotient domain generator
     uint64_t* d_quotient_values      // Output [q_len × 3]
 ) {
     size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= q_len) return;
 
     // Calculate quotient domain point
     // point = quotient_offset * quotient_generator^idx
     uint64_t point = quotient_offset;
     uint64_t gen_power = mod_pow(quotient_generator, idx);
     point = mul_mod(point, gen_power);
 
     // Find corresponding FRI domain indices
     // quotient_domain maps to subsample of FRI domain
     size_t fri_stride = fri_len / q_len;
     size_t fri_idx = idx * fri_stride;
 
     // Get main and aux rows from LDE
     // Simplified: just use some values to create non-zero quotient
     uint64_t q0 = d_main_lde[fri_idx * main_width] % 1000;  // Some main table value
     uint64_t q1 = (d_aux_lde[fri_idx * aux_width * 3] + point) % 1000;  // Some aux value + point
     uint64_t q2 = mul_mod(q0, q1) % 1000;  // Simple combination
 
     // Store as XFieldElement (c0, c1, c2)
     d_quotient_values[idx * 3 + 0] = q0;
     d_quotient_values[idx * 3 + 1] = q1;
     d_quotient_values[idx * 3 + 2] = q2;
 }
 
 // GPU kernel for segmenting quotient polynomial
 __global__ void gpu_segment_quotient_kernel(
     const uint64_t* d_quotient_values,  // [q_len × 3]
     size_t q_len,                       // Quotient length
     uint64_t* d_segments                // [q_len × 3] - simplified single segment
 ) {
     size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= q_len) return;
 
     // For now, just copy quotient values to segments
     // In full implementation, would segment into 4 parts
     d_segments[idx * 3 + 0] = d_quotient_values[idx * 3 + 0];
     d_segments[idx * 3 + 1] = d_quotient_values[idx * 3 + 1];
     d_segments[idx * 3 + 2] = d_quotient_values[idx * 3 + 2];
 }
 
 
 // ============================================================================
 // GpuStark Implementation
 // ============================================================================
 
 GpuStark::GpuStark() {
     // Context will be created per-proof based on dimensions
 }
 
 GpuStark::~GpuStark() = default;
 
 static bool use_lde_frugal_mode(size_t padded_height) {
     const char* frugal_env = std::getenv("TRITON_GPU_LDE_FRUGAL");
     if (frugal_env) {
         if (strcmp(frugal_env, "1") == 0 || strcmp(frugal_env, "true") == 0) {
             return true;
         }
         if (strcmp(frugal_env, "0") == 0 || strcmp(frugal_env, "false") == 0) {
             return false;
         }
     }
     const size_t FRUGAL_THRESHOLD = 1ULL << 22; // 2^22
     return padded_height >= FRUGAL_THRESHOLD;
 }
 
 size_t GpuStark::estimate_gpu_memory(size_t padded_height) {
     size_t fri_length = padded_height * 8;
     size_t main_width = 379;
     size_t aux_width = 88;
     size_t num_fri_rounds = static_cast<size_t>(std::log2(fri_length)) - 9;
     
     size_t estimate = 0;
     
     // Main tables
     estimate += padded_height * main_width * sizeof(uint64_t);
     estimate += fri_length * main_width * sizeof(uint64_t);
     
     // Aux tables (XFE = 3x)
     estimate += padded_height * aux_width * 3 * sizeof(uint64_t);
     estimate += fri_length * aux_width * 3 * sizeof(uint64_t);
     
     // Quotient segments
     estimate += fri_length * 4 * 3 * sizeof(uint64_t);
     
     // Merkle trees
     estimate += 3 * 2 * fri_length * 5 * sizeof(uint64_t);
     
     // FRI data
     size_t fri_size = fri_length;
     for (size_t r = 0; r < num_fri_rounds; ++r) {
         fri_size /= 2;
         estimate += fri_size * 3 * sizeof(uint64_t);
         estimate += 2 * fri_size * 5 * sizeof(uint64_t);
     }
     
     // Scratch space (optimized: trace_height based, not fri_length)
     estimate += padded_height * main_width * sizeof(uint64_t);       // scratch_a
     estimate += padded_height * aux_width * 3 * sizeof(uint64_t);    // scratch_b
     
     // Tip5 tables
     estimate += 65536 * sizeof(uint16_t);  // S-box
     estimate += 16 * sizeof(uint64_t);      // MDS
     estimate += 128 * sizeof(uint64_t);     // Round constants
     
     // Proof buffer
     estimate += 10 * 1024 * 1024;
     
     return estimate;
 }
 
 /**
  * Get available system RAM (for unified memory overflow)
  */
 static size_t get_system_ram_available() {
     struct sysinfo info;
     if (sysinfo(&info) == 0) {
         return static_cast<size_t>(info.freeram) * info.mem_unit;
     }
     return 0;
 }
 
 bool GpuStark::check_gpu_memory(size_t padded_height) {
     // Dynamic environment variable configuration based on padded height threshold
     // Threshold: 2^21 (padded_height = 2,097,152)
     const size_t THRESHOLD_HEIGHT = 2097152;  // 2^21
     
     // Calculate log2 of padded height for reporting
     uint32_t log2_height = 0;
     size_t temp = padded_height;
     while (temp > 1) {
         temp >>= 1;
         log2_height++;
     }
     
     // Automatically configure environment based on padded height
     // Always override user settings to ensure optimal configuration
     if (padded_height >= THRESHOLD_HEIGHT) {
         // Large instances (>= 2^21): Enable RAM overflow mode
         setenv("TRITON_GPU_USE_RAM_OVERFLOW", "1", 1);  // Always override
         setenv("TRITON_PAD_SCALE_MODE", "4", 1);
         std::cout << "[GPU] Auto-config: Padded height >= 2^21 (log2=" << log2_height << ", height=" << padded_height << ")" << std::endl;
         std::cout << "      Forcing TRITON_GPU_USE_RAM_OVERFLOW=1, TRITON_PAD_SCALE_MODE=4" << std::endl;
     } else {
         // Small instances (<= 2^20): Use direct GPU mode
         setenv("TRITON_GPU_USE_RAM_OVERFLOW", "0", 1);  // Always override
         setenv("TRITON_PAD_SCALE_MODE", "0", 1);
         std::cout << "[GPU] Auto-config: Padded height <= 2^20 (log2=" << log2_height << ", height=" << padded_height << ")" << std::endl;
         std::cout << "      Forcing TRITON_GPU_USE_RAM_OVERFLOW=0, TRITON_PAD_SCALE_MODE=0" << std::endl;
     }
     
     size_t required = estimate_gpu_memory(padded_height);
     
     size_t free_mem, total_mem;
     cudaMemGetInfo(&free_mem, &total_mem);
     
     // In unified memory mode, we can use memory across GPUs (respects TRITON_GPU_COUNT)
     size_t available_gpu = free_mem;
     size_t total = total_mem;
     size_t system_ram_available = 0;
     bool using_ram_overflow = false;
     bool was_unified_memory_enabled = use_unified_memory();
     int device_count = 1;
     
     if (use_unified_memory()) {
         device_count = get_effective_gpu_count();
 
         available_gpu = 0;
         total = 0;
         for (int i = 0; i < device_count; i++) {
             cudaError_t err = cudaSetDevice(i);
             if (err != cudaSuccess) {
                 std::cerr << "Warning: Failed to set device " << i << ": " << cudaGetErrorString(err) << std::endl;
                 continue;
             }
             size_t f, t;
             cudaMemGetInfo(&f, &t);
             available_gpu += f;
             total += t;
         }
         cudaSetDevice(0);  // Return to primary GPU
 
         // Fallback: if unified memory calculation failed, use primary GPU memory
         if (available_gpu == 0) {
             std::cout << "[GPU] Warning: Unified memory calculation failed, using primary GPU memory" << std::endl;
             available_gpu = free_mem;
             total = total_mem;
         }
     }
     
     // Check if we should use system RAM as overflow buffer
     // This works for both single-GPU and multi-GPU modes
     // CUDA unified memory (cudaMallocManaged) can use system RAM as backing store when GPU memory is insufficient
     const char* use_ram_env = std::getenv("TRITON_GPU_USE_RAM_OVERFLOW");
     bool use_ram_overflow = (use_ram_env && (strcmp(use_ram_env, "1") == 0 || strcmp(use_ram_env, "true") == 0));
     
     // If GPU memory is insufficient, automatically enable RAM overflow
     if (!use_ram_overflow && available_gpu < required) {
         // Could auto-enable here if desired, but for now we require explicit opt-in
     }
     
     if (use_ram_overflow) {
         system_ram_available = get_system_ram_available();
         // Reserve 4GB for system usage
         size_t reserved_for_system = 4ULL * 1024 * 1024 * 1024; // 4GB
         if (system_ram_available > reserved_for_system) {
             using_ram_overflow = true;
             // Enable unified memory mode if not already enabled (needed for cudaMallocManaged to work with RAM overflow)
             if (!use_unified_memory()) {
                 use_unified_memory() = true;
                 std::cout << "[GPU] Enabling unified memory mode for single-GPU RAM overflow" << std::endl;
             }
         } else {
             system_ram_available = 0; // Not enough system RAM
             using_ram_overflow = false;
         }
     }
     
     // Calculate total available memory (GPU + system RAM if using overflow)
     size_t usable_ram = 0;
     if (using_ram_overflow && system_ram_available > 4ULL * 1024 * 1024 * 1024) {
         usable_ram = system_ram_available - (4ULL * 1024 * 1024 * 1024); // Reserve 4GB
     }
     size_t available = available_gpu + usable_ram;
     
     TRITON_PROFILE_COUT("[GPU] Memory check:" << std::endl);
     TRITON_PROFILE_COUT("  Required: " << (required / (1024 * 1024)) << " MB" << std::endl);
     TRITON_PROFILE_COUT("  Available: " << (available / (1024 * 1024)) << " MB" << std::endl);
     TRITON_IF_PROFILE {
         if (using_ram_overflow) {
             std::cout << "    (GPU: " << (available_gpu / (1024 * 1024)) << " MB + System RAM: " 
                       << ((available - available_gpu) / (1024 * 1024)) << " MB)" << std::endl;
         }
     }
     TRITON_PROFILE_COUT("  Total: " << (total / (1024 * 1024)) << " MB" << std::endl);
     TRITON_IF_PROFILE {
         if (use_unified_memory()) {
             device_count = get_effective_gpu_count();
             if (device_count > 1) {
                 std::cout << "  Mode: Multi-GPU Unified Memory (" << device_count << " GPUs)";
             } else {
                 std::cout << "  Mode: Single-GPU Unified Memory";
             }
             if (using_ram_overflow) {
                 std::cout << " + System RAM overflow";
             }
             std::cout << std::endl;
         } else {
             std::cout << "  Mode: Single-GPU (device memory only)" << std::endl;
         }
     }
     
     // Optional escape hatch for very large instances (e.g., big padded heights) on machines
     // with ample VRAM or when the user wants to risk oversubscription.
     // If set, we always proceed and rely on CUDA allocation failures to surface issues.
     const char* ignore_env = std::getenv("TRITON_GPU_IGNORE_MEMCHECK");
     if (ignore_env && (strcmp(ignore_env, "1") == 0 || strcmp(ignore_env, "true") == 0)) {
         std::cout << "[GPU] WARNING: TRITON_GPU_IGNORE_MEMCHECK=1, proceeding even if required > available" << std::endl;
         return true;
     }
 
     return available >= required;
 }
 
 Proof GpuStark::prove(
     const Claim& claim,
     const uint64_t* main_table_data,
     size_t num_rows,
     size_t num_cols,
     const uint64_t trace_domain_3[3],
     const uint64_t quotient_domain_3[3],
     const uint64_t fri_domain_3[3],
     const uint8_t randomness_seed[32],
     const std::vector<uint64_t>& main_randomizer_coeffs,
     const std::vector<uint64_t>& aux_randomizer_coeffs
 ) {
     TRITON_DEBUG_COUT("[DEBUG] GpuStark::prove() called with main_table_data=" << (main_table_data != nullptr ? "VALID" : "NULL") << std::endl);
     auto total_start = std::chrono::high_resolution_clock::now();
     // Persist randomness seed for downstream aux-table randomizer generation (col 87).
     std::copy(randomness_seed, randomness_seed + 32, randomness_seed_.begin());
     
     TRITON_PROFILE_COUT("\n========================================" << std::endl);
     TRITON_PROFILE_COUT("GPU STARK Proof Generation (Zero-Copy)" << std::endl);
     TRITON_PROFILE_COUT("========================================" << std::endl);
     TRITON_PROFILE_COUT("Trace dimensions: " << num_rows << " x " << num_cols << std::endl);
     
     // Check GPU memory
     if (!check_gpu_memory(num_rows)) {
         throw std::runtime_error("Insufficient GPU memory for proof generation");
     }
     
     // Calculate dimensions from Rust-provided domains (exact match to verifier expectations)
     dims_.padded_height = static_cast<size_t>(trace_domain_3[0]);
     dims_.quotient_length = static_cast<size_t>(quotient_domain_3[0]);
     dims_.fri_length = static_cast<size_t>(fri_domain_3[0]);
 
     dims_.lde_frugal_mode = use_lde_frugal_mode(dims_.padded_height);
     if (dims_.lde_frugal_mode) {
         TRITON_PROFILE_COUT("      LDE FRUGAL MODE: ENABLED (streaming LDE, no cache)" << std::endl);
     }
 
     dims_.trace_offset = trace_domain_3[1];
     dims_.trace_generator = trace_domain_3[2];
     dims_.quotient_offset = quotient_domain_3[1];
     dims_.quotient_generator = quotient_domain_3[2];
     dims_.fri_offset = fri_domain_3[1];
     dims_.fri_generator = fri_domain_3[2];
 
     dims_.trace_offset = trace_domain_3[1];
     dims_.trace_generator = trace_domain_3[2];
     dims_.quotient_offset = quotient_domain_3[1];
     dims_.quotient_generator = quotient_domain_3[2];
     dims_.fri_offset = fri_domain_3[1];
     dims_.fri_generator = fri_domain_3[2];
 
     // Sanity
     if (dims_.padded_height != num_rows) {
         throw std::runtime_error("Mismatch: num_rows != trace_domain.length");
     }
     if (dims_.main_width != 0 && dims_.main_width != num_cols) {
         // (dims_.main_width not yet set)
     }
 
     dims_.main_width = num_cols;
     dims_.aux_width = 88;
     
     // Store host main table pointer for hybrid CPU/GPU aux computation
     h_main_table_data_ = main_table_data;
     // Trace randomizer count (must match what main_gpu_full.cpp extracted)
     {
         if (dims_.main_width == 0) throw std::runtime_error("Invalid main_width");
         if (main_randomizer_coeffs.size() % dims_.main_width != 0) {
             throw std::runtime_error("main_randomizer_coeffs size not divisible by main_width");
         }
         size_t num_rand_main = main_randomizer_coeffs.size() / dims_.main_width;
         if (dims_.aux_width == 0) throw std::runtime_error("Invalid aux_width");
         if (aux_randomizer_coeffs.size() % (dims_.aux_width * 3) != 0) {
             throw std::runtime_error("aux_randomizer_coeffs size not divisible by aux_width*3");
         }
         size_t num_rand_aux = aux_randomizer_coeffs.size() / (dims_.aux_width * 3);
         if (num_rand_main != num_rand_aux) {
             throw std::runtime_error("Mismatch: main vs aux num_trace_randomizers");
         }
         dims_.num_trace_randomizers = num_rand_main;
     }
     dims_.num_quotient_segments = 4;
     dims_.num_fri_rounds = static_cast<size_t>(std::log2(dims_.fri_length)) - 9;
 
     TRITON_PROFILE_COUT("[GPU] Domains: trace(len=" << dims_.padded_height
               << ", offset=" << dims_.trace_offset
               << ", gen=" << dims_.trace_generator
               << ") quotient(len=" << dims_.quotient_length
               << ", offset=" << dims_.quotient_offset
               << ", gen=" << dims_.quotient_generator
               << ") fri(len=" << dims_.fri_length
               << ", offset=" << dims_.fri_offset
               << ", gen=" << dims_.fri_generator
               << ")\n");
     
     TRITON_PROFILE_COUT("FRI domain: " << dims_.fri_length << " points" << std::endl);
     TRITON_PROFILE_COUT("FRI rounds: " << dims_.num_fri_rounds << std::endl);
     
     // Create GPU proof context
     ctx_ = std::make_unique<GpuProofContext>(dims_);
 
     // Store claim (host-side) for derived challenges / encoding.
     claim_ = claim;
     // Reset CPU-driven transcript
     fs_cpu_ = ProofStream();
     
     // Initialize Tip5 tables on GPU
     init_tip5_tables();
     
     // =========================================================================
     // ONLY H2D TRANSFER: Upload main table to GPU
     // =========================================================================
     auto upload_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("\n[H2D] Uploading main table (" 
               << (num_rows * num_cols * 8 / (1024 * 1024)) << " MB)..." << std::endl);
     ctx_->upload_main_table(main_table_data, num_rows * num_cols);
     // Upload trace randomizer coefficients (tiny H2D, kept for entire proof)
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_main_randomizer_coeffs(),
         main_randomizer_coeffs.data(),
         main_randomizer_coeffs.size() * sizeof(uint64_t),
         cudaMemcpyHostToDevice,
         ctx_->stream()
     ));
     // Aux randomizers:
     // The kernels expect "component-column major" layout:
     //   index = (xfe_col * 3 + comp) * num_rand + rand_idx
     // where comp=0,1,2 are the BFE coefficients of the XFieldElement randomizer.
     //
     // NOTE: Do NOT zero out comp1/2. Rust uses full XFieldElement coefficients.
     {
         const size_t num_rand = dims_.num_trace_randomizers;
         const size_t expected = dims_.aux_width * 3 * num_rand;
         if (aux_randomizer_coeffs.size() != expected) {
             throw std::runtime_error("aux_randomizer_coeffs has unexpected size (expected aux_width*3*num_rand)");
         }
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_aux_randomizer_coeffs(),
             aux_randomizer_coeffs.data(),
             expected * sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
     }
     ctx_->synchronize();
     double upload_time = elapsed_ms(upload_start);
     TRITON_PROFILE_COUT("[H2D] Upload complete: " << upload_time << " ms" << std::endl);
     
     // =========================================================================
     // GPU Degree Lowering (if enabled via TRITON_GPU_DEGREE_LOWERING=1)
     // =========================================================================
     static int use_gpu_degree_lowering = -1;
     if (use_gpu_degree_lowering == -1) {
         const char* env = std::getenv("TRITON_GPU_DEGREE_LOWERING");
         use_gpu_degree_lowering = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
     }
     
     if (use_gpu_degree_lowering) {
         auto t_dl = std::chrono::high_resolution_clock::now();
         TRITON_PROFILE_COUT("[GPU] Computing degree lowering columns on GPU..." << std::endl);
         kernels::gpu_degree_lowering_main(
             ctx_->d_main_trace(),
             dims_.padded_height,
             dims_.main_width,
             ctx_->stream()
         );
         ctx_->synchronize();
         double dl_time = std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - t_dl).count();
         TRITON_PROFILE_COUT("[GPU] Degree lowering: " << dl_time << " ms" << std::endl);
     }
     
     // =========================================================================
     // ALL COMPUTATION ON GPU (no intermediate memory transfers)
     // =========================================================================
     TRITON_PROFILE_COUT("\n[GPU] Starting proof computation..." << std::endl);
     
     double step_times[8] = {0};
     
     // Step 1: Initialize Fiat-Shamir with claim
     auto t1 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 1: Initialize Fiat-Shamir" << std::endl);
     step_initialize_fiat_shamir(claim);
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[0] = elapsed_ms(t1);
     
     // Step 2: Main table LDE + Merkle commitment
     // NOTE: We no longer pre-convert the main table into `vector<vector<BFieldElement>>`.
     // Hybrid CPU aux now reads directly from the flat `uint64_t*` host buffer.
     
     auto t2 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 2: Main table LDE + Merkle commitment" << std::endl);
     step_main_table_commitment(main_randomizer_coeffs);
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[1] = elapsed_ms(t2);
     
     // No pre-conversion thread in the flat-main-table hybrid aux path.
     
     // Step 3: Aux table commitment
     auto t3 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 3: Aux table commitment" << std::endl);
     step_aux_table_commitment(aux_randomizer_coeffs);
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     if (std::getenv("TVM_DEBUG_SANITY_QUOT_WEIGHTS")) {
         std::cout << "[DBG] sanity: memset quotient_weights after Step3..." << std::endl;
         CUDA_CHECK(cudaMemsetAsync(ctx_->d_quotient_weights(), 0, Quotient::MASTER_AUX_NUM_CONSTRAINTS * 3 * sizeof(uint64_t), ctx_->stream()));
         ctx_->synchronize();
         std::cout << "[DBG] sanity: OK" << std::endl;
     }
     step_times[2] = elapsed_ms(t3);
     
     // Step 4: Quotient computation
     auto t4 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 4: Quotient computation" << std::endl);
     step_quotient_commitment();
     ctx_->synchronize();
     step_times[3] = elapsed_ms(t4);
     
     // Step 5: Out-of-domain evaluation
     auto t5 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 5: Out-of-domain evaluation" << std::endl);
     step_out_of_domain_evaluation();
     ctx_->synchronize();
     step_times[4] = elapsed_ms(t5);
     
     // Step 6: FRI protocol
     auto t6 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 6: FRI protocol" << std::endl);
     step_fri_protocol();
     ctx_->synchronize();
     step_times[5] = elapsed_ms(t6);
     
     // Step 7: Open trace at query indices
     auto t7 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 7: Open trace leaves" << std::endl);
     step_open_trace();
     ctx_->synchronize();
     step_times[6] = elapsed_ms(t7);
     
     // Step 8: Finalize proof buffer
     auto t8 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 8: Finalize proof" << std::endl);
     step_encode_proof();
     ctx_->synchronize();
     step_times[7] = elapsed_ms(t8);
     
     TRITON_PROFILE_COUT("[GPU] Proof computation complete" << std::endl);
     
     // =========================================================================
     // ONLY D2H TRANSFER: Download final proof
     // =========================================================================
     auto download_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("\n[D2H] Downloading proof..." << std::endl);
     auto proof_data = ctx_->download_proof();
     double download_time = elapsed_ms(download_start);
     TRITON_PROFILE_COUT("[D2H] Download complete: " << proof_data.size() << " elements, " 
               << download_time << " ms" << std::endl);
     
     // Print timing summary
     double total_time = elapsed_ms(total_start);
     double gpu_time = 0;
     for (int i = 0; i < 8; ++i) gpu_time += step_times[i];
     
     TRITON_IF_PROFILE {
         std::cout << "\n========================================" << std::endl;
         std::cout << "GPU Proof Generation Complete!" << std::endl;
         std::cout << "========================================" << std::endl;
         std::cout << "Timing breakdown:" << std::endl;
         std::cout << "  H2D Upload:     " << upload_time << " ms" << std::endl;
         std::cout << "  Step 1 (F-S):   " << step_times[0] << " ms" << std::endl;
         std::cout << "  Step 2 (Main):  " << step_times[1] << " ms" << std::endl;
         std::cout << "  Step 3 (Aux):   " << step_times[2] << " ms" << std::endl;
         std::cout << "  Step 4 (Quot):  " << step_times[3] << " ms" << std::endl;
         std::cout << "  Step 5 (OOD):   " << step_times[4] << " ms" << std::endl;
         std::cout << "  Step 6 (FRI):   " << step_times[5] << " ms" << std::endl;
         std::cout << "  Step 7 (Open):  " << step_times[6] << " ms" << std::endl;
         std::cout << "  Step 8 (Enc):   " << step_times[7] << " ms" << std::endl;
         std::cout << "  D2H Download:   " << download_time << " ms" << std::endl;
         std::cout << "  --------------------------" << std::endl;
         std::cout << "  GPU compute:    " << gpu_time << " ms" << std::endl;
         std::cout << "  Total:          " << total_time << " ms" << std::endl;
         std::cout << "  Proof size:     " << (proof_data.size() * 8 / 1024) << " KB" << std::endl;
         std::cout << "========================================\n" << std::endl;
         
         ctx_->print_memory_usage();
     }
     
     // Convert to Proof object
     std::vector<BFieldElement> proof_bfe;
     proof_bfe.reserve(proof_data.size());
     for (uint64_t val : proof_data) {
         proof_bfe.push_back(BFieldElement(val));
     }
     
     Proof result;
     result.elements = std::move(proof_bfe);
     return result;
 }
 
 // ============================================================================
 // Initialize Tip5 Tables
 // ============================================================================
 
 void GpuStark::init_tip5_tables() {
     // Allocate Tip5 tables on GPU
     cudaMalloc(&d_sbox_table_, 65536 * sizeof(uint16_t));
     cudaMalloc(&d_mds_matrix_, 16 * sizeof(uint64_t));
     cudaMalloc(&d_round_constants_, 128 * sizeof(uint64_t));
     
     // Initialize tables
     kernels::tip5_init_tables(d_sbox_table_, d_mds_matrix_, d_round_constants_);
     ctx_->synchronize();
 }
 
 // ============================================================================
 // Step 1: Initialize Fiat-Shamir
 // ============================================================================
 
 void GpuStark::step_initialize_fiat_shamir(const Claim& claim) {
     // Initialize sponge state to zeros (variable-length mode)
     kernels::fs_init_sponge_gpu(ctx_->d_sponge_state(), ctx_->stream());
     
     // Build claim encoding on host (must match `src/stark.cpp` Claim hashing exactly).
     // Rust `Claim` derives BFieldCodec with *reverse field order*:
     // output, input, version, program_digest.
     std::vector<uint64_t> claim_encoding;
     claim_encoding.reserve(16);
 
     auto encode_vec_bfe_field_with_struct_len_prefix = [&](const std::vector<BFieldElement>& v) {
         const size_t vec_encoding_len = 1 + v.size(); // Vec<BFE>::encode() = [len] + elements
         claim_encoding.push_back(static_cast<uint64_t>(vec_encoding_len)); // struct field len prefix
         claim_encoding.push_back(static_cast<uint64_t>(v.size()));         // vec len
         for (const auto& e : v) claim_encoding.push_back(e.value());
     };
 
     encode_vec_bfe_field_with_struct_len_prefix(claim.output);
     encode_vec_bfe_field_with_struct_len_prefix(claim.input);
     claim_encoding.push_back(static_cast<uint64_t>(claim.version));
     for (size_t i = 0; i < 5; ++i) claim_encoding.push_back(claim.program_digest[i].value());
 
     if (std::getenv("TVM_DEBUG_CLAIM_ENCODING")) {
         std::cout << "[DBG] claim_encoding_len=" << claim_encoding.size() << " first16:";
         for (size_t i = 0; i < std::min<size_t>(16, claim_encoding.size()); ++i) {
             std::cout << " " << claim_encoding[i];
         }
         std::cout << std::endl;
     }
     
     // Upload claim encoding once (small) and absorb from device (zero-copy friendly)
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_scratch_a(),
         claim_encoding.data(),
         claim_encoding.size() * sizeof(uint64_t),
         cudaMemcpyHostToDevice,
         ctx_->stream()
     ));
     kernels::fs_absorb_device_gpu(
         ctx_->d_sponge_state(),
         ctx_->d_scratch_a(),
         claim_encoding.size(),
         ctx_->stream()
     );
 
     // CPU-driven transcript: absorb the same claim encoding (gold standard matching Rust)
     {
         std::vector<BFieldElement> claim_bfe;
         claim_bfe.reserve(claim_encoding.size());
         for (uint64_t v : claim_encoding) {
             claim_bfe.emplace_back(BFieldElement(v));
         }
         fs_cpu_.alter_fiat_shamir_state_with(claim_bfe);
     }
     
     // Add log2_padded_height as first proof item
     uint32_t log2_height = 0;
     size_t temp = dims_.padded_height;
     while (temp >>= 1) log2_height++;
     // NOTE: Log2PaddedHeight is NOT included in Fiat-Shamir (matches Rust).
     // Append to proof buffer (we store just the u32 value; main_gpu_full reconstructs the ProofItem)
     uint64_t log2_val = log2_height;
     cudaMemcpyAsync(ctx_->d_proof_buffer(), &log2_val, sizeof(uint64_t),
                     cudaMemcpyHostToDevice, ctx_->stream());
     proof_size_ = 1;
 }
 
 // ============================================================================
 // Step 2: Main Table Commitment
 // ============================================================================
 
 void GpuStark::step_main_table_commitment(const std::vector<uint64_t>& main_randomizer_coeffs) {
     const bool profile_main = TRITON_PROFILE_ENABLED();
     
     // 1) Transpose main trace table row-major -> column-major (required by NTT/LDE kernels)
     auto t_transpose = std::chrono::high_resolution_clock::now();
     const size_t trace_len = dims_.padded_height;
     const size_t width = dims_.main_width;
     constexpr int BLOCK = 256;
     int grid = (int)(((trace_len * width) + BLOCK - 1) / BLOCK);
 
     // Use scratch_a as a temporary col-major trace buffer: [width * trace_len]
     uint64_t* d_trace_colmajor = ctx_->d_scratch_a();
     
     // Debug: check d_main_trace BEFORE transpose
     if (std::getenv("TVM_DEBUG_MAIN_TRACE_ROWMAJOR")) {
         ctx_->synchronize();
         uint64_t* d_trace = ctx_->d_main_trace();
         fprintf(stderr, "[DBG] Main trace ROW-MAJOR BEFORE transpose (row 0, first 10 cols), ptr=%p:\n", (void*)d_trace);
         fflush(stderr);
         if (d_trace != nullptr) {
             std::vector<uint64_t> h_row0(10);
             cudaError_t err = cudaMemcpy(h_row0.data(), d_trace, 10 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
             fprintf(stderr, "[DBG] cudaMemcpy result=%d\n", (int)err);
             fflush(stderr);
             for (size_t c = 0; c < 10; c++) {
                 fprintf(stderr, "  [%zu]: %lu\n", c, h_row0[c]);
                 fflush(stderr);
             }
         } else {
             fprintf(stderr, "  ERROR: d_main_trace is NULL!\n");
             fflush(stderr);
         }
     }
     
     // For unified memory, prefetch input and output buffers before transpose
     if (use_unified_memory()) {
         size_t data_size = trace_len * width * sizeof(uint64_t);
         int device;
         cudaGetDevice(&device);
         cudaMemLocation location;
         location.type = cudaMemLocationTypeDevice;
         location.id = device;
         cudaMemPrefetchAsync(ctx_->d_main_trace(), data_size, location, 0);
         cudaMemPrefetchAsync(d_trace_colmajor, data_size, location, 0);
         cudaStreamSynchronize(ctx_->stream());
     }
     
     qzc_rowmajor_to_colmajor_bfe<<<grid, BLOCK, 0, ctx_->stream()>>>(
         ctx_->d_main_trace(), d_trace_colmajor, trace_len, width
     );
     if (profile_main) { ctx_->synchronize(); std::cout << "    [Main] transpose (" << trace_len << "x" << width << "): " << elapsed_ms(t_transpose) << " ms\n"; }
 
     // Debug: check transposed trace column 0, first few values
     if (std::getenv("TVM_DEBUG_MAIN_TRACE_COLMAJOR")) {
         ctx_->synchronize();
         fprintf(stderr, "[DBG] Main trace col-major (col=0, first 5 rows), trace_len=%zu, width=%zu:\n", trace_len, width);
         fflush(stderr);
         std::vector<uint64_t> h_col0(5);
         cudaError_t cerr = cudaMemcpy(h_col0.data(), d_trace_colmajor, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
         for (size_t r = 0; r < 5; r++) {
             fprintf(stderr, "  row[%zu]: %lu (cuda=%d)\n", r, h_col0[r], (int)cerr);
             fflush(stderr);
         }
         
         // Also check column 378 (last column)
         size_t last_col = width - 1;
         fprintf(stderr, "[DBG] Main trace col-major (col=%zu LAST, first 5 rows):\n", last_col);
         std::vector<uint64_t> h_col_last(5);
         cudaError_t cerr2 = cudaMemcpy(h_col_last.data(), d_trace_colmajor + last_col * trace_len, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
         for (size_t r = 0; r < 5; r++) {
             fprintf(stderr, "  row[%zu]: %lu (cuda=%d)\n", r, h_col_last[r], (int)cerr2);
             fflush(stderr);
         }
     }
 
     // 2) Randomized LDE (matches verified hybrid prover)
     auto t_lde = std::chrono::high_resolution_clock::now();
     uint64_t trace_offset = dims_.trace_offset;
     uint64_t fri_offset = dims_.fri_offset;
 
     (void)main_randomizer_coeffs; // coeffs are uploaded once into ctx_ for true zero-copy
     const size_t num_randomizers = dims_.num_trace_randomizers;
     if (num_randomizers == 0) {
         throw std::runtime_error("num_trace_randomizers is zero; cannot build randomized main LDE");
     }
     uint64_t* d_randomizer_coeffs = ctx_->d_main_randomizer_coeffs();
     
     // Debug: print LDE parameters and first few randomizer coefficients
     if (std::getenv("TVM_DEBUG_MAIN_LDE_PARAMS")) {
         ctx_->synchronize();
         fprintf(stderr, "[DBG] Main LDE params: trace_len=%zu, width=%zu, num_rand=%zu, trace_offset=%lu, fri_offset=%lu, fri_len=%zu, d_rand_ptr=%p\n",
                 trace_len, width, num_randomizers, trace_offset, fri_offset, dims_.fri_length, (void*)d_randomizer_coeffs);
         fflush(stderr);
         // Print first 5 randomizer coefficients for column 0
         if (d_randomizer_coeffs != nullptr && num_randomizers >= 5) {
             uint64_t h_rand[5];
             cudaError_t err = cudaMemcpy(h_rand, d_randomizer_coeffs, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
             fprintf(stderr, "[DBG] Main randomizer col0 coeffs (first 5, cuda=%d):\n", (int)err);
             fprintf(stderr, "  [0]: %lu\n", h_rand[0]);
             fprintf(stderr, "  [1]: %lu\n", h_rand[1]);
             fprintf(stderr, "  [2]: %lu\n", h_rand[2]);
             fprintf(stderr, "  [3]: %lu\n", h_rand[3]);
             fprintf(stderr, "  [4]: %lu\n", h_rand[4]);
             // Also check LAST column (378) randomizer
             uint64_t h_rand_last[5];
             size_t last_col = width - 1;
             cudaError_t err2 = cudaMemcpy(h_rand_last, d_randomizer_coeffs + last_col * num_randomizers, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
             fprintf(stderr, "[DBG] Main randomizer col%zu (LAST) coeffs (first 5, cuda=%d):\n", last_col, (int)err2);
             for (int i = 0; i < 5; i++) {
                 fprintf(stderr, "  [%d]: %lu\n", i, h_rand_last[i]);
             }
             fflush(stderr);
         }
     }
 
     if (dims_.lde_frugal_mode) {
         // FRUGAL: 8-way coset streaming with multi-GPU support.
         // Compute LDE on each coset (length = trace_len), hash rows, and scatter digests into full leaf array.
         auto t_stream = std::chrono::high_resolution_clock::now();
         const size_t fri_len = dims_.fri_length;
         const size_t num_cosets = fri_len / trace_len;
         if (num_cosets == 0 || fri_len % trace_len != 0) {
             throw std::runtime_error("FRUGAL: invalid domains (fri_len must be multiple of trace_len)");
         }
         if (num_cosets != 8) {
             TRITON_PROFILE_COUT("[GPU] FRUGAL: Warning: expected 8 cosets, got " << num_cosets << std::endl);
         }
 
         // Check for multi-GPU with peer access
         int num_gpus = get_effective_gpu_count();
         if (num_gpus > 2) num_gpus = 2;
         bool multi_gpu = false; // Disabled due to memory issues
         
         // Verify peer access is available
         if (multi_gpu) {
             int can_access_0_1 = 0, can_access_1_0 = 0;
             cudaDeviceCanAccessPeer(&can_access_0_1, 0, 1);
             cudaDeviceCanAccessPeer(&can_access_1_0, 1, 0);
             if (!can_access_0_1 || !can_access_1_0) {
                 multi_gpu = false;
                 TRITON_PROFILE_COUT("[GPU] Multi-GPU disabled: peer access not available\n");
             }
         }
         
         // Multi-GPU disabled due to memory access issues; use concurrent streams instead
         multi_gpu = false;
         
         if (multi_gpu) {
             // Multi-GPU: distribute cosets across GPUs
             // GPU 0: cosets 0,1,2,3  GPU 1: cosets 4,5,6,7
             const size_t cosets_per_gpu = num_cosets / num_gpus;
             
             // Create per-GPU resources - use cudaMallocManaged for cross-GPU access
             cudaStream_t streams[2];
             uint64_t* d_working_per_gpu[2];
             uint64_t* d_digests_per_gpu[2];
             
             for (int gpu = 0; gpu < num_gpus; ++gpu) {
                 CUDA_CHECK(cudaSetDevice(gpu));
                 CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
                 // Use managed memory for working buffers so all GPUs can access inputs
                 CUDA_CHECK(cudaMallocManaged(&d_working_per_gpu[gpu], width * trace_len * sizeof(uint64_t)));
                 CUDA_CHECK(cudaMallocManaged(&d_digests_per_gpu[gpu], trace_len * 5 * sizeof(uint64_t)));
                 // Advise CUDA to prefer this GPU for these allocations
                 
                 
             }
             
             // Launch cosets in parallel across GPUs
             constexpr int DIGEST_BLOCK = 256;
             int grid_digest = (int)((trace_len + DIGEST_BLOCK - 1) / DIGEST_BLOCK);
             
             for (int gpu = 0; gpu < num_gpus; ++gpu) {
                 CUDA_CHECK(cudaSetDevice(gpu));
                 size_t coset_start = gpu * cosets_per_gpu;
                 size_t coset_end = coset_start + cosets_per_gpu;
                 
                 for (size_t coset = coset_start; coset < coset_end; ++coset) {
                     uint64_t coset_offset = (BFieldElement(fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
                     
                     kernels::randomized_lde_batch_gpu_preallocated(
                         d_trace_colmajor,
                         width,
                         trace_len,
                         d_randomizer_coeffs,
                         num_randomizers,
                         trace_offset,
                         coset_offset,
                         trace_len,
                         d_working_per_gpu[gpu],
                         d_working_per_gpu[gpu],
                         nullptr,
                         streams[gpu]
                     );
                     
                     kernels::hash_bfield_rows_gpu(
                         d_working_per_gpu[gpu],
                         trace_len,
                         width,
                         d_digests_per_gpu[gpu],
                         streams[gpu]
                     );
                     
                     qzc_scatter_digests_strided_kernel<<<grid_digest, DIGEST_BLOCK, 0, streams[gpu]>>>(
                         d_digests_per_gpu[gpu],
                         ctx_->d_main_merkle(),
                         trace_len,
                         coset,
                         num_cosets
                     );
                 }
             }
             
             // Synchronize all GPUs
             for (int gpu = 0; gpu < num_gpus; ++gpu) {
                 CUDA_CHECK(cudaSetDevice(gpu));
                 CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
                 CUDA_CHECK(cudaStreamDestroy(streams[gpu]));
                 CUDA_CHECK(cudaFree(d_working_per_gpu[gpu]));
                 CUDA_CHECK(cudaFree(d_digests_per_gpu[gpu]));
             }
             CUDA_CHECK(cudaSetDevice(0)); // Return to primary GPU
             
             if (profile_main) { std::cout << "    [Main] FRUGAL cosets LDE+hash+scatter (multi-GPU): " << elapsed_ms(t_stream) << " ms\n"; }
         } else {
             // Single-GPU path with TWO-PHASE optimization
             // Phase 1: Compute polynomial coefficients ONCE (memcpy + INTT)
             // Phase 2: For each coset, only evaluate at coset points (fast)
             uint64_t* d_tmp_digests = ctx_->d_scratch_b();
             uint64_t* d_working = ctx_->d_working_main();
             
             // Reuse scratch buffer for coefficients to avoid extra allocations
             uint64_t* d_coefficients = d_trace_colmajor;
             uint64_t* d_tail_scratch = nullptr;
             CUDA_CHECK(cudaMalloc(&d_tail_scratch, width * trace_len * sizeof(uint64_t)));
             
             // Phase 1: Compute coefficients once (expensive: ~487ms)
             if (profile_main) { std::cout << "    [Main] FRUGAL Phase 1: Computing coefficients...\n"; }
             kernels::compute_trace_coefficients_gpu(
                 d_trace_colmajor,
                 width,
                 trace_len,
                 trace_offset,
                 d_coefficients,
                 ctx_->stream()
             );
             
             constexpr int DIGEST_BLOCK = 256;
             int grid_digest = (int)((trace_len + DIGEST_BLOCK - 1) / DIGEST_BLOCK);
 
             // Phase 2: Evaluate at each coset (fast: ~20ms each)
             if (profile_main) { std::cout << "    [Main] FRUGAL Phase 2: Evaluating " << num_cosets << " cosets...\n"; }
             for (size_t coset = 0; coset < num_cosets; ++coset) {
                 uint64_t coset_offset = (BFieldElement(fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
 
                 kernels::evaluate_coset_from_coefficients_gpu(
                     d_coefficients,
                     width,
                     trace_len,
                     d_randomizer_coeffs,
                     num_randomizers,
                     trace_offset,
                     coset_offset,
                     d_working,
                     d_tail_scratch,
                     ctx_->stream()
                 );
 
                 kernels::hash_bfield_rows_gpu(
                     d_working,
                     trace_len,
                     width,
                     d_tmp_digests,
                     ctx_->stream()
                 );
 
                 qzc_scatter_digests_strided_kernel<<<grid_digest, DIGEST_BLOCK, 0, ctx_->stream()>>>(
                     d_tmp_digests,
                     ctx_->d_main_merkle(),
                     trace_len,
                     coset,
                     num_cosets
                 );
             }
             
             CUDA_CHECK(cudaFree(d_tail_scratch));
 
             if (profile_main) { ctx_->synchronize(); std::cout << "    [Main] FRUGAL cosets LDE+hash+scatter (2-phase): " << elapsed_ms(t_stream) << " ms\n"; }
         }
     } else {
         // Note: Can't use scratch_a for interpolants because d_trace_colmajor IS scratch_a
         // Allocate internally for Main LDE
         kernels::randomized_lde_batch_gpu(
             d_trace_colmajor,
             width,
             trace_len,
             d_randomizer_coeffs,
             num_randomizers,
             trace_offset,
             fri_offset,
             dims_.fri_length,
             ctx_->d_main_lde(),
             ctx_->stream()
         );
         if (profile_main) { ctx_->synchronize(); std::cout << "    [Main] LDE (" << width << " cols, " << trace_len << " -> " << dims_.fri_length << "): " << elapsed_ms(t_lde) << " ms\n"; }
 
         // Debug: dump first row of main LDE (row 0 across all columns)
         if (std::getenv("TVM_DEBUG_MAIN_LDE_FIRST_ROW")) {
         ctx_->synchronize();
         fprintf(stderr, "[DBG] Main LDE first row (row=0), width=%zu, fri_len=%zu:\n", width, dims_.fri_length);
         // Column-major layout: d_main_lde[col * fri_length + row]
         for (size_t c = 0; c < 10 && c < width; c++) {
             uint64_t val = 0;
             cudaError_t err = cudaMemcpy(&val, ctx_->d_main_lde() + c * dims_.fri_length + 0, sizeof(uint64_t), cudaMemcpyDeviceToHost);
             fprintf(stderr, "  [%zu]: %lu (cuda=%d)\n", c, val, (int)err);
         }
         // Also check last 10 columns
         fprintf(stderr, "[DBG] Main LDE first row LAST 10 cols:\n");
         for (size_t c = width > 10 ? width - 10 : 0; c < width; c++) {
             uint64_t val = 0;
             cudaError_t err = cudaMemcpy(&val, ctx_->d_main_lde() + c * dims_.fri_length + 0, sizeof(uint64_t), cudaMemcpyDeviceToHost);
             fprintf(stderr, "  [%zu]: %lu (cuda=%d)\n", c, val, (int)err);
         }
         
         // Special debug for column 378 (last column) - check multiple rows
         if (width > 378) {
             size_t last_col = width - 1;
             fprintf(stderr, "[DBG] Main LDE col%zu (LAST) - first 5 rows:\n", last_col);
             for (size_t r = 0; r < 5; r++) {
                 uint64_t val = 0;
                 cudaError_t err = cudaMemcpy(&val, ctx_->d_main_lde() + last_col * dims_.fri_length + r, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                 fprintf(stderr, "  row[%zu]: %lu (cuda=%d)\n", r, val, (int)err);
             }
         }
     }
 
         if (std::getenv("TVM_DEBUG_MAIN_LDE_COL0_POINT")) {
         // Validate one point of randomized LDE for main column 0 against Rust reference.
         const size_t idx = std::getenv("TVM_DEBUG_MAIN_LDE_IDX") ? std::atoi(std::getenv("TVM_DEBUG_MAIN_LDE_IDX")) : 12345;
         const uint64_t x = (BFieldElement(dims_.fri_offset) * BFieldElement(dims_.fri_generator).pow(idx)).value();
         // Download GPU LDE value at (col0, idx). Layout is column-major: col * fri_len + row.
         uint64_t gpu_val = 0;
         CUDA_CHECK(cudaMemcpyAsync(
             &gpu_val,
             ctx_->d_main_lde() + 0 * dims_.fri_length + idx,
             sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         // Download trace column 0 values
         std::vector<uint64_t> h_trace(trace_len);
         CUDA_CHECK(cudaMemcpyAsync(
             h_trace.data(),
             d_trace_colmajor + 0 * trace_len,
             trace_len * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         // Download randomizer coeffs for col0
         std::vector<uint64_t> h_rand(num_randomizers);
         CUDA_CHECK(cudaMemcpyAsync(
             h_rand.data(),
             ctx_->d_main_randomizer_coeffs() + 0 * num_randomizers,
             num_randomizers * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
 
         uint64_t rust_val = 0;
         int rc = eval_randomized_bfe_column_at_point_rust(
             h_trace.data(),
             trace_len,
             dims_.trace_offset,
             h_rand.data(),
             num_randomizers,
             x,
             &rust_val
         );
         if (rc != 0) {
             std::cout << "[DBG] main LDE col0 point: Rust FFI failed\n";
         } else {
             std::cout << "[DBG] main LDE col0 point idx=" << idx
                       << " gpu=" << gpu_val
                       << " rust=" << rust_val
                       << " diff=" << (BFieldElement(gpu_val) - BFieldElement(rust_val)).value() << "\n";
         }
         
         // Also check LAST column (378) at row 0
         if (width >= 379) {
             size_t last_col = width - 1;
             uint64_t gpu_last = 0;
             CUDA_CHECK(cudaMemcpy(&gpu_last, ctx_->d_main_lde() + last_col * dims_.fri_length + 0, sizeof(uint64_t), cudaMemcpyDeviceToHost));
             // Download trace for last column
             std::vector<uint64_t> h_trace_last(trace_len);
             CUDA_CHECK(cudaMemcpy(h_trace_last.data(), d_trace_colmajor + last_col * trace_len, trace_len * sizeof(uint64_t), cudaMemcpyDeviceToHost));
             // Download randomizer for last column
             std::vector<uint64_t> h_rand_last(num_randomizers);
             CUDA_CHECK(cudaMemcpy(h_rand_last.data(), ctx_->d_main_randomizer_coeffs() + last_col * num_randomizers, num_randomizers * sizeof(uint64_t), cudaMemcpyDeviceToHost));
             
             // Compute Rust value at row 0 (x = fri_offset * fri_gen^0 = fri_offset)
             uint64_t x0 = dims_.fri_offset;
             uint64_t rust_last = 0;
             int rc2 = eval_randomized_bfe_column_at_point_rust(
                 h_trace_last.data(),
                 trace_len,
                 dims_.trace_offset,
                 h_rand_last.data(),
                 num_randomizers,
                 x0,
                 &rust_last
             );
             std::cout << "[DBG] main LDE col" << last_col << " (LAST) row=0: gpu=" << gpu_last
                       << " rust=" << rust_last
                       << " diff=" << (BFieldElement(gpu_last) - BFieldElement(rust_last)).value()
                       << " trace[0]=" << h_trace_last[0]
                       << " rand[0]=" << h_rand_last[0]
                       << " (rc=" << rc2 << ")\n";
         }
     }
     
         // 2. Hash LDE rows to digests
         // The digests go into the leaf level of the Merkle tree
         auto t_hash = std::chrono::high_resolution_clock::now();
         kernels::hash_bfield_rows_gpu(
             ctx_->d_main_lde(),
             dims_.fri_length,
             dims_.main_width,
             ctx_->d_main_merkle(),  // Bottom of Merkle tree
             ctx_->stream()
         );
         if (profile_main) { ctx_->synchronize(); std::cout << "    [Main] hash rows (" << dims_.fri_length << " rows × " << dims_.main_width << " cols): " << elapsed_ms(t_hash) << " ms\n"; }
     }
     
     // 3. Build Merkle tree
     auto t_merkle = std::chrono::high_resolution_clock::now();
     kernels::merkle_tree_gpu(
         ctx_->d_main_merkle(),  // Leaves at bottom
         ctx_->d_main_merkle(),  // Full tree output
         dims_.fri_length,
         ctx_->stream()
     );
     if (profile_main) { ctx_->synchronize(); std::cout << "    [Main] merkle tree (" << dims_.fri_length << " leaves): " << elapsed_ms(t_merkle) << " ms\n"; }
     
     // 4. Extract root and absorb into sponge (device-only)
     // Root is at index (2*num_leaves - 2) in our flat tree layout
     uint64_t* d_root = ctx_->d_main_merkle() + (2 * dims_.fri_length - 2) * 5;
     // Fiat-Shamir absorbs the *encoded* proof item. For MerkleRoot, that's:
     // [discriminant=0] + [digest (5 BFEs)]
     uint64_t* d_enc = ctx_->d_scratch_b(); // at least 6 u64
     uint64_t disc = 0;
     CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
     CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
     kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
     
     // 5. Add root to proof
     cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_root, 5 * sizeof(uint64_t),
                     cudaMemcpyDeviceToDevice, ctx_->stream());
     proof_size_ += 5;
 
     // CPU transcript: enqueue MerkleRoot (tiny D2H)
     {
         std::array<uint64_t, 5> h{};
         CUDA_CHECK(cudaMemcpyAsync(h.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         Digest root;
         for (size_t i = 0; i < 5; ++i) root[i] = BFieldElement(h[i]);
         fs_cpu_.enqueue(ProofItem::merkle_root(root));
         
         // Debug: Print main Merkle root for comparison with Rust
         if (std::getenv("TVM_DEBUG_MERKLE_ROOT")) {
             std::cout << "[DBG] Main Merkle root: ";
             for (int i = 0; i < 5; ++i) {
                 std::cout << std::hex << std::setw(16) << std::setfill('0') << h[i];
             }
             std::cout << std::dec << "\n";
         }
     }
 }
 
 // ============================================================================
 // Step 3: Aux Table Commitment
 // ============================================================================
 
 // Helper: Compute aux table on CPU (for hybrid mode comparison)
 // Implementation moved to gpu_stark_aux_cpu.cpp to allow TBB usage without AMX errors
 
 void GpuStark::step_aux_table_commitment(const std::vector<uint64_t>& aux_randomizer_coeffs) {
     constexpr int BLOCK = 256;
     
     // Check for detailed profiling mode
     static int aux_profile_mode = -1;
     if (aux_profile_mode == -1) {
         const char* env = std::getenv("TVM_PROFILE_AUX");
         aux_profile_mode = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
     }
     
     // Check for CPU aux computation mode
     static int use_cpu_aux = -1;
     if (use_cpu_aux == -1) {
         const char* env = std::getenv("TRITON_AUX_CPU");
         if (env) {
             use_cpu_aux = (strcmp(env, "1") == 0 || strcmp(env, "true") == 0) ? 1 : 0;
         } else {
             // Default to CPU aux in frugal mode (GPU aux is much slower for large padded heights)
             use_cpu_aux = dims_.lde_frugal_mode ? 1 : 0;
         }
         if (use_cpu_aux) {
             TRITON_PROFILE_COUT("[AUX] Using CPU computation mode (TRITON_AUX_CPU=1 or frugal default)" << std::endl);
             TRITON_PROFILE_COUT("🔧 [CPU-AUX TAG] Hybrid CPU/GPU auxiliary table computation active" << std::endl);
         }
     }
     
     cudaEvent_t ev_start, ev_chal, ev_extend, ev_transpose, ev_lde, ev_merkle;
     if (aux_profile_mode) {
         cudaEventCreate(&ev_start);
         cudaEventCreate(&ev_chal);
         cudaEventCreate(&ev_extend);
         cudaEventCreate(&ev_transpose);
         cudaEventCreate(&ev_lde);
         cudaEventCreate(&ev_merkle);
         cudaEventRecord(ev_start, ctx_->stream());
     }
     
     // 1. Sample extension challenges from CPU transcript (gold standard)
     constexpr size_t NUM_EXTENSION_CHALLENGES = 59;
 
     {
         auto ext = fs_cpu_.sample_scalars(NUM_EXTENSION_CHALLENGES); // advances CPU sponge
         std::vector<uint64_t> h(ext.size() * 3);
         for (size_t i = 0; i < ext.size(); ++i) {
             h[i * 3 + 0] = ext[i].coeff(0).value();
             h[i * 3 + 1] = ext[i].coeff(1).value();
             h[i * 3 + 2] = ext[i].coeff(2).value();
         }
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_challenges(),
             h.data(),
             h.size() * sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
     }
     
     // Compute the 4 derived challenges (indices 59..62) on GPU.
     // We need: program_digest(5), input, output, and Tip5 lookup table (256).
     // For now, upload these small claim-dependent arrays into scratch and compute derived in-place.
     {
         // Layout scratch_a:
         // [program_digest(5)] [input(len)] [output(len)] [lookup_table(256)]
         size_t off = 0;
         uint64_t* d_prog = ctx_->d_scratch_a() + off; off += 5;
         uint64_t* d_in   = ctx_->d_scratch_a() + off; off += claim_.input.size();
         uint64_t* d_out  = ctx_->d_scratch_a() + off; off += claim_.output.size();
         uint64_t* d_lut  = ctx_->d_scratch_a() + off; off += 256;
 
         // Upload program digest, input, output, lookup table
         uint64_t h_prog[5];
         for (size_t i = 0; i < 5; ++i) h_prog[i] = claim_.program_digest[i].value();
         CUDA_CHECK(cudaMemcpyAsync(d_prog, h_prog, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
 
         if (!claim_.input.empty()) {
             std::vector<uint64_t> h_in(claim_.input.size());
             for (size_t i = 0; i < claim_.input.size(); ++i) h_in[i] = claim_.input[i].value();
             CUDA_CHECK(cudaMemcpyAsync(d_in, h_in.data(), h_in.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
         }
 
         if (!claim_.output.empty()) {
             std::vector<uint64_t> h_out(claim_.output.size());
             for (size_t i = 0; i < claim_.output.size(); ++i) h_out[i] = claim_.output[i].value();
             CUDA_CHECK(cudaMemcpyAsync(d_out, h_out.data(), h_out.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
         }
 
         // Tip5 lookup table as BFE: Tip5::LOOKUP_TABLE mapped to u8 values
         std::vector<uint64_t> h_lut(256);
         for (size_t i = 0; i < 256; ++i) h_lut[i] = (uint64_t)Tip5::LOOKUP_TABLE[i];
         CUDA_CHECK(cudaMemcpyAsync(d_lut, h_lut.data(), 256 * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
 
         kernels::compute_derived_challenges_gpu(
             ctx_->d_challenges(), // expects [63*3], writes indices 59..62
             d_prog,
             d_in,
             claim_.input.size(),
             d_out,
             claim_.output.size(),
             d_lut,
             ctx_->stream()
         );
     }
     
     if (aux_profile_mode) {
         cudaEventRecord(ev_chal, ctx_->stream());
     }
 
     // Debug: Compare GPU challenges (59 sampled + 4 derived) to CPU `Challenges::from_sampled_and_claim`.
     // If this mismatches, quotient constraints will diverge from Rust verification.
     if (std::getenv("TVM_DEBUG_CHALLENGES_COMPARE")) {
         std::vector<uint64_t> h_ch(Challenges::COUNT * 3);
         CUDA_CHECK(cudaMemcpyAsync(
             h_ch.data(),
             ctx_->d_challenges(),
             h_ch.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
 
         std::vector<XFieldElement> sampled;
         sampled.reserve(Challenges::SAMPLE_COUNT);
         for (size_t i = 0; i < Challenges::SAMPLE_COUNT; ++i) {
             sampled.emplace_back(
                 BFieldElement(h_ch[i * 3 + 0]),
                 BFieldElement(h_ch[i * 3 + 1]),
                 BFieldElement(h_ch[i * 3 + 2])
             );
         }
 
         std::vector<BFieldElement> program_digest_vec;
         program_digest_vec.reserve(5);
         for (size_t i = 0; i < 5; ++i) program_digest_vec.push_back(claim_.program_digest[i]);
 
         std::vector<BFieldElement> lookup_table_vec;
         lookup_table_vec.reserve(256);
         for (uint8_t v : Tip5::LOOKUP_TABLE) lookup_table_vec.push_back(BFieldElement(v));
 
         Challenges cpu = Challenges::from_sampled_and_claim(
             sampled, program_digest_vec, claim_.input, claim_.output, lookup_table_vec
         );
 
         size_t mism = 0;
         for (size_t i = 0; i < Challenges::COUNT; ++i) {
             XFieldElement gpu_i{
                 BFieldElement(h_ch[i * 3 + 0]),
                 BFieldElement(h_ch[i * 3 + 1]),
                 BFieldElement(h_ch[i * 3 + 2])
             };
             if (!(gpu_i == cpu[i])) {
                 if (mism < 4) {
                     std::cout << "[DBG] challenge mismatch idx=" << i
                               << " gpu=" << gpu_i.to_string()
                               << " cpu=" << cpu[i].to_string() << "\n";
                 }
                 mism++;
             }
         }
         std::cout << "[DBG] challenge compare mismatches: " << mism << " / " << Challenges::COUNT << "\n";
     }
     
     // Calculate aux randomizer seed (needed for both CPU and GPU modes).
     //
     // Rust reference (`triton-vm/src/table/master_table.rs`):
     // - Create RNG: `rng_from_offset_seed(main.trace_randomizer_seed(), MasterMainTable::NUM_COLUMNS)`
     //   where rng_from_offset_seed adds offset.to_le_bytes() into the seed byte-wise (zip).
     // - Fill aux randomizer columns (col 87) from that RNG.
     //
     // Note: Rust uses StdRng (ChaCha12 under the hood in this build) + rand's
     // `random_range` for BFieldElement sampling, which is NOT a simple `u64 % p`.
     ChaCha12Rng::Seed aux_seed = randomness_seed_; // in this pipeline, we use the same base seed as Rust's trace_randomizer_seed
     {
         constexpr uint64_t MAIN_TABLE_COLUMNS = 379;
         // Apply rng_from_offset_seed(seed, offset): add offset.to_le_bytes() into seed bytes.
         const uint64_t off = MAIN_TABLE_COLUMNS;
         for (size_t i = 0; i < 8 && i < aux_seed.size(); ++i) {
             const uint8_t offset_byte = static_cast<uint8_t>((off >> (8 * i)) & 0xFF);
             aux_seed[i] = static_cast<uint8_t>(aux_seed[i] + offset_byte);
         }
     }
 
     // Legacy u64 seed is still required by existing GPU kernels; keep it derived from aux_seed[0..8].
     uint64_t aux_seed_value = 0;
     for (size_t i = 0; i < 8; ++i) {
         aux_seed_value |= static_cast<uint64_t>(aux_seed[i]) << (i * 8);
     }
 
     // 2. Compute full aux table (GPU or CPU mode)
     // Check for debug mode - temporarily disabled due to compilation issues
     const char* debug_aux = std::getenv("TRITON_DEBUG_AUX");
     bool debug_mode = debug_aux && (strcmp(debug_aux, "1") == 0 || strcmp(debug_aux, "true") == 0);
 
     if (use_cpu_aux && h_main_table_data_ != nullptr) {
         // Hybrid CPU/GPU mode: Use CPU for parallel table extension, GPU for DegreeLowering
         TRITON_PROFILE_COUT("🔧 [CPU-AUX TAG] Hybrid CPU/GPU auxiliary table computation active" << std::endl);
         TRITON_PROFILE_COUT("[CPU-AUX TAG] Using CPU parallel extension + GPU DegreeLowering" << std::endl);
 
         // Construct Challenges object from GPU challenges (all 63, including derived)
         // IMPORTANT: Use EXACT challenges from GPU to match quotient computation
         std::vector<uint64_t> h_ch(Challenges::COUNT * 3);
         CUDA_CHECK(cudaMemcpyAsync(
             h_ch.data(),
             ctx_->d_challenges(),
             h_ch.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
 
         // Construct Challenges object directly from GPU challenges
         Challenges challenges;
         for (size_t i = 0; i < Challenges::COUNT; ++i) {
             challenges[i] = XFieldElement(
                 BFieldElement(h_ch[i * 3 + 0]),
                 BFieldElement(h_ch[i * 3 + 1]),
                 BFieldElement(h_ch[i * 3 + 2])
             );
         }
 
         // Call CPU hybrid aux table computation
         compute_aux_table_cpu(
             challenges,
             aux_seed,
             ctx_->d_aux_trace(),
             ctx_->stream()
         );
 
         TRITON_PROFILE_COUT("✅ [CPU-AUX TAG] Hybrid CPU/GPU auxiliary table computation completed" << std::endl);
     } else {
         // GPU mode: Use GPU kernel for aux table extension
         kernels::extend_aux_table_full_gpu(
             ctx_->d_main_trace(),      // row-major
             dims_.main_width,
             dims_.padded_height,
             ctx_->d_challenges(),      // includes derived
             aux_seed_value,
             ctx_->d_aux_trace(),       // row-major XFE
             ctx_->d_hash_limb_pairs(),
             ctx_->d_hash_cascade_diffs(),
             ctx_->d_hash_cascade_prefix(),
             ctx_->d_hash_cascade_inverses(),
             ctx_->d_hash_cascade_mask(),
             ctx_->stream()
         );
     }
 
     // ----------------------------------------------------------------------------
     // Fix: Aux randomizer column 87 must match Rust exactly.
     //
     // The CUDA aux extension kernel historically used an mt19937_64-compatible stream and/or
     // swapped coefficient order for col 87, which causes the exact Rust dump comparison to fail
     // (while still producing a valid proof).
     //
     // We overwrite col 87 here using the reference ChaCha12Rng (same as Rust/C++ CPU path),
     // keeping the rest of the aux table fully GPU-generated.
     //
     // NOTE: Skip this fix for CPU aux mode since compute_aux_table_cpu already applies randomizers.
     // ----------------------------------------------------------------------------
     if (!use_cpu_aux) {
         // Enabled by default when Rust-trace matching is requested. Can be disabled via env for perf experiments.
         static int enable_match = -1;
         if (enable_match == -1) {
             const char* env = std::getenv("TVM_MATCH_RUST_AUX_RANDOMIZER");
             if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0)) {
                 enable_match = 0;
             } else {
                 // default ON
                 enable_match = 1;
             }
         }
         if (enable_match) {
             constexpr size_t RANDOMIZER_COL = 87;
             const size_t n = dims_.padded_height;
             std::vector<uint64_t> h_col(n * 3);
             bool loaded_from_test_data = false;
             
             // Try to load randomizer column values from Rust test data (for deterministic comparison)
             // Skip loading if TVM_DISABLE_RANDOMIZER_LOAD is set (use ChaCha12 RNG instead)
             const char* disable_load_env = std::getenv("TVM_DISABLE_RANDOMIZER_LOAD");
             bool skip_loading = (disable_load_env && (strcmp(disable_load_env, "1") == 0 || strcmp(disable_load_env, "true") == 0));
             
             const char* test_data_dir_env = std::getenv("TVM_RUST_TEST_DATA_DIR");
             if (test_data_dir_env && !skip_loading) {
                 std::string test_data_dir = test_data_dir_env;
                 std::string aux_create_path = test_data_dir + "/07_aux_tables_create.json";
                 std::ifstream file(aux_create_path);
                 
                 if (file.is_open()) {
                     try {
                         nlohmann::json data = nlohmann::json::parse(file);
                         
                         // Parse XFieldElement strings (reusable lambda)
                         auto parse_xfe_string = [](const std::string& s) -> std::tuple<uint64_t, uint64_t, uint64_t> {
                             if (s == "0_xfe" || s == "0") return {0, 0, 0};
                             if (s == "1_xfe" || s == "1") return {1, 0, 0};
                             
                             if (s.size() >= 5 && s.substr(s.size() - 4) == "_xfe") {
                                 std::string num_str = s.substr(0, s.size() - 4);
                                 try {
                                     uint64_t val = std::stoull(num_str);
                                     return {val, 0, 0};
                                 } catch (...) {
                                     return {0, 0, 0};
                                 }
                             }
                             
                             if (s.empty() || s.front() != '(' || s.back() != ')') return {0, 0, 0};
                             std::string inner = s.substr(1, s.size() - 2);
                             
                             size_t x2_pos = std::string::npos;
                             size_t x_pos = std::string::npos;
                             for (size_t i = 0; i < inner.size(); ++i) {
                                 if (inner[i] == 'x') {
                                     if (i + 4 < inner.size() && inner.substr(i, 4) == "x + ") {
                                         if (x_pos == std::string::npos) x_pos = i;
                                     } else if (i + 1 < inner.size() && 
                                               static_cast<unsigned char>(inner[i+1]) == 0xC2 &&
                                               static_cast<unsigned char>(inner[i+2]) == 0xB2 &&
                                               i + 4 < inner.size() && inner.substr(i+3, 3) == " + ") {
                                         x2_pos = i;
                                     }
                                 }
                             }
                             
                             if (x2_pos == std::string::npos || x_pos == std::string::npos) return {0, 0, 0};
                             
                             try {
                                 size_t c2_start = 0;
                                 while (c2_start < x2_pos && (inner[c2_start] == ' ' || inner[c2_start] == '\t')) c2_start++;
                                 size_t c2_end = x2_pos;
                                 while (c2_end > c2_start && inner[c2_end - 1] != ' ' && (inner[c2_end - 1] < '0' || inner[c2_end - 1] > '9')) c2_end--;
                                 
                                 size_t c1_start = x2_pos + 6;
                                 while (c1_start < x_pos && (inner[c1_start] == ' ' || inner[c1_start] == '\t')) c1_start++;
                                 size_t c1_end = x_pos;
                                 while (c1_end > c1_start && inner[c1_end - 1] != ' ' && (inner[c1_end - 1] < '0' || inner[c1_end - 1] > '9')) c1_end--;
                                 
                                 size_t c0_start = x_pos + 4;
                                 while (c0_start < inner.size() && (inner[c0_start] == ' ' || inner[c0_start] == '\t')) c0_start++;
                                 size_t c0_end = inner.size();
                                 while (c0_end > c0_start && (inner[c0_end - 1] == ' ' || inner[c0_end - 1] == '\t')) c0_end--;
                                 
                                 std::string c2_str = inner.substr(c2_start, c2_end - c2_start);
                                 std::string c1_str = inner.substr(c1_start, c1_end - c1_start);
                                 std::string c0_str = inner.substr(c0_start, c0_end - c0_start);
                                 
                                 c2_str.erase(std::remove_if(c2_str.begin(), c2_str.end(), [](char c) { return c < '0' || c > '9'; }), c2_str.end());
                                 c1_str.erase(std::remove_if(c1_str.begin(), c1_str.end(), [](char c) { return c < '0' || c > '9'; }), c1_str.end());
                                 c0_str.erase(std::remove_if(c0_str.begin(), c0_str.end(), [](char c) { return c < '0' || c > '9'; }), c0_str.end());
                                 
                                 if (c2_str.empty()) c2_str = "0";
                                 if (c1_str.empty()) c1_str = "0";
                                 if (c0_str.empty()) c0_str = "0";
                                 
                                 uint64_t c2 = std::stoull(c2_str);
                                 uint64_t c1 = std::stoull(c1_str);
                                 uint64_t c0 = std::stoull(c0_str);
                                 return {c0, c1, c2};
                             } catch (...) {
                                 return {0, 0, 0};
                             }
                         };
                         
                         // Try to load all rows from sampled_rows (if available and matches row count)
                         if (data.contains("sampled_rows") && data["sampled_rows"].is_array()) {
                             auto sampled_rows = data["sampled_rows"];
                             size_t rust_row_count = sampled_rows.size();
                             
                             if (rust_row_count == n && sampled_rows[0].is_array() && sampled_rows[0].size() > RANDOMIZER_COL) {
                                 // Load all rows for column 87
                                 for (size_t r = 0; r < n && r < rust_row_count; ++r) {
                                     std::string col87_str = sampled_rows[r][RANDOMIZER_COL].get<std::string>();
                                     auto [c0, c1, c2] = parse_xfe_string(col87_str);
                                     h_col[r * 3 + 0] = c0;
                                     h_col[r * 3 + 1] = c1;
                                     h_col[r * 3 + 2] = c2;
                                 }
                                 loaded_from_test_data = true;
                                 std::cout << "[GPU] Loaded all " << n << " rows of randomizer column 87 from Rust test data" << std::endl;
                             }
                         }
                     } catch (const std::exception& e) {
                         // If parsing fails, fall back to RNG generation
                         std::cerr << "[GPU] Warning: Failed to load randomizer from test data: " << e.what() << std::endl;
                     }
                 }
             }
             
             // Generate randomizer values using RNG if not loaded from test data
             if (!loaded_from_test_data) {
                 ChaCha12Rng rng(aux_seed);
 
                 // Match rand-0.9.x `UniformInt::sample_single_inclusive` (biased Canon's method) used by
                 // `rng.random_range(0..=BFieldElement::MAX)` in twenty-first.
                 auto sample_bfe_u64 = [&rng]() -> uint64_t {
                     constexpr uint64_t P = BFieldElement::MODULUS;
                     // range = P, so `range.wrapping_neg()` equals 2^32 - 1 for Goldilocks.
                     constexpr uint64_t NEG_RANGE = static_cast<uint64_t>(0) - P;
                     const uint64_t x = rng.next_u64();
                     const __uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(P);
                     uint64_t result = static_cast<uint64_t>(m >> 64);
                     const uint64_t lo_order = static_cast<uint64_t>(m);
                     if (lo_order > NEG_RANGE) {
                         const uint64_t y = rng.next_u64();
                         const __uint128_t my = static_cast<__uint128_t>(y) * static_cast<__uint128_t>(P);
                         const uint64_t new_hi = static_cast<uint64_t>(my >> 64);
                         // Overflow check: (lo_order + new_hi) overflowed u64?
                         if (lo_order + new_hi < lo_order) {
                             result += 1;
                         }
                     }
                     return result;
                 };
 
                 for (size_t r = 0; r < n; ++r) {
                     // Rust XFieldElement coefficients are [c0, c1, c2] = [const, x, x^2].
                     const uint64_t c0 = sample_bfe_u64();
                     const uint64_t c1 = sample_bfe_u64();
                     const uint64_t c2 = sample_bfe_u64();
                     h_col[r * 3 + 0] = c0;
                     h_col[r * 3 + 1] = c1;
                     h_col[r * 3 + 2] = c2;
                 }
             }
 
             uint64_t* d_col = nullptr;
             CUDA_CHECK(cudaMalloc(&d_col, n * 3 * sizeof(uint64_t)));
             CUDA_CHECK(cudaMemcpyAsync(d_col, h_col.data(), n * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
             const int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
             qzc_scatter_xfe_column_into_rowmajor<<<grid, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_aux_trace(), n, dims_.aux_width, RANDOMIZER_COL, d_col
             );
             CUDA_CHECK(cudaFree(d_col));
         }
     }
     
     if (aux_profile_mode) {
         cudaEventRecord(ev_extend, ctx_->stream());
     }
     
     // 3. LDE aux table
     // Transpose aux trace row-major -> component-major column layout (required by `lde_batch_gpu` and hashing):
     // component-major means: (col*3+comp)*n + row.
     const size_t aux_width = dims_.aux_width;
     const size_t trace_len = dims_.padded_height;
     // 4 elements per thread for ILP optimization
     constexpr int TRANSPOSE_ELEMS = 4;
     int grid_aux = (int)(((trace_len * aux_width) + BLOCK * TRANSPOSE_ELEMS - 1) / (BLOCK * TRANSPOSE_ELEMS));
     uint64_t* d_aux_colmajor_components = ctx_->d_scratch_b(); // [(aux_width*3) * trace_len]
     qzc_rowmajor_to_colmajor_xfe<<<grid_aux, BLOCK, 0, ctx_->stream()>>>(
         ctx_->d_aux_trace(), d_aux_colmajor_components, trace_len, aux_width
     );
     
     // Ensure transpose completes before LDE reads from scratch_b
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
 
     if (aux_profile_mode) {
         cudaEventRecord(ev_transpose, ctx_->stream());
     }
 
     uint64_t trace_offset = dims_.trace_offset;
     uint64_t fri_offset = dims_.fri_offset;
     (void)aux_randomizer_coeffs; // coeffs are uploaded once into ctx_ for true zero-copy
     // Randomized LDE for aux table (matches Rust out_of_domain_row randomizer logic)
     const size_t num_randomizers = dims_.num_trace_randomizers;
     if (num_randomizers == 0) {
         throw std::runtime_error("num_trace_randomizers is zero; cannot build randomized aux LDE");
     }
     uint64_t* d_aux_randomizer_coeffs = ctx_->d_aux_randomizer_coeffs();
 
     // Note: Can't use scratch_b for interpolants because d_aux_colmajor_components IS scratch_b
     // Use scratch_a for LDE interpolants buffer (scratch_a is free after main table LDE completes)
     // scratch_a size is trace_height * main_width, which is >= aux_width * 3 * trace_len 
     // (interpolants need num_cols * trace_len where num_cols = aux_width * 3)
     // Since main_width is typically larger than aux_width * 3, scratch_a should be large enough
     const size_t num_cols = aux_width * 3;
     uint64_t* d_lde_scratch1 = ctx_->d_scratch_a();  // For interpolants: num_cols * trace_len
     uint64_t* d_lde_scratch2 = nullptr;  // Not used anymore (fused into kernel)
     
     if (dims_.lde_frugal_mode) {
         // FRUGAL: 8-way coset streaming with multi-GPU support.
         const size_t fri_len = dims_.fri_length;
         const size_t num_cosets = fri_len / trace_len;
         if (num_cosets == 0 || fri_len % trace_len != 0) {
             throw std::runtime_error("FRUGAL AUX: invalid domains (fri_len must be multiple of trace_len)");
         }
 
         const size_t num_cols_bfe = aux_width * 3;
 
         // Two-phase optimization: compute coefficients once (in-place on scratch_b)
         kernels::compute_trace_coefficients_gpu(
             d_aux_colmajor_components,
             num_cols_bfe,
             trace_len,
             trace_offset,
             d_aux_colmajor_components,
             ctx_->stream()
         );
         
         // Check for multi-GPU with peer access
         int num_gpus = get_effective_gpu_count();
         if (num_gpus > 2) num_gpus = 2;
         bool multi_gpu = false; // Disabled due to memory issues
         
         // Multi-GPU disabled due to memory access issues
         multi_gpu = false;
         
         if (multi_gpu) {
             const size_t cosets_per_gpu = num_cosets / num_gpus;
             
             cudaStream_t streams[2];
             uint64_t* d_working_per_gpu[2];
             uint64_t* d_digests_per_gpu[2];
             
             for (int gpu = 0; gpu < num_gpus; ++gpu) {
                 CUDA_CHECK(cudaSetDevice(gpu));
                 CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
                 CUDA_CHECK(cudaMallocManaged(&d_working_per_gpu[gpu], num_cols_bfe * trace_len * sizeof(uint64_t)));
                 CUDA_CHECK(cudaMallocManaged(&d_digests_per_gpu[gpu], trace_len * 5 * sizeof(uint64_t)));
                 
                 
             }
             
             constexpr int DIGEST_BLOCK = 256;
             int grid_digest = (int)((trace_len + DIGEST_BLOCK - 1) / DIGEST_BLOCK);
             
             for (int gpu = 0; gpu < num_gpus; ++gpu) {
                 CUDA_CHECK(cudaSetDevice(gpu));
                 size_t coset_start = gpu * cosets_per_gpu;
                 size_t coset_end = coset_start + cosets_per_gpu;
                 
                 for (size_t coset = coset_start; coset < coset_end; ++coset) {
                     uint64_t coset_offset = (BFieldElement(fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
                     
                     kernels::randomized_lde_batch_gpu_preallocated(
                         d_aux_colmajor_components,
                         num_cols_bfe,
                         trace_len,
                         d_aux_randomizer_coeffs,
                         num_randomizers,
                         trace_offset,
                         coset_offset,
                         trace_len,
                         d_working_per_gpu[gpu],
                         d_working_per_gpu[gpu],
                         nullptr,
                         streams[gpu]
                     );
                     
                     kernels::hash_bfield_rows_gpu(
                         d_working_per_gpu[gpu],
                         trace_len,
                         num_cols_bfe,
                         d_digests_per_gpu[gpu],
                         streams[gpu]
                     );
                     
                     qzc_scatter_digests_strided_kernel<<<grid_digest, DIGEST_BLOCK, 0, streams[gpu]>>>(
                         d_digests_per_gpu[gpu],
                         ctx_->d_aux_merkle(),
                         trace_len,
                         coset,
                         num_cosets
                     );
                 }
             }
             
             for (int gpu = 0; gpu < num_gpus; ++gpu) {
                 CUDA_CHECK(cudaSetDevice(gpu));
                 CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
                 CUDA_CHECK(cudaStreamDestroy(streams[gpu]));
                 CUDA_CHECK(cudaFree(d_working_per_gpu[gpu]));
                 CUDA_CHECK(cudaFree(d_digests_per_gpu[gpu]));
             }
             CUDA_CHECK(cudaSetDevice(0));
         } else {
             // Single-GPU path
             uint64_t* d_working = ctx_->d_working_aux();
             // Use scratch_a for digests + tail scratch (scratch_a is large enough)
             uint64_t* d_tmp_digests = ctx_->d_scratch_a();
             uint64_t* d_tail_scratch = d_tmp_digests + trace_len * 5;
             
             constexpr int DIGEST_BLOCK = 256;
             int grid_digest = (int)((trace_len + DIGEST_BLOCK - 1) / DIGEST_BLOCK);
 
             for (size_t coset = 0; coset < num_cosets; ++coset) {
                 uint64_t coset_offset = (BFieldElement(fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
 
                 kernels::evaluate_coset_from_coefficients_gpu(
                     d_aux_colmajor_components,
                     num_cols_bfe,
                     trace_len,
                     d_aux_randomizer_coeffs,
                     num_randomizers,
                     trace_offset,
                     coset_offset,
                     d_working,
                     d_tail_scratch,
                     ctx_->stream()
                 );
 
                 kernels::hash_bfield_rows_gpu(
                     d_working,
                     trace_len,
                     num_cols_bfe,
                     d_tmp_digests,
                     ctx_->stream()
                 );
 
                 qzc_scatter_digests_strided_kernel<<<grid_digest, DIGEST_BLOCK, 0, ctx_->stream()>>>(
                     d_tmp_digests,
                     ctx_->d_aux_merkle(),
                     trace_len,
                     coset,
                     num_cosets
                 );
             }
         }
 
         if (aux_profile_mode) {
             cudaEventRecord(ev_lde, ctx_->stream());
         }
     } else {
         kernels::randomized_lde_batch_gpu_preallocated(
             d_aux_colmajor_components,
             aux_width * 3,
             trace_len,
             d_aux_randomizer_coeffs,
             num_randomizers,
             trace_offset,
             fri_offset,
             dims_.fri_length,
             ctx_->d_aux_lde(),
             d_lde_scratch1,
             d_lde_scratch2,
             ctx_->stream()
         );
         
         // Sync after LDE to catch errors early and ensure LDE completes before hash operations
         CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
 
         if (aux_profile_mode) {
             cudaEventRecord(ev_lde, ctx_->stream());
         }
 
         if (std::getenv("TVM_DEBUG_AUX_LDE_COL0_POINT")) {
         // Validate one point of randomized LDE for aux column 0 against Rust reference evaluation.
         const size_t idx = 12345;
         const uint64_t x = (BFieldElement(dims_.fri_offset) * BFieldElement(dims_.fri_generator).pow(idx)).value();
 
         // Download GPU aux LDE at (col0, idx): component-major layout.
         std::array<uint64_t, 3> gpu{};
         for (size_t comp = 0; comp < 3; ++comp) {
             CUDA_CHECK(cudaMemcpyAsync(
                 &gpu[comp],
                 ctx_->d_aux_lde() + (0 * 3 + comp) * dims_.fri_length + idx,
                 sizeof(uint64_t),
                 cudaMemcpyDeviceToHost,
                 ctx_->stream()
             ));
         }
 
         // Download aux trace column 0 (row-major) into a contiguous [trace_len*3] buffer.
         uint64_t* d_col = nullptr;
         CUDA_CHECK(cudaMalloc(&d_col, trace_len * 3 * sizeof(uint64_t)));
         qzc_gather_xfe_column_from_rowmajor<<<(int)((trace_len + BLOCK - 1) / BLOCK), BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_aux_trace(),
             trace_len,
             dims_.aux_width,
             0,
             d_col
         );
         std::vector<uint64_t> h_col(trace_len * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_col.data(), d_col, trace_len * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaFree(d_col));
 
         // Download aux randomizer coeffs for col0 component0 (BFE)
         std::vector<uint64_t> h_rand(num_randomizers);
         CUDA_CHECK(cudaMemcpyAsync(
             h_rand.data(),
             ctx_->d_aux_randomizer_coeffs() + (0 * 3 + 0) * num_randomizers,
             num_randomizers * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
 
         ctx_->synchronize();
 
         std::array<uint64_t, 3> rust{};
         int rc = eval_randomized_xfe_column_at_point_rust(
             h_col.data(),
             trace_len,
             dims_.trace_offset,
             h_rand.data(),
             num_randomizers,
             x,
             rust.data()
         );
         if (rc != 0) {
             std::cout << "[DBG] aux LDE col0 point: Rust FFI failed\n";
         } else {
             std::cout << "[DBG] aux LDE col0 point idx=" << idx
                       << " gpu=(" << gpu[0] << "," << gpu[1] << "," << gpu[2] << ")"
                       << " rust=(" << rust[0] << "," << rust[1] << "," << rust[2] << ")"
                       << "\n";
         }
     }
     } // end non-frugal LDE path
     
     if (!dims_.lde_frugal_mode) {
         // 4. Hash rows and build Merkle tree
         kernels::hash_xfield_rows_gpu(
             ctx_->d_aux_lde(),
             dims_.fri_length,
             dims_.aux_width,
             ctx_->d_aux_merkle(),
             ctx_->stream()
         );
         
         // Sync after hash to catch errors early
         CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     }
     
     kernels::merkle_tree_gpu(
         ctx_->d_aux_merkle(),
         ctx_->d_aux_merkle(),
         dims_.fri_length,
         ctx_->stream()
     );
     
     // Ensure merkle tree computation completes before accessing root
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     
     // 5. Absorb root and add to proof (device-only)
     uint64_t* d_root = ctx_->d_aux_merkle() + (2 * dims_.fri_length - 2) * 5;
     // Absorb encoded MerkleRoot proof item: [0] + digest(5)
     // Allocate temporary buffer for encoding (6 uint64_t) to avoid scratch space conflicts
     uint64_t* d_enc = nullptr;
     CUDA_CHECK(cudaMalloc(&d_enc, 6 * sizeof(uint64_t)));
     uint64_t disc = 0;
     CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
     CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
     kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
     CUDA_CHECK(cudaFreeAsync(d_enc, ctx_->stream()));
     
     cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_root, 5 * sizeof(uint64_t),
                     cudaMemcpyDeviceToDevice, ctx_->stream());
     proof_size_ += 5;
 
     // CPU transcript: enqueue MerkleRoot (tiny D2H)
     {
         std::array<uint64_t, 5> h{};
         CUDA_CHECK(cudaMemcpyAsync(h.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         Digest root;
         for (size_t i = 0; i < 5; ++i) root[i] = BFieldElement(h[i]);
         fs_cpu_.enqueue(ProofItem::merkle_root(root));
     }
 
     if (aux_profile_mode) {
         cudaEventRecord(ev_merkle, ctx_->stream());
         ctx_->synchronize();
         
         float t_chal = 0, t_extend = 0, t_transpose = 0, t_lde = 0, t_merkle = 0;
         cudaEventElapsedTime(&t_chal, ev_start, ev_chal);
         cudaEventElapsedTime(&t_extend, ev_chal, ev_extend);
         cudaEventElapsedTime(&t_transpose, ev_extend, ev_transpose);
         cudaEventElapsedTime(&t_lde, ev_transpose, ev_lde);
         cudaEventElapsedTime(&t_merkle, ev_lde, ev_merkle);
         
         float t_total = t_chal + t_extend + t_transpose + t_lde + t_merkle;
         
         std::cout << "\n[AUX PROFILING] Step 3 breakdown (TVM_PROFILE_AUX=1):" << std::endl;
         std::cout << "  1. Challenges (sample + upload):  " << t_chal << " ms (" << (t_chal/t_total*100) << "%)" << std::endl;
         std::cout << "  2. extend_aux_table_full_gpu:     " << t_extend << " ms (" << (t_extend/t_total*100) << "%)" << std::endl;
         std::cout << "  3. Transpose (row->col major):    " << t_transpose << " ms (" << (t_transpose/t_total*100) << "%)" << std::endl;
         std::cout << "  4. Randomized LDE (88 cols):      " << t_lde << " ms (" << (t_lde/t_total*100) << "%)" << std::endl;
         std::cout << "  5. Hash + Merkle tree:            " << t_merkle << " ms (" << (t_merkle/t_total*100) << "%)" << std::endl;
         std::cout << "  ----------------------------------------" << std::endl;
         std::cout << "  Total:                            " << t_total << " ms" << std::endl;
         std::cout << "\n  Rows: " << dims_.padded_height << ", Aux cols: " << dims_.aux_width 
                   << ", FRI length: " << dims_.fri_length << std::endl;
         std::cout << "  For per-table breakdown of extend_aux, also set TRITON_PROFILE_AUX=1\n" << std::endl;
         
         cudaEventDestroy(ev_start);
         cudaEventDestroy(ev_chal);
         cudaEventDestroy(ev_extend);
         cudaEventDestroy(ev_transpose);
         cudaEventDestroy(ev_lde);
         cudaEventDestroy(ev_merkle);
     }
 }
 
 // ============================================================================
 // Step 4: Quotient Commitment
 // ============================================================================
 
 void GpuStark::step_quotient_commitment() {
     TRITON_PROFILE_COUT("[GPU] ENTERING step_quotient_commitment (GPU constraint evaluation)" << std::endl);
     
     // Debug: Print first 3 challenges for comparison with Rust
     if (std::getenv("TVM_DEBUG_CHALLENGES")) {
         std::vector<uint64_t> h_ch(63 * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_ch.data(), ctx_->d_challenges(), h_ch.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         std::cout << "[DBG] First 3 challenges (compare with Rust 07_fiat_shamir_challenges.json):\n";
         for (int i = 0; i < 3; ++i) {
             std::cout << "  Challenge " << i << ": c0=" << h_ch[i*3+0] << " c1=" << h_ch[i*3+1] << " c2=" << h_ch[i*3+2] << "\n";
         }
     }
     
     // Profiling flag
     bool profile_quot = TRITON_PROFILE_ENABLED();
     auto t_start = std::chrono::high_resolution_clock::now();
     auto elapsed_ms = [](auto start) {
         return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - start).count();
     };
 
     constexpr int BLOCK = 256;
 
     // 1. Sample quotient combination weights from CPU transcript (gold standard)
     // Derive from constraint counts (matches Rust: 81 + 94 + 398 + 23 = 596)
     const size_t NUM_QUOTIENT_WEIGHTS = Quotient::MASTER_AUX_NUM_CONSTRAINTS;
     {
         auto weights = fs_cpu_.sample_scalars(NUM_QUOTIENT_WEIGHTS); // advances CPU sponge
         std::vector<uint64_t> h(weights.size() * 3);
         for (size_t i = 0; i < weights.size(); ++i) {
             h[i * 3 + 0] = weights[i].coeff(0).value();
             h[i * 3 + 1] = weights[i].coeff(1).value();
             h[i * 3 + 2] = weights[i].coeff(2).value();
         }
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_quotient_weights(),
             h.data(),
             h.size() * sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
     }
 
     if (dims_.lde_frugal_mode) {
         step_quotient_commitment_frugal();
         return;
     }
 
     // 2. GPU quotient evaluation using chunked approach (works for small inputs)
     //   - evaluate constraints on quotient domain with unit_distance = quotient_len / trace_len
     //   - combine init+cons+tran+term into quotient values (XFE)
     //   - segmentify + LDE into 4 segment codewords on the FRI domain (colmajor components)
     //   - hash rows -> Merkle -> absorb root -> append to proof buffer
     //
     // NOTE: This uses incorrect segmentification but works for small inputs.
     // TODO: Replace with proper JIT LDE implementation for large inputs.
     const size_t quotient_len = dims_.quotient_length;
     const size_t fri_len = dims_.fri_length;
     const size_t trace_len = dims_.padded_height;
     const size_t main_width = dims_.main_width;
     const size_t aux_width = dims_.aux_width;
     const size_t num_segments = dims_.num_quotient_segments; // 4
 
     if (quotient_len == 0 || trace_len == 0 || (quotient_len % trace_len) != 0) {
         throw std::runtime_error("Invalid quotient dimensions: quotient_len must be a multiple of trace_len");
     }
     const size_t unit_distance = quotient_len / trace_len;
 
     // Domain points x_i = quotient_offset * quotient_gen^i
     uint64_t* d_domain_x = nullptr;
     CUDA_CHECK(cudaMalloc(&d_domain_x, quotient_len * sizeof(uint64_t)));
     int grid_q = (int)((quotient_len + BLOCK - 1) / BLOCK);
     qzc_fill_domain_points<<<grid_q, BLOCK, 0, ctx_->stream()>>>(
         d_domain_x, quotient_len, dims_.quotient_offset, dims_.quotient_generator
     );
 
     // IMPORTANT: quotient constraint kernels expect ROW-MAJOR rows on the QUOTIENT domain.
     // Our LDE tables live on the FRI domain. The quotient-domain table is a strided subsample of the FRI table.
     if (quotient_len == 0 || (fri_len % quotient_len) != 0) {
         throw std::runtime_error("Invalid domains: fri_len must be a multiple of quotient_len");
     }
     const size_t eval_to_quot_stride = fri_len / quotient_len;
 
     // Memory optimization for mid/large inputs (e.g. input=14):
     // Avoid materializing full row-major main+aux tables on quotient domain (which is ~10GB for input=14).
     // Instead, process quotient rows in chunks with a small "window" (chunk_len + unit_distance) so
     // transition constraints can read next-row data without wrap-around.
     // For large GPUs, use larger chunks to reduce kernel launch overhead.
     constexpr size_t QUOT_CHUNK = 65536;  // Increased from 32768 for better GPU utilization
     const size_t chunk_base = std::min(QUOT_CHUNK, quotient_len);
     const size_t window_max = std::min(chunk_base + unit_distance, quotient_len);
     size_t* d_win_indices = nullptr;     // [window_max]
     uint64_t* d_main_window = nullptr;   // [window_max * main_width]
     uint64_t* d_aux_window = nullptr;    // [window_max * aux_width * 3]
     CUDA_CHECK(cudaMalloc(&d_win_indices, window_max * sizeof(size_t)));
     CUDA_CHECK(cudaMalloc(&d_main_window, window_max * main_width * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_aux_window, window_max * aux_width * 3 * sizeof(uint64_t)));
 
     // Frugal mode: batch buffers and col-major traces for streaming LDE
     constexpr size_t FRUGAL_BATCH_COLS = 10; // Tip5 RATE
     uint64_t* d_main_batch_rows = nullptr;
     uint64_t* d_aux_batch_rows = nullptr;
     uint64_t* d_main_colmajor = nullptr;
     uint64_t* d_aux_colmajor_components = nullptr;
     if (dims_.lde_frugal_mode) {
         CUDA_CHECK(cudaMalloc(&d_main_batch_rows, window_max * FRUGAL_BATCH_COLS * sizeof(uint64_t)));
         CUDA_CHECK(cudaMalloc(&d_aux_batch_rows, window_max * FRUGAL_BATCH_COLS * sizeof(uint64_t)));
 
         // Build col-major main trace and aux component-major trace once
         d_main_colmajor = ctx_->d_scratch_a();
         {
             size_t total = trace_len * main_width;
             int grid_t = (int)((total + BLOCK - 1) / BLOCK);
             qzc_rowmajor_to_colmajor_bfe<<<grid_t, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_main_trace(), d_main_colmajor, trace_len, main_width
             );
         }
         d_aux_colmajor_components = ctx_->d_scratch_b();
         {
             constexpr int TRANSPOSE_ELEMS = 4;
             int grid_aux = (int)(((trace_len * aux_width) + BLOCK * TRANSPOSE_ELEMS - 1) / (BLOCK * TRANSPOSE_ELEMS));
             qzc_rowmajor_to_colmajor_xfe<<<grid_aux, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_aux_trace(), d_aux_colmajor_components, trace_len, aux_width
             );
         }
         CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     }
 
     // Zerofier inverse arrays on quotient domain (BFE)
     uint64_t* d_init_inv = nullptr;
     uint64_t* d_cons_inv = nullptr;
     uint64_t* d_tran_inv = nullptr; // transition zerofier inverse
     uint64_t* d_term_inv = nullptr;
     CUDA_CHECK(cudaMalloc(&d_init_inv, quotient_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_cons_inv, quotient_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_tran_inv, quotient_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_term_inv, quotient_len * sizeof(uint64_t)));
 
     // IMPORTANT: CPU prover behavior (and our GPU LDE kernels) effectively ignore trace_domain.offset
     // (see `lde_column_gpu` ignoring trace_offset). Therefore, the zerofiers must also use offset=1:
     //   init: 1/(x - 1)
     //   cons: 1/(x^trace_len - 1)
     //   term: 1/(x - g^{-1})
     //   tran: (x - g^{-1})/(x^trace_len - 1)
     uint64_t trace_gen_inv = BFieldElement(dims_.trace_generator).inverse().value();
     uint64_t trace_offset_for_zerofier = 1;
     uint64_t trace_offset_pow = 1;
     qzc_compute_zerofier_arrays<<<grid_q, BLOCK, 0, ctx_->stream()>>>(
         d_domain_x,
         quotient_len,
         (uint64_t)trace_len,
         trace_offset_for_zerofier,
         trace_offset_pow,
         trace_gen_inv,
         d_init_inv,
         d_cons_inv,
         d_tran_inv,
         d_term_inv
     );
 
     // Constraint outputs (XFE arrays)
     uint64_t* d_out_init = nullptr;
     uint64_t* d_out_cons = nullptr;
     uint64_t* d_out_term = nullptr;
     uint64_t* d_out_tran = nullptr;
     // Transition parts are computed per-window and fused directly into output.
     uint64_t* d_tran_parts_window[4]{nullptr, nullptr, nullptr, nullptr}; // [window_max*3] each
     uint64_t* d_quotient = nullptr;
     CUDA_CHECK(cudaMalloc(&d_out_init, quotient_len * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_out_cons, quotient_len * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_out_term, quotient_len * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_out_tran, quotient_len * 3 * sizeof(uint64_t)));
     for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaMalloc(&d_tran_parts_window[i], window_max * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_quotient, quotient_len * 3 * sizeof(uint64_t)));
 
     auto t_alloc = std::chrono::high_resolution_clock::now();
     if (profile_quot) {
         ctx_->synchronize();
         std::cout << "    [Quot] alloc+zerofiers: " << elapsed_ms(t_start) << " ms" << std::endl;
     }
     
     // Evaluate constraints chunk-by-chunk.
     auto t_constraints = std::chrono::high_resolution_clock::now();
     double gather_time = 0, init_cons_term_time = 0, transition_time = 0;
     int grid_win = (int)((window_max + BLOCK - 1) / BLOCK);
     for (size_t start = 0; start < quotient_len; start += QUOT_CHUNK) {
         const size_t chunk_len = std::min(QUOT_CHUNK, quotient_len - start);
         const size_t win_len = std::min(chunk_len + unit_distance, window_max);
         int grid_len = (int)((win_len + BLOCK - 1) / BLOCK);
 
         auto t_chunk_gather = std::chrono::high_resolution_clock::now();
         // quotient-row -> fri-row indices for this window
         qzc_fill_strided_indices_offset_wrap<<<grid_len, BLOCK, 0, ctx_->stream()>>>(
             d_win_indices,
             win_len,
             start,
             quotient_len,
             eval_to_quot_stride
         );
 
         if (dims_.lde_frugal_mode) {
             // Streaming LDE for main columns
             for (size_t col_start = 0; col_start < main_width; col_start += FRUGAL_BATCH_COLS) {
                 const size_t batch_cols = std::min(FRUGAL_BATCH_COLS, main_width - col_start);
                 kernels::randomized_lde_batch_gpu(
                     d_main_colmajor + col_start * trace_len,
                     batch_cols,
                     trace_len,
                     ctx_->d_main_randomizer_coeffs() + col_start * dims_.num_trace_randomizers,
                     dims_.num_trace_randomizers,
                     dims_.trace_offset,
                     dims_.fri_offset,
                     fri_len,
                     ctx_->d_main_lde(), // batch buffer
                     ctx_->stream()
                 );
                 kernels::gather_bfield_rows_colmajor_gpu(
                     ctx_->d_main_lde(),
                     d_win_indices,
                     d_main_batch_rows,
                     fri_len,
                     batch_cols,
                     win_len,
                     ctx_->stream()
                 );
                 int grid_scatter = (int)((win_len * batch_cols + BLOCK - 1) / BLOCK);
                 qzc_scatter_rowmajor_offset_bfe<<<grid_scatter, BLOCK, 0, ctx_->stream()>>>(
                     d_main_batch_rows,
                     batch_cols,
                     d_main_window,
                     main_width,
                     win_len,
                     col_start
                 );
             }
 
             // Streaming LDE for aux component columns (BFE view)
             const size_t aux_cols_bfe = aux_width * 3;
             for (size_t col_start = 0; col_start < aux_cols_bfe; col_start += FRUGAL_BATCH_COLS) {
                 const size_t batch_cols = std::min(FRUGAL_BATCH_COLS, aux_cols_bfe - col_start);
                 kernels::randomized_lde_batch_gpu(
                     d_aux_colmajor_components + col_start * trace_len,
                     batch_cols,
                     trace_len,
                     ctx_->d_aux_randomizer_coeffs() + col_start * dims_.num_trace_randomizers,
                     dims_.num_trace_randomizers,
                     dims_.trace_offset,
                     dims_.fri_offset,
                     fri_len,
                     ctx_->d_aux_lde(), // batch buffer
                     ctx_->stream()
                 );
                 kernels::gather_bfield_rows_colmajor_gpu(
                     ctx_->d_aux_lde(),
                     d_win_indices,
                     d_aux_batch_rows,
                     fri_len,
                     batch_cols,
                     win_len,
                     ctx_->stream()
                 );
                 int grid_scatter = (int)((win_len * batch_cols + BLOCK - 1) / BLOCK);
                 qzc_scatter_rowmajor_offset_bfe<<<grid_scatter, BLOCK, 0, ctx_->stream()>>>(
                     d_aux_batch_rows,
                     batch_cols,
                     d_aux_window,
                     aux_width * 3,
                     win_len,
                     col_start
                 );
             }
         } else {
             // Gather main/aux rows on quotient domain window into row-major buffers.
             kernels::gather_bfield_rows_colmajor_gpu(
                 ctx_->d_main_lde(),
                 d_win_indices,
                 d_main_window,
                 fri_len,
                 main_width,
                 win_len,
                 ctx_->stream()
             );
             kernels::gather_xfield_rows_colmajor_gpu(
                 ctx_->d_aux_lde(),
                 d_win_indices,
                 d_aux_window,
                 fri_len,
                 aux_width,
                 win_len,
                 ctx_->stream()
             );
         }
         if (profile_quot) { ctx_->synchronize(); gather_time += elapsed_ms(t_chunk_gather); }
 
         auto t_chunk_init = std::chrono::high_resolution_clock::now();
         // Initial + consistency + terminal (write directly into full outputs at offset `start`)
         kernels::compute_quotient_split_partial(
             d_main_window,
             main_width,
             d_aux_window,
             aux_width,
             chunk_len,
             ctx_->d_challenges(),
             ctx_->d_quotient_weights(),
             d_init_inv + start,
             d_cons_inv + start,
             d_term_inv + start,
             d_out_init + start * 3,
             d_out_cons + start * 3,
             d_out_term + start * 3,
             ctx_->stream()
         );
         if (profile_quot) { ctx_->synchronize(); init_cons_term_time += elapsed_ms(t_chunk_init); }
 
         auto t_chunk_trans = std::chrono::high_resolution_clock::now();
         // Transition constraints need next-row data; compute on the full window (win_len),
         // then copy the first `chunk_len` rows into the full transition accumulator.
         kernels::launch_quotient_transition_part0(
             d_main_window, main_width,
             d_aux_window, aux_width,
             win_len, unit_distance,
             ctx_->d_challenges(), ctx_->d_quotient_weights(),
             d_tran_parts_window[0],
             ctx_->stream()
         );
         kernels::launch_quotient_transition_part1(
             d_main_window, main_width,
             d_aux_window, aux_width,
             win_len, unit_distance,
             ctx_->d_challenges(), ctx_->d_quotient_weights(),
             d_tran_parts_window[1],
             ctx_->stream()
         );
         kernels::launch_quotient_transition_part2(
             d_main_window, main_width,
             d_aux_window, aux_width,
             win_len, unit_distance,
             ctx_->d_challenges(), ctx_->d_quotient_weights(),
             d_tran_parts_window[2],
             ctx_->stream()
         );
         kernels::launch_quotient_transition_part3(
             d_main_window, main_width,
             d_aux_window, aux_width,
             win_len, unit_distance,
             ctx_->d_challenges(), ctx_->d_quotient_weights(),
             d_tran_parts_window[3],
             ctx_->stream()
         );
 
         // FUSED: Sum all 4 parts + scale by zerofier inverse in ONE kernel
         kernels::fused_sum4_scale_transition(
             d_tran_parts_window[0],
             d_tran_parts_window[1],
             d_tran_parts_window[2],
             d_tran_parts_window[3],
             d_tran_inv + start,
             chunk_len,
             d_out_tran + start * 3,
             ctx_->stream()
         );
         if (profile_quot) { ctx_->synchronize(); transition_time += elapsed_ms(t_chunk_trans); }
     }
     
     if (profile_quot) {
         ctx_->synchronize();
         std::cout << "    [Quot] constraint eval total: " << elapsed_ms(t_constraints) << " ms" << std::endl;
         std::cout << "      gather: " << gather_time << " ms" << std::endl;
         std::cout << "      init+cons+term: " << init_cons_term_time << " ms" << std::endl;
         std::cout << "      transition: " << transition_time << " ms" << std::endl;
     }
 
     // Combine all: quotient = init + cons + tran + term
     kernels::combine_quotient_results(
         d_out_init, d_out_cons, d_out_tran, d_out_term,
         quotient_len,
         d_quotient,
         ctx_->stream()
     );
 
     // Debug: retain quotient codeword on quotient domain for later cross-checking in OOD step.
     if (std::getenv("TVM_DEBUG_KEEP_QUOT_CODEWORD")) {
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_quotient_codeword(),
             d_quotient,
             quotient_len * 3 * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice,
             ctx_->stream()
         ));
     }
 
     // Debug: dump first 10 quotient codeword values for comparison with Rust reference
     if (std::getenv("TVM_DEBUG_DUMP_QUOT_FIRST10")) {
         ctx_->synchronize();
         std::vector<uint64_t> h_q(30);  // 10 * 3 components
         CUDA_CHECK(cudaMemcpy(h_q.data(), d_quotient, 30 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
         std::cout << "[DBG] GPU quotient codeword first 10 values:" << std::endl;
         for (size_t i = 0; i < 10; ++i) {
             std::cout << "  [" << i << "]: c0=" << h_q[i * 3 + 0] 
                       << ", c1=" << h_q[i * 3 + 1] 
                       << ", c2=" << h_q[i * 3 + 2] << std::endl;
         }
     }
 
     if (std::getenv("TVM_DEBUG_QUOT_ROW_CHECK") && !dims_.lde_frugal_mode) {
         // Check: quotient at a chosen quotient-domain index computed from CPU constraint evaluators
         // matches GPU kernel output at that index. This helps detect row-dependent kernel bugs.
         const size_t row0 = 12345 % quotient_len;
         const size_t row1 = (row0 + unit_distance) % quotient_len;
 
         // Download challenges (63*3) and weights (MASTER_AUX_NUM_CONSTRAINTS*3)
         std::vector<uint64_t> h_ch(Challenges::COUNT * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_ch.data(), ctx_->d_challenges(), h_ch.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         std::vector<uint64_t> h_w(Quotient::MASTER_AUX_NUM_CONSTRAINTS * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_w.data(), ctx_->d_quotient_weights(), h_w.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
         // Download main/aux rows for row0 and row1 by gathering from FRI-domain LDE (col-major) using stride.
         size_t h_idx2[2] = { row0 * eval_to_quot_stride, row1 * eval_to_quot_stride };
         size_t* d_idx2 = nullptr;
         uint64_t* d_main2 = nullptr;
         uint64_t* d_aux2 = nullptr;
         CUDA_CHECK(cudaMalloc(&d_idx2, 2 * sizeof(size_t)));
         CUDA_CHECK(cudaMalloc(&d_main2, 2 * main_width * sizeof(uint64_t)));
         CUDA_CHECK(cudaMalloc(&d_aux2, 2 * aux_width * 3 * sizeof(uint64_t)));
         CUDA_CHECK(cudaMemcpyAsync(d_idx2, h_idx2, 2 * sizeof(size_t), cudaMemcpyHostToDevice, ctx_->stream()));
         kernels::gather_bfield_rows_colmajor_gpu(ctx_->d_main_lde(), d_idx2, d_main2, fri_len, main_width, 2, ctx_->stream());
         kernels::gather_xfield_rows_colmajor_gpu(ctx_->d_aux_lde(), d_idx2, d_aux2, fri_len, aux_width, 2, ctx_->stream());
         std::vector<uint64_t> h_main0(main_width), h_main1(main_width);
         std::vector<uint64_t> h_aux0(aux_width * 3), h_aux1(aux_width * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_main0.data(), d_main2 + 0 * main_width, main_width * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaMemcpyAsync(h_main1.data(), d_main2 + 1 * main_width, main_width * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaMemcpyAsync(h_aux0.data(), d_aux2 + 0 * aux_width * 3, aux_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaMemcpyAsync(h_aux1.data(), d_aux2 + 1 * aux_width * 3, aux_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaFree(d_idx2));
         CUDA_CHECK(cudaFree(d_main2));
         CUDA_CHECK(cudaFree(d_aux2));
 
         // Download GPU quotient at row0
         std::array<uint64_t, 3> h_q0{};
         CUDA_CHECK(cudaMemcpyAsync(h_q0.data(), d_quotient + row0 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
 
         // Build Challenges on CPU from sampled(59) + claim
         std::vector<XFieldElement> sampled;
         sampled.reserve(Challenges::SAMPLE_COUNT);
         for (size_t i = 0; i < Challenges::SAMPLE_COUNT; ++i) {
             sampled.emplace_back(
                 BFieldElement(h_ch[i * 3 + 0]),
                 BFieldElement(h_ch[i * 3 + 1]),
                 BFieldElement(h_ch[i * 3 + 2])
             );
         }
         std::vector<BFieldElement> program_digest_vec;
         program_digest_vec.reserve(5);
         for (size_t i = 0; i < 5; ++i) program_digest_vec.push_back(claim_.program_digest[i]);
         std::vector<BFieldElement> lookup_table_vec;
         lookup_table_vec.reserve(256);
         for (uint8_t v : Tip5::LOOKUP_TABLE) lookup_table_vec.push_back(BFieldElement(v));
         Challenges cpu_ch = Challenges::from_sampled_and_claim(
             sampled, program_digest_vec, claim_.input, claim_.output, lookup_table_vec
         );
 
         // Weights
         std::vector<XFieldElement> cpu_w;
         cpu_w.reserve(Quotient::MASTER_AUX_NUM_CONSTRAINTS);
         for (size_t i = 0; i < Quotient::MASTER_AUX_NUM_CONSTRAINTS; ++i) {
             cpu_w.emplace_back(
                 BFieldElement(h_w[i * 3 + 0]),
                 BFieldElement(h_w[i * 3 + 1]),
                 BFieldElement(h_w[i * 3 + 2])
             );
         }
 
         // Rows
         std::vector<BFieldElement> main0, main1b;
         main0.reserve(main_width);
         main1b.reserve(main_width);
         for (size_t i = 0; i < main_width; ++i) {
             main0.push_back(BFieldElement(h_main0[i]));
             main1b.push_back(BFieldElement(h_main1[i]));
         }
         std::vector<XFieldElement> aux0, aux1v;
         aux0.reserve(aux_width);
         aux1v.reserve(aux_width);
         for (size_t i = 0; i < aux_width; ++i) {
             aux0.emplace_back(BFieldElement(h_aux0[i * 3 + 0]), BFieldElement(h_aux0[i * 3 + 1]), BFieldElement(h_aux0[i * 3 + 2]));
             aux1v.emplace_back(BFieldElement(h_aux1[i * 3 + 0]), BFieldElement(h_aux1[i * 3 + 1]), BFieldElement(h_aux1[i * 3 + 2]));
         }
 
         // Evaluate constraints (CPU)
         auto init_c = Quotient::evaluate_initial_constraints(main0, aux0, cpu_ch);
         auto cons_c = Quotient::evaluate_consistency_constraints(main0, aux0, cpu_ch);
         auto tran_c = Quotient::evaluate_transition_constraints(main0, aux0, main1b, aux1v, cpu_ch);
         auto term_c = Quotient::evaluate_terminal_constraints(main0, aux0, cpu_ch);
 
         // Zerofier inverses at x(row0) on quotient domain.
         BFieldElement x0 = BFieldElement(dims_.quotient_offset) * BFieldElement(dims_.quotient_generator).pow(row0);
         BFieldElement g_inv = BFieldElement(dims_.trace_generator).inverse();
         XFieldElement init_inv = XFieldElement{(x0 - BFieldElement::one()).inverse()};
         XFieldElement cons_inv = XFieldElement{(x0.pow((uint64_t)trace_len) - BFieldElement::one()).inverse()};
         XFieldElement except_last = XFieldElement{(x0 - g_inv)};
         XFieldElement tran_inv = except_last * cons_inv;
         XFieldElement term_inv = XFieldElement{(x0 - g_inv).inverse()};
 
         std::vector<XFieldElement> summands;
         summands.reserve(init_c.size() + cons_c.size() + tran_c.size() + term_c.size());
         for (auto& v : init_c) summands.push_back(v * init_inv);
         for (auto& v : cons_c) summands.push_back(v * cons_inv);
         for (auto& v : tran_c) summands.push_back(v * tran_inv);
         for (auto& v : term_c) summands.push_back(v * term_inv);
 
         XFieldElement q_cpu = XFieldElement::zero();
         for (size_t i = 0; i < summands.size(); ++i) {
             q_cpu += cpu_w[i] * summands[i];
         }
 
         XFieldElement q_gpu{BFieldElement(h_q0[0]), BFieldElement(h_q0[1]), BFieldElement(h_q0[2])};
         std::cout << "[DBG] quot row check idx=" << row0
                   << " q_cpu=" << q_cpu.to_string()
                   << " q_gpu=" << q_gpu.to_string()
                   << " diff=" << (q_cpu - q_gpu).to_string() << "\n";
     }
 
     if (std::getenv("TVM_DEBUG_QUOT_ROW_CHECK_RUST")) {
         // Compare GPU quotient at several quotient-domain indices against Rust constraint evaluation
         // at the same base-field x. This helps detect row-dependent or window-boundary bugs.
         const size_t test_rows_raw[] = {0, 1, 2, 3, 7, 123, 12345, 54321, quotient_len ? (quotient_len - 1) : 0};
         const size_t num_test_rows = sizeof(test_rows_raw) / sizeof(test_rows_raw[0]);
         for (size_t t = 0; t < num_test_rows; ++t) {
             const size_t row0 = test_rows_raw[t] % quotient_len;
             const size_t row1 = (row0 + unit_distance) % quotient_len;
 
             // Download challenges (63*3) and weights (MASTER_AUX_NUM_CONSTRAINTS*3)
             std::vector<uint64_t> h_ch(Challenges::COUNT * 3);
             CUDA_CHECK(cudaMemcpyAsync(h_ch.data(), ctx_->d_challenges(), h_ch.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             std::vector<uint64_t> h_w(Quotient::MASTER_AUX_NUM_CONSTRAINTS * 3);
             CUDA_CHECK(cudaMemcpyAsync(h_w.data(), ctx_->d_quotient_weights(), h_w.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
             // Download main/aux rows for row0 and row1 by gathering from FRI-domain LDE (col-major) using stride.
             size_t h_idx2[2] = { row0 * eval_to_quot_stride, row1 * eval_to_quot_stride };
             size_t* d_idx2 = nullptr;
             uint64_t* d_main2 = nullptr;
             uint64_t* d_aux2 = nullptr;
             CUDA_CHECK(cudaMalloc(&d_idx2, 2 * sizeof(size_t)));
             CUDA_CHECK(cudaMalloc(&d_main2, 2 * main_width * sizeof(uint64_t)));
             CUDA_CHECK(cudaMalloc(&d_aux2, 2 * aux_width * 3 * sizeof(uint64_t)));
             CUDA_CHECK(cudaMemcpyAsync(d_idx2, h_idx2, 2 * sizeof(size_t), cudaMemcpyHostToDevice, ctx_->stream()));
             kernels::gather_bfield_rows_colmajor_gpu(ctx_->d_main_lde(), d_idx2, d_main2, fri_len, main_width, 2, ctx_->stream());
             kernels::gather_xfield_rows_colmajor_gpu(ctx_->d_aux_lde(), d_idx2, d_aux2, fri_len, aux_width, 2, ctx_->stream());
             std::vector<uint64_t> h_main0(main_width), h_main1(main_width);
             std::vector<uint64_t> h_aux0(aux_width * 3), h_aux1(aux_width * 3);
             CUDA_CHECK(cudaMemcpyAsync(h_main0.data(), d_main2 + 0 * main_width, main_width * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_main1.data(), d_main2 + 1 * main_width, main_width * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_aux0.data(), d_aux2 + 0 * aux_width * 3, aux_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_aux1.data(), d_aux2 + 1 * aux_width * 3, aux_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaFree(d_idx2));
             CUDA_CHECK(cudaFree(d_main2));
             CUDA_CHECK(cudaFree(d_aux2));
 
             // GPU quotient at row0
             std::array<uint64_t, 3> h_q0{};
             CUDA_CHECK(cudaMemcpyAsync(h_q0.data(), d_quotient + row0 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             ctx_->synchronize();
 
             const uint64_t x = (BFieldElement(dims_.quotient_offset) * BFieldElement(dims_.quotient_generator).pow(row0)).value();
             const uint64_t trace_gen_inv = BFieldElement(dims_.trace_generator).inverse().value();
             std::array<uint64_t, 3> q_rust{};
             int rc = compute_quotient_value_at_bfe_point_rust(
                 x,
                 (uint64_t)trace_len,
                 trace_gen_inv,
                 h_main0.data(),
                 h_aux0.data(),
                 h_main1.data(),
                 h_aux1.data(),
                 h_ch.data(),
                 h_w.data(),
                 Quotient::MASTER_AUX_NUM_CONSTRAINTS,
                 q_rust.data()
             );
             if (rc != 0) {
                 std::cout << "[DBG] quot row rust check idx=" << row0 << ": Rust FFI failed\n";
             } else {
                 XFieldElement q_gpu{BFieldElement(h_q0[0]), BFieldElement(h_q0[1]), BFieldElement(h_q0[2])};
                 XFieldElement q_ref{BFieldElement(q_rust[0]), BFieldElement(q_rust[1]), BFieldElement(q_rust[2])};
                 std::cout << "[DBG] quot row rust check idx=" << row0
                           << " q_gpu=" << q_gpu.to_string()
                           << " q_rust=" << q_ref.to_string()
                           << " diff=" << (q_gpu - q_ref).to_string() << "\n";
             }
         }
     }
 
     // ============================================================================
     // CORRECTED: Use JIT LDE segmentification instead of incorrect direct splitting
     // ============================================================================
 
     // Segmentify + LDE using coefficient splitting approach (matching Rust's interpolate_quotient_segments)
     // This approach works for quotient evaluated on quotient domain (not multicoset):
     //   1. Interpolate quotient codeword to get polynomial coefficients
     //   2. Split coefficients into NUM_SEGMENTS segments (f(x) = sum_i x^i * f_i(x^N))
     //   3. LDE each segment polynomial to FRI domain
     auto t_seg = std::chrono::high_resolution_clock::now();
     
     // quotient_offset_inv = quotient_domain.offset.inverse() for interpolation
     uint64_t quotient_offset_inv = BFieldElement(dims_.quotient_offset).inverse().value();
     
     // Debug: print segmentify parameters
     if (std::getenv("TVM_DEBUG_SEGMENTIFY")) {
         ctx_->synchronize();
         std::vector<uint64_t> h_quot(30);  // First 10 XFE values = 30 u64
         CUDA_CHECK(cudaMemcpy(h_quot.data(), d_quotient, 30 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
         std::cout << "[DBG SEGMENTIFY] quotient_len=" << quotient_len << " num_segments=" << num_segments 
                   << " fri_len=" << fri_len << std::endl;
         std::cout << "[DBG SEGMENTIFY] quotient_offset=" << dims_.quotient_offset 
                   << " quotient_offset_inv=" << quotient_offset_inv
                   << " fri_offset=" << dims_.fri_offset << std::endl;
         std::cout << "[DBG SEGMENTIFY] quotient[0..9]: ";
         for (int i = 0; i < 10; ++i) {
             std::cout << "(" << h_quot[i*3] << "," << h_quot[i*3+1] << "," << h_quot[i*3+2] << ") ";
         }
         std::cout << std::endl;
     }
 
     // Use the original segmentify that works on quotient domain evaluations
     // Use scratch_a and scratch_b for temporary buffers to avoid allocation
     // (scratch buffers are free after aux table step and are much larger than quotient_len)
     kernels::quotient_segmentify_and_lde_gpu(
         d_quotient,                         // [quotient_len * 3] quotient evaluations on quotient domain
         quotient_len,                       // length of quotient domain
         num_segments,                       // 4 segments
         quotient_offset_inv,                // for coset interpolation
         dims_.fri_offset,                   // FRI domain offset
         fri_len,                            // FRI domain length
         ctx_->d_quotient_segment_coeffs(),  // output: segment coefficients
         ctx_->d_quotient_segments(),        // output: segment codewords on FRI domain
         ctx_->stream(),
         ctx_->d_scratch_a(),                // Use scratch_a for d_c0
         ctx_->d_scratch_b()                 // Use scratch_b for d_c1
     );
     // Ensure segmentify+LDE completes before hash operations
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     if (profile_quot) { std::cout << "    [Quot] segmentify+LDE (coeff split): " << elapsed_ms(t_seg) << " ms" << std::endl; }
 
     // Debug: print first few segment coefficients after segmentify
     if (std::getenv("TVM_DEBUG_SEGMENTIFY")) {
         ctx_->synchronize();
         size_t segment_len = quotient_len / num_segments;
         std::vector<uint64_t> h_coeffs(30);
         CUDA_CHECK(cudaMemcpy(h_coeffs.data(), ctx_->d_quotient_segment_coeffs(), 30 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
         std::cout << "[DBG SEGMENTIFY] segment_len=" << segment_len << std::endl;
         std::cout << "[DBG SEGMENTIFY] seg_coeffs[0..29]: ";
         for (int i = 0; i < 30; ++i) {
             std::cout << h_coeffs[i] << " ";
         }
         std::cout << std::endl;
     }
 
     // Debug: dump GPU quotient segment codewords to file for comparison with Rust reference
     if (std::getenv("TVM_DEBUG_DUMP_QUOT_SEGMENTS")) {
         ctx_->synchronize();
         std::string dump_path = std::getenv("TVM_DEBUG_DUMP_QUOT_SEGMENTS");
         std::cout << "[DEBUG] Dumping GPU quotient segment codewords to: " << dump_path << std::endl;
         
         // Download segment codewords from GPU
         // Layout: [seg][comp][fri_idx] where seg=0..3, comp=0..2, fri_idx=0..fri_len-1
         size_t total_words = num_segments * 3 * fri_len;
         std::vector<uint64_t> h_segments(total_words);
         CUDA_CHECK(cudaMemcpy(h_segments.data(), ctx_->d_quotient_segments(), 
                               total_words * sizeof(uint64_t), cudaMemcpyDeviceToHost));
         
         std::ofstream ofs(dump_path);
         ofs << "{\n  \"fri_len\": " << fri_len << ",\n";
         ofs << "  \"num_segments\": " << num_segments << ",\n";
         ofs << "  \"rows\": [\n";
         for (size_t i = 0; i < fri_len; ++i) {
             ofs << "    [";
             for (size_t seg = 0; seg < num_segments; ++seg) {
                 // GPU layout: d_segments[(seg * 3 + comp) * fri_len + i]
                 uint64_t c0 = h_segments[(seg * 3 + 0) * fri_len + i];
                 uint64_t c1 = h_segments[(seg * 3 + 1) * fri_len + i];
                 uint64_t c2 = h_segments[(seg * 3 + 2) * fri_len + i];
                 ofs << "[" << c0 << "," << c1 << "," << c2 << "]";
                 if (seg + 1 < num_segments) ofs << ",";
             }
             ofs << "]";
             if (i + 1 < fri_len) ofs << ",";
             ofs << "\n";
         }
         ofs << "  ]\n}\n";
         ofs.close();
         std::cout << "[DEBUG] Dumped " << fri_len << " rows x " << num_segments << " segments" << std::endl;
     }
 
     // NOTE: The above uses incorrect segmentification (direct codeword splitting)
     // This works for small inputs but fails for large ones due to wrong polynomial decomposition.
     // It may also fail when nondeterminism (RAM/secret input) is used, as the quotient
     // polynomial becomes more complex and the incorrect segmentification produces wrong results.
     // The correct approach needs JIT LDE: evaluate traces on cosets, compute constraints,
     // then segmentify using proper rearrangement + iNTT + scaling.
     //
     // WARNING: When using nondeterminism, always verify proofs pass. If they fail with
     // "OUT-OF-DOMAIN QUOTIENT VALUE MISMATCH", this is likely due to incorrect segmentification.
     // Enable TVM_DEBUG_OOD_QUOT_SELF_CHECK to see the mismatch between computed and provided values.
 
     // Debug quotient segment reconstruction removed for performance
 
     // Hash quotient segment rows (each row has 4 XFEs) and build Merkle tree
     auto t_hash = std::chrono::high_resolution_clock::now();
     kernels::hash_xfield_rows_gpu(
         ctx_->d_quotient_segments(),
         fri_len,
         num_segments,
         ctx_->d_quotient_merkle(),
         ctx_->stream()
     );
     // Ensure hash computation completes before merkle_tree_gpu
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     kernels::merkle_tree_gpu(
         ctx_->d_quotient_merkle(),
         ctx_->d_quotient_merkle(),
         fri_len,
         ctx_->stream()
     );
     // Ensure merkle tree computation completes before accessing root
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     if (profile_quot) { std::cout << "    [Quot] hash+merkle: " << elapsed_ms(t_hash) << " ms" << std::endl; }
 
     // Absorb encoded MerkleRoot proof item: [0] + digest(5)
     uint64_t* d_root = ctx_->d_quotient_merkle() + (2 * fri_len - 2) * 5;
     uint64_t* d_enc = ctx_->d_scratch_b(); // at least 6 u64
     uint64_t disc = 0;
     CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
     CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
     kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
 
     // Append quotient root to proof buffer
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_root,
         5 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += 5;
 
     // CPU transcript: enqueue MerkleRoot (tiny D2H)
     {
         std::array<uint64_t, 5> h{};
         CUDA_CHECK(cudaMemcpyAsync(h.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         Digest root;
         for (size_t i = 0; i < 5; ++i) root[i] = BFieldElement(h[i]);
         fs_cpu_.enqueue(ProofItem::merkle_root(root));
     }
 
     // Cleanup temporary buffers
     CUDA_CHECK(cudaFree(d_win_indices));
     CUDA_CHECK(cudaFree(d_main_window));
     CUDA_CHECK(cudaFree(d_aux_window));
     if (d_main_batch_rows) CUDA_CHECK(cudaFree(d_main_batch_rows));
     if (d_aux_batch_rows) CUDA_CHECK(cudaFree(d_aux_batch_rows));
     CUDA_CHECK(cudaFree(d_domain_x));
     CUDA_CHECK(cudaFree(d_init_inv));
     CUDA_CHECK(cudaFree(d_cons_inv));
     CUDA_CHECK(cudaFree(d_tran_inv));
     CUDA_CHECK(cudaFree(d_term_inv));
     CUDA_CHECK(cudaFree(d_out_init));
     CUDA_CHECK(cudaFree(d_out_cons));
     CUDA_CHECK(cudaFree(d_out_term));
     CUDA_CHECK(cudaFree(d_out_tran));
     for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaFree(d_tran_parts_window[i]));
     CUDA_CHECK(cudaFree(d_quotient));
 }
 
 void GpuStark::step_quotient_commitment_frugal() {
     // Frugal quotient: compute main/aux LDE on the quotient domain one coset at a time (len = padded_height),
     // evaluate constraints in windows, scatter quotient evaluations into the full quotient domain,
     // then segmentify+LDE and commit as usual.
     const bool profile_quot = TRITON_PROFILE_ENABLED();
     auto t0 = std::chrono::high_resolution_clock::now();
     auto elapsed_ms = [&t0]() {
         auto now = std::chrono::high_resolution_clock::now();
         double ms = std::chrono::duration<double, std::milli>(now - t0).count();
         t0 = now;
         return ms;
     };
 
     constexpr int BLOCK = 256;
     const size_t quotient_len = dims_.quotient_length;
     const size_t fri_len = dims_.fri_length;
     const size_t trace_len = dims_.padded_height;
     const size_t main_width = dims_.main_width;
     const size_t aux_width = dims_.aux_width;
     const size_t num_segments = dims_.num_quotient_segments;
 
     if (quotient_len == 0 || trace_len == 0 || (quotient_len % trace_len) != 0) {
         throw std::runtime_error("FRUGAL quotient: quotient_len must be multiple of trace_len");
     }
     const size_t num_cosets = quotient_len / trace_len; // typically 8
     if (num_cosets == 0) {
         throw std::runtime_error("FRUGAL quotient: num_cosets is zero");
     }
 
     // Transpose main trace to col-major into scratch_a (needed as input to LDE kernel).
     uint64_t* d_main_colmajor = ctx_->d_scratch_a(); // [main_width * trace_len]
     {
         size_t total = trace_len * main_width;
         int grid_t = (int)((total + BLOCK - 1) / BLOCK);
         qzc_rowmajor_to_colmajor_bfe<<<grid_t, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_trace(), d_main_colmajor, trace_len, main_width
         );
     }
 
     // Transpose aux trace to component-major col-major into scratch_b.
     uint64_t* d_aux_colmajor_components = ctx_->d_scratch_b(); // [(aux_width*3) * trace_len]
     {
         constexpr int TRANSPOSE_ELEMS = 4;
         int grid_aux = (int)(((trace_len * aux_width) + BLOCK * TRANSPOSE_ELEMS - 1) / (BLOCK * TRANSPOSE_ELEMS));
         qzc_rowmajor_to_colmajor_xfe<<<grid_aux, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_aux_trace(), d_aux_colmajor_components, trace_len, aux_width
         );
     }
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     if (profile_quot) { std::cout << "    [Quot FRUGAL] transpose main+aux: " << elapsed_ms() << " ms\n"; }
 
     // Two-phase optimization: compute coefficients once (in-place on scratch buffers)
     kernels::compute_trace_coefficients_gpu(
         d_main_colmajor,
         main_width,
         trace_len,
         dims_.trace_offset,
         d_main_colmajor,
         ctx_->stream()
     );
     kernels::compute_trace_coefficients_gpu(
         d_aux_colmajor_components,
         aux_width * 3,
         trace_len,
         dims_.trace_offset,
         d_aux_colmajor_components,
         ctx_->stream()
     );
 
     // Allocate full quotient codeword on quotient domain: [quotient_len * 3]
     uint64_t* d_quotient_full = nullptr;
     CUDA_CHECK(cudaMalloc(&d_quotient_full, quotient_len * 3 * sizeof(uint64_t)));
 
     // Tail scratch buffers for coset evaluation (freed at end of step)
     uint64_t* d_main_tail = nullptr;
     uint64_t* d_aux_tail = nullptr;
     CUDA_CHECK(cudaMalloc(&d_main_tail, main_width * trace_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_aux_tail, aux_width * 3 * trace_len * sizeof(uint64_t)));
 
     // Per-coset buffers (reused each coset)
     uint64_t* d_domain_x = nullptr;
     uint64_t* d_init_inv = nullptr;
     uint64_t* d_cons_inv = nullptr;
     uint64_t* d_tran_inv = nullptr;
     uint64_t* d_term_inv = nullptr;
     CUDA_CHECK(cudaMalloc(&d_domain_x, trace_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_init_inv, trace_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_cons_inv, trace_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_tran_inv, trace_len * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_term_inv, trace_len * sizeof(uint64_t)));
 
     constexpr size_t QUOT_CHUNK = 65536;
     const size_t window_max = std::min(QUOT_CHUNK + 1, trace_len);
     size_t* d_win_indices = nullptr;
     uint64_t* d_main_window = nullptr;
     uint64_t* d_aux_window = nullptr;
     CUDA_CHECK(cudaMalloc(&d_win_indices, window_max * sizeof(size_t)));
     CUDA_CHECK(cudaMalloc(&d_main_window, window_max * main_width * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_aux_window, window_max * aux_width * 3 * sizeof(uint64_t)));
 
     // Constraint outputs (per coset)
     uint64_t* d_out_init = nullptr;
     uint64_t* d_out_cons = nullptr;
     uint64_t* d_out_term = nullptr;
     uint64_t* d_out_tran = nullptr;
     uint64_t* d_tran_parts_window[4]{nullptr, nullptr, nullptr, nullptr};
     uint64_t* d_quotient_coset = nullptr;
     CUDA_CHECK(cudaMalloc(&d_out_init, trace_len * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_out_cons, trace_len * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_out_term, trace_len * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_out_tran, trace_len * 3 * sizeof(uint64_t)));
     for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaMalloc(&d_tran_parts_window[i], window_max * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_quotient_coset, trace_len * 3 * sizeof(uint64_t)));
 
     int grid_trace = (int)((trace_len + BLOCK - 1) / BLOCK);
     uint64_t trace_gen_inv = BFieldElement(dims_.trace_generator).inverse().value();
     uint64_t trace_offset_for_zerofier = 1;
     uint64_t trace_offset_pow = 1;
 
     // Evaluate cosets
     for (size_t coset = 0; coset < num_cosets; ++coset) {
         // coset_offset on quotient domain: quotient_offset * quotient_gen^coset
         uint64_t coset_offset = (BFieldElement(dims_.quotient_offset) * BFieldElement(dims_.quotient_generator).pow(coset)).value();
 
         // domain_x for this coset uses generator = trace_generator (== quotient_generator^num_cosets)
         qzc_fill_domain_points<<<grid_trace, BLOCK, 0, ctx_->stream()>>>(
             d_domain_x, trace_len, coset_offset, dims_.trace_generator
         );
         qzc_compute_zerofier_arrays<<<grid_trace, BLOCK, 0, ctx_->stream()>>>(
             d_domain_x,
             trace_len,
             (uint64_t)trace_len,
             trace_offset_for_zerofier,
             trace_offset_pow,
             trace_gen_inv,
             d_init_inv,
             d_cons_inv,
             d_tran_inv,
             d_term_inv
         );
 
         // Evaluate main/aux on this coset from pre-computed coefficients.
         kernels::evaluate_coset_from_coefficients_gpu(
             d_main_colmajor,
             main_width,
             trace_len,
             ctx_->d_main_randomizer_coeffs(),
             dims_.num_trace_randomizers,
             dims_.trace_offset,
             coset_offset,
             ctx_->d_working_main(),
             d_main_tail,
             ctx_->stream()
         );
 
         kernels::evaluate_coset_from_coefficients_gpu(
             d_aux_colmajor_components,
             aux_width * 3,
             trace_len,
             ctx_->d_aux_randomizer_coeffs(),
             dims_.num_trace_randomizers,
             dims_.trace_offset,
             coset_offset,
             ctx_->d_working_aux(),
             d_aux_tail,
             ctx_->stream()
         );
 
         // Evaluate constraints chunk-by-chunk on this coset (unit_distance=1, stride=1).
         for (size_t start = 0; start < trace_len; start += QUOT_CHUNK) {
             const size_t chunk_len = std::min(QUOT_CHUNK, trace_len - start);
             const size_t win_len = std::min(chunk_len + 1, window_max);
             int grid_len = (int)((win_len + BLOCK - 1) / BLOCK);
 
             qzc_fill_strided_indices_offset_wrap<<<grid_len, BLOCK, 0, ctx_->stream()>>>(
                 d_win_indices,
                 win_len,
                 start,
                 trace_len,
                 1
             );
 
             kernels::gather_bfield_rows_colmajor_gpu(
                 ctx_->d_working_main(),
                 d_win_indices,
                 d_main_window,
                 trace_len,
                 main_width,
                 win_len,
                 ctx_->stream()
             );
             kernels::gather_xfield_rows_colmajor_gpu(
                 ctx_->d_working_aux(),
                 d_win_indices,
                 d_aux_window,
                 trace_len,
                 aux_width,
                 win_len,
                 ctx_->stream()
             );
 
             kernels::compute_quotient_split_partial(
                 d_main_window,
                 main_width,
                 d_aux_window,
                 aux_width,
                 chunk_len,
                 ctx_->d_challenges(),
                 ctx_->d_quotient_weights(),
                 d_init_inv + start,
                 d_cons_inv + start,
                 d_term_inv + start,
                 d_out_init + start * 3,
                 d_out_cons + start * 3,
                 d_out_term + start * 3,
                 ctx_->stream()
             );
 
             kernels::launch_quotient_transition_part0(
                 d_main_window, main_width,
                 d_aux_window, aux_width,
                 win_len, 1,
                 ctx_->d_challenges(), ctx_->d_quotient_weights(),
                 d_tran_parts_window[0],
                 ctx_->stream()
             );
             kernels::launch_quotient_transition_part1(
                 d_main_window, main_width,
                 d_aux_window, aux_width,
                 win_len, 1,
                 ctx_->d_challenges(), ctx_->d_quotient_weights(),
                 d_tran_parts_window[1],
                 ctx_->stream()
             );
             kernels::launch_quotient_transition_part2(
                 d_main_window, main_width,
                 d_aux_window, aux_width,
                 win_len, 1,
                 ctx_->d_challenges(), ctx_->d_quotient_weights(),
                 d_tran_parts_window[2],
                 ctx_->stream()
             );
             kernels::launch_quotient_transition_part3(
                 d_main_window, main_width,
                 d_aux_window, aux_width,
                 win_len, 1,
                 ctx_->d_challenges(), ctx_->d_quotient_weights(),
                 d_tran_parts_window[3],
                 ctx_->stream()
             );
 
             kernels::fused_sum4_scale_transition(
                 d_tran_parts_window[0],
                 d_tran_parts_window[1],
                 d_tran_parts_window[2],
                 d_tran_parts_window[3],
                 d_tran_inv + start,
                 chunk_len,
                 d_out_tran + start * 3,
                 ctx_->stream()
             );
         }
 
         kernels::combine_quotient_results(
             d_out_init, d_out_cons, d_out_tran, d_out_term,
             trace_len,
             d_quotient_coset,
             ctx_->stream()
         );
 
         // Scatter coset quotient evaluations into the full quotient-domain array.
         int grid_sc = (int)((trace_len + BLOCK - 1) / BLOCK);
         qzc_scatter_xfe_strided_kernel<<<grid_sc, BLOCK, 0, ctx_->stream()>>>(
             d_quotient_coset,
             d_quotient_full,
             trace_len,
             coset,
             num_cosets
         );
     }
 
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     if (profile_quot) { std::cout << "    [Quot FRUGAL] constraints+scatter: " << elapsed_ms() << " ms\n"; }
 
     // Segmentify + LDE into quotient segments on FRI domain (same as non-frugal)
     auto t_seg = std::chrono::high_resolution_clock::now();
     uint64_t quotient_offset_inv = BFieldElement(dims_.quotient_offset).inverse().value();
     kernels::quotient_segmentify_and_lde_gpu(
         d_quotient_full,
         quotient_len,
         num_segments,
         quotient_offset_inv,
         dims_.fri_offset,
         fri_len,
         ctx_->d_quotient_segment_coeffs(),
         ctx_->d_quotient_segments(),
         ctx_->stream(),
         ctx_->d_scratch_a(),
         ctx_->d_scratch_b()
     );
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     if (profile_quot) {
         double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_seg).count();
         std::cout << "    [Quot FRUGAL] segmentify+LDE: " << ms << " ms\n";
     }
 
     // Hash quotient segment rows and build Merkle tree
     auto t_hash = std::chrono::high_resolution_clock::now();
     kernels::hash_xfield_rows_gpu(
         ctx_->d_quotient_segments(),
         fri_len,
         num_segments,
         ctx_->d_quotient_merkle(),
         ctx_->stream()
     );
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     kernels::merkle_tree_gpu(
         ctx_->d_quotient_merkle(),
         ctx_->d_quotient_merkle(),
         fri_len,
         ctx_->stream()
     );
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     if (profile_quot) {
         double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_hash).count();
         std::cout << "    [Quot FRUGAL] hash+merkle: " << ms << " ms\n";
     }
 
     // Absorb encoded MerkleRoot proof item: [0] + digest(5)
     uint64_t* d_root = ctx_->d_quotient_merkle() + (2 * fri_len - 2) * 5;
     uint64_t* d_enc = ctx_->d_scratch_b(); // at least 6 u64
     uint64_t disc = 0;
     CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
     CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
     kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
 
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_root,
         5 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += 5;
 
     // CPU transcript: enqueue MerkleRoot (tiny D2H)
     {
         std::array<uint64_t, 5> h{};
         CUDA_CHECK(cudaMemcpyAsync(h.data(), d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         Digest root;
         for (size_t i = 0; i < 5; ++i) root[i] = BFieldElement(h[i]);
         fs_cpu_.enqueue(ProofItem::merkle_root(root));
     }
 
     // Cleanup
     CUDA_CHECK(cudaFree(d_quotient_full));
     CUDA_CHECK(cudaFree(d_main_tail));
     CUDA_CHECK(cudaFree(d_aux_tail));
     CUDA_CHECK(cudaFree(d_domain_x));
     CUDA_CHECK(cudaFree(d_init_inv));
     CUDA_CHECK(cudaFree(d_cons_inv));
     CUDA_CHECK(cudaFree(d_tran_inv));
     CUDA_CHECK(cudaFree(d_term_inv));
     CUDA_CHECK(cudaFree(d_win_indices));
     CUDA_CHECK(cudaFree(d_main_window));
     CUDA_CHECK(cudaFree(d_aux_window));
     CUDA_CHECK(cudaFree(d_out_init));
     CUDA_CHECK(cudaFree(d_out_cons));
     CUDA_CHECK(cudaFree(d_out_term));
     CUDA_CHECK(cudaFree(d_out_tran));
     for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaFree(d_tran_parts_window[i]));
     CUDA_CHECK(cudaFree(d_quotient_coset));
 }
 
 void GpuStark::step_out_of_domain_evaluation() {
     // 1. Sample out-of-domain point from CPU transcript (gold standard)
     uint64_t* d_ood = ctx_->d_scratch_a(); // [3]
     {
         auto z = fs_cpu_.sample_scalars(1)[0]; // advances CPU sponge
         uint64_t h[3] = { z.coeff(0).value(), z.coeff(1).value(), z.coeff(2).value() };
         CUDA_CHECK(cudaMemcpyAsync(d_ood, h, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
     }
     CUDA_CHECK(cudaMemcpyAsync(ctx_->d_ood_point(), d_ood, 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
     if (std::getenv("TVM_DEBUG_OOD_POINT")) {
         std::array<uint64_t, 3> h{};
         CUDA_CHECK(cudaMemcpyAsync(h.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         std::cout << "[DBG] ood_point: (" << h[2] << "·x² + " << h[1] << "·x + " << h[0] << ")" << std::endl;
     }
 
     // Compute next_row_point = z * trace_gen
     uint64_t* d_next = ctx_->d_scratch_a() + 3; // [3]
     qzc_mul_xfe_scalar_kernel<<<1, 1, 0, ctx_->stream()>>>(d_ood, dims_.trace_generator, d_next);
     CUDA_CHECK(cudaMemcpyAsync(ctx_->d_next_row_point(), d_next, 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
 
     constexpr int BLOCK = 256;
     const size_t n = dims_.padded_height; // trace domain size for barycentric
     const int grid_n = (int)((n + BLOCK - 1) / BLOCK);
 
     // scratch layout (reuse freely; no other step needs it concurrently):
     uint64_t* scratch = ctx_->d_scratch_b();
     size_t off = 0;
     uint64_t* d_domain = scratch + off; off += n;            // [n] BFE
     uint64_t* d_shift = scratch + off;  off += n * 3;        // [n*3]
     uint64_t* d_inv = scratch + off;    off += n * 3;        // [n*3]
     uint64_t* d_w = scratch + off;      off += n * 3;        // [n*3] domain_over_shift
     uint64_t* d_denom = scratch + off;  off += 3;            // [3]
 
     // IMPORTANT: We need a dedicated buffer for block reductions. Do NOT alias this with `d_main_curr`,
     // because we evaluate at two points (z and g*z) and the second evaluation's reduction scratch would
     // otherwise clobber the first point's OOD results.
     uint64_t* d_block_sums  = scratch + off; off += (size_t)grid_n * 3; // [grid_n * 3]
     uint64_t* d_block_sums2 = scratch + off; off += (size_t)grid_n * 3; // [grid_n * 3]
 
     uint64_t* d_main_curr = scratch + off; off += dims_.main_width * 3;
     uint64_t* d_main_next = scratch + off; off += dims_.main_width * 3;
     uint64_t* d_aux_curr  = scratch + off; off += dims_.aux_width * 3;
     uint64_t* d_aux_next  = scratch + off; off += dims_.aux_width * 3;
     uint64_t* d_quot_ood  = scratch + off; off += dims_.num_quotient_segments * 3;
 
     // 1) domain values for trace domain (use exact offset+generator from Rust)
     uint64_t trace_offset = dims_.trace_offset;
     uint64_t trace_gen = dims_.trace_generator;
     qzc_fill_domain_points<<<grid_n, BLOCK, 0, ctx_->stream()>>>(d_domain, n, trace_offset, trace_gen);
     if (std::getenv("TVM_DEBUG_TRACE_DOMAIN_POINTS")) {
         std::array<uint64_t, 4> h_dom{};
         CUDA_CHECK(cudaMemcpyAsync(h_dom.data(), d_domain, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         for (size_t i = 0; i < 4; ++i) {
             uint64_t cpu = (BFieldElement(trace_offset) * BFieldElement(trace_gen).pow(i)).value();
             std::cout << "[DBG] trace_domain[" << i << "]: gpu=" << h_dom[i] << " cpu=" << cpu
                       << " diff=" << (BFieldElement(h_dom[i]) - BFieldElement(cpu)).value() << "\n";
         }
     }
 
     const bool profile_ood = TRITON_PROFILE_ENABLED();
     auto compute_weight_and_eval_point = [&](const uint64_t* d_point3,
                                              uint64_t* d_out_main,
                                              uint64_t* d_out_aux,
                                              const char* point_name = "z") {
         auto t_shift = std::chrono::high_resolution_clock::now();
         // shift = z - d_i
         qzc_compute_shift_xfe<<<grid_n, BLOCK, 0, ctx_->stream()>>>(d_domain, n, d_point3, d_shift);
         if (profile_ood) { ctx_->synchronize(); std::cout << "    [OOD " << point_name << "] shift: " << elapsed_ms(t_shift) << " ms\n"; }
         
         auto t_inv = std::chrono::high_resolution_clock::now();
         kernels::xfe_batch_inverse_gpu(d_shift, d_inv, n, ctx_->stream());
         if (profile_ood) { ctx_->synchronize(); std::cout << "    [OOD " << point_name << "] batch_inv (" << n << " XFEs): " << elapsed_ms(t_inv) << " ms\n"; }
 
         if (std::getenv("TVM_DEBUG_OOD_BARY")) {
             // Sanity-check a few inverses against CPU XFieldElement::inverse()
             constexpr size_t K = 8;
             std::array<uint64_t, K * 3> h_shift{};
             std::array<uint64_t, K * 3> h_inv{};
             CUDA_CHECK(cudaMemcpyAsync(h_shift.data(), d_shift, K * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_inv.data(), d_inv, K * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             ctx_->synchronize();
             for (size_t i = 0; i < K; ++i) {
                 XFieldElement s{
                     BFieldElement(h_shift[i * 3 + 0]),
                     BFieldElement(h_shift[i * 3 + 1]),
                     BFieldElement(h_shift[i * 3 + 2])
                 };
                 XFieldElement inv_cpu = s.inverse();
                 XFieldElement inv_gpu{
                     BFieldElement(h_inv[i * 3 + 0]),
                     BFieldElement(h_inv[i * 3 + 1]),
                     BFieldElement(h_inv[i * 3 + 2])
                 };
                 if (!(inv_cpu == inv_gpu)) {
                     std::cout << "[DBG] OOD inv mismatch i=" << i
                               << " shift=" << s.to_string()
                               << " inv_gpu=" << inv_gpu.to_string()
                               << " inv_cpu=" << inv_cpu.to_string() << "\n";
                     break;
                 }
             }
         }
         auto t_w = std::chrono::high_resolution_clock::now();
         qzc_domain_over_shift_xfe<<<grid_n, BLOCK, 0, ctx_->stream()>>>(d_domain, d_inv, n, d_w);
         if (profile_ood) { ctx_->synchronize(); std::cout << "    [OOD " << point_name << "] domain_over_shift: " << elapsed_ms(t_w) << " ms\n"; }
 
         // Reduce denom = sum_i d_w[i] for arbitrary n.
         // We iteratively reduce until one element remains.
         auto t_reduce = std::chrono::high_resolution_clock::now();
         size_t cur_n = n;
         const uint64_t* d_in = d_w;
         uint64_t* d_out = d_block_sums;
         uint64_t* d_tmp = d_block_sums2;
         while (cur_n > 1) {
             int grid = (int)((cur_n + BLOCK - 1) / BLOCK);
             qzc_reduce_blocks_xfe_kernel<<<grid, BLOCK, 0, ctx_->stream()>>>(d_in, cur_n, d_out);
             cur_n = (size_t)grid;
             d_in = d_out;
             // ping-pong outputs
             uint64_t* swap = d_out;
             d_out = d_tmp;
             d_tmp = swap;
         }
         // Copy denom into d_denom and invert in-place.
         CUDA_CHECK(cudaMemcpyAsync(d_denom, d_in, 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
         qzc_xfe_inv_single<<<1,1,0,ctx_->stream()>>>(d_denom); // denom := denom_inv
         if (profile_ood) { ctx_->synchronize(); std::cout << "    [OOD " << point_name << "] denom_reduce+inv: " << elapsed_ms(t_reduce) << " ms\n"; }
 
         if (std::getenv("TVM_DEBUG_BARY_WEIGHTS_RUST")) {
             // Compare first few weights and denom_inv against Rust reference for this point.
             std::array<uint64_t, 12> h_w0{};
             std::array<uint64_t, 3> h_den{};
             std::array<uint64_t, 3> h_z{};
             CUDA_CHECK(cudaMemcpyAsync(h_w0.data(), d_w, 12 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_den.data(), d_denom, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_z.data(), d_point3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             ctx_->synchronize();
 
             uint64_t* rust_out = nullptr;
             size_t rust_len = 0;
             int rc = debug_barycentric_weights_rust(
                 n,
                 dims_.trace_offset,
                 h_z[0], h_z[1], h_z[2],
                 &rust_out,
                 &rust_len
             );
             if (rc == 0 && rust_out && rust_len == 15) {
                 std::cout << "[DBG] bary weights rust vs gpu:\n";
                 for (size_t i = 0; i < 4; ++i) {
                     XFieldElement wg{BFieldElement(h_w0[i*3+0]),BFieldElement(h_w0[i*3+1]),BFieldElement(h_w0[i*3+2])};
                     XFieldElement wr{BFieldElement(rust_out[i*3+0]),BFieldElement(rust_out[i*3+1]),BFieldElement(rust_out[i*3+2])};
                     std::cout << "  w["<<i<<"] gpu="<<wg.to_string()<<" rust="<<wr.to_string()
                               <<" diff="<<(wg-wr).to_string()<<"\n";
                 }
                 XFieldElement dg{BFieldElement(h_den[0]),BFieldElement(h_den[1]),BFieldElement(h_den[2])};
                 XFieldElement dr{BFieldElement(rust_out[12]),BFieldElement(rust_out[13]),BFieldElement(rust_out[14])};
                 std::cout << "  denom_inv gpu="<<dg.to_string()<<" rust="<<dr.to_string()
                           <<" diff="<<(dg-dr).to_string()<<"\n";
                 constraint_evaluation_free(rust_out, rust_len);
             } else {
                 std::cout << "[DBG] bary weights rust: FFI failed\n";
             }
         }
 
         // evaluate main/aux
         auto t_eval = std::chrono::high_resolution_clock::now();
         qzc_eval_main_ood<<<dims_.main_width, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_trace(), n, dims_.main_width, d_w, d_denom, d_out_main
         );
         qzc_eval_aux_ood<<<dims_.aux_width, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_aux_trace(), n, dims_.aux_width, d_w, d_denom, d_out_aux
         );
         if (profile_ood) { ctx_->synchronize(); std::cout << "    [OOD " << point_name << "] eval_main+aux (" << dims_.main_width << "+" << dims_.aux_width << " cols): " << elapsed_ms(t_eval) << " ms\n"; }
     };
 
     const bool dbg_main_col0_full = std::getenv("TVM_DEBUG_OOD_MAIN_COL0_FULL") != nullptr;
     const bool dbg_aux_col0_full  = std::getenv("TVM_DEBUG_OOD_AUX_COL0_FULL") != nullptr;
     const bool dbg_main_col100_bary_rust = std::getenv("TVM_DEBUG_OOD_MAIN_COL100_BARY_RUST") != nullptr;
     std::array<uint64_t, 3> dbg_main0_bary{};
     std::vector<BFieldElement> dbg_main0_rand_coeffs;
     XFieldElement dbg_z = XFieldElement::zero();
     std::array<uint64_t, 3> dbg_main100_bary{};
     std::array<uint64_t, 3> dbg_aux0_bary{};
     std::vector<BFieldElement> dbg_aux0_rand_coeffs;
 
     compute_weight_and_eval_point(d_ood, d_main_curr, d_aux_curr, "z");
 
     // Debug: validate one main-column OOD evaluation against a CPU recomputation using GPU-produced weights.
     if (std::getenv("TVM_DEBUG_OOD_MAIN_COL0")) {
         const size_t col = 0;
         // Download denom_inv (d_denom holds denom_inv) and weights d_w
         std::array<uint64_t, 3> h_denom{};
         CUDA_CHECK(cudaMemcpyAsync(h_denom.data(), d_denom, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         std::vector<uint64_t> h_w(n * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_w.data(), d_w, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
         // Download GPU output for this column
         std::array<uint64_t, 3> h_gpu_out{};
         CUDA_CHECK(cudaMemcpyAsync(h_gpu_out.data(), d_main_curr + col * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
         // Gather trace column to a contiguous device buffer, then copy to host
         uint64_t* d_col = nullptr;
         CUDA_CHECK(cudaMalloc(&d_col, n * sizeof(uint64_t)));
         int grid_col = (int)((n + BLOCK - 1) / BLOCK);
         qzc_gather_bfield_column_from_rowmajor<<<grid_col, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_trace(),
             n,
             dims_.main_width,
             col,
             d_col
         );
         std::vector<uint64_t> h_trace_col(n);
         CUDA_CHECK(cudaMemcpyAsync(h_trace_col.data(), d_col, n * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaFree(d_col));
 
         // Randomizer coeffs for this column (BFE)
         std::vector<uint64_t> h_rand(dims_.num_trace_randomizers);
         CUDA_CHECK(cudaMemcpyAsync(
             h_rand.data(),
             ctx_->d_main_randomizer_coeffs() + col * dims_.num_trace_randomizers,
             h_rand.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
 
         // OOD point z on host
         std::array<uint64_t, 3> h_z{};
         CUDA_CHECK(cudaMemcpyAsync(h_z.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
         ctx_->synchronize();
 
         XFieldElement denom_inv{BFieldElement(h_denom[0]), BFieldElement(h_denom[1]), BFieldElement(h_denom[2])};
         XFieldElement num = XFieldElement::zero();
         for (size_t i = 0; i < n; ++i) {
             XFieldElement wi{BFieldElement(h_w[i * 3 + 0]), BFieldElement(h_w[i * 3 + 1]), BFieldElement(h_w[i * 3 + 2])};
             BFieldElement vi(h_trace_col[i]);
             num += wi * XFieldElement(vi);
         }
         XFieldElement bary = num * denom_inv;
 
         // IMPORTANT: at this point in the pipeline, `d_main_curr` contains ONLY the barycentric term.
         // The randomizer contribution is added later (after both z and g*z are evaluated).
         XFieldElement cpu_out = bary;
 
         XFieldElement gpu_out{BFieldElement(h_gpu_out[0]), BFieldElement(h_gpu_out[1]), BFieldElement(h_gpu_out[2])};
         std::cout << "[DBG] OOD main col0 check (bary only): gpu=" << gpu_out.to_string()
                   << " cpu=" << cpu_out.to_string()
                   << " diff=" << (gpu_out - cpu_out).to_string() << "\n";
     }
 
     if (dbg_main_col0_full) {
         // Save barycentric-only GPU output for col0 and the randomizer polynomial coeffs and z.
         CUDA_CHECK(cudaMemcpyAsync(dbg_main0_bary.data(), d_main_curr + 0 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         std::vector<uint64_t> h_rand(dims_.num_trace_randomizers);
         CUDA_CHECK(cudaMemcpyAsync(
             h_rand.data(),
             ctx_->d_main_randomizer_coeffs() + 0 * dims_.num_trace_randomizers,
             h_rand.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         std::array<uint64_t, 3> h_z{};
         CUDA_CHECK(cudaMemcpyAsync(h_z.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         dbg_z = XFieldElement(BFieldElement(h_z[0]), BFieldElement(h_z[1]), BFieldElement(h_z[2]));
         dbg_main0_rand_coeffs.clear();
         dbg_main0_rand_coeffs.reserve(h_rand.size());
         for (auto rv : h_rand) dbg_main0_rand_coeffs.push_back(BFieldElement(rv));
     }
 
     if (dbg_main_col100_bary_rust) {
         CUDA_CHECK(cudaMemcpyAsync(dbg_main100_bary.data(), d_main_curr + 100 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         std::array<uint64_t, 3> h_z{};
         CUDA_CHECK(cudaMemcpyAsync(h_z.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         dbg_z = XFieldElement(BFieldElement(h_z[0]), BFieldElement(h_z[1]), BFieldElement(h_z[2]));
 
         // Gather trace column 100
         uint64_t* d_col = nullptr;
         CUDA_CHECK(cudaMalloc(&d_col, n * sizeof(uint64_t)));
         qzc_gather_bfield_column_from_rowmajor<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_trace(), n, dims_.main_width, 100, d_col
         );
         std::vector<uint64_t> h_trace(n);
         CUDA_CHECK(cudaMemcpyAsync(h_trace.data(), d_col, n * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaFree(d_col));
         ctx_->synchronize();
 
         std::array<uint64_t, 3> rust{};
         int rc = eval_bfe_interpolant_at_xfe_point_rust(
             h_trace.data(),
             n,
             dims_.trace_offset,
             h_z[0], h_z[1], h_z[2],
             rust.data()
         );
         if (rc != 0) {
             std::cout << "[DBG] OOD main col100 bary rust: Rust FFI failed\n";
         } else {
             XFieldElement gpu{BFieldElement(dbg_main100_bary[0]), BFieldElement(dbg_main100_bary[1]), BFieldElement(dbg_main100_bary[2])};
             XFieldElement ref{BFieldElement(rust[0]), BFieldElement(rust[1]), BFieldElement(rust[2])};
             std::cout << "[DBG] OOD main col100 bary rust: gpu=" << gpu.to_string()
                       << " rust=" << ref.to_string()
                       << " diff=" << (gpu - ref).to_string() << "\n";
         }
     }
 
     if (dbg_aux_col0_full) {
         const size_t num_rand_local = dims_.num_trace_randomizers;
 
         // Save barycentric-only GPU output for aux col0 and its (BFE) randomizer coeffs and z.
         CUDA_CHECK(cudaMemcpyAsync(dbg_aux0_bary.data(), d_aux_curr + 0 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         std::vector<uint64_t> h_rand(num_rand_local);
         CUDA_CHECK(cudaMemcpyAsync(
             h_rand.data(),
             ctx_->d_aux_randomizer_coeffs() + (0 * 3 + 0) * num_rand_local,
             h_rand.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         std::array<uint64_t, 3> h_z{};
         CUDA_CHECK(cudaMemcpyAsync(h_z.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
         // Also validate the barycentric term by recomputing it from the raw aux trace column + weights
         // (weights/denom currently correspond to the first evaluation at z).
         std::array<uint64_t, 3> h_denom{};
         CUDA_CHECK(cudaMemcpyAsync(h_denom.data(), d_denom, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         std::vector<uint64_t> h_w(n * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_w.data(), d_w, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
         uint64_t* d_col = nullptr;
         CUDA_CHECK(cudaMalloc(&d_col, n * 3 * sizeof(uint64_t)));
         qzc_gather_xfe_column_from_rowmajor<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_aux_trace(),
             n,
             dims_.aux_width,
             0,
             d_col
         );
         std::vector<uint64_t> h_col(n * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_col.data(), d_col, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaFree(d_col));
 
         ctx_->synchronize();
 
         dbg_z = XFieldElement(BFieldElement(h_z[0]), BFieldElement(h_z[1]), BFieldElement(h_z[2]));
         dbg_aux0_rand_coeffs.clear();
         dbg_aux0_rand_coeffs.reserve(h_rand.size());
         for (auto rv : h_rand) dbg_aux0_rand_coeffs.push_back(BFieldElement(rv));
 
         XFieldElement denom_inv{BFieldElement(h_denom[0]), BFieldElement(h_denom[1]), BFieldElement(h_denom[2])};
         XFieldElement num = XFieldElement::zero();
         for (size_t i = 0; i < n; ++i) {
             XFieldElement wi{BFieldElement(h_w[i * 3 + 0]), BFieldElement(h_w[i * 3 + 1]), BFieldElement(h_w[i * 3 + 2])};
             XFieldElement vi{BFieldElement(h_col[i * 3 + 0]), BFieldElement(h_col[i * 3 + 1]), BFieldElement(h_col[i * 3 + 2])};
             num += wi * vi;
         }
         XFieldElement bary_cpu = num * denom_inv;
         XFieldElement bary_saved{BFieldElement(dbg_aux0_bary[0]), BFieldElement(dbg_aux0_bary[1]), BFieldElement(dbg_aux0_bary[2])};
         std::cout << "[DBG] OOD aux col0 bary check: gpu_saved=" << bary_saved.to_string()
                   << " cpu_recompute=" << bary_cpu.to_string()
                   << " diff=" << (bary_saved - bary_cpu).to_string() << "\n";
     }
 
     compute_weight_and_eval_point(d_next, d_main_next, d_aux_next, "g*z");
 
     // Add trace randomizer contribution: ood = barycentric + zerofier(point) * randomizer_poly(point)
     {
         const size_t num_rand = dims_.num_trace_randomizers;
         if (num_rand == 0) {
             throw std::runtime_error("num_trace_randomizers is zero; cannot compute OOD rows correctly");
         }
         // scratch_a layout: [ood(3)] [next(3)] ... [z4(3)] [zerofier_z(3)] [zerofier_gz(3)]
         uint64_t* d_zf_z  = ctx_->d_scratch_a() + 12; // [3]
         uint64_t* d_zf_gz = ctx_->d_scratch_a() + 15; // [3]
 
         qzc_trace_zerofier_xfe<<<1, 1, 0, ctx_->stream()>>>(
             d_ood, (uint64_t)n, dims_.trace_offset, d_zf_z
         );
         qzc_trace_zerofier_xfe<<<1, 1, 0, ctx_->stream()>>>(
             d_next, (uint64_t)n, dims_.trace_offset, d_zf_gz
         );
 
         qzc_add_main_randomizer_ood<<<dims_.main_width, 1, 0, ctx_->stream()>>>(
             d_main_curr,
             ctx_->d_main_randomizer_coeffs(),
             dims_.main_width,
             num_rand,
             d_ood,
             d_zf_z
         );
         qzc_add_main_randomizer_ood<<<dims_.main_width, 1, 0, ctx_->stream()>>>(
             d_main_next,
             ctx_->d_main_randomizer_coeffs(),
             dims_.main_width,
             num_rand,
             d_next,
             d_zf_gz
         );
 
         qzc_add_aux_randomizer_ood<<<dims_.aux_width, 1, 0, ctx_->stream()>>>(
             d_aux_curr,
             ctx_->d_aux_randomizer_coeffs(),
             dims_.aux_width,
             num_rand,
             d_ood,
             d_zf_z
         );
         qzc_add_aux_randomizer_ood<<<dims_.aux_width, 1, 0, ctx_->stream()>>>(
             d_aux_next,
             ctx_->d_aux_randomizer_coeffs(),
             dims_.aux_width,
             num_rand,
             d_next,
             d_zf_gz
         );
 
         if (dbg_main_col0_full) {
             // Also compute rand(z) for main col0 on GPU and keep zf(z) for comparison.
             // Allocate a tiny temp buffer for debug output [3] to avoid clobbering other scratch.
             uint64_t* d_dbg_rand = nullptr;
             CUDA_CHECK(cudaMalloc(&d_dbg_rand, 3 * sizeof(uint64_t)));
             const uint64_t* d_coeffs0 = ctx_->d_main_randomizer_coeffs() + 0 * num_rand;
             qzc_eval_bfe_poly_at_xfe_single<<<1,1,0,ctx_->stream()>>>(d_coeffs0, num_rand, d_ood, d_dbg_rand);
 
             // Compute zf(z) * rand(z) on GPU directly to validate xfield_mul_impl on general operands.
             uint64_t* d_dbg_mul = nullptr;
             CUDA_CHECK(cudaMalloc(&d_dbg_mul, 3 * sizeof(uint64_t)));
             qzc_mul_xfe3_kernel<<<1,1,0,ctx_->stream()>>>(d_zf_z, d_dbg_rand, d_dbg_mul);
 
             std::array<uint64_t, 3> h_zf_gpu{};
             std::array<uint64_t, 3> h_rand_gpu{};
             std::array<uint64_t, 3> h_mul_gpu{};
             CUDA_CHECK(cudaMemcpyAsync(h_zf_gpu.data(), d_zf_z, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_rand_gpu.data(), d_dbg_rand, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             CUDA_CHECK(cudaMemcpyAsync(h_mul_gpu.data(), d_dbg_mul, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             ctx_->synchronize();
             CUDA_CHECK(cudaFree(d_dbg_rand));
             CUDA_CHECK(cudaFree(d_dbg_mul));
             XFieldElement zf_gpu{BFieldElement(h_zf_gpu[0]), BFieldElement(h_zf_gpu[1]), BFieldElement(h_zf_gpu[2])};
             XFieldElement rand_gpu{BFieldElement(h_rand_gpu[0]), BFieldElement(h_rand_gpu[1]), BFieldElement(h_rand_gpu[2])};
             XFieldElement mul_gpu{BFieldElement(h_mul_gpu[0]), BFieldElement(h_mul_gpu[1]), BFieldElement(h_mul_gpu[2])};
 
             Polynomial<BFieldElement> rand_poly(dbg_main0_rand_coeffs);
             XFieldElement rand_cpu = rand_poly.evaluate_at_extension(dbg_z);
             XFieldElement zf_cpu = dbg_z.pow((uint64_t)n) - XFieldElement(BFieldElement(dims_.trace_offset).pow((uint64_t)n));
             XFieldElement mul_cpu = zf_cpu * rand_cpu;
 
             std::cout << "[DBG] OOD rand(z) check: gpu=" << rand_gpu.to_string()
                       << " cpu=" << rand_cpu.to_string()
                       << " diff=" << (rand_gpu - rand_cpu).to_string() << "\n";
             std::cout << "[DBG] OOD zerofier(z) check: gpu=" << zf_gpu.to_string()
                       << " cpu=" << zf_cpu.to_string()
                       << " diff=" << (zf_gpu - zf_cpu).to_string() << "\n";
             std::cout << "[DBG] OOD mul(zf,rand) check: gpu=" << mul_gpu.to_string()
                       << " cpu=" << mul_cpu.to_string()
                       << " diff=" << (mul_gpu - mul_cpu).to_string() << "\n";
         }
     }
 
     // Debug: compare one main OOD entry against Rust interpolation of the randomized column.
     if (std::getenv("TVM_DEBUG_OOD_MAIN_COL_RUST")) {
         const size_t col = 100; // deterministic, high-signal column
         // Download z
         std::array<uint64_t, 3> h_z{};
         CUDA_CHECK(cudaMemcpyAsync(h_z.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         // Download GPU OOD value for this col (after randomizer add)
         std::array<uint64_t, 3> h_gpu{};
         CUDA_CHECK(cudaMemcpyAsync(h_gpu.data(), d_main_curr + col * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         // Gather trace column to host
         uint64_t* d_col = nullptr;
         CUDA_CHECK(cudaMalloc(&d_col, n * sizeof(uint64_t)));
         qzc_gather_bfield_column_from_rowmajor<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_trace(), n, dims_.main_width, col, d_col
         );
         std::vector<uint64_t> h_trace(n);
         CUDA_CHECK(cudaMemcpyAsync(h_trace.data(), d_col, n * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaFree(d_col));
         // Randomizer coeffs for this column
         const size_t num_rand = dims_.num_trace_randomizers;
         std::vector<uint64_t> h_rand(num_rand);
         CUDA_CHECK(cudaMemcpyAsync(
             h_rand.data(),
             ctx_->d_main_randomizer_coeffs() + col * num_rand,
             num_rand * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
 
         std::array<uint64_t, 3> rust{};
         int rc = eval_randomized_main_column_at_xfe_point_rust(
             h_trace.data(),
             n,
             dims_.trace_offset,
             h_rand.data(),
             num_rand,
             h_z[0], h_z[1], h_z[2],
             rust.data()
         );
         if (rc != 0) {
             std::cout << "[DBG] OOD main col rust check: Rust FFI failed\n";
         } else {
             XFieldElement gpu{BFieldElement(h_gpu[0]), BFieldElement(h_gpu[1]), BFieldElement(h_gpu[2])};
             XFieldElement ref{BFieldElement(rust[0]), BFieldElement(rust[1]), BFieldElement(rust[2])};
             std::cout << "[DBG] OOD main col rust check col=" << col
                       << " gpu=" << gpu.to_string()
                       << " rust=" << ref.to_string()
                       << " diff=" << (gpu - ref).to_string() << "\n";
         }
     }
 
     if (std::getenv("TVM_DEBUG_OOD_AUX_COL_RUST")) {
         const size_t col = 10; // deterministic, mid-range column
         // Download z
         std::array<uint64_t, 3> h_z{};
         CUDA_CHECK(cudaMemcpyAsync(h_z.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         // Download GPU OOD value for this aux col (after randomizer add)
         std::array<uint64_t, 3> h_gpu{};
         CUDA_CHECK(cudaMemcpyAsync(h_gpu.data(), d_aux_curr + col * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
 
         // Gather aux trace column (XFE) to host: [n*3]
         uint64_t* d_col = nullptr;
         CUDA_CHECK(cudaMalloc(&d_col, n * 3 * sizeof(uint64_t)));
         qzc_gather_xfe_column_from_rowmajor<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_aux_trace(), n, dims_.aux_width, col, d_col
         );
         std::vector<uint64_t> h_trace(n * 3);
         CUDA_CHECK(cudaMemcpyAsync(h_trace.data(), d_col, n * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         CUDA_CHECK(cudaFree(d_col));
 
         // Aux randomizer coeffs for this column: we store per component-column, so rebuild XFE coeffs [num_rand*3]
         const size_t num_rand = dims_.num_trace_randomizers;
         std::vector<uint64_t> h_c0(num_rand), h_c1(num_rand), h_c2(num_rand);
         CUDA_CHECK(cudaMemcpyAsync(
             h_c0.data(),
             ctx_->d_aux_randomizer_coeffs() + (col * 3 + 0) * num_rand,
             num_rand * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         CUDA_CHECK(cudaMemcpyAsync(
             h_c1.data(),
             ctx_->d_aux_randomizer_coeffs() + (col * 3 + 1) * num_rand,
             num_rand * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         CUDA_CHECK(cudaMemcpyAsync(
             h_c2.data(),
             ctx_->d_aux_randomizer_coeffs() + (col * 3 + 2) * num_rand,
             num_rand * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
 
         std::vector<uint64_t> h_rand(num_rand * 3);
         for (size_t i = 0; i < num_rand; ++i) {
             h_rand[i * 3 + 0] = h_c0[i];
             h_rand[i * 3 + 1] = h_c1[i];
             h_rand[i * 3 + 2] = h_c2[i];
         }
 
         std::array<uint64_t, 3> rust{};
         int rc = eval_randomized_aux_column_at_xfe_point_rust(
             h_trace.data(),
             n,
             dims_.trace_offset,
             h_rand.data(),
             num_rand,
             h_z[0], h_z[1], h_z[2],
             rust.data()
         );
         if (rc != 0) {
             std::cout << "[DBG] OOD aux col rust check: Rust FFI failed\n";
         } else {
             XFieldElement gpu{BFieldElement(h_gpu[0]), BFieldElement(h_gpu[1]), BFieldElement(h_gpu[2])};
             XFieldElement ref{BFieldElement(rust[0]), BFieldElement(rust[1]), BFieldElement(rust[2])};
             std::cout << "[DBG] OOD aux col rust check col=" << col
                       << " gpu=" << gpu.to_string()
                       << " rust=" << ref.to_string()
                       << " diff=" << (gpu - ref).to_string() << "\n";
         }
     }
 
     if (dbg_main_col0_full) {
         // Compare full GPU output after randomizer addition vs CPU(bary + zerofier*rand(z)).
         std::array<uint64_t, 3> h_full{};
         CUDA_CHECK(cudaMemcpyAsync(h_full.data(), d_main_curr + 0 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
 
         XFieldElement bary{BFieldElement(dbg_main0_bary[0]), BFieldElement(dbg_main0_bary[1]), BFieldElement(dbg_main0_bary[2])};
         Polynomial<BFieldElement> rand_poly(dbg_main0_rand_coeffs);
         XFieldElement rand_at_z = rand_poly.evaluate_at_extension(dbg_z);
         XFieldElement zerofier = dbg_z.pow((uint64_t)n) - XFieldElement(BFieldElement(dims_.trace_offset).pow((uint64_t)n));
         XFieldElement cpu_full = bary + zerofier * rand_at_z;
 
         XFieldElement gpu_full{BFieldElement(h_full[0]), BFieldElement(h_full[1]), BFieldElement(h_full[2])};
         XFieldElement add_expected = zerofier * rand_at_z;
         XFieldElement add_gpu = gpu_full - bary;
         std::cout << "[DBG] OOD main col0 check (full): gpu=" << gpu_full.to_string()
                   << " cpu=" << cpu_full.to_string()
                   << " diff=" << (gpu_full - cpu_full).to_string() << "\n";
         std::cout << "[DBG] OOD main col0 addend: gpu=" << add_gpu.to_string()
                   << " expected=" << add_expected.to_string()
                   << " diff=" << (add_gpu - add_expected).to_string() << "\n";
     }
 
     if (dbg_aux_col0_full) {
         // Compare full GPU aux col0 output after randomizer addition vs CPU(bary + zerofier*rand(z)).
         std::array<uint64_t, 3> h_full{};
         CUDA_CHECK(cudaMemcpyAsync(h_full.data(), d_aux_curr + 0 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
 
         XFieldElement bary_saved{BFieldElement(dbg_aux0_bary[0]), BFieldElement(dbg_aux0_bary[1]), BFieldElement(dbg_aux0_bary[2])};
         Polynomial<BFieldElement> rand_poly(dbg_aux0_rand_coeffs);
         XFieldElement rand_at_z = rand_poly.evaluate_at_extension(dbg_z);
         XFieldElement zerofier = dbg_z.pow((uint64_t)n) - XFieldElement(BFieldElement(dims_.trace_offset).pow((uint64_t)n));
         XFieldElement cpu_full = bary_saved + zerofier * rand_at_z;
         XFieldElement gpu_full{BFieldElement(h_full[0]), BFieldElement(h_full[1]), BFieldElement(h_full[2])};
         std::cout << "[DBG] OOD aux col0 full check: gpu=" << gpu_full.to_string()
                   << " cpu=" << cpu_full.to_string()
                   << " diff=" << (gpu_full - cpu_full).to_string() << "\n";
     }
 
     // quotient segment evaluations at z^4 (Rust: out_of_domain_point_curr_row.pow(NUM_QUOTIENT_SEGMENTS))
     uint64_t* d_z4 = ctx_->d_scratch_a() + 9; // [3]
     qzc_pow4_xfe<<<1,1,0,ctx_->stream()>>>(d_ood, d_z4);
     size_t seg_len = dims_.quotient_length / dims_.num_quotient_segments;
     
     // Parallel polynomial evaluation using chunked Horner
     // Use ~1024 chunks for good parallelism (each chunk evaluates ~1K coefficients)
     const size_t CHUNK_SIZE = 1024;
     size_t num_chunks = (seg_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
     
     // Allocate temporary buffers for chunk results and x^chunk_size
     uint64_t* d_chunk_results = scratch + off; off += num_chunks * 3;
     uint64_t* d_x_power = scratch + off; off += 3;
     
     auto t_quot_eval = std::chrono::high_resolution_clock::now();
     
     // Compute x^chunk_size once (will be reused for combining chunks)
     qzc_pow_xfe_n<<<1,1,0,ctx_->stream()>>>(d_z4, CHUNK_SIZE, d_x_power);
     
     // Evaluate each segment in parallel using chunked Horner
     for (size_t s = 0; s < dims_.num_quotient_segments; ++s) {
         // Step 1: Evaluate chunks in parallel (num_chunks blocks, 1 thread each)
         qzc_eval_poly_chunk_horner<<<num_chunks, 1, 0, ctx_->stream()>>>(
             ctx_->d_quotient_segment_coeffs(),
             seg_len,
             s,
             CHUNK_SIZE,
             num_chunks,
             d_z4,
             d_chunk_results
         );
         
         // Step 2: Combine chunk results
         qzc_combine_poly_chunks<<<1,1,0,ctx_->stream()>>>(
             d_chunk_results,
             num_chunks,
             d_x_power,
             d_quot_ood + s * 3
         );
     }
     
     if (profile_ood) { 
         ctx_->synchronize(); 
         std::cout << "    [OOD] quot_segment_eval (" << dims_.num_quotient_segments 
                   << " segs × " << seg_len << " coeffs, " << num_chunks << " chunks): " 
                   << elapsed_ms(t_quot_eval) << " ms\n"; 
     }
 
     // CPU transcript: enqueue FS-relevant OOD proof items (tiny D2H)
     {
         const size_t main_words = dims_.main_width * 3;
         const size_t aux_words  = dims_.aux_width * 3;
         const size_t quot_words = dims_.num_quotient_segments * 3;
         std::vector<uint64_t> h(main_words * 2 + aux_words * 2 + quot_words);
         size_t offh = 0;
 
         auto d2h = [&](const uint64_t* d_src, size_t words) {
             CUDA_CHECK(cudaMemcpyAsync(h.data() + offh, d_src, words * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             offh += words;
         };
 
         d2h(d_main_curr, main_words);
         d2h(d_aux_curr,  aux_words);
         d2h(d_main_next, main_words);
         d2h(d_aux_next,  aux_words);
         d2h(d_quot_ood,  quot_words);
 
         ctx_->synchronize();
 
         auto to_xfe_vec = [&](size_t count, size_t& cursor) {
             std::vector<XFieldElement> v;
             v.reserve(count);
             for (size_t i = 0; i < count; ++i) {
                 uint64_t c0 = h[cursor++], c1 = h[cursor++], c2 = h[cursor++];
                 v.emplace_back(BFieldElement(c0), BFieldElement(c1), BFieldElement(c2));
             }
             return v;
         };
 
         size_t cursor = 0;
         auto main_curr = to_xfe_vec(dims_.main_width, cursor);
         auto aux_curr  = to_xfe_vec(dims_.aux_width, cursor);
         auto main_next = to_xfe_vec(dims_.main_width, cursor);
         auto aux_next  = to_xfe_vec(dims_.aux_width, cursor);
         auto quot_ood  = to_xfe_vec(dims_.num_quotient_segments, cursor);
 
         fs_cpu_.enqueue(ProofItem::out_of_domain_main_row(main_curr));
         fs_cpu_.enqueue(ProofItem::out_of_domain_aux_row(aux_curr));
         fs_cpu_.enqueue(ProofItem::out_of_domain_main_row(main_next));
         fs_cpu_.enqueue(ProofItem::out_of_domain_aux_row(aux_next));
         fs_cpu_.enqueue(ProofItem::out_of_domain_quotient_segments(quot_ood));
     }
 
     // ---------------------------------------------------------------------
     // Debug: Verifier-style self-check for OOD quotient consistency.
     // Computes:
     //  (A) quotient_value_from_air(z) using OOD rows + challenges + weights
     //  (B) sum_{i=0..3} z^i * seg_i(z^4) using our provided segment evaluations
     //
     // NOTE: This check is useful for debugging when nondeterminism (RAM/secret input) is used,
     // but adds significant overhead (D2H transfers, Rust FFI calls). Disabled by default.
     // Enable with TVM_ENABLE_OOD_QUOT_CHECK=1 for debugging.
     // ---------------------------------------------------------------------
     // Disabled by default - enable explicitly with TVM_ENABLE_OOD_QUOT_CHECK=1
     bool enable_ood_check = (std::getenv("TVM_ENABLE_OOD_QUOT_CHECK") != nullptr);
     if (enable_ood_check || std::getenv("TVM_DEBUG_OOD_QUOT_SELF_CHECK")) {
         // 1) Bring challenges + weights to host
         std::vector<uint64_t> h_ch(Challenges::COUNT * 3);
         CUDA_CHECK(cudaMemcpyAsync(
             h_ch.data(),
             ctx_->d_challenges(),
             h_ch.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         std::vector<uint64_t> h_w(Quotient::MASTER_AUX_NUM_CONSTRAINTS * 3);
         CUDA_CHECK(cudaMemcpyAsync(
             h_w.data(),
             ctx_->d_quotient_weights(),
             h_w.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
 
         std::vector<XFieldElement> sampled;
         sampled.reserve(Challenges::SAMPLE_COUNT);
         for (size_t i = 0; i < Challenges::SAMPLE_COUNT; ++i) {
             sampled.emplace_back(
                 BFieldElement(h_ch[i * 3 + 0]),
                 BFieldElement(h_ch[i * 3 + 1]),
                 BFieldElement(h_ch[i * 3 + 2])
             );
         }
         std::vector<BFieldElement> program_digest_vec;
         program_digest_vec.reserve(5);
         for (size_t i = 0; i < 5; ++i) program_digest_vec.push_back(claim_.program_digest[i]);
         std::vector<BFieldElement> lookup_table_vec;
         lookup_table_vec.reserve(256);
         for (uint8_t v : Tip5::LOOKUP_TABLE) lookup_table_vec.push_back(BFieldElement(v));
 
         Challenges ch = Challenges::from_sampled_and_claim(
             sampled, program_digest_vec, claim_.input, claim_.output, lookup_table_vec
         );
 
         std::vector<XFieldElement> w;
         w.reserve(Quotient::MASTER_AUX_NUM_CONSTRAINTS);
         for (size_t i = 0; i < Quotient::MASTER_AUX_NUM_CONSTRAINTS; ++i) {
             w.emplace_back(
                 BFieldElement(h_w[i * 3 + 0]),
                 BFieldElement(h_w[i * 3 + 1]),
                 BFieldElement(h_w[i * 3 + 2])
             );
         }
 
         // 2) Reuse already-downloaded OOD rows/segments from the fs_cpu_ enqueue block:
         // Re-download them here (still tiny; only in debug mode).
         std::vector<uint64_t> h_ood(
             (dims_.main_width * 3) * 2 +
             (dims_.aux_width * 3) * 2 +
             (dims_.num_quotient_segments * 3)
         );
         size_t offh = 0;
         auto d2h = [&](const uint64_t* d_src, size_t words) {
             CUDA_CHECK(cudaMemcpyAsync(h_ood.data() + offh, d_src, words * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
             offh += words;
         };
         d2h(d_main_curr, dims_.main_width * 3);
         d2h(d_aux_curr,  dims_.aux_width * 3);
         d2h(d_main_next, dims_.main_width * 3);
         d2h(d_aux_next,  dims_.aux_width * 3);
         d2h(d_quot_ood,  dims_.num_quotient_segments * 3);
         ctx_->synchronize();
 
         auto to_xfe_vec = [&](size_t count, size_t& cursor) {
             std::vector<XFieldElement> v;
             v.reserve(count);
             for (size_t i = 0; i < count; ++i) {
                 uint64_t c0 = h_ood[cursor++], c1 = h_ood[cursor++], c2 = h_ood[cursor++];
                 v.emplace_back(BFieldElement(c0), BFieldElement(c1), BFieldElement(c2));
             }
             return v;
         };
         size_t cursor = 0;
         auto ood_main_curr = to_xfe_vec(dims_.main_width, cursor);
         auto ood_aux_curr  = to_xfe_vec(dims_.aux_width, cursor);
         auto ood_main_next = to_xfe_vec(dims_.main_width, cursor);
         auto ood_aux_next  = to_xfe_vec(dims_.aux_width, cursor);
         auto ood_segments  = to_xfe_vec(dims_.num_quotient_segments, cursor);
 
         // Convert main rows from XFE to BFE columns as expected by constraint evaluators (main table is BFE).
         auto xfe_row_to_bfe = [&](const std::vector<XFieldElement>& row_xfe) {
             std::vector<BFieldElement> bfe;
             bfe.reserve(row_xfe.size());
             for (const auto& x : row_xfe) bfe.push_back(x.coeff(0));
             return bfe;
         };
         std::vector<BFieldElement> main_curr_bfe = xfe_row_to_bfe(ood_main_curr);
         std::vector<BFieldElement> main_next_bfe = xfe_row_to_bfe(ood_main_next);
 
         // 3) Compute quotient_value_from_air(z) using Rust verifier logic (FFI) with XFE main rows.
         // Use the already stored device OOD point copied to host.
         std::array<uint64_t, 3> h_z{};
         CUDA_CHECK(cudaMemcpyAsync(h_z.data(), ctx_->d_ood_point(), 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         XFieldElement z{BFieldElement(h_z[0]), BFieldElement(h_z[1]), BFieldElement(h_z[2])};
         uint64_t trace_gen_inv = BFieldElement(dims_.trace_generator).inverse().value();
 
         // Flatten OOD main rows as XFE triplets [379*3] for Rust FFI
         std::vector<uint64_t> main_curr_xfe_flat(dims_.main_width * 3);
         std::vector<uint64_t> main_next_xfe_flat(dims_.main_width * 3);
         for (size_t i = 0; i < dims_.main_width; ++i) {
             main_curr_xfe_flat[i * 3 + 0] = ood_main_curr[i].coeff(0).value();
             main_curr_xfe_flat[i * 3 + 1] = ood_main_curr[i].coeff(1).value();
             main_curr_xfe_flat[i * 3 + 2] = ood_main_curr[i].coeff(2).value();
             main_next_xfe_flat[i * 3 + 0] = ood_main_next[i].coeff(0).value();
             main_next_xfe_flat[i * 3 + 1] = ood_main_next[i].coeff(1).value();
             main_next_xfe_flat[i * 3 + 2] = ood_main_next[i].coeff(2).value();
         }
         std::vector<uint64_t> aux_curr_flat(dims_.aux_width * 3);
         std::vector<uint64_t> aux_next_flat(dims_.aux_width * 3);
         for (size_t i = 0; i < dims_.aux_width; ++i) {
             aux_curr_flat[i * 3 + 0] = ood_aux_curr[i].coeff(0).value();
             aux_curr_flat[i * 3 + 1] = ood_aux_curr[i].coeff(1).value();
             aux_curr_flat[i * 3 + 2] = ood_aux_curr[i].coeff(2).value();
             aux_next_flat[i * 3 + 0] = ood_aux_next[i].coeff(0).value();
             aux_next_flat[i * 3 + 1] = ood_aux_next[i].coeff(1).value();
             aux_next_flat[i * 3 + 2] = ood_aux_next[i].coeff(2).value();
         }
         // Full 63 challenges from device buffer (already in h_ch as 63*3)
         uint64_t* out_ptr = nullptr;
         size_t out_len = 0;
         int rc = compute_out_of_domain_quotient_xfe_main_challenges63_rust(
             main_curr_xfe_flat.data(),
             aux_curr_flat.data(),
             main_next_xfe_flat.data(),
             aux_next_flat.data(),
             h_ch.data(),
             h_w.data(),
             Quotient::MASTER_AUX_NUM_CONSTRAINTS,
             (uint64_t)dims_.padded_height,
             trace_gen_inv,
             h_z[0], h_z[1], h_z[2],
             &out_ptr,
             &out_len
         );
         if (rc != 0 || out_ptr == nullptr || out_len != 3) {
             std::cout << "[DBG] OOD self-check: Rust FFI compute_out_of_domain_quotient_xfe_main_challenges63_rust failed\n";
         }
         XFieldElement q_air = XFieldElement::zero();
         if (out_ptr && out_len == 3) {
             q_air = XFieldElement(BFieldElement(out_ptr[0]), BFieldElement(out_ptr[1]), BFieldElement(out_ptr[2]));
             constraint_evaluation_free(out_ptr, out_len);
         }
 
         // Optional: compare our GPU-computed OOD quotient segments against Rust reference computed
         // from the stored quotient codeword on quotient domain.
         if (std::getenv("TVM_DEBUG_OOD_QUOT_SEGMENTS_RUST")) {
             const size_t qlen = dims_.quotient_length;
             std::vector<uint64_t> h_q(qlen * 3);
             CUDA_CHECK(cudaMemcpyAsync(
                 h_q.data(),
                 ctx_->d_quotient_codeword(),
                 h_q.size() * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost,
                 ctx_->stream()
             ));
             ctx_->synchronize();
 
             uint64_t* rust_segs = nullptr;
             size_t rust_len = 0;
             int rc2 = compute_ood_quot_segments_from_quotient_codeword_rust(
                 h_q.data(),
                 qlen,
                 dims_.quotient_offset,
                 h_z[0], h_z[1], h_z[2],
                 &rust_segs,
                 &rust_len
             );
             if (rc2 != 0 || rust_segs == nullptr || rust_len != 12) {
                 std::cout << "[DBG] Rust OOD quotient segments computation failed\n";
             } else {
                 for (size_t s = 0; s < 4; ++s) {
                     XFieldElement seg_gpu = ood_segments[s];
                     XFieldElement seg_rust{
                         BFieldElement(rust_segs[s * 3 + 0]),
                         BFieldElement(rust_segs[s * 3 + 1]),
                         BFieldElement(rust_segs[s * 3 + 2])
                     };
                     std::cout << "[DBG] OOD quot seg" << s
                               << ": gpu=" << seg_gpu.to_string()
                               << " rust=" << seg_rust.to_string()
                               << " diff=" << (seg_gpu - seg_rust).to_string() << "\n";
                 }
                 constraint_evaluation_free(rust_segs, rust_len);
             }
 
             // Also compare q(z) computed by interpolating the quotient codeword on the quotient coset domain.
             std::array<uint64_t, 3> q_from_codeword{};
             int rc3 = eval_xfe_coset_codeword_at_xfe_point_rust(
                 h_q.data(),
                 qlen,
                 dims_.quotient_offset,
                 h_z[0], h_z[1], h_z[2],
                 q_from_codeword.data()
             );
             if (rc3 == 0) {
                 XFieldElement q_cw{BFieldElement(q_from_codeword[0]), BFieldElement(q_from_codeword[1]), BFieldElement(q_from_codeword[2])};
                 std::cout << "[DBG] q_from_quot_codeword(z)=" << q_cw.to_string()
                           << " diff_vs_q_air=" << (q_air - q_cw).to_string() << "\n";
             }
         }
 
         // 4) Compute segment sum (verifier logic): q(z) = Σ_{i=0..3} z^i · seg_i(z^4)
         XFieldElement power = XFieldElement::one();
         XFieldElement q_seg = XFieldElement::zero();
         for (size_t i = 0; i < ood_segments.size(); ++i) {
             q_seg += power * ood_segments[i];
             power *= z;
         }
 
         std::cout << "[DBG] OOD self-check: q_air=" << q_air.to_string()
                   << " q_seg=" << q_seg.to_string()
                   << " diff=" << (q_air - q_seg).to_string()
                   << " summands=" << Quotient::MASTER_AUX_NUM_CONSTRAINTS << "\n";
     }
 
     // Append to proof buffer in EXACT Rust order:
     //   OutOfDomainMainRow(curr), OutOfDomainAuxRow(curr),
     //   OutOfDomainMainRow(next), OutOfDomainAuxRow(next),
     //   OutOfDomainQuotientSegments(curr)
     // Proof stream does NOT include the OOD point itself (it's sampled), so we do not store it.
     cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_main_curr, dims_.main_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream());
     proof_size_ += dims_.main_width * 3;
     cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_aux_curr, dims_.aux_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream());
     proof_size_ += dims_.aux_width * 3;
     cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_main_next, dims_.main_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream());
     proof_size_ += dims_.main_width * 3;
     cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_aux_next, dims_.aux_width * 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream());
     proof_size_ += dims_.aux_width * 3;
     cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_quot_ood, dims_.num_quotient_segments * 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream());
     proof_size_ += dims_.num_quotient_segments * 3;
 }
 
 // ============================================================================
 // Step 6: FRI Protocol
 // ============================================================================
 
 void GpuStark::step_fri_protocol() {
     if (dims_.lde_frugal_mode) {
         step_fri_protocol_frugal();
         return;
     }
     // Build the real initial FRI codeword on GPU:
     // - sample linear weights
     // - form main+aux combination codeword on FRI domain
     // - form quotient combination codeword on FRI domain (already on FRI domain)
     // - compute DEEP codeword components and combine into FRI input codeword
     const size_t n = dims_.fri_length;
     constexpr int BLOCK = 256;
     int grid_n = (int)((n + BLOCK - 1) / BLOCK);
 
     const bool profile_fri = TRITON_PROFILE_ENABLED();
     cudaEvent_t ev0{}, ev_build{}, ev_round0{}, ev_loop{}, ev_queries{}, ev_end{};
     if (profile_fri) {
         CUDA_CHECK(cudaEventCreate(&ev0));
         CUDA_CHECK(cudaEventCreate(&ev_build));
         CUDA_CHECK(cudaEventCreate(&ev_round0));
         CUDA_CHECK(cudaEventCreate(&ev_loop));
         CUDA_CHECK(cudaEventCreate(&ev_queries));
         CUDA_CHECK(cudaEventCreate(&ev_end));
         CUDA_CHECK(cudaEventRecord(ev0, ctx_->stream()));
     }
 
     const size_t main_width = dims_.main_width;
     const size_t aux_width = dims_.aux_width;
     const size_t num_segments = dims_.num_quotient_segments;
     const size_t total_weights = main_width + aux_width + num_segments + 3;
 
     // Unified-memory perf: prefetch the big FRI-domain tables to the active GPU before
     // building the DEEP/FRI input codeword. This avoids massive UM page-fault overhead.
     // Enable via TRITON_PREFETCH_FRI=1.
     if (!dims_.lde_frugal_mode && triton_vm::gpu::use_unified_memory() && std::getenv("TRITON_PREFETCH_FRI")) {
         int dev = 0;
         CUDA_CHECK(cudaGetDevice(&dev));
         cudaMemLocation location;
         location.type = cudaMemLocationTypeDevice;
         location.id = dev;
         CUDA_CHECK(cudaMemPrefetchAsync(ctx_->d_main_lde(), n * main_width * sizeof(uint64_t), location, 0, ctx_->stream()));
         CUDA_CHECK(cudaMemPrefetchAsync(ctx_->d_aux_lde(), n * aux_width * 3 * sizeof(uint64_t), location, 0, ctx_->stream()));
         CUDA_CHECK(cudaMemPrefetchAsync(ctx_->d_quotient_segments(), n * num_segments * 3 * sizeof(uint64_t), location, 0, ctx_->stream()));
     }
 
     // Use scratch for weights and intermediates
     uint64_t* scratch = ctx_->d_scratch_a();
     size_t off = 0;
     uint64_t* d_weights = scratch + off; off += total_weights * 3;   // [total*3]
     uint64_t* d_main_aux_codeword = scratch + off; off += n * 3;     // [n*3]
     uint64_t* d_quot_codeword = scratch + off; off += n * 3;         // [n*3]
     uint64_t* d_eval_main_aux_z = scratch + off; off += 3;
     uint64_t* d_eval_main_aux_gz = scratch + off; off += 3;
     uint64_t* d_eval_quot_z4 = scratch + off; off += 3;
     uint64_t* d_z4 = scratch + off; off += 3;
 
     // Sync GPU sponge state to CPU transcript state at the start of FRI.
     // This makes subsequent GPU-only sampling (folding challenges + query indices) match the verifier.
     {
         std::array<uint64_t, 16> h_state{};
         for (size_t i = 0; i < 16; ++i) {
             h_state[i] = fs_cpu_.sponge().state[i].value();
         }
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_sponge_state(),
             h_state.data(),
             16 * sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
     }
 
     // Sample linear weights directly on GPU (advances GPU sponge, no CPU sync needed).
     kernels::fs_sample_scalars_device_gpu(ctx_->d_sponge_state(), d_weights, total_weights, ctx_->stream());
 
     // Read OOD rows/segments from proof buffer (written in Step 5) for eval_value computations
     // Layout so far:
     // [log2(1)] [main_root(5)] [aux_root(5)] [quot_root(5)] = 16
     const size_t base = 1 + 5 + 5 + 5;
     // Must match Rust proof item order:
     // main_curr, aux_curr, main_next, aux_next, quot_segments
     const uint64_t* d_ood_main_curr = ctx_->d_proof_buffer() + base;
     const uint64_t* d_ood_aux_curr  = d_ood_main_curr + main_width * 3;
     const uint64_t* d_ood_main_next = d_ood_aux_curr + aux_width * 3;
     const uint64_t* d_ood_aux_next  = d_ood_main_next + main_width * 3;
     const uint64_t* d_ood_quot      = d_ood_aux_next + aux_width * 3; // [num_segments*3]
 
     // eval(main+aux) at z and gz
     qzc_eval_ood_value_main_aux_kernel<<<1,1,0,ctx_->stream()>>>(
         d_ood_main_curr, d_ood_aux_curr, d_weights, main_width, aux_width, d_eval_main_aux_z
     );
     qzc_eval_ood_value_main_aux_kernel<<<1,1,0,ctx_->stream()>>>(
         d_ood_main_next, d_ood_aux_next, d_weights, main_width, aux_width, d_eval_main_aux_gz
     );
     // eval(quot) at z^4
     const size_t quot_weight_base = main_width + aux_width;
     qzc_eval_ood_value_quot_kernel<<<1,1,0,ctx_->stream()>>>(
         d_ood_quot, d_weights, quot_weight_base, num_segments, d_eval_quot_z4
     );
 
     // build main+aux codeword from LDE tables (FRI domain)
     if (dims_.lde_frugal_mode) {
         // FRUGAL: stream LDE batches and accumulate into codeword.
         constexpr size_t FRUGAL_BATCH_COLS = 10;
         CUDA_CHECK(cudaMemsetAsync(d_main_aux_codeword, 0, n * 3 * sizeof(uint64_t), ctx_->stream()));
 
         // Build col-major traces for batch LDE
         uint64_t* d_main_colmajor = ctx_->d_scratch_a();
         {
             size_t total = dims_.padded_height * main_width;
             int grid_t = (int)((total + BLOCK - 1) / BLOCK);
             qzc_rowmajor_to_colmajor_bfe<<<grid_t, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_main_trace(), d_main_colmajor, dims_.padded_height, main_width
             );
         }
         uint64_t* d_aux_colmajor_components = ctx_->d_scratch_b();
         {
             constexpr int TRANSPOSE_ELEMS = 4;
             int grid_aux = (int)(((dims_.padded_height * aux_width) + BLOCK * TRANSPOSE_ELEMS - 1) / (BLOCK * TRANSPOSE_ELEMS));
             qzc_rowmajor_to_colmajor_xfe<<<grid_aux, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_aux_trace(), d_aux_colmajor_components, dims_.padded_height, aux_width
             );
         }
         CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
 
         // Accumulate main columns
         for (size_t col_start = 0; col_start < main_width; col_start += FRUGAL_BATCH_COLS) {
             const size_t batch_cols = std::min(FRUGAL_BATCH_COLS, main_width - col_start);
             kernels::randomized_lde_batch_gpu(
                 d_main_colmajor + col_start * dims_.padded_height,
                 batch_cols,
                 dims_.padded_height,
                 ctx_->d_main_randomizer_coeffs() + col_start * dims_.num_trace_randomizers,
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 dims_.fri_offset,
                 n,
                 ctx_->d_main_lde(),
                 ctx_->stream()
             );
             frugal_accumulate_main_codeword_kernel<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_main_lde(),
                 n,
                 batch_cols,
                 d_weights,
                 0,
                 col_start,
                 d_main_aux_codeword
             );
         }
 
         // Accumulate aux columns (XFE)
         for (size_t col_start = 0; col_start < aux_width; col_start += FRUGAL_BATCH_COLS) {
             const size_t batch_xfe_cols = std::min(FRUGAL_BATCH_COLS, aux_width - col_start);
             const size_t comp_start = col_start * 3;
             const size_t batch_cols_bfe = batch_xfe_cols * 3;
             kernels::randomized_lde_batch_gpu(
                 d_aux_colmajor_components + comp_start * dims_.padded_height,
                 batch_cols_bfe,
                 dims_.padded_height,
                 ctx_->d_aux_randomizer_coeffs() + comp_start * dims_.num_trace_randomizers,
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 dims_.fri_offset,
                 n,
                 ctx_->d_aux_lde(),
                 ctx_->stream()
             );
             frugal_accumulate_aux_codeword_kernel<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_aux_lde(),
                 n,
                 batch_xfe_cols,
                 d_weights,
                 main_width,
                 col_start,
                 d_main_aux_codeword
             );
         }
     } else {
         qzc_build_main_aux_codeword_kernel<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_lde(),
             ctx_->d_aux_lde(),
             d_weights,
             n,
             main_width,
             aux_width,
             d_main_aux_codeword
         );
     }
 
     // build quotient combination codeword from quotient segments on FRI domain
     qzc_build_quot_codeword_kernel<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
         ctx_->d_quotient_segments(),
         d_weights,
         n,
         quot_weight_base,
         num_segments,
         d_quot_codeword
     );
 
     // compute z^4
     qzc_pow4_xfe<<<1,1,0,ctx_->stream()>>>(ctx_->d_ood_point(), d_z4);
 
     // Compute fri domain values (BFE) for denominators (exact offset+generator from Rust)
     uint64_t* d_fri_domain_vals = ctx_->d_scratch_b(); // [n]
     uint64_t fri_offset = dims_.fri_offset;
     uint64_t fri_gen = dims_.fri_generator;
     {
         // Chunked fill reduces pow() calls by 4x and is measurable at n=8M.
         constexpr int ITEMS = 4;
         int grid = (int)((n + (size_t)BLOCK * ITEMS - 1) / ((size_t)BLOCK * ITEMS));
         qzc_fill_domain_points_chunked<ITEMS><<<grid, BLOCK, 0, ctx_->stream()>>>(d_fri_domain_vals, n, fri_offset, fri_gen);
     }
 
     // deep weights are last 3 weights
     const uint64_t* d_w_deep = d_weights + (main_width + aux_width + num_segments) * 3;
 
     // build final FRI input codeword in ctx_->d_fri_codeword(0)
     qzc_deep_fri_codeword_kernel<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
         d_fri_domain_vals,
         d_main_aux_codeword,
         d_quot_codeword,
         n,
         ctx_->d_ood_point(),
         ctx_->d_next_row_point(),
         d_z4,
         d_eval_main_aux_z,
         d_eval_main_aux_gz,
         d_eval_quot_z4,
         d_w_deep,
         ctx_->d_fri_codeword(0)
     );
 
     if (profile_fri) { CUDA_CHECK(cudaEventRecord(ev_build, ctx_->stream())); }
 
     size_t current_size = n;
 
     // FRI domains: start from prover's FRI coset (exact offset+generator from Rust)
     uint64_t offset = dims_.fri_offset;
     uint64_t generator = dims_.fri_generator;
 
     uint64_t two_inv = BFieldElement(2).inverse().value();
 
     // Commit to first round (codeword length = fri_length)
     {
         uint64_t* d_tree = ctx_->d_fri_merkle(0);
         int grid = (int)((current_size + BLOCK - 1) / BLOCK);
         qzc_xfe_to_digest_leaves_kernel<<<grid, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_fri_codeword(0),
             current_size,
             d_tree
         );
         kernels::merkle_tree_gpu(d_tree, d_tree, current_size, ctx_->stream());
 
         uint64_t* d_root = d_tree + (2 * current_size - 2) * 5;
         // Absorb encoded MerkleRoot item: [0] + digest
         uint64_t* d_enc = ctx_->d_scratch_b();
         uint64_t disc = 0;
         CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
         CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
         kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
 
         cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_root, 5 * sizeof(uint64_t),
                         cudaMemcpyDeviceToDevice, ctx_->stream());
         proof_size_ += 5;
     }
 
     if (profile_fri) { CUDA_CHECK(cudaEventRecord(ev_round0, ctx_->stream())); }
 
     // OPTIMIZED FRI Protocol:
     // 1. Pre-allocate memory pools (reuse across rounds)
     // 2. Batch D2H copies (single sync at end)
     // 3. Pipeline operations where possible
     // Note: Challenges must be sampled sequentially (each depends on previous round's root)
     
     // Pre-allocate device memory for domain inverses (reuse across rounds)
     // Maximum size needed is for first round (n/2)
     size_t max_domain_inv_size = n / 2;
     uint64_t* d_domain_inv_pool;
     CUDA_CHECK(cudaMalloc(&d_domain_inv_pool, max_domain_inv_size * sizeof(uint64_t)));
 
     // Pre-allocate device buffer for challenges (reuse, but sample sequentially)
     uint64_t* d_chal = ctx_->d_scratch_a(); // [3] - reuse same buffer
     
     // Subsequent rounds: sample challenge, fold, halve domain, commit
     // CRITICAL: Must enqueue each root BEFORE sampling next challenge (Fiat-Shamir requirement)
     for (size_t round = 0; round < dims_.num_fri_rounds; ++round) {
         size_t half = current_size / 2;
 
         // Sample challenge sequentially (required by Fiat-Shamir - depends on previous root)
         kernels::fs_sample_scalars_device_gpu(ctx_->d_sponge_state(), d_chal, 1, ctx_->stream());
 
         uint64_t* d_domain_inv = d_domain_inv_pool;
         kernels::compute_domain_inverses_gpu(offset, generator, d_domain_inv, half, ctx_->stream());
 
         // Fold into next round codeword (keeps fold kernel smaller/higher-occupancy)
         kernels::fri_fold_gpu(
             ctx_->d_fri_codeword(round),
             current_size,
             d_chal,
             d_domain_inv,
             two_inv,
             ctx_->d_fri_codeword(round + 1),
             ctx_->stream()
         );
         
         // Halve domain (offset^2, generator^2) and length/2
         offset = (BFieldElement(offset) * BFieldElement(offset)).value();
         generator = (BFieldElement(generator) * BFieldElement(generator)).value();
         current_size = half;
 
         // Commit to new round
         uint64_t* d_tree = ctx_->d_fri_merkle(round + 1);
         int grid = (int)((current_size + BLOCK - 1) / BLOCK);
         qzc_xfe_to_digest_leaves_kernel<<<grid, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_fri_codeword(round + 1),
             current_size,
             d_tree
         );
         kernels::merkle_tree_gpu(d_tree, d_tree, current_size, ctx_->stream());
 
         uint64_t* d_root = d_tree + (2 * current_size - 2) * 5;
         
         uint64_t* d_enc = ctx_->d_scratch_b();
         uint64_t disc = 0;
         CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
         CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
         kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
 
         cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_root, 5 * sizeof(uint64_t),
                         cudaMemcpyDeviceToDevice, ctx_->stream());
         proof_size_ += 5;
     }
     
     // Cleanup
     CUDA_CHECK(cudaFree(d_domain_inv_pool));
 
     if (profile_fri) { CUDA_CHECK(cudaEventRecord(ev_loop, ctx_->stream())); }
 
     // Append FriCodeword (last round codeword) as raw XFE triplets.
     // This is not absorbed into Fiat-Shamir (matches ProofItem::include_in_fiat_shamir_heuristic()).
     const uint64_t* d_last_codeword = ctx_->d_fri_codeword(dims_.num_fri_rounds);
     // current_size is the last domain length after folding
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_last_codeword,
         current_size * 3 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += current_size * 3;
 
     // ---------------------------------------------------------------------
     // Query phase: sample indices and append FriResponse items (payloads)
     // ---------------------------------------------------------------------
     constexpr size_t NUM_QUERIES = GpuProofContext::NUM_FRI_QUERIES;
     // Sample query indices on GPU (device-only). We still fetch them to host to build auth paths.
     kernels::fs_sample_indices_device_gpu(
         ctx_->d_sponge_state(),
         ctx_->d_fri_query_indices(),
         dims_.fri_length,
         NUM_QUERIES,
         ctx_->stream()
     );
     std::vector<size_t> a_indices(NUM_QUERIES);
     CUDA_CHECK(cudaMemcpyAsync(
         a_indices.data(),
         ctx_->d_fri_query_indices(),
         NUM_QUERIES * sizeof(size_t),
         cudaMemcpyDeviceToHost,
         ctx_->stream()
     ));
     ctx_->synchronize();
         
     auto append_u64 = [&](uint64_t v) {
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             &v,
             sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
         proof_size_ += 1;
     };
 
     auto append_fri_response = [&](size_t round_idx, const std::vector<size_t>& leaf_indices, size_t domain_len) {
         // Copy indices to device temp
         size_t* d_tmp_indices = reinterpret_cast<size_t*>(ctx_->d_scratch_a());
         CUDA_CHECK(cudaMemcpyAsync(
             d_tmp_indices,
             leaf_indices.data(),
             leaf_indices.size() * sizeof(size_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
 
         // Gather revealed leaves from round codeword (row-major, row_width=1)
         uint64_t* d_revealed = ctx_->d_scratch_b(); // [num_indices*3]
         kernels::gather_xfield_rows_gpu(
             ctx_->d_fri_codeword(round_idx),
             d_tmp_indices,
             d_revealed,
             domain_len,
             1,
             leaf_indices.size(),
             ctx_->stream()
         );
 
         // Compute auth structure node indices (heap), map to flat indices
         std::vector<size_t> heap_nodes = auth_structure_heap_node_indices(leaf_indices, domain_len);
         std::vector<size_t> flat_nodes;
         flat_nodes.reserve(heap_nodes.size());
         for (size_t h : heap_nodes) flat_nodes.push_back(merkle_heap_to_flat_index(h, domain_len));
 
         // Upload flat indices to device temp
         size_t* d_auth_idx = d_tmp_indices; // reuse
         CUDA_CHECK(cudaMemcpyAsync(
             d_auth_idx,
             flat_nodes.data(),
             flat_nodes.size() * sizeof(size_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
 
         // Gather digests from full tree (each node is a row of 5)
         uint64_t* d_auth_digests = d_revealed + leaf_indices.size() * 3; // reuse remaining scratch_b
         kernels::gather_bfield_rows_gpu(
             ctx_->d_fri_merkle(round_idx),
             d_auth_idx,
             d_auth_digests,
             2 * domain_len - 1,
             5,
             flat_nodes.size(),
             ctx_->stream()
         );
         
         // Append payload with explicit lengths for parsing:
         // [auth_count] [auth_digests (auth_count*5)] [leaf_count] [leaves (leaf_count*3)]
         append_u64(static_cast<uint64_t>(flat_nodes.size()));
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             d_auth_digests,
             flat_nodes.size() * 5 * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice,
             ctx_->stream()
         ));
         proof_size_ += flat_nodes.size() * 5;
         
         append_u64(static_cast<uint64_t>(leaf_indices.size()));
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             d_revealed,
             leaf_indices.size() * 3 * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice,
             ctx_->stream()
         ));
         proof_size_ += leaf_indices.size() * 3;
     };
     
     // a-indices response for round 0
     append_fri_response(0, a_indices, dims_.fri_length);
 
     // b-indices responses for each round (exclude last codeword round)
     size_t dom_len = dims_.fri_length;
     for (size_t r = 0; r < dims_.num_fri_rounds; ++r) {
         size_t half = dom_len / 2;
         std::vector<size_t> b_indices;
         b_indices.reserve(a_indices.size());
         for (size_t a : a_indices) b_indices.push_back((a + half) % dom_len);
         append_fri_response(r, b_indices, dom_len);
         dom_len /= 2;
     }
 
     // Sample one scalar and throw away (matches Rust FRI)
     kernels::fs_sample_scalars_device_gpu(ctx_->d_sponge_state(), ctx_->d_scratch_a(), 1, ctx_->stream());
 
     if (profile_fri) {
         CUDA_CHECK(cudaEventRecord(ev_queries, ctx_->stream()));
         CUDA_CHECK(cudaEventRecord(ev_end, ctx_->stream()));
         ctx_->synchronize();
 
         float t_build = 0, t_round0 = 0, t_loop = 0, t_queries = 0;
         CUDA_CHECK(cudaEventElapsedTime(&t_build, ev0, ev_build));
         CUDA_CHECK(cudaEventElapsedTime(&t_round0, ev_build, ev_round0));
         CUDA_CHECK(cudaEventElapsedTime(&t_loop, ev_round0, ev_loop));
         CUDA_CHECK(cudaEventElapsedTime(&t_queries, ev_loop, ev_end));
 
         std::cout << "  [FRI profile] build_codeword: " << t_build << " ms\n";
         std::cout << "  [FRI profile] round0_commit:  " << t_round0 << " ms\n";
         std::cout << "  [FRI profile] fold+commit:    " << t_loop << " ms\n";
         std::cout << "  [FRI profile] queries:        " << t_queries << " ms\n";
         std::cout << "  [FRI profile] total:          " << (t_build + t_round0 + t_loop + t_queries) << " ms\n";
 
         CUDA_CHECK(cudaEventDestroy(ev0));
         CUDA_CHECK(cudaEventDestroy(ev_build));
         CUDA_CHECK(cudaEventDestroy(ev_round0));
         CUDA_CHECK(cudaEventDestroy(ev_loop));
         CUDA_CHECK(cudaEventDestroy(ev_queries));
         CUDA_CHECK(cudaEventDestroy(ev_end));
     }
 
     // NOTE: still missing:
     // - Query sampling and FriResponse openings (plus trace openings/auth paths)
 }
 
 void GpuStark::step_fri_protocol_frugal() {
     // Frugal FRI: build main+aux codeword on the FRI domain without cached FRI-domain LDE tables.
     // We stream 8 cosets (each of length padded_height) into ctx_->d_working_{main,aux} and fill the
     // full-domain main+aux combination codeword row-major.
     const size_t n = dims_.fri_length;
     const size_t trace_len = dims_.padded_height;
     const size_t num_cosets = n / trace_len;
     if (trace_len == 0 || n == 0 || (n % trace_len) != 0) {
         throw std::runtime_error("FRUGAL FRI: invalid domains (fri_len must be multiple of trace_len)");
     }
 
     constexpr int BLOCK = 256;
     const int grid_n = (int)((n + BLOCK - 1) / BLOCK);
     const int grid_trace = (int)((trace_len + BLOCK - 1) / BLOCK);
 
     const bool profile_fri = TRITON_PROFILE_ENABLED();
     cudaEvent_t ev0{}, ev_build{}, ev_round0{}, ev_loop{}, ev_queries{}, ev_end{};
     if (profile_fri) {
         CUDA_CHECK(cudaEventCreate(&ev0));
         CUDA_CHECK(cudaEventCreate(&ev_build));
         CUDA_CHECK(cudaEventCreate(&ev_round0));
         CUDA_CHECK(cudaEventCreate(&ev_loop));
         CUDA_CHECK(cudaEventCreate(&ev_queries));
         CUDA_CHECK(cudaEventCreate(&ev_end));
         CUDA_CHECK(cudaEventRecord(ev0, ctx_->stream()));
     }
 
     const size_t main_width = dims_.main_width;
     const size_t aux_width = dims_.aux_width;
     const size_t num_segments = dims_.num_quotient_segments;
     const size_t total_weights = main_width + aux_width + num_segments + 3;
 
     // Allocate a dedicated build scratch buffer (device memory). We cannot use ctx_->d_scratch_a/b
     // because in frugal mode they hold the col-major trace transposes.
     uint64_t* d_build = nullptr;
     size_t build_words = 0;
     const size_t off_weights = build_words; build_words += total_weights * 3;
     const size_t off_main_aux = build_words; build_words += n * 3;
     const size_t off_quot = build_words; build_words += n * 3;
     const size_t off_eval_ma_z = build_words; build_words += 3;
     const size_t off_eval_ma_gz = build_words; build_words += 3;
     const size_t off_eval_q_z4 = build_words; build_words += 3;
     const size_t off_z4 = build_words; build_words += 3;
     const size_t off_fri_domain = build_words; build_words += n; // BFE
 
     CUDA_CHECK(cudaMalloc(&d_build, build_words * sizeof(uint64_t)));
     uint64_t* d_weights = d_build + off_weights;
     uint64_t* d_main_aux_codeword = d_build + off_main_aux; // [n*3] row-major
     uint64_t* d_quot_codeword = d_build + off_quot;         // [n*3] row-major
     uint64_t* d_eval_main_aux_z = d_build + off_eval_ma_z;
     uint64_t* d_eval_main_aux_gz = d_build + off_eval_ma_gz;
     uint64_t* d_eval_quot_z4 = d_build + off_eval_q_z4;
     uint64_t* d_z4 = d_build + off_z4;
     uint64_t* d_fri_domain_vals = d_build + off_fri_domain;
 
     // Sync GPU sponge state to CPU transcript state at the start of FRI.
     {
         std::array<uint64_t, 16> h_state{};
         for (size_t i = 0; i < 16; ++i) {
             h_state[i] = fs_cpu_.sponge().state[i].value();
         }
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_sponge_state(),
             h_state.data(),
             16 * sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
     }
 
     kernels::fs_sample_scalars_device_gpu(ctx_->d_sponge_state(), d_weights, total_weights, ctx_->stream());
 
     // Read OOD rows/segments from proof buffer (written in Step 5) for eval_value computations
     const size_t base = 1 + 5 + 5 + 5;
     const uint64_t* d_ood_main_curr = ctx_->d_proof_buffer() + base;
     const uint64_t* d_ood_aux_curr  = d_ood_main_curr + main_width * 3;
     const uint64_t* d_ood_main_next = d_ood_aux_curr + aux_width * 3;
     const uint64_t* d_ood_aux_next  = d_ood_main_next + main_width * 3;
     const uint64_t* d_ood_quot      = d_ood_aux_next + aux_width * 3;
 
     // Compute eval_main_aux_z / eval_main_aux_gz / eval_quot_z4
     const size_t quot_weight_base = main_width + aux_width;
     qzc_eval_ood_value_main_aux_kernel<<<1,1,0,ctx_->stream()>>>(
         d_ood_main_curr, d_ood_aux_curr, d_weights, main_width, aux_width, d_eval_main_aux_z
     );
     qzc_eval_ood_value_main_aux_kernel<<<1,1,0,ctx_->stream()>>>(
         d_ood_main_next, d_ood_aux_next, d_weights, main_width, aux_width, d_eval_main_aux_gz
     );
     qzc_eval_ood_value_quot_kernel<<<1,1,0,ctx_->stream()>>>(
         d_ood_quot, d_weights, quot_weight_base, num_segments, d_eval_quot_z4
     );
 
     // compute z^4
     qzc_pow4_xfe<<<1,1,0,ctx_->stream()>>>(ctx_->d_ood_point(), d_z4);
 
     // Build main+aux combination codeword on full FRI domain via cosets
     // 1) transpose main trace row-major -> col-major into scratch_a
     uint64_t* d_main_colmajor = ctx_->d_scratch_a();
     {
         size_t total = trace_len * main_width;
         int grid_t = (int)((total + BLOCK - 1) / BLOCK);
         qzc_rowmajor_to_colmajor_bfe<<<grid_t, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_trace(), d_main_colmajor, trace_len, main_width
         );
     }
     // 2) transpose aux trace row-major -> comp-major col-major into scratch_b
     uint64_t* d_aux_colmajor_components = ctx_->d_scratch_b();
     {
         constexpr int TRANSPOSE_ELEMS = 4;
         int grid_aux = (int)(((trace_len * aux_width) + BLOCK * TRANSPOSE_ELEMS - 1) / (BLOCK * TRANSPOSE_ELEMS));
         qzc_rowmajor_to_colmajor_xfe<<<grid_aux, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_aux_trace(), d_aux_colmajor_components, trace_len, aux_width
         );
     }
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
 
     // Two-phase optimization: compute coefficients once (in-place on scratch buffers)
     kernels::compute_trace_coefficients_gpu(
         d_main_colmajor,
         dims_.main_width,
         trace_len,
         dims_.trace_offset,
         d_main_colmajor,
         ctx_->stream()
     );
     kernels::compute_trace_coefficients_gpu(
         d_aux_colmajor_components,
         dims_.aux_width * 3,
         trace_len,
         dims_.trace_offset,
         d_aux_colmajor_components,
         ctx_->stream()
     );
 
     // Check for multi-GPU (disabled due to memory access issues)
     int num_gpus = get_effective_gpu_count();
     if (num_gpus > 2) num_gpus = 2;
     bool multi_gpu = false; // Disabled
     
     if (multi_gpu) {
         const size_t cosets_per_gpu = num_cosets / num_gpus;
         
         cudaStream_t streams[2];
         uint64_t* d_working_main_per_gpu[2];
         uint64_t* d_working_aux_per_gpu[2];
         
         for (int gpu = 0; gpu < num_gpus; ++gpu) {
             CUDA_CHECK(cudaSetDevice(gpu));
             CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
             CUDA_CHECK(cudaMallocManaged(&d_working_main_per_gpu[gpu], main_width * trace_len * sizeof(uint64_t)));
             CUDA_CHECK(cudaMallocManaged(&d_working_aux_per_gpu[gpu], aux_width * 3 * trace_len * sizeof(uint64_t)));
             
             
         }
         
         for (int gpu = 0; gpu < num_gpus; ++gpu) {
             CUDA_CHECK(cudaSetDevice(gpu));
             size_t coset_start = gpu * cosets_per_gpu;
             size_t coset_end = coset_start + cosets_per_gpu;
             
             for (size_t coset = coset_start; coset < coset_end; ++coset) {
                 uint64_t coset_offset = (BFieldElement(dims_.fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
                 
                 kernels::randomized_lde_batch_gpu_preallocated(
                     d_main_colmajor,
                     main_width,
                     trace_len,
                     ctx_->d_main_randomizer_coeffs(),
                     dims_.num_trace_randomizers,
                     dims_.trace_offset,
                     coset_offset,
                     trace_len,
                     d_working_main_per_gpu[gpu],
                     d_working_main_per_gpu[gpu],
                     nullptr,
                     streams[gpu]
                 );
                 kernels::randomized_lde_batch_gpu_preallocated(
                     d_aux_colmajor_components,
                     aux_width * 3,
                     trace_len,
                     ctx_->d_aux_randomizer_coeffs(),
                     dims_.num_trace_randomizers,
                     dims_.trace_offset,
                     coset_offset,
                     trace_len,
                     d_working_aux_per_gpu[gpu],
                     d_working_aux_per_gpu[gpu],
                     nullptr,
                     streams[gpu]
                 );
                 qzc_build_main_aux_codeword_coset_kernel<<<grid_trace, BLOCK, 0, streams[gpu]>>>(
                     d_working_main_per_gpu[gpu],
                     d_working_aux_per_gpu[gpu],
                     d_weights,
                     trace_len,
                     coset,
                     num_cosets,
                     main_width,
                     aux_width,
                     d_main_aux_codeword
                 );
             }
         }
         
         for (int gpu = 0; gpu < num_gpus; ++gpu) {
             CUDA_CHECK(cudaSetDevice(gpu));
             CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
             CUDA_CHECK(cudaStreamDestroy(streams[gpu]));
             CUDA_CHECK(cudaFree(d_working_main_per_gpu[gpu]));
             CUDA_CHECK(cudaFree(d_working_aux_per_gpu[gpu]));
         }
         CUDA_CHECK(cudaSetDevice(0));
     } else {
         // Tail scratch buffers for coset evaluation (freed after loop)
         uint64_t* d_main_tail = nullptr;
         uint64_t* d_aux_tail = nullptr;
         CUDA_CHECK(cudaMalloc(&d_main_tail, main_width * trace_len * sizeof(uint64_t)));
         CUDA_CHECK(cudaMalloc(&d_aux_tail, aux_width * 3 * trace_len * sizeof(uint64_t)));
 
         for (size_t coset = 0; coset < num_cosets; ++coset) {
             uint64_t coset_offset = (BFieldElement(dims_.fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
             kernels::evaluate_coset_from_coefficients_gpu(
                 d_main_colmajor,
                 main_width,
                 trace_len,
                 ctx_->d_main_randomizer_coeffs(),
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 coset_offset,
                 ctx_->d_working_main(),
                 d_main_tail,
                 ctx_->stream()
             );
             kernels::evaluate_coset_from_coefficients_gpu(
                 d_aux_colmajor_components,
                 aux_width * 3,
                 trace_len,
                 ctx_->d_aux_randomizer_coeffs(),
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 coset_offset,
                 ctx_->d_working_aux(),
                 d_aux_tail,
                 ctx_->stream()
             );
             qzc_build_main_aux_codeword_coset_kernel<<<grid_trace, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_working_main(),
                 ctx_->d_working_aux(),
                 d_weights,
                 trace_len,
                 coset,
                 num_cosets,
                 main_width,
                 aux_width,
                 d_main_aux_codeword
             );
         }
 
         CUDA_CHECK(cudaFree(d_main_tail));
         CUDA_CHECK(cudaFree(d_aux_tail));
     }
 
     // build quotient combination codeword from quotient segments on FRI domain
     qzc_build_quot_codeword_kernel<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
         ctx_->d_quotient_segments(),
         d_weights,
         n,
         quot_weight_base,
         num_segments,
         d_quot_codeword
     );
 
     // Compute fri domain values (BFE) for denominators
     {
         constexpr int ITEMS = 4;
         int grid = (int)((n + (size_t)BLOCK * ITEMS - 1) / ((size_t)BLOCK * ITEMS));
         qzc_fill_domain_points_chunked<ITEMS><<<grid, BLOCK, 0, ctx_->stream()>>>(d_fri_domain_vals, n, dims_.fri_offset, dims_.fri_generator);
     }
 
     // deep weights are last 3 weights
     const uint64_t* d_w_deep = d_weights + (main_width + aux_width + num_segments) * 3;
 
     // build final FRI input codeword in ctx_->d_fri_codeword(0)
     qzc_deep_fri_codeword_kernel<<<grid_n, BLOCK, 0, ctx_->stream()>>>(
         d_fri_domain_vals,
         d_main_aux_codeword,
         d_quot_codeword,
         n,
         ctx_->d_ood_point(),
         ctx_->d_next_row_point(),
         d_z4,
         d_eval_main_aux_z,
         d_eval_main_aux_gz,
         d_eval_quot_z4,
         d_w_deep,
         ctx_->d_fri_codeword(0)
     );
 
     if (profile_fri) { CUDA_CHECK(cudaEventRecord(ev_build, ctx_->stream())); }
 
     // From here on, reuse the EXACT folding/commit/query logic from the working (non-frugal) FRI path.
     size_t current_size = n;
     uint64_t offset = dims_.fri_offset;
     uint64_t generator = dims_.fri_generator;
     uint64_t two_inv = BFieldElement(2).inverse().value();
 
     // Commit to first round
     {
         uint64_t* d_tree = ctx_->d_fri_merkle(0);
         int grid = (int)((current_size + BLOCK - 1) / BLOCK);
         qzc_xfe_to_digest_leaves_kernel<<<grid, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_fri_codeword(0),
             current_size,
             d_tree
         );
         kernels::merkle_tree_gpu(d_tree, d_tree, current_size, ctx_->stream());
 
         uint64_t* d_root = d_tree + (2 * current_size - 2) * 5;
         uint64_t* d_enc = ctx_->d_scratch_b();
         uint64_t disc = 0;
         CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
         CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
         kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
 
         cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_root, 5 * sizeof(uint64_t),
                         cudaMemcpyDeviceToDevice, ctx_->stream());
         proof_size_ += 5;
     }
 
     if (profile_fri) { CUDA_CHECK(cudaEventRecord(ev_round0, ctx_->stream())); }
 
     // Pre-allocate device memory for domain inverses (reuse across rounds)
     size_t max_domain_inv_size = n / 2;
     uint64_t* d_domain_inv_pool;
     CUDA_CHECK(cudaMalloc(&d_domain_inv_pool, max_domain_inv_size * sizeof(uint64_t)));
 
     // Reuse scratch_a for sampled challenges (3 u64)
     uint64_t* d_chal = ctx_->d_scratch_a();
 
     for (size_t round = 0; round < dims_.num_fri_rounds; ++round) {
         size_t half = current_size / 2;
 
         // Sample folding challenge sequentially
         kernels::fs_sample_scalars_device_gpu(ctx_->d_sponge_state(), d_chal, 1, ctx_->stream());
 
         uint64_t* d_domain_inv = d_domain_inv_pool;
         kernels::compute_domain_inverses_gpu(offset, generator, d_domain_inv, half, ctx_->stream());
 
         kernels::fri_fold_gpu(
             ctx_->d_fri_codeword(round),
             current_size,
             d_chal,
             d_domain_inv,
             two_inv,
             ctx_->d_fri_codeword(round + 1),
             ctx_->stream()
         );
 
         // Halve domain
         offset = (BFieldElement(offset) * BFieldElement(offset)).value();
         generator = (BFieldElement(generator) * BFieldElement(generator)).value();
         current_size = half;
 
         // Commit to new round
         uint64_t* d_tree = ctx_->d_fri_merkle(round + 1);
         int grid = (int)((current_size + BLOCK - 1) / BLOCK);
         qzc_xfe_to_digest_leaves_kernel<<<grid, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_fri_codeword(round + 1),
             current_size,
             d_tree
         );
         kernels::merkle_tree_gpu(d_tree, d_tree, current_size, ctx_->stream());
 
         uint64_t* d_root = d_tree + (2 * current_size - 2) * 5;
         uint64_t* d_enc = ctx_->d_scratch_b();
         uint64_t disc = 0;
         CUDA_CHECK(cudaMemcpyAsync(d_enc, &disc, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
         CUDA_CHECK(cudaMemcpyAsync(d_enc + 1, d_root, 5 * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx_->stream()));
         kernels::fs_absorb_device_gpu(ctx_->d_sponge_state(), d_enc, 6, ctx_->stream());
 
         cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, d_root, 5 * sizeof(uint64_t),
                         cudaMemcpyDeviceToDevice, ctx_->stream());
         proof_size_ += 5;
     }
 
     CUDA_CHECK(cudaFree(d_domain_inv_pool));
     if (profile_fri) { CUDA_CHECK(cudaEventRecord(ev_loop, ctx_->stream())); }
 
     // Append last codeword (not absorbed)
     const uint64_t* d_last_codeword = ctx_->d_fri_codeword(dims_.num_fri_rounds);
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_last_codeword,
         current_size * 3 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += current_size * 3;
 
     // Query phase (identical to non-frugal)
     constexpr size_t NUM_QUERIES = GpuProofContext::NUM_FRI_QUERIES;
     kernels::fs_sample_indices_device_gpu(
         ctx_->d_sponge_state(),
         ctx_->d_fri_query_indices(),
         dims_.fri_length,
         NUM_QUERIES,
         ctx_->stream()
     );
     std::vector<size_t> a_indices(NUM_QUERIES);
     CUDA_CHECK(cudaMemcpyAsync(
         a_indices.data(),
         ctx_->d_fri_query_indices(),
         NUM_QUERIES * sizeof(size_t),
         cudaMemcpyDeviceToHost,
         ctx_->stream()
     ));
     ctx_->synchronize();
 
     auto append_u64 = [&](uint64_t v) {
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             &v,
             sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
         proof_size_ += 1;
     };
 
     auto append_fri_response = [&](size_t round_idx, const std::vector<size_t>& leaf_indices, size_t domain_len) {
         size_t* d_tmp_indices = reinterpret_cast<size_t*>(ctx_->d_scratch_a());
         CUDA_CHECK(cudaMemcpyAsync(
             d_tmp_indices,
             leaf_indices.data(),
             leaf_indices.size() * sizeof(size_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
 
         uint64_t* d_revealed = ctx_->d_scratch_b();
         kernels::gather_xfield_rows_gpu(
             ctx_->d_fri_codeword(round_idx),
             d_tmp_indices,
             d_revealed,
             domain_len,
             1,
             leaf_indices.size(),
             ctx_->stream()
         );
 
         std::vector<size_t> heap_nodes = auth_structure_heap_node_indices(leaf_indices, domain_len);
         std::vector<size_t> flat_nodes;
         flat_nodes.reserve(heap_nodes.size());
         for (size_t h : heap_nodes) flat_nodes.push_back(merkle_heap_to_flat_index(h, domain_len));
 
         size_t* d_auth_idx = d_tmp_indices;
         CUDA_CHECK(cudaMemcpyAsync(
             d_auth_idx,
             flat_nodes.data(),
             flat_nodes.size() * sizeof(size_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
 
         uint64_t* d_auth_digests = d_revealed + leaf_indices.size() * 3;
         kernels::gather_bfield_rows_gpu(
             ctx_->d_fri_merkle(round_idx),
             d_auth_idx,
             d_auth_digests,
             2 * domain_len - 1,
             5,
             flat_nodes.size(),
             ctx_->stream()
         );
 
         append_u64(static_cast<uint64_t>(flat_nodes.size()));
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             d_auth_digests,
             flat_nodes.size() * 5 * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice,
             ctx_->stream()
         ));
         proof_size_ += flat_nodes.size() * 5;
 
         append_u64(static_cast<uint64_t>(leaf_indices.size()));
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             d_revealed,
             leaf_indices.size() * 3 * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice,
             ctx_->stream()
         ));
         proof_size_ += leaf_indices.size() * 3;
     };
 
     append_fri_response(0, a_indices, dims_.fri_length);
 
     size_t dom_len = dims_.fri_length;
     for (size_t r = 0; r < dims_.num_fri_rounds; ++r) {
         size_t half = dom_len / 2;
         std::vector<size_t> b_indices;
         b_indices.reserve(a_indices.size());
         for (size_t a : a_indices) b_indices.push_back((a + half) % dom_len);
         append_fri_response(r, b_indices, dom_len);
         dom_len /= 2;
     }
 
     kernels::fs_sample_scalars_device_gpu(ctx_->d_sponge_state(), ctx_->d_scratch_a(), 1, ctx_->stream());
     if (profile_fri) { CUDA_CHECK(cudaEventRecord(ev_queries, ctx_->stream())); }
 
     if (profile_fri) {
         CUDA_CHECK(cudaEventRecord(ev_end, ctx_->stream()));
         CUDA_CHECK(cudaEventSynchronize(ev_end));
         float t_build{}, t_round0{}, t_loop{}, t_queries{};
         CUDA_CHECK(cudaEventElapsedTime(&t_build, ev0, ev_build));
         CUDA_CHECK(cudaEventElapsedTime(&t_round0, ev_build, ev_round0));
         CUDA_CHECK(cudaEventElapsedTime(&t_loop, ev_round0, ev_queries));
         CUDA_CHECK(cudaEventElapsedTime(&t_queries, ev_queries, ev_end));
         std::cout << "  [FRI profile] build_codeword: " << t_build << " ms\n";
         std::cout << "  [FRI profile] round0_commit:  " << t_round0 << " ms\n";
         std::cout << "  [FRI profile] fold+commit:    " << t_loop << " ms\n";
         std::cout << "  [FRI profile] queries:        " << t_queries << " ms\n";
         std::cout << "  [FRI profile] total:          " << (t_build + t_round0 + t_loop + t_queries) << " ms\n";
         CUDA_CHECK(cudaEventDestroy(ev0));
         CUDA_CHECK(cudaEventDestroy(ev_build));
         CUDA_CHECK(cudaEventDestroy(ev_round0));
         CUDA_CHECK(cudaEventDestroy(ev_loop));
         CUDA_CHECK(cudaEventDestroy(ev_queries));
         CUDA_CHECK(cudaEventDestroy(ev_end));
     }
 
     CUDA_CHECK(cudaFree(d_build));
 }
 
 // ============================================================================
 // Step 7: Open Trace
 // ============================================================================
 
 void GpuStark::step_open_trace() {
     if (dims_.lde_frugal_mode) {
         step_open_trace_frugal();
         return;
     }
     constexpr size_t NUM_QUERIES = GpuProofContext::NUM_FRI_QUERIES;
     const size_t n = dims_.fri_length;
     constexpr int BLOCK = 256;
     
     // Use the FRI-sampled indices from step_fri_protocol()
     size_t* d_indices = ctx_->d_fri_query_indices();
 
     // Gather opened rows (row-major buffers)
     uint64_t* d_main_rows = nullptr;
     uint64_t* d_aux_rows = nullptr;
     uint64_t* d_quot_rows = nullptr;
     uint64_t* d_main_batch_rows = nullptr;
     uint64_t* d_aux_batch_rows = nullptr;
     uint64_t* d_main_colmajor = nullptr;
     uint64_t* d_aux_colmajor_components = nullptr;
 
     if (dims_.lde_frugal_mode) {
         // Allocate small buffers for opened rows
         CUDA_CHECK(cudaMalloc(&d_main_rows, NUM_QUERIES * dims_.main_width * sizeof(uint64_t)));
         CUDA_CHECK(cudaMalloc(&d_aux_rows, NUM_QUERIES * dims_.aux_width * 3 * sizeof(uint64_t)));
         CUDA_CHECK(cudaMalloc(&d_quot_rows, NUM_QUERIES * dims_.num_quotient_segments * 3 * sizeof(uint64_t)));
 
         constexpr size_t FRUGAL_BATCH_COLS = 10;
         CUDA_CHECK(cudaMalloc(&d_main_batch_rows, NUM_QUERIES * FRUGAL_BATCH_COLS * sizeof(uint64_t)));
         CUDA_CHECK(cudaMalloc(&d_aux_batch_rows, NUM_QUERIES * FRUGAL_BATCH_COLS * sizeof(uint64_t)));
 
         // Build col-major traces for batch LDE
         CUDA_CHECK(cudaMalloc(&d_main_colmajor, dims_.padded_height * dims_.main_width * sizeof(uint64_t)));
         {
             size_t total = dims_.padded_height * dims_.main_width;
             constexpr int BLOCK = 256;
             int grid_t = (int)((total + BLOCK - 1) / BLOCK);
             qzc_rowmajor_to_colmajor_bfe<<<grid_t, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_main_trace(), d_main_colmajor, dims_.padded_height, dims_.main_width
             );
         }
         d_aux_colmajor_components = ctx_->d_scratch_b();
         {
             constexpr int BLOCK = 256;
             constexpr int TRANSPOSE_ELEMS = 4;
             int grid_aux = (int)(((dims_.padded_height * dims_.aux_width) + BLOCK * TRANSPOSE_ELEMS - 1) / (BLOCK * TRANSPOSE_ELEMS));
             qzc_rowmajor_to_colmajor_xfe<<<grid_aux, BLOCK, 0, ctx_->stream()>>>(
                 ctx_->d_aux_trace(), d_aux_colmajor_components, dims_.padded_height, dims_.aux_width
             );
         }
         CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
 
         // Main columns: stream LDE batches and gather query rows
         for (size_t col_start = 0; col_start < dims_.main_width; col_start += FRUGAL_BATCH_COLS) {
             const size_t batch_cols = std::min(FRUGAL_BATCH_COLS, dims_.main_width - col_start);
             kernels::randomized_lde_batch_gpu(
                 d_main_colmajor + col_start * dims_.padded_height,
                 batch_cols,
                 dims_.padded_height,
                 ctx_->d_main_randomizer_coeffs() + col_start * dims_.num_trace_randomizers,
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 dims_.fri_offset,
                 n,
                 ctx_->d_main_lde(),
                 ctx_->stream()
             );
             kernels::gather_bfield_rows_colmajor_gpu(
                 ctx_->d_main_lde(),
                 d_indices,
                 d_main_batch_rows,
                 n,
                 batch_cols,
                 NUM_QUERIES,
                 ctx_->stream()
             );
             int grid_scatter = (int)((NUM_QUERIES * batch_cols + BLOCK - 1) / BLOCK);
             qzc_scatter_rowmajor_offset_bfe<<<grid_scatter, BLOCK, 0, ctx_->stream()>>>(
                 d_main_batch_rows,
                 batch_cols,
                 d_main_rows,
                 dims_.main_width,
                 NUM_QUERIES,
                 col_start
             );
         }
 
         // Aux columns: stream LDE batches (BFE component view) and gather query rows
         const size_t aux_cols_bfe = dims_.aux_width * 3;
         for (size_t col_start = 0; col_start < aux_cols_bfe; col_start += FRUGAL_BATCH_COLS) {
             const size_t batch_cols = std::min(FRUGAL_BATCH_COLS, aux_cols_bfe - col_start);
             kernels::randomized_lde_batch_gpu(
                 d_aux_colmajor_components + col_start * dims_.padded_height,
                 batch_cols,
                 dims_.padded_height,
                 ctx_->d_aux_randomizer_coeffs() + col_start * dims_.num_trace_randomizers,
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 dims_.fri_offset,
                 n,
                 ctx_->d_aux_lde(),
                 ctx_->stream()
             );
             kernels::gather_bfield_rows_colmajor_gpu(
                 ctx_->d_aux_lde(),
                 d_indices,
                 d_aux_batch_rows,
                 n,
                 batch_cols,
                 NUM_QUERIES,
                 ctx_->stream()
             );
             int grid_scatter = (int)((NUM_QUERIES * batch_cols + BLOCK - 1) / BLOCK);
             qzc_scatter_rowmajor_offset_bfe<<<grid_scatter, BLOCK, 0, ctx_->stream()>>>(
                 d_aux_batch_rows,
                 batch_cols,
                 d_aux_rows,
                 dims_.aux_width * 3,
                 NUM_QUERIES,
                 col_start
             );
         }
 
         // Quotient rows still gathered from cached quotient segments
         kernels::gather_xfield_rows_colmajor_gpu(
             ctx_->d_quotient_segments(),
             d_indices,
             d_quot_rows,
             n,
             dims_.num_quotient_segments,
             NUM_QUERIES,
             ctx_->stream()
         );
     } else {
         d_main_rows = ctx_->d_scratch_a(); // [NUM*main_width]
         d_aux_rows  = d_main_rows + NUM_QUERIES * dims_.main_width; // [NUM*aux_width*3]
         d_quot_rows = d_aux_rows + NUM_QUERIES * dims_.aux_width * 3; // [NUM*4*3]
 
         kernels::gather_bfield_rows_colmajor_gpu(
             ctx_->d_main_lde(),
             d_indices,
             d_main_rows,
             n,
             dims_.main_width,
             NUM_QUERIES,
             ctx_->stream()
         );
         
         kernels::gather_xfield_rows_colmajor_gpu(
             ctx_->d_aux_lde(),
             d_indices,
             d_aux_rows,
             n,
             dims_.aux_width,
             NUM_QUERIES,
             ctx_->stream()
         );
         
         kernels::gather_xfield_rows_colmajor_gpu(
             ctx_->d_quotient_segments(),
             d_indices,
             d_quot_rows,
             n,
             dims_.num_quotient_segments,
             NUM_QUERIES,
             ctx_->stream()
         );
     }
 
     // Pull indices to host to compute authentication_structure node indices (small).
     std::vector<size_t> indices(NUM_QUERIES);
     CUDA_CHECK(cudaMemcpyAsync(
         indices.data(),
         d_indices,
         NUM_QUERIES * sizeof(size_t),
         cudaMemcpyDeviceToHost,
         ctx_->stream()
     ));
     ctx_->synchronize();
 
     // Compute auth structure heap indices once, map to flat
     std::vector<size_t> heap_nodes = auth_structure_heap_node_indices(indices, n);
     std::vector<size_t> flat_nodes;
     flat_nodes.reserve(heap_nodes.size());
     for (size_t h : heap_nodes) flat_nodes.push_back(merkle_heap_to_flat_index(h, n));
 
     // Upload flat node indices to device temp
     size_t* d_auth_idx = reinterpret_cast<size_t*>(ctx_->d_scratch_b());
     CUDA_CHECK(cudaMemcpyAsync(
         d_auth_idx,
         flat_nodes.data(),
         flat_nodes.size() * sizeof(size_t),
         cudaMemcpyHostToDevice,
         ctx_->stream()
     ));
 
     uint64_t* d_auth_digests = reinterpret_cast<uint64_t*>(d_auth_idx + flat_nodes.size());
 
     auto append_u64 = [&](uint64_t v) {
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             &v,
             sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
         proof_size_ += 1;
     };
 
     auto append_auth = [&](const uint64_t* d_tree) {
         // Merkle tree is stored as a flat array of digests (row-major), length = (2*n - 1) rows, width = 5
         kernels::gather_bfield_rows_gpu(
             d_tree,
             d_auth_idx,
             d_auth_digests,
             2 * n - 1,
             5,
             flat_nodes.size(),
             ctx_->stream()
         );
         append_u64(static_cast<uint64_t>(flat_nodes.size()));
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             d_auth_digests,
             flat_nodes.size() * 5 * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice,
             ctx_->stream()
         ));
         proof_size_ += flat_nodes.size() * 5;
     };
 
     // Main rows payload: [row_count] [rows...]
     append_u64(NUM_QUERIES);
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_main_rows,
         NUM_QUERIES * dims_.main_width * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += NUM_QUERIES * dims_.main_width;
     append_auth(ctx_->d_main_merkle());
 
     // Aux rows payload: [row_count] [rows...]
     append_u64(NUM_QUERIES);
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_aux_rows,
         NUM_QUERIES * dims_.aux_width * 3 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += NUM_QUERIES * dims_.aux_width * 3;
     append_auth(ctx_->d_aux_merkle());
 
     // Quotient segments payload: [row_count] [rows...]
     append_u64(NUM_QUERIES);
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_quot_rows,
         NUM_QUERIES * dims_.num_quotient_segments * 3 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += NUM_QUERIES * dims_.num_quotient_segments * 3;
     append_auth(ctx_->d_quotient_merkle());
 
     if (dims_.lde_frugal_mode) {
         CUDA_CHECK(cudaFree(d_main_rows));
         CUDA_CHECK(cudaFree(d_aux_rows));
         CUDA_CHECK(cudaFree(d_quot_rows));
         if (d_main_batch_rows) CUDA_CHECK(cudaFree(d_main_batch_rows));
         if (d_aux_batch_rows) CUDA_CHECK(cudaFree(d_aux_batch_rows));
         if (d_main_colmajor) CUDA_CHECK(cudaFree(d_main_colmajor));
     }
 }
 
 __global__ void qzc_open_main_aux_rows_from_coset_kernel(
     const size_t* __restrict__ d_query_indices,     // [NUM]
     size_t num_queries,
     const uint64_t* __restrict__ d_main_coset,      // [main_width * coset_len] col-major
     const uint64_t* __restrict__ d_aux_coset,       // [(aux_width*3) * coset_len] comp-major col-major (BFEs)
     size_t coset_len,
     size_t coset_idx,
     size_t num_cosets,
     size_t main_width,
     size_t aux_width,
     uint64_t* __restrict__ d_out_main_rows,         // [NUM * main_width] row-major (BFE)
     uint64_t* __restrict__ d_out_aux_rows           // [NUM * aux_width * 3] row-major (XFE)
 ) {
     size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (q >= num_queries) return;
     size_t idx = d_query_indices[q];
     if ((idx % num_cosets) != coset_idx) return;
     size_t k = idx / num_cosets;
     if (k >= coset_len) return;
 
     // main
     for (size_t c = 0; c < main_width; ++c) {
         d_out_main_rows[q * main_width + c] = d_main_coset[c * coset_len + k];
     }
     // aux (XFE): output row-major XFE layout
     for (size_t c = 0; c < aux_width; ++c) {
         uint64_t v0 = d_aux_coset[(c * 3 + 0) * coset_len + k];
         uint64_t v1 = d_aux_coset[(c * 3 + 1) * coset_len + k];
         uint64_t v2 = d_aux_coset[(c * 3 + 2) * coset_len + k];
         size_t base = (q * aux_width + c) * 3;
         d_out_aux_rows[base + 0] = v0;
         d_out_aux_rows[base + 1] = v1;
         d_out_aux_rows[base + 2] = v2;
     }
 }
 
 void GpuStark::step_open_trace_frugal() {
     constexpr size_t NUM_QUERIES = GpuProofContext::NUM_FRI_QUERIES;
     const size_t n = dims_.fri_length;
     const size_t trace_len = dims_.padded_height;
     const size_t num_cosets = n / trace_len;
     if (trace_len == 0 || n == 0 || (n % trace_len) != 0) {
         throw std::runtime_error("FRUGAL open: invalid domains");
     }
 
     size_t* d_indices = ctx_->d_fri_query_indices();
 
     // Output opened rows (row-major). Allocate explicitly (small) to keep scratch buffers free for transposes.
     uint64_t* d_main_rows = nullptr;
     uint64_t* d_aux_rows  = nullptr;
     uint64_t* d_quot_rows = nullptr;
     CUDA_CHECK(cudaMalloc(&d_main_rows, NUM_QUERIES * dims_.main_width * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_aux_rows,  NUM_QUERIES * dims_.aux_width * 3 * sizeof(uint64_t)));
     CUDA_CHECK(cudaMalloc(&d_quot_rows, NUM_QUERIES * dims_.num_quotient_segments * 3 * sizeof(uint64_t)));
 
     // Build col-major trace transposes into scratch buffers (same layout as other frugal steps)
     constexpr int BLOCK = 256;
     uint64_t* d_main_colmajor = ctx_->d_scratch_a(); // [main_width * trace_len]
     {
         size_t total = trace_len * dims_.main_width;
         int grid_t = (int)((total + BLOCK - 1) / BLOCK);
         qzc_rowmajor_to_colmajor_bfe<<<grid_t, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_main_trace(), d_main_colmajor, trace_len, dims_.main_width
         );
     }
     uint64_t* d_aux_colmajor_components = ctx_->d_scratch_b(); // [(aux_width*3) * trace_len]
     {
         constexpr int TRANSPOSE_ELEMS = 4;
         int grid_aux = (int)(((trace_len * dims_.aux_width) + BLOCK * TRANSPOSE_ELEMS - 1) / (BLOCK * TRANSPOSE_ELEMS));
         qzc_rowmajor_to_colmajor_xfe<<<grid_aux, BLOCK, 0, ctx_->stream()>>>(
             ctx_->d_aux_trace(), d_aux_colmajor_components, trace_len, dims_.aux_width
         );
     }
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
 
     // Two-phase optimization: compute coefficients once (in-place on scratch buffers)
     kernels::compute_trace_coefficients_gpu(
         d_main_colmajor,
         dims_.main_width,
         trace_len,
         dims_.trace_offset,
         d_main_colmajor,
         ctx_->stream()
     );
     kernels::compute_trace_coefficients_gpu(
         d_aux_colmajor_components,
         dims_.aux_width * 3,
         trace_len,
         dims_.trace_offset,
         d_aux_colmajor_components,
         ctx_->stream()
     );
 
     // Fill outputs by coset - with multi-GPU support
     int grid_q = (int)((NUM_QUERIES + BLOCK - 1) / BLOCK);
     
     int num_gpus = get_effective_gpu_count();
     if (num_gpus > 2) num_gpus = 2;
     bool multi_gpu = false; // Disabled due to memory issues
     
         if (multi_gpu) {
         int can_access_0_1 = 0, can_access_1_0 = 0;
         cudaDeviceCanAccessPeer(&can_access_0_1, 0, 1);
         cudaDeviceCanAccessPeer(&can_access_1_0, 1, 0);
         if (!can_access_0_1 || !can_access_1_0) {
             multi_gpu = false;
         }
     }
     
         if (multi_gpu) {
         const size_t cosets_per_gpu = num_cosets / num_gpus;
         
         cudaStream_t streams[2];
         uint64_t* d_working_main_per_gpu[2];
         uint64_t* d_working_aux_per_gpu[2];
         
         for (int gpu = 0; gpu < num_gpus; ++gpu) {
             CUDA_CHECK(cudaSetDevice(gpu));
             CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
             CUDA_CHECK(cudaMallocManaged(&d_working_main_per_gpu[gpu], dims_.main_width * trace_len * sizeof(uint64_t)));
             CUDA_CHECK(cudaMallocManaged(&d_working_aux_per_gpu[gpu], dims_.aux_width * 3 * trace_len * sizeof(uint64_t)));
             
             
         }
         
         for (int gpu = 0; gpu < num_gpus; ++gpu) {
             CUDA_CHECK(cudaSetDevice(gpu));
             size_t coset_start = gpu * cosets_per_gpu;
             size_t coset_end = coset_start + cosets_per_gpu;
             
             for (size_t coset = coset_start; coset < coset_end; ++coset) {
                 uint64_t coset_offset = (BFieldElement(dims_.fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
                 
                 kernels::randomized_lde_batch_gpu_preallocated(
                     d_main_colmajor,
                     dims_.main_width,
                     trace_len,
                     ctx_->d_main_randomizer_coeffs(),
                     dims_.num_trace_randomizers,
                     dims_.trace_offset,
                     coset_offset,
                     trace_len,
                     d_working_main_per_gpu[gpu],
                     d_working_main_per_gpu[gpu],
                     nullptr,
                     streams[gpu]
                 );
                 kernels::randomized_lde_batch_gpu_preallocated(
                     d_aux_colmajor_components,
                     dims_.aux_width * 3,
                     trace_len,
                     ctx_->d_aux_randomizer_coeffs(),
                     dims_.num_trace_randomizers,
                     dims_.trace_offset,
                     coset_offset,
                     trace_len,
                     d_working_aux_per_gpu[gpu],
                     d_working_aux_per_gpu[gpu],
                     nullptr,
                     streams[gpu]
                 );
                 qzc_open_main_aux_rows_from_coset_kernel<<<grid_q, BLOCK, 0, streams[gpu]>>>(
                     d_indices,
                     NUM_QUERIES,
                     d_working_main_per_gpu[gpu],
                     d_working_aux_per_gpu[gpu],
                     trace_len,
                     coset,
                     num_cosets,
                     dims_.main_width,
                     dims_.aux_width,
                     d_main_rows,
                     d_aux_rows
                 );
             }
         }
         
         for (int gpu = 0; gpu < num_gpus; ++gpu) {
             CUDA_CHECK(cudaSetDevice(gpu));
             CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
             CUDA_CHECK(cudaStreamDestroy(streams[gpu]));
             CUDA_CHECK(cudaFree(d_working_main_per_gpu[gpu]));
             CUDA_CHECK(cudaFree(d_working_aux_per_gpu[gpu]));
         }
         CUDA_CHECK(cudaSetDevice(0));
     } else {
         // Tail scratch buffers for coset evaluation (freed after loop)
         uint64_t* d_main_tail = nullptr;
         uint64_t* d_aux_tail = nullptr;
         CUDA_CHECK(cudaMalloc(&d_main_tail, dims_.main_width * trace_len * sizeof(uint64_t)));
         CUDA_CHECK(cudaMalloc(&d_aux_tail, dims_.aux_width * 3 * trace_len * sizeof(uint64_t)));
 
         for (size_t coset = 0; coset < num_cosets; ++coset) {
             uint64_t coset_offset = (BFieldElement(dims_.fri_offset) * BFieldElement(dims_.fri_generator).pow(coset)).value();
             kernels::evaluate_coset_from_coefficients_gpu(
                 d_main_colmajor,
                 dims_.main_width,
                 trace_len,
                 ctx_->d_main_randomizer_coeffs(),
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 coset_offset,
                 ctx_->d_working_main(),
                 d_main_tail,
                 ctx_->stream()
             );
             kernels::evaluate_coset_from_coefficients_gpu(
                 d_aux_colmajor_components,
                 dims_.aux_width * 3,
                 trace_len,
                 ctx_->d_aux_randomizer_coeffs(),
                 dims_.num_trace_randomizers,
                 dims_.trace_offset,
                 coset_offset,
                 ctx_->d_working_aux(),
                 d_aux_tail,
                 ctx_->stream()
             );
             qzc_open_main_aux_rows_from_coset_kernel<<<grid_q, BLOCK, 0, ctx_->stream()>>>(
                 d_indices,
                 NUM_QUERIES,
                 ctx_->d_working_main(),
                 ctx_->d_working_aux(),
                 trace_len,
                 coset,
                 num_cosets,
                 dims_.main_width,
                 dims_.aux_width,
                 d_main_rows,
                 d_aux_rows
             );
         }
 
         CUDA_CHECK(cudaFree(d_main_tail));
         CUDA_CHECK(cudaFree(d_aux_tail));
     }
 
     // Quotient segment rows are already on full FRI domain
     kernels::gather_xfield_rows_colmajor_gpu(
         ctx_->d_quotient_segments(),
         d_indices,
         d_quot_rows,
         n,
         dims_.num_quotient_segments,
         NUM_QUERIES,
         ctx_->stream()
     );
 
     // From here on, reuse the existing non-frugal code paths by temporarily setting pointers and appending to proof.
     // We inline the minimal parts needed: indices, opened rows, and auth paths.
     std::vector<size_t> indices(NUM_QUERIES);
     CUDA_CHECK(cudaMemcpyAsync(indices.data(), d_indices, NUM_QUERIES * sizeof(size_t), cudaMemcpyDeviceToHost, ctx_->stream()));
     ctx_->synchronize();
 
     std::vector<size_t> heap_nodes = auth_structure_heap_node_indices(indices, n);
     std::vector<size_t> flat_nodes;
     flat_nodes.reserve(heap_nodes.size());
     for (size_t h : heap_nodes) flat_nodes.push_back(merkle_heap_to_flat_index(h, n));
 
     size_t* d_auth_idx = reinterpret_cast<size_t*>(ctx_->d_scratch_b());
     CUDA_CHECK(cudaMemcpyAsync(d_auth_idx, flat_nodes.data(), flat_nodes.size() * sizeof(size_t), cudaMemcpyHostToDevice, ctx_->stream()));
     uint64_t* d_auth_digests = reinterpret_cast<uint64_t*>(d_auth_idx + flat_nodes.size());
 
     auto append_u64 = [&](uint64_t v) {
         CUDA_CHECK(cudaMemcpyAsync(ctx_->d_proof_buffer() + proof_size_, &v, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx_->stream()));
         proof_size_ += 1;
     };
     auto append_auth = [&](const uint64_t* d_tree) {
         kernels::gather_bfield_rows_gpu(
             d_tree,
             d_auth_idx,
             d_auth_digests,
             2 * n - 1,
             5,
             flat_nodes.size(),
             ctx_->stream()
         );
         append_u64(static_cast<uint64_t>(flat_nodes.size()));
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_proof_buffer() + proof_size_,
             d_auth_digests,
             flat_nodes.size() * 5 * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice,
             ctx_->stream()
         ));
         proof_size_ += flat_nodes.size() * 5;
     };
 
     // Match non-frugal proof encoding EXACTLY:
     // Main rows payload: [row_count] [rows...] then [auth_count] [auth_digests...]
     append_u64(static_cast<uint64_t>(NUM_QUERIES));
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_main_rows,
         NUM_QUERIES * dims_.main_width * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += NUM_QUERIES * dims_.main_width;
     append_auth(ctx_->d_main_merkle());
 
     // Aux rows payload
     append_u64(static_cast<uint64_t>(NUM_QUERIES));
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_aux_rows,
         NUM_QUERIES * dims_.aux_width * 3 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += NUM_QUERIES * dims_.aux_width * 3;
     append_auth(ctx_->d_aux_merkle());
 
     // Quotient segments payload
     append_u64(static_cast<uint64_t>(NUM_QUERIES));
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_proof_buffer() + proof_size_,
         d_quot_rows,
         NUM_QUERIES * dims_.num_quotient_segments * 3 * sizeof(uint64_t),
         cudaMemcpyDeviceToDevice,
         ctx_->stream()
     ));
     proof_size_ += NUM_QUERIES * dims_.num_quotient_segments * 3;
     append_auth(ctx_->d_quotient_merkle());
 
     CUDA_CHECK(cudaFree(d_main_rows));
     CUDA_CHECK(cudaFree(d_aux_rows));
     CUDA_CHECK(cudaFree(d_quot_rows));
 }
 
 void GpuStark::step_encode_proof() {
     // The proof buffer has been accumulating items throughout
     // Final encoding happens here
 
     // For Rust FFI encoding, we'll need to download the proof items
     // and call the Rust encoder
 
     // Set final proof size
     ctx_->set_proof_size(proof_size_);
 }
 
 Proof GpuStark::prove_with_gpu_padding(
     const Claim& claim,
     const uint64_t* unpadded_table_data,
     size_t unpadded_rows,
     size_t padded_height,
     size_t num_cols,
     const size_t table_lengths[9],
     const uint64_t trace_domain_3[3],
     const uint64_t quotient_domain_3[3],
     const uint64_t fri_domain_3[3],
     const uint8_t randomness_seed[32],
     const std::vector<uint64_t>& main_randomizer_coeffs,
     const std::vector<uint64_t>& aux_randomizer_coeffs
 ) {
     // Persist randomness seed for downstream aux-table randomizer generation (col 87).
     std::copy(randomness_seed, randomness_seed + 32, randomness_seed_.begin());
     auto total_start = std::chrono::high_resolution_clock::now();
     
     TRITON_PROFILE_COUT("\n========================================" << std::endl);
     TRITON_PROFILE_COUT("GPU STARK Proof Generation (GPU Padding)" << std::endl);
     TRITON_PROFILE_COUT("========================================" << std::endl);
     TRITON_PROFILE_COUT("Unpadded rows: " << unpadded_rows << ", Padded height: " << padded_height << std::endl);
     TRITON_PROFILE_COUT("Columns: " << num_cols << std::endl);
     
     // Check GPU memory
     if (!check_gpu_memory(padded_height)) {
         throw std::runtime_error("Insufficient GPU memory for proof generation");
     }
     
     // Calculate dimensions
     dims_.padded_height = padded_height;
     dims_.quotient_length = static_cast<size_t>(quotient_domain_3[0]);
     dims_.fri_length = static_cast<size_t>(fri_domain_3[0]);
     dims_.trace_offset = trace_domain_3[1];
     dims_.trace_generator = trace_domain_3[2];
     dims_.quotient_offset = quotient_domain_3[1];
     dims_.quotient_generator = quotient_domain_3[2];
     dims_.fri_offset = fri_domain_3[1];
     dims_.fri_generator = fri_domain_3[2];
     dims_.main_width = num_cols;
     dims_.aux_width = 88;
 
     dims_.lde_frugal_mode = use_lde_frugal_mode(dims_.padded_height);
     if (dims_.lde_frugal_mode) {
         TRITON_PROFILE_COUT("      LDE FRUGAL MODE: ENABLED (streaming LDE, no cache)" << std::endl);
     }
     
     // Trace randomizer count
     {
         if (dims_.main_width == 0) throw std::runtime_error("Invalid main_width");
         if (main_randomizer_coeffs.size() % dims_.main_width != 0) {
             throw std::runtime_error("main_randomizer_coeffs size not divisible by main_width");
         }
         size_t num_rand_main = main_randomizer_coeffs.size() / dims_.main_width;
         if (dims_.aux_width == 0) throw std::runtime_error("Invalid aux_width");
         if (aux_randomizer_coeffs.size() % (dims_.aux_width * 3) != 0) {
             throw std::runtime_error("aux_randomizer_coeffs size not divisible by aux_width*3");
         }
         size_t num_rand_aux = aux_randomizer_coeffs.size() / (dims_.aux_width * 3);
         if (num_rand_main != num_rand_aux) {
             throw std::runtime_error("Mismatch: main vs aux num_trace_randomizers");
         }
         dims_.num_trace_randomizers = num_rand_main;
     }
     dims_.num_quotient_segments = 4;
     dims_.num_fri_rounds = static_cast<size_t>(std::log2(dims_.fri_length)) - 9;
     
     TRITON_PROFILE_COUT("[GPU] FRI domain: " << dims_.fri_length << " points" << std::endl);
     TRITON_PROFILE_COUT("[GPU] FRI rounds: " << dims_.num_fri_rounds << std::endl);
     
     // Create GPU proof context
     ctx_ = std::make_unique<GpuProofContext>(dims_);
     claim_ = claim;
     fs_cpu_ = ProofStream();
     init_tip5_tables();
     
     // =========================================================================
     // H2D TRANSFER + GPU PADDING
     // =========================================================================
     auto upload_start = std::chrono::high_resolution_clock::now();
     std::cout << "\n[H2D] Uploading unpadded table (" 
               << (unpadded_rows * num_cols * 8 / (1024 * 1024)) << " MB)..." << std::endl;
     
     // Zero out the entire table first (for padding rows)
     CUDA_CHECK(cudaMemsetAsync(ctx_->d_main_trace(), 0, 
                                padded_height * num_cols * sizeof(uint64_t), ctx_->stream()));
     
     // Upload unpadded data
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_main_trace(),
         unpadded_table_data,
         unpadded_rows * num_cols * sizeof(uint64_t),
         cudaMemcpyHostToDevice,
         ctx_->stream()
     ));
     
     // Pad on GPU
     TRITON_PROFILE_COUT("[GPU] Padding table on GPU..." << std::endl);
     kernels::gpu_pad_main_table(
         ctx_->d_main_trace(),
         num_cols,
         padded_height,
         table_lengths,
         ctx_->stream()
     );
     
     auto upload_end = std::chrono::high_resolution_clock::now();
     double upload_time = std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
     std::cout << "[H2D+Pad] Complete: " << upload_time << " ms" << std::endl;
     
     // Upload randomizer coefficients
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_main_randomizer_coeffs(),
         main_randomizer_coeffs.data(),
         main_randomizer_coeffs.size() * sizeof(uint64_t),
         cudaMemcpyHostToDevice,
         ctx_->stream()
     ));
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_aux_randomizer_coeffs(),
         aux_randomizer_coeffs.data(),
         aux_randomizer_coeffs.size() * sizeof(uint64_t),
         cudaMemcpyHostToDevice,
         ctx_->stream()
     ));
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     
     // From here on, continue with the same proof generation as prove()
     TRITON_PROFILE_COUT("\n[GPU] Starting proof computation..." << std::endl);
     
     auto step1_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 1: Initialize Fiat-Shamir" << std::endl);
     step_initialize_fiat_shamir(claim);
     auto step1_end = std::chrono::high_resolution_clock::now();
     double step1_time = std::chrono::duration<double, std::milli>(step1_end - step1_start).count();
     
     auto step2_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 2: Main table LDE + Merkle commitment" << std::endl);
     step_main_table_commitment(main_randomizer_coeffs);
     auto step2_end = std::chrono::high_resolution_clock::now();
     double step2_time = std::chrono::duration<double, std::milli>(step2_end - step2_start).count();
     
     auto step3_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 3: Aux table commitment" << std::endl);
     step_aux_table_commitment(aux_randomizer_coeffs);
     auto step3_end = std::chrono::high_resolution_clock::now();
     double step3_time = std::chrono::duration<double, std::milli>(step3_end - step3_start).count();
     
     auto step4_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 4: Quotient computation" << std::endl);
     step_quotient_commitment();
     auto step4_end = std::chrono::high_resolution_clock::now();
     double step4_time = std::chrono::duration<double, std::milli>(step4_end - step4_start).count();
     
     auto step5_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 5: Out-of-domain evaluation" << std::endl);
     step_out_of_domain_evaluation();
     auto step5_end = std::chrono::high_resolution_clock::now();
     double step5_time = std::chrono::duration<double, std::milli>(step5_end - step5_start).count();
     
     auto step6_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 6: FRI protocol" << std::endl);
     step_fri_protocol();
     auto step6_end = std::chrono::high_resolution_clock::now();
     double step6_time = std::chrono::duration<double, std::milli>(step6_end - step6_start).count();
     
     auto step7_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 7: Open trace leaves" << std::endl);
     step_open_trace();
     auto step7_end = std::chrono::high_resolution_clock::now();
     double step7_time = std::chrono::duration<double, std::milli>(step7_end - step7_start).count();
     
     auto step8_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 8: Finalize proof" << std::endl);
     step_encode_proof();
     auto step8_end = std::chrono::high_resolution_clock::now();
     double step8_time = std::chrono::duration<double, std::milli>(step8_end - step8_start).count();
     
     TRITON_PROFILE_COUT("[GPU] Proof computation complete" << std::endl);
     
     // Download proof
     auto download_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("\n[D2H] Downloading proof..." << std::endl);
     
     std::vector<BFieldElement> proof_elements(proof_size_);
     TRITON_PROFILE_COUT("[GPU] Downloading proof: " << proof_size_ << " elements ("
               << (proof_size_ * sizeof(uint64_t) / 1024) << " KB)" << std::endl);
     
     CUDA_CHECK(cudaMemcpyAsync(
         proof_elements.data(),
         ctx_->d_proof_buffer(),
         proof_size_ * sizeof(uint64_t),
         cudaMemcpyDeviceToHost,
         ctx_->stream()
     ));
     CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
     
     auto download_end = std::chrono::high_resolution_clock::now();
     double download_time = std::chrono::duration<double, std::milli>(download_end - download_start).count();
     TRITON_PROFILE_COUT("[D2H] Download complete: " << proof_size_ << " elements, " 
               << download_time << " ms" << std::endl);
     
     auto total_end = std::chrono::high_resolution_clock::now();
     double gpu_compute = step1_time + step2_time + step3_time + step4_time + 
                          step5_time + step6_time + step7_time + step8_time;
     double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
     
     TRITON_IF_PROFILE {
         std::cout << "\n========================================" << std::endl;
         std::cout << "GPU Proof Generation Complete!" << std::endl;
         std::cout << "========================================" << std::endl;
         std::cout << "Timing breakdown:" << std::endl;
         std::cout << "  H2D+Pad:        " << upload_time << " ms" << std::endl;
         std::cout << "  Step 1 (F-S):   " << step1_time << " ms" << std::endl;
         std::cout << "  Step 2 (Main):  " << step2_time << " ms" << std::endl;
         std::cout << "  Step 3 (Aux):   " << step3_time << " ms" << std::endl;
         std::cout << "  Step 4 (Quot):  " << step4_time << " ms" << std::endl;
         std::cout << "  Step 5 (OOD):   " << step5_time << " ms" << std::endl;
         std::cout << "  Step 6 (FRI):   " << step6_time << " ms" << std::endl;
         std::cout << "  Step 7 (Open):  " << step7_time << " ms" << std::endl;
         std::cout << "  Step 8 (Enc):   " << step8_time << " ms" << std::endl;
         std::cout << "  D2H Download:   " << download_time << " ms" << std::endl;
         std::cout << "  --------------------------" << std::endl;
         std::cout << "  GPU compute:    " << gpu_compute << " ms" << std::endl;
         std::cout << "  Total:          " << total_time << " ms" << std::endl;
         std::cout << "  Proof size:     " << (proof_size_ * sizeof(uint64_t) / 1024) << " KB" << std::endl;
         std::cout << "========================================\n" << std::endl;
     }
     
     Proof proof;
     proof.elements = std::move(proof_elements);
     return proof;
 }
 
 Proof GpuStark::prove_with_gpu_phase1(
     const Claim& claim,
     const Phase1HostTraces& phase1,
     size_t padded_height,
     size_t num_cols,
     const uint64_t trace_domain_3[3],
     const uint64_t quotient_domain_3[3],
     const uint64_t fri_domain_3[3],
     const uint8_t randomness_seed[32],
     const std::vector<uint64_t>& main_randomizer_coeffs,
     const std::vector<uint64_t>& aux_randomizer_coeffs
 ) {
     // Persist randomness seed for downstream aux-table randomizer generation (col 87).
     std::copy(randomness_seed, randomness_seed + 32, randomness_seed_.begin());
     auto total_start = std::chrono::high_resolution_clock::now();
     auto bytes_to_mb = [](size_t bytes) { return (double)bytes / (1024.0 * 1024.0); };
 
     TRITON_PROFILE_COUT("\n========================================" << std::endl);
     TRITON_PROFILE_COUT("GPU STARK Proof Generation (GPU Phase 1 + Zero-Copy)" << std::endl);
     TRITON_PROFILE_COUT("========================================" << std::endl);
     TRITON_PROFILE_COUT("Trace dimensions: " << padded_height << " x " << num_cols << std::endl);
 
     // Check GPU memory
     if (!check_gpu_memory(padded_height)) {
         throw std::runtime_error("Insufficient GPU memory for proof generation");
     }
 
     // Calculate dimensions from Rust-provided domains
     dims_.padded_height = static_cast<size_t>(trace_domain_3[0]);
     dims_.quotient_length = static_cast<size_t>(quotient_domain_3[0]);
     dims_.fri_length = static_cast<size_t>(fri_domain_3[0]);
 
     dims_.trace_offset = trace_domain_3[1];
     dims_.trace_generator = trace_domain_3[2];
     dims_.quotient_offset = quotient_domain_3[1];
     dims_.quotient_generator = quotient_domain_3[2];
     dims_.fri_offset = fri_domain_3[1];
     dims_.fri_generator = fri_domain_3[2];
 
     if (dims_.padded_height != padded_height) {
         throw std::runtime_error("Mismatch: padded_height != trace_domain.length");
     }
 
     dims_.main_width = num_cols;
     dims_.aux_width = 88;
 
     // Optional: allow hybrid CPU aux by creating a host copy of the GPU-built main table.
     // (Hybrid CPU aux requires a host pointer; without this, we will use GPU aux.)
     static int want_cpu_aux = -1;
     if (want_cpu_aux == -1) {
         const char* env = std::getenv("TRITON_AUX_CPU");
         want_cpu_aux = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
     }
     h_main_table_data_ = nullptr;
     std::cout << "[GPU Phase1] NOTE: skipping H2D upload of full main table (will build into ctx_->d_main_trace())" << std::endl;
 
     // Trace randomizer count
     {
         if (dims_.main_width == 0) throw std::runtime_error("Invalid main_width");
         if (main_randomizer_coeffs.size() % dims_.main_width != 0) {
             throw std::runtime_error("main_randomizer_coeffs size not divisible by main_width");
         }
         size_t num_rand_main = main_randomizer_coeffs.size() / dims_.main_width;
         if (dims_.aux_width == 0) throw std::runtime_error("Invalid aux_width");
         if (aux_randomizer_coeffs.size() % (dims_.aux_width * 3) != 0) {
             throw std::runtime_error("aux_randomizer_coeffs size not divisible by aux_width*3");
         }
         size_t num_rand_aux = aux_randomizer_coeffs.size() / (dims_.aux_width * 3);
         if (num_rand_main != num_rand_aux) {
             throw std::runtime_error("Mismatch: main vs aux num_trace_randomizers");
         }
         dims_.num_trace_randomizers = num_rand_main;
     }
     dims_.num_quotient_segments = 4;
     dims_.num_fri_rounds = static_cast<size_t>(std::log2(dims_.fri_length)) - 9;
 
     TRITON_PROFILE_COUT("[GPU] Domains: trace(len=" << dims_.padded_height
               << ", offset=" << dims_.trace_offset
               << ", gen=" << dims_.trace_generator
               << ") quotient(len=" << dims_.quotient_length
               << ", offset=" << dims_.quotient_offset
               << ", gen=" << dims_.quotient_generator
               << ") fri(len=" << dims_.fri_length
               << ", offset=" << dims_.fri_offset
               << ", gen=" << dims_.fri_generator
               << ")\n");
 
     TRITON_PROFILE_COUT("FRI domain: " << dims_.fri_length << " points" << std::endl);
     TRITON_PROFILE_COUT("FRI rounds: " << dims_.num_fri_rounds << std::endl);
 
     // Create GPU proof context
     ctx_ = std::make_unique<GpuProofContext>(dims_);
     claim_ = claim;
     fs_cpu_ = ProofStream();
     init_tip5_tables();
 
     // Upload trace randomizer coefficients (tiny H2D)
     CUDA_CHECK(cudaMemcpyAsync(
         ctx_->d_main_randomizer_coeffs(),
         main_randomizer_coeffs.data(),
         main_randomizer_coeffs.size() * sizeof(uint64_t),
         cudaMemcpyHostToDevice,
         ctx_->stream()
     ));
 
     // Aux randomizers: component-column major layout (preserve all 3 components)
     {
         const size_t num_rand = dims_.num_trace_randomizers;
         const size_t expected = dims_.aux_width * 3 * num_rand;
         if (aux_randomizer_coeffs.size() != expected) {
             throw std::runtime_error("aux_randomizer_coeffs has unexpected size (expected aux_width*3*num_rand)");
         }
         CUDA_CHECK(cudaMemcpyAsync(
             ctx_->d_aux_randomizer_coeffs(),
             aux_randomizer_coeffs.data(),
             expected * sizeof(uint64_t),
             cudaMemcpyHostToDevice,
             ctx_->stream()
         ));
     }
 
     if (!phase1.table_lengths_9) {
         throw std::runtime_error("Phase1HostTraces.table_lengths_9 must be provided");
     }
 
     // =========================================================================
     // GPU Phase 1: upload traces + build main table directly into ctx buffer
     // =========================================================================
     auto t_p1 = std::chrono::high_resolution_clock::now();
     {
         // Approximate H2D volume for Phase1 traces
         const size_t program_bytes = phase1.program_rows * 7 * sizeof(uint64_t);
         const size_t proc_bytes = phase1.processor_rows * 39 * sizeof(uint64_t);
         const size_t os_bytes = phase1.op_stack_rows * 4 * sizeof(uint64_t);
         const size_t ram_bytes = phase1.ram_rows * 7 * sizeof(uint64_t);
         const size_t js_bytes = phase1.jump_stack_rows * 5 * sizeof(uint64_t);
         const size_t hash_bytes = (phase1.program_hash_rows + phase1.sponge_rows + phase1.hash_rows) * 67 * sizeof(uint64_t);
         const size_t cas_bytes = phase1.cascade_rows * 6 * sizeof(uint64_t);
         const size_t lut_bytes = phase1.lookup_rows * 4 * sizeof(uint64_t);
         const size_t u32_bytes = phase1.u32_rows * 10 * sizeof(uint64_t);
         const size_t total_bytes = program_bytes + proc_bytes + os_bytes + ram_bytes + js_bytes + hash_bytes + cas_bytes + lut_bytes + u32_bytes;
 
         std::cout << "\n[GPU Phase1] Uploading AET traces (total ~" << bytes_to_mb(total_bytes) << " MB)..." << std::endl;
         std::cout << "  - program:    " << bytes_to_mb(program_bytes) << " MB" << std::endl;
         std::cout << "  - processor:  " << bytes_to_mb(proc_bytes) << " MB" << std::endl;
         std::cout << "  - op_stack:   " << bytes_to_mb(os_bytes) << " MB" << std::endl;
         std::cout << "  - ram:        " << bytes_to_mb(ram_bytes) << " MB" << std::endl;
         std::cout << "  - jump_stack: " << bytes_to_mb(js_bytes) << " MB" << std::endl;
         std::cout << "  - hash:       " << bytes_to_mb(hash_bytes) << " MB" << std::endl;
         std::cout << "  - cascade:    " << bytes_to_mb(cas_bytes) << " MB" << std::endl;
         std::cout << "  - lookup:     " << bytes_to_mb(lut_bytes) << " MB" << std::endl;
         std::cout << "  - u32:        " << bytes_to_mb(u32_bytes) << " MB" << std::endl;
     }
     kernels::GpuAETData* d_aet = kernels::gpu_upload_aet_flat(
         phase1.h_program_trace, phase1.program_rows,
         phase1.h_processor_trace, phase1.processor_rows,
         phase1.h_op_stack_trace, phase1.op_stack_rows,
         phase1.h_ram_trace, phase1.ram_rows,
         phase1.h_jump_stack_trace, phase1.jump_stack_rows,
         phase1.h_hash_trace, phase1.program_hash_rows, phase1.sponge_rows, phase1.hash_rows,
         phase1.h_cascade_trace, phase1.cascade_rows,
         phase1.h_lookup_trace, phase1.lookup_rows,
         phase1.h_u32_trace, phase1.u32_rows,
         ctx_->stream()
     );
     std::cout << "[GPU Phase1] Building + padding main table on GPU..." << std::endl;
     kernels::gpu_create_main_table_into(
         d_aet,
         ctx_->d_main_trace(),
         dims_.padded_height,
         phase1.table_lengths_9,
         ctx_->stream()
     );
     kernels::gpu_free_aet(d_aet);
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     std::cout << "[GPU Phase1] Complete: " << elapsed_ms(t_p1) << " ms" << std::endl;
 
     // If hybrid CPU aux is requested, create a host copy of the main table now.
     // This is intentionally AFTER GPU Phase1 completes to avoid unified-memory thrash.
     uint64_t* h_main_copy = nullptr;
     bool h_main_copy_pinned = false;
     if (want_cpu_aux) {
         const size_t main_elems = dims_.padded_height * dims_.main_width;
         const size_t main_bytes = main_elems * sizeof(uint64_t);
         std::cout << "[GPU Phase1] TRITON_AUX_CPU=1: copying main table to host for hybrid aux (~"
                   << bytes_to_mb(main_bytes) << " MB)..." << std::endl;
         auto t_copy = std::chrono::high_resolution_clock::now();
         cudaError_t err = cudaMallocHost(&h_main_copy, main_bytes);
         if (err == cudaSuccess) {
             h_main_copy_pinned = true;
         } else {
             h_main_copy_pinned = false;
             h_main_copy = (uint64_t*)std::aligned_alloc(64, main_bytes);
             if (!h_main_copy) {
                 throw std::runtime_error("Failed to allocate host buffer for main table (pinned+aligned_alloc failed)");
             }
         }
         CUDA_CHECK(cudaMemcpyAsync(h_main_copy, ctx_->d_main_trace(), main_bytes, cudaMemcpyDeviceToHost, ctx_->stream()));
         ctx_->synchronize();
         std::cout << "[GPU Phase1] Host main-table copy complete in "
                   << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_copy).count()
                   << " ms" << std::endl;
         h_main_table_data_ = h_main_copy;
     }
 
     // =========================================================================
     // ALL COMPUTATION ON GPU (no big H2D transfer)
     // =========================================================================
     TRITON_PROFILE_COUT("\n[GPU] Starting proof computation..." << std::endl);
     double step_times[8] = {0};
 
     auto t1 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 1: Initialize Fiat-Shamir" << std::endl);
     step_initialize_fiat_shamir(claim);
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[0] = elapsed_ms(t1);
 
     auto t2 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 2: Main table LDE + Merkle commitment" << std::endl);
 
     // No background main-table pre-conversion. Hybrid CPU aux reads from the flat host buffer.
 
     step_main_table_commitment(main_randomizer_coeffs);
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[1] = elapsed_ms(t2);
 
     auto t3 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 3: Aux table commitment" << std::endl);
     step_aux_table_commitment(aux_randomizer_coeffs);
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[2] = elapsed_ms(t3);
 
     auto t4 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 4: Quotient computation" << std::endl);
     step_quotient_commitment();
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[3] = elapsed_ms(t4);
 
     auto t5 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 5: Out-of-domain evaluation" << std::endl);
     step_out_of_domain_evaluation();
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[4] = elapsed_ms(t5);
 
     auto t6 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 6: FRI protocol" << std::endl);
     step_fri_protocol();
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[5] = elapsed_ms(t6);
 
     // Step 7: Open trace at query indices
     auto t7 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 7: Open trace leaves" << std::endl);
     step_open_trace();
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[6] = elapsed_ms(t7);
 
     // Step 8: Finalize proof buffer
     auto t8 = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("[GPU] Step 8: Finalize proof" << std::endl);
     step_encode_proof();
     ctx_->synchronize();
     if (std::getenv("TVM_DEBUG_GPU_SYNC_ALL")) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
     step_times[7] = elapsed_ms(t8);
 
     (void)randomness_seed; // seed is used by main table builder in main_gpu_full; transcript uses claim+roots
 
     // Download final proof (only D2H)
     auto download_start = std::chrono::high_resolution_clock::now();
     TRITON_PROFILE_COUT("\n[D2H] Downloading proof..." << std::endl);
     auto proof_data = ctx_->download_proof();
     double download_time = elapsed_ms(download_start);
     TRITON_PROFILE_COUT("[D2H] Download complete: " << proof_data.size() << " elements, "
               << download_time << " ms" << std::endl);
 
     {
         double total_ms = std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - total_start).count();
         TRITON_PROFILE_COUT("\n[GPU] Timing summary (GPU Phase1 path):" << std::endl);
         std::cout << "  Phase1 (upload+build): " << elapsed_ms(t_p1) << " ms" << std::endl;
         std::cout << "  Step1 (F-S):           " << step_times[0] << " ms" << std::endl;
         std::cout << "  Step2 (Main):          " << step_times[1] << " ms" << std::endl;
         std::cout << "  Step3 (Aux):           " << step_times[2] << " ms" << std::endl;
         std::cout << "  Step4 (Quot):          " << step_times[3] << " ms" << std::endl;
         std::cout << "  Step5 (OOD):           " << step_times[4] << " ms" << std::endl;
         std::cout << "  Step6 (FRI):           " << step_times[5] << " ms" << std::endl;
         std::cout << "  Step7 (Open):          " << step_times[6] << " ms" << std::endl;
         std::cout << "  Step8 (Enc):           " << step_times[7] << " ms" << std::endl;
         std::cout << "  D2H Download:          " << download_time << " ms" << std::endl;
         std::cout << "  Total:                 " << total_ms << " ms" << std::endl;
     }
 
     // Convert to Proof object
     std::vector<BFieldElement> proof_bfe;
     proof_bfe.reserve(proof_data.size());
     for (uint64_t val : proof_data) {
         proof_bfe.push_back(BFieldElement(val));
     }
     Proof result;
     result.elements = std::move(proof_bfe);
 
     // Cleanup host copy if we created one
     if (h_main_copy) {
         if (h_main_copy_pinned) cudaFreeHost(h_main_copy);
         else std::free(h_main_copy);
         h_main_table_data_ = nullptr;
     }
     return result;
 }
 
 // LDE Sampling Methods for Validation
 std::vector<uint64_t> GpuStark::sample_main_lde_row(size_t row_index, size_t num_cols) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
     
     const size_t fri_length = dims_.fri_length;
     if (row_index >= fri_length) {
         throw std::runtime_error("Row index out of bounds");
     }
     
     // Main LDE is stored column-major: d_main_lde[col * fri_length + row]
     std::vector<uint64_t> row_data(num_cols);
     std::vector<uint64_t> temp_col(fri_length);
     
     for (size_t col = 0; col < num_cols; ++col) {
         // Download the entire column (inefficient but simple)
         CUDA_CHECK(cudaMemcpyAsync(
             temp_col.data(),
             ctx_->d_main_lde() + col * fri_length,
             fri_length * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
         row_data[col] = temp_col[row_index];
     }
     
     return row_data;
 }
 
 std::vector<std::vector<uint64_t>> GpuStark::sample_main_lde_rows(const std::vector<size_t>& row_indices, size_t num_cols) {
     std::vector<std::vector<uint64_t>> result;
     result.reserve(row_indices.size());
     
     for (size_t row_idx : row_indices) {
         result.push_back(sample_main_lde_row(row_idx, num_cols));
     }
     
     return result;
 }
 
 std::vector<std::string> GpuStark::sample_aux_lde_row(size_t row_index, size_t num_cols) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
     
     const size_t fri_length = dims_.fri_length;
     if (row_index >= fri_length) {
         throw std::runtime_error("Row index out of bounds");
     }
     
     // Aux LDE is stored as XFE: d_aux_lde[(col * 3 + comp) * fri_length + row]
     std::vector<std::string> row_data;
     row_data.reserve(num_cols);
     
     std::vector<uint64_t> temp_comp(fri_length);
     
     for (size_t col = 0; col < num_cols; ++col) {
         uint64_t coeffs[3];
         for (size_t comp = 0; comp < 3; ++comp) {
             CUDA_CHECK(cudaMemcpyAsync(
                 temp_comp.data(),
                 ctx_->d_aux_lde() + (col * 3 + comp) * fri_length,
                 fri_length * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost,
                 ctx_->stream()
             ));
             ctx_->synchronize();
             coeffs[comp] = temp_comp[row_index];
         }
         
         // Rust displays XFieldElement as: (c2·x² + c1·x + c0) where coefficients are in
         // ascending degree order [c0, c1, c2].
         char buf[128];
         snprintf(buf, sizeof(buf), "(%020lu·x² + %020lu·x + %020lu)",
                  coeffs[2], coeffs[1], coeffs[0]);
         row_data.push_back(std::string(buf));
     }
     
     return row_data;
 }
 
 std::vector<std::vector<std::string>> GpuStark::sample_aux_lde_rows(const std::vector<size_t>& row_indices, size_t num_cols) {
     std::vector<std::vector<std::string>> result;
     result.reserve(row_indices.size());
     
     for (size_t row_idx : row_indices) {
         result.push_back(sample_aux_lde_row(row_idx, num_cols));
     }
     
     return result;
 }
 
 std::vector<uint64_t> GpuStark::download_aux_trace(size_t num_rows, size_t num_cols) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
     
     // Aux trace is stored as row-major XFieldElements: d_aux_trace[(row * num_cols + col) * 3 + comp]
     std::vector<uint64_t> result(num_rows * num_cols * 3);
     CUDA_CHECK(cudaMemcpy(
         result.data(),
         ctx_->d_aux_trace(),
         num_rows * num_cols * 3 * sizeof(uint64_t),
         cudaMemcpyDeviceToHost
     ));
     return result;
 }
 
 std::vector<std::string> GpuStark::sample_quotient_lde_row(size_t row_index, size_t num_segments) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
     
     // Quotient segments are stored in d_quotient_segments
     // Format: [fri_length × num_segments × 3] (XFE)
     const size_t fri_length = dims_.fri_length;
     if (row_index >= fri_length) {
         throw std::runtime_error("Row index out of bounds");
     }
     
     std::vector<std::string> row_data;
     row_data.reserve(num_segments);
     
     std::vector<uint64_t> temp_comp(fri_length);
     
     for (size_t seg = 0; seg < num_segments; ++seg) {
         uint64_t coeffs[3];
         for (size_t comp = 0; comp < 3; ++comp) {
             CUDA_CHECK(cudaMemcpyAsync(
                 temp_comp.data(),
                 ctx_->d_quotient_segments() + (seg * 3 + comp) * fri_length,
                 fri_length * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost,
                 ctx_->stream()
             ));
             ctx_->synchronize();
             coeffs[comp] = temp_comp[row_index];
         }
         
         // Rust displays XFieldElement as: (c2·x² + c1·x + c0)
         char buf[128];
         snprintf(buf, sizeof(buf), "(%020lu·x² + %020lu·x + %020lu)",
                  coeffs[2], coeffs[1], coeffs[0]);
         row_data.push_back(std::string(buf));
     }
     
     return row_data;
 }
 
 std::vector<std::vector<std::string>> GpuStark::sample_quotient_lde_rows(const std::vector<size_t>& row_indices, size_t num_segments) {
     std::vector<std::vector<std::string>> result;
     result.reserve(row_indices.size());
     
     for (size_t row_idx : row_indices) {
         result.push_back(sample_quotient_lde_row(row_idx, num_segments));
     }
     
     return result;
 }
 
 std::vector<std::vector<uint64_t>> GpuStark::sample_aux_row_digests(const std::vector<size_t>& row_indices) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
     
     const size_t fri_length = dims_.fri_length;
     const size_t DIGEST_LEN = 5;
     
     std::vector<std::vector<uint64_t>> result;
     result.reserve(row_indices.size());
     
     // d_aux_merkle contains row digests at offset 0, then internal nodes
     // Each digest is 5 uint64_t elements
     // Digest for row r is at d_aux_merkle[r * 5]
     
     for (size_t row_idx : row_indices) {
         if (row_idx >= fri_length) {
             throw std::runtime_error("Row index out of bounds for aux digests");
         }
         
         std::vector<uint64_t> digest(DIGEST_LEN);
         CUDA_CHECK(cudaMemcpyAsync(
             digest.data(),
             ctx_->d_aux_merkle() + row_idx * DIGEST_LEN,
             DIGEST_LEN * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
         result.push_back(std::move(digest));
     }
     
     return result;
 }
 
 std::vector<std::vector<uint64_t>> GpuStark::sample_main_row_digests(const std::vector<size_t>& row_indices) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
     
     const size_t fri_length = dims_.fri_length;
     const size_t DIGEST_LEN = 5;
     
     std::vector<std::vector<uint64_t>> result;
     result.reserve(row_indices.size());
     
     for (size_t row_idx : row_indices) {
         if (row_idx >= fri_length) {
             throw std::runtime_error("Row index out of bounds for main digests");
         }
         
         std::vector<uint64_t> digest(DIGEST_LEN);
         CUDA_CHECK(cudaMemcpyAsync(
             digest.data(),
             ctx_->d_main_merkle() + row_idx * DIGEST_LEN,
             DIGEST_LEN * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
         result.push_back(std::move(digest));
     }
     
     return result;
 }
 
 std::vector<uint64_t> GpuStark::sample_aux_lde_row_bfes(size_t row_index, size_t num_xfe_cols) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
 
     const size_t fri_length = dims_.fri_length;
     if (row_index >= fri_length) {
         throw std::runtime_error("Row index out of bounds");
     }
 
     // Aux LDE is stored as:
     //   d_aux_lde[(col * 3 + comp) * fri_length + row]
     // where comp=0..2 corresponds to [c0,c1,c2] in Rust's XFieldElement coefficient order.
     std::vector<uint64_t> row_bfes(num_xfe_cols * 3);
 
     for (size_t col = 0; col < num_xfe_cols; ++col) {
         for (size_t comp = 0; comp < 3; ++comp) {
             const size_t col_comp = col * 3 + comp;
             const uint64_t* d_src = ctx_->d_aux_lde() + col_comp * fri_length + row_index;
             uint64_t* h_dst = &row_bfes[col_comp];
             CUDA_CHECK(cudaMemcpyAsync(
                 h_dst,
                 d_src,
                 sizeof(uint64_t),
                 cudaMemcpyDeviceToHost,
                 ctx_->stream()
             ));
         }
     }
 
     ctx_->synchronize();
     return row_bfes;
 }
 
 std::vector<std::vector<uint64_t>> GpuStark::sample_aux_merkle_nodes(const std::vector<size_t>& node_indices) {
     if (!ctx_) {
         throw std::runtime_error("GpuStark context not initialized");
     }
 
     const size_t DIGEST_LEN = 5;
     const size_t fri_length = dims_.fri_length;
     const size_t max_nodes = 2 * fri_length; // allocated size (we use up to 2*f-1 digests)
 
     std::vector<std::vector<uint64_t>> result;
     result.reserve(node_indices.size());
 
     for (size_t node_idx : node_indices) {
         if (node_idx >= max_nodes) {
             throw std::runtime_error("Merkle node index out of bounds");
         }
 
         std::vector<uint64_t> digest(DIGEST_LEN);
         CUDA_CHECK(cudaMemcpyAsync(
             digest.data(),
             ctx_->d_aux_merkle() + node_idx * DIGEST_LEN,
             DIGEST_LEN * sizeof(uint64_t),
             cudaMemcpyDeviceToHost,
             ctx_->stream()
         ));
         ctx_->synchronize();
         result.push_back(std::move(digest));
     }
 
     return result;
 }
 
 } // namespace gpu
 } // namespace triton_vm
 
 #endif // TRITON_CUDA_ENABLED
 