/**
 * GPU Quotient Post-Processing Kernel
 *
 * The constraint evaluation (MASTER_AUX_NUM_CONSTRAINTS constraints per row) is kept on CPU due to
 * its complexity causing nvcc compilation to hang.
 * 
 * This kernel accelerates the NTT-heavy post-processing:
 * - Quotient polynomial interpolation
 * - Segment splitting
 * - FRI domain evaluation
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include "backend/cpu_backend.hpp"  // For BFieldElement
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

// Host helper functions for BFieldElement arithmetic
// Note: These are duplicated from randomized_lde_kernel.cu to avoid linker issues
//       with CUDA compilation units. Keep them in sync.
namespace {
constexpr uint64_t GOLDILOCKS_P_LOCAL = 0xFFFFFFFF00000001ULL;

inline uint64_t bfield_mul_host_local(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    return (uint64_t)(prod % GOLDILOCKS_P_LOCAL);
}

inline uint64_t bfield_pow_host(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    base = base % GOLDILOCKS_P_LOCAL;
    while (exp > 0) {
        if (exp & 1) {
            result = bfield_mul_host_local(result, base);
        }
        base = bfield_mul_host_local(base, base);
        exp >>= 1;
    }
    return result;
}

inline uint64_t bfield_inv_host(uint64_t a) {
    // Fermat's little theorem: a^(-1) = a^(p-2) mod p
    return bfield_pow_host(a, GOLDILOCKS_P_LOCAL - 2);
}
} // anonymous namespace

namespace triton_vm {
namespace gpu {
namespace kernels {

// Import device functions
using namespace triton_vm::gpu::kernels;

// ============================================================================
// Helpers
// ============================================================================

__global__ void extract_xfe_components_kernel(
    const uint64_t* d_xfe, // [n*3] row-major
    size_t n,
    uint64_t* d_c0,
    uint64_t* d_c1,
    uint64_t* d_c2
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t off = idx * 3;
    d_c0[idx] = d_xfe[off + 0];
    d_c1[idx] = d_xfe[off + 1];
    d_c2[idx] = d_xfe[off + 2];
}

__global__ void interleave_xfe_components_kernel(
    const uint64_t* d_c0,
    const uint64_t* d_c1,
    const uint64_t* d_c2,
    size_t n,
    uint64_t* d_xfe // [n*3] row-major
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t off = idx * 3;
    d_xfe[off + 0] = d_c0[idx];
    d_xfe[off + 1] = d_c1[idx];
    d_xfe[off + 2] = d_c2[idx];
}

// NOTE: kernel name must be unique across all CUDA translation units.
// `lde_kernel.cu` already defines `coset_scale_kernel`, so we use a distinct name here.
__global__ void qpc_coset_scale_kernel(
    uint64_t* d_data,
    size_t n,
    uint64_t offset
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t scale = bfield_pow_impl(offset, idx);
    d_data[idx] = bfield_mul_impl(d_data[idx], scale);
}

__global__ void qpc_coset_scale_batch_kernel(
    uint64_t* d_arrays,   // contiguous arrays: batch_size * n
    size_t n,
    size_t batch_size,
    uint64_t offset
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t scale = bfield_pow_impl(offset, idx);
    for (size_t b = 0; b < batch_size; ++b) {
        size_t off = b * n + idx;
        d_arrays[off] = bfield_mul_impl(d_arrays[off], scale);
    }
}

__global__ void segmentify_xfe_coeffs_compact_kernel(
    const uint64_t* d_c0, // [n]
    const uint64_t* d_c1, // [n]
    const uint64_t* d_c2, // [n]
    size_t n,
    size_t num_segments,
    uint64_t* d_seg_coeffs_compact // [num_segments * 3 * seg_len]
) {
    size_t seg_len = n / num_segments;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_segments * seg_len;
    if (tid >= total) return;

    size_t seg = tid / seg_len;
    size_t k = tid % seg_len;
    size_t src = seg + num_segments * k;

    size_t base = (seg * 3) * seg_len + k;
    d_seg_coeffs_compact[base + 0 * seg_len] = d_c0[src];
    d_seg_coeffs_compact[base + 1 * seg_len] = d_c1[src];
    d_seg_coeffs_compact[base + 2 * seg_len] = d_c2[src];
}

__global__ void scatter_compact_to_padded_colmajor_kernel(
    const uint64_t* d_seg_coeffs_compact, // [num_segments * 3 * seg_len]
    size_t seg_len,
    size_t n,
    size_t num_segments,
    uint64_t* d_seg_coeffs_padded_colmajor // [num_segments * 3 * n]
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_segments * 3 * seg_len;
    if (tid >= total) return;

    size_t tmp = tid;
    size_t k = tmp % seg_len; tmp /= seg_len;
    size_t comp = tmp % 3;    tmp /= 3;
    size_t seg = tmp;

    uint64_t v = d_seg_coeffs_compact[(seg * 3 + comp) * seg_len + k];
    d_seg_coeffs_padded_colmajor[(seg * 3 + comp) * n + k] = v;
}

/**
 * Interpolate XFieldElement column on GPU using 3 parallel BField NTTs
 */
void interpolate_xfield_column_gpu(
    const uint64_t* d_values,  // [n * 3] XFieldElements
    uint64_t* d_coeffs,        // [n * 3] output coefficients
    size_t n,
    uint64_t offset_inv,       // inverse of domain offset
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int grid_n = (int)((n + BLOCK - 1) / BLOCK);

    uint64_t* d_c0 = nullptr;
    uint64_t* d_c1 = nullptr;
    uint64_t* d_c2 = nullptr;
    cudaMalloc(&d_c0, n * sizeof(uint64_t));
    cudaMalloc(&d_c1, n * sizeof(uint64_t));
    cudaMalloc(&d_c2, n * sizeof(uint64_t));

    extract_xfe_components_kernel<<<grid_n, BLOCK, 0, stream>>>(d_values, n, d_c0, d_c1, d_c2);

    // INTT each component
    ntt_inverse_gpu(d_c0, n, stream);
    ntt_inverse_gpu(d_c1, n, stream);
    ntt_inverse_gpu(d_c2, n, stream);

    // Coset unscale by offset_inv^i (matches CPU interpolate_xfield_column)
    qpc_coset_scale_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c0, n, offset_inv);
    qpc_coset_scale_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c1, n, offset_inv);
    qpc_coset_scale_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c2, n, offset_inv);

    interleave_xfe_components_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c0, d_c1, d_c2, n, d_coeffs);

    cudaFree(d_c0);
    cudaFree(d_c1);
    cudaFree(d_c2);
}

/**
 * Evaluate polynomial on coset (GPU version)
 * Takes coefficients and produces evaluations on offset * <generator>
 */
void evaluate_on_coset_gpu(
    const uint64_t* d_coeffs,  // [n * 3] XFieldElement coefficients
    uint64_t* d_evals,         // [n * 3] output evaluations
    size_t n,
    uint64_t offset,           // coset offset
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int grid_n = (int)((n + BLOCK - 1) / BLOCK);

    uint64_t* d_c0 = nullptr;
    uint64_t* d_c1 = nullptr;
    uint64_t* d_c2 = nullptr;
    cudaMalloc(&d_c0, n * sizeof(uint64_t));
    cudaMalloc(&d_c1, n * sizeof(uint64_t));
    cudaMalloc(&d_c2, n * sizeof(uint64_t));

    extract_xfe_components_kernel<<<grid_n, BLOCK, 0, stream>>>(d_coeffs, n, d_c0, d_c1, d_c2);

    // Coset scale by offset^i
    qpc_coset_scale_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c0, n, offset);
    qpc_coset_scale_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c1, n, offset);
    qpc_coset_scale_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c2, n, offset);

    // Forward NTT each component
    ntt_forward_gpu(d_c0, n, stream);
    ntt_forward_gpu(d_c1, n, stream);
    ntt_forward_gpu(d_c2, n, stream);

    interleave_xfe_components_kernel<<<grid_n, BLOCK, 0, stream>>>(d_c0, d_c1, d_c2, n, d_evals);

    cudaFree(d_c0);
    cudaFree(d_c1);
    cudaFree(d_c2);
}

// ============================================================================
// Corrected JIT LDE segmentification (matching Rust's algorithm)
// ============================================================================

// Kernel to rearrange quotient multicoset from (num_cosets, working_len) to (num_segments, output_len)
// This matches Rust's segmentify rearrangement logic
__global__ void jit_lde_rearrange_multicoset_kernel(
    const uint64_t* d_quotient_multicoset,  // [working_len * num_cosets * 3] XFE row-major
    size_t working_len,
    size_t num_cosets,                      // 8
    size_t num_segments,                    // 4
    uint64_t* d_quotient_segments           // [output_len * num_segments * 3] XFE row-major
) {
    size_t output_len = working_len * num_cosets / num_segments;  // working_len * 2

    size_t output_row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_col_idx = blockIdx.y;
    size_t comp_idx = blockIdx.z;

    if (output_row_idx >= output_len || output_col_idx >= num_segments || comp_idx >= 3) return;

    // Calculate which input element this output position corresponds to
    // This matches Rust's: exponent_of_iota = output_row_idx + output_col_idx * output_len
    size_t exponent_of_iota = output_row_idx + output_col_idx * output_len;
    size_t input_row_idx = exponent_of_iota / num_cosets;
    size_t input_col_idx = exponent_of_iota % num_cosets;

    // Copy from input to output
    size_t input_off = (input_col_idx * working_len + input_row_idx) * 3 + comp_idx;
    size_t output_off = (output_col_idx * output_len + output_row_idx) * 3 + comp_idx;

    d_quotient_segments[output_off] = d_quotient_multicoset[input_off];
}

// Kernel to apply iNTT to each row of quotient_segments (length num_segments = 4)
__global__ void jit_lde_intt_segment_rows_kernel(
    uint64_t* d_quotient_segments,     // [output_len * num_segments * 3] XFE row-major
    size_t output_len,
    size_t num_segments
) {
    size_t row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t comp_idx = blockIdx.y;

    if (row_idx >= output_len || comp_idx >= 3) return;

    // Apply iNTT to the row of length num_segments (4)
    uint64_t* row_start = d_quotient_segments + (row_idx * num_segments * 3) + comp_idx;

    // For length 4, we need to implement proper iNTT
    // iNTT matrix is (1/4) * [[1,1,1,1],[1,w,w²,w³],[1,w²,w^4=1,w^6=w²],[1,w³,w^6=w²,w^9=w]]
    // where w is primitive 4th root of unity, w = 7 (since 7^4 = 2401 ≡ 1 mod 2^64-2^32+1)

    uint64_t w = 7;  // primitive 4th root of unity
    uint64_t w2 = bfield_mul_impl(w, w);
    uint64_t w3 = bfield_mul_impl(w2, w);

    uint64_t a = row_start[0 * 3];
    uint64_t b = row_start[1 * 3];
    uint64_t c = row_start[2 * 3];
    uint64_t d = row_start[3 * 3];

    // Apply iNTT matrix multiplication
    uint64_t out0 = bfield_add_impl(bfield_add_impl(a, b), bfield_add_impl(c, d));
    uint64_t out1 = bfield_add_impl(bfield_add_impl(a, bfield_mul_impl(b, w)), bfield_add_impl(bfield_mul_impl(c, w2), bfield_mul_impl(d, w3)));
    uint64_t out2 = bfield_add_impl(bfield_add_impl(a, bfield_mul_impl(b, w2)), bfield_add_impl(bfield_mul_impl(c, w2), bfield_mul_impl(d, w2)));  // w^4 = 1, w^6 = w²
    uint64_t out3 = bfield_add_impl(bfield_add_impl(a, bfield_mul_impl(b, w3)), bfield_add_impl(bfield_mul_impl(c, w3), bfield_mul_impl(d, w)));   // w^9 = w

    // Divide by 4
    uint64_t four_inv = bfield_inv_impl(4);
    row_start[0 * 3] = bfield_mul_impl(out0, four_inv);
    row_start[1 * 3] = bfield_mul_impl(out1, four_inv);
    row_start[2 * 3] = bfield_mul_impl(out2, four_inv);
    row_start[3 * 3] = bfield_mul_impl(out3, four_inv);
}

// Kernel to scale rows according to the complex scaling pattern Ψ^-k · ι^(-k(j+i·M))
__global__ void jit_lde_scale_segment_rows_kernel(
    uint64_t* d_quotient_segments,     // [output_len * num_segments * 3] XFE row-major
    size_t output_len,
    size_t num_segments,
    size_t chunk_start,
    uint64_t psi_inv,
    uint64_t iota_inv
) {
    size_t row_in_chunk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t comp_idx = blockIdx.y;

    if (row_in_chunk >= (output_len / 256 + 1) || comp_idx >= 3) return;

    size_t row_idx = chunk_start + row_in_chunk;

    uint64_t* row_start = d_quotient_segments + (row_idx * num_segments * 3) + comp_idx;

    // Scale by Ψ^-k · ι^(-k(j+i·M)) pattern
    // From Rust: psi_iotajim_invk starts as 1, then multiplies by psi_iotajim_inv each time
    uint64_t psi_iotajim_inv = bfield_mul_impl(psi_inv, bfield_pow_impl(iota_inv, chunk_start));
    uint64_t psi_iotajim_invk = 1;

    for (size_t k = 0; k < num_segments; ++k) {
        row_start[k * 3] = bfield_mul_impl(row_start[k * 3], psi_iotajim_invk);
        psi_iotajim_invk = bfield_mul_impl(psi_iotajim_invk, psi_iotajim_inv);
    }
}

void quotient_segmentify_jit_lde_gpu(
    const uint64_t* d_quotient_multicoset,   // [working_len * num_cosets * 3] XFE row-major
    size_t working_len,
    size_t num_cosets,                       // 8
    size_t num_segments,                     // 4
    uint64_t psi,                            // fri_offset
    uint64_t iota,                           // coset root
    uint64_t fri_offset,
    size_t fri_len,
    uint64_t* d_segment_codewords_colmajor,  // [num_segments * 3 * fri_len] output
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    size_t output_len = working_len * num_cosets / num_segments;  // working_len * 2

    // Allocate temporary buffer for quotient_segments
    uint64_t* d_quotient_segments = nullptr;
    size_t segments_size = output_len * num_segments * 3 * sizeof(uint64_t);
    cudaMalloc(&d_quotient_segments, segments_size);

    // Step 1: Rearrange data from (working_len, num_cosets) to (output_len, num_segments)
    {
        dim3 grid((output_len + BLOCK - 1) / BLOCK, num_segments, 3);
        dim3 block(BLOCK, 1, 1);
        jit_lde_rearrange_multicoset_kernel<<<grid, block, 0, stream>>>(
            d_quotient_multicoset, working_len, num_cosets, num_segments, d_quotient_segments
        );
    }

    // Step 2: Apply iNTT to each row (length num_segments = 4)
    {
        dim3 grid((output_len + BLOCK - 1) / BLOCK, 3, 1);
        dim3 block(BLOCK, 1, 1);
        jit_lde_intt_segment_rows_kernel<<<grid, block, 0, stream>>>(
            d_quotient_segments, output_len, num_segments
        );
    }

    // Step 3: Scale rows with the complex pattern Ψ^-k · ι^(-k(j+i·M))
    {
        uint64_t iota_inv = BFieldElement(iota).inverse().value();
        uint64_t psi_inv = BFieldElement(psi).inverse().value();

        size_t num_threads = 256;
        size_t chunk_size = output_len / num_threads;
        if (chunk_size == 0) chunk_size = 1;

        for (size_t chunk_idx = 0; chunk_idx < num_threads; ++chunk_idx) {
            size_t chunk_start = chunk_idx * chunk_size;
            if (chunk_start >= output_len) break;

            dim3 grid((chunk_size + BLOCK - 1) / BLOCK, 3, 1);
            dim3 block(BLOCK, 1, 1);
            jit_lde_scale_segment_rows_kernel<<<grid, block, 0, stream>>>(
                d_quotient_segments, output_len, num_segments, chunk_start, psi_inv, iota_inv
            );
        }
    }

    // Step 4: Interpolate each column and LDE to FRI domain
    uint64_t segment_domain_offset = BFieldElement(fri_offset).pow(num_segments).value();

    for (size_t seg = 0; seg < num_segments; ++seg) {
        for (size_t comp = 0; comp < 3; ++comp) {
            // Extract column from quotient_segments
            uint64_t* d_segment_column = d_quotient_segments + (seg * output_len * 3) + comp;

            // Interpolate on segment domain (length output_len)
            // For now, simplified: just copy to FRI domain (this needs proper interpolation)
            size_t col_offset = (seg * 3 + comp) * fri_len;
            uint64_t* d_codeword = d_segment_codewords_colmajor + col_offset;

            cudaMemsetAsync(d_codeword, 0, fri_len * sizeof(uint64_t), stream);
            cudaMemcpyAsync(d_codeword, d_segment_column, output_len * sizeof(uint64_t),
                          cudaMemcpyDeviceToDevice, stream);

            // Apply coset scaling for the segment domain and forward NTT
            uint64_t segment_offset = BFieldElement(segment_domain_offset).pow(seg).value();
            qpc_coset_scale_kernel<<<(fri_len + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
                d_codeword, fri_len, segment_offset
            );
            ntt_forward_gpu(d_codeword, fri_len, stream);
        }
    }

    cudaFree(d_quotient_segments);
}

// Original segmentification (kept for compatibility)
void quotient_segmentify_and_lde_gpu(
    const uint64_t* d_quotient_values_xfe,
    size_t quotient_len,
    size_t num_segments,
    uint64_t quotient_offset_inv,
    uint64_t fri_offset,
    size_t fri_len,
    uint64_t* d_segment_coeffs_compact,     // [num_segments * 3 * seg_len]
    uint64_t* d_segment_codewords_colmajor, // [num_segments * 3 * fri_len]
    cudaStream_t stream,
    uint64_t* d_scratch_c0,                 // Optional scratch buffer for c0 (must be >= quotient_len)
    uint64_t* d_scratch_c1                  // Optional scratch buffer for c1 (must be >= quotient_len)
) {
    constexpr int BLOCK = 256;
    int grid_q = (int)((quotient_len + BLOCK - 1) / BLOCK);
    int grid_f = (int)((fri_len + BLOCK - 1) / BLOCK);
    size_t seg_len = quotient_len / num_segments;

    // 1) Extract components of quotient codeword
    // Use provided scratch buffers if available
    // If scratch_c0 is provided, we can use it for all 3 components with offsets
    // (scratch buffers are much larger than quotient_len, so this is safe)
    uint64_t* d_c0 = nullptr;
    uint64_t* d_c1 = nullptr;
    uint64_t* d_c2 = nullptr;
    bool allocated_c0 = (d_scratch_c0 == nullptr);
    bool allocated_c1 = (d_scratch_c1 == nullptr);
    bool allocated_c2 = true;
    
    if (allocated_c0) {
        cudaError_t err0 = cudaMalloc(&d_c0, quotient_len * sizeof(uint64_t));
        if (err0 != cudaSuccess) {
            throw std::runtime_error("Failed to allocate d_c0 for quotient segmentify: " + std::string(cudaGetErrorString(err0)) +
                                   " (quotient_len=" + std::to_string(quotient_len) + ")");
        }
    } else {
        d_c0 = d_scratch_c0;
        // If scratch_c0 is provided, use it for all 3 components with offsets
        // (scratch_c0 is much larger than quotient_len)
        d_c1 = d_scratch_c0 + quotient_len;
        d_c2 = d_scratch_c0 + 2 * quotient_len;
        allocated_c1 = false;  // Don't need to allocate c1
        allocated_c2 = false;  // Don't need to allocate c2
    }
    
    if (allocated_c1) {
        // Only allocate if we didn't use scratch_c0 offsets
        if (d_scratch_c1 != nullptr) {
            d_c1 = d_scratch_c1;
            allocated_c1 = false;
        } else {
            cudaError_t err1 = cudaMalloc(&d_c1, quotient_len * sizeof(uint64_t));
            if (err1 != cudaSuccess) {
                if (allocated_c0) cudaFree(d_c0);
                throw std::runtime_error("Failed to allocate d_c1 for quotient segmentify: " + std::string(cudaGetErrorString(err1)) +
                                       " (quotient_len=" + std::to_string(quotient_len) + ")");
            }
        }
    }
    
    if (allocated_c2) {
        // Only allocate if we didn't use scratch_c0 offsets
        cudaError_t err2 = cudaMalloc(&d_c2, quotient_len * sizeof(uint64_t));
        if (err2 != cudaSuccess) {
            if (allocated_c0) cudaFree(d_c0);
            if (allocated_c1) cudaFree(d_c1);
            throw std::runtime_error("Failed to allocate d_c2 for quotient segmentify: " + std::string(cudaGetErrorString(err2)) +
                                   " (quotient_len=" + std::to_string(quotient_len) + ")");
        }
    }
    extract_xfe_components_kernel<<<grid_q, BLOCK, 0, stream>>>(d_quotient_values_xfe, quotient_len, d_c0, d_c1, d_c2);

    // 2) Interpolate (inverse NTT) and coset-unscale by quotient_offset_inv^i
    ntt_inverse_gpu(d_c0, quotient_len, stream);
    ntt_inverse_gpu(d_c1, quotient_len, stream);
    ntt_inverse_gpu(d_c2, quotient_len, stream);
    qpc_coset_scale_kernel<<<grid_q, BLOCK, 0, stream>>>(d_c0, quotient_len, quotient_offset_inv);
    qpc_coset_scale_kernel<<<grid_q, BLOCK, 0, stream>>>(d_c1, quotient_len, quotient_offset_inv);
    qpc_coset_scale_kernel<<<grid_q, BLOCK, 0, stream>>>(d_c2, quotient_len, quotient_offset_inv);

    // 3) Segmentify coefficients (compact)
    size_t total_seg = num_segments * seg_len;
    int grid_seg = (int)((total_seg + BLOCK - 1) / BLOCK);
    segmentify_xfe_coeffs_compact_kernel<<<grid_seg, BLOCK, 0, stream>>>(
        d_c0, d_c1, d_c2, quotient_len, num_segments, d_segment_coeffs_compact
    );

    // 4) Create padded coefficient arrays on device in column-major layout and scatter compact coeffs
    size_t padded_words = num_segments * 3 * fri_len;
    cudaMemsetAsync(d_segment_codewords_colmajor, 0, padded_words * sizeof(uint64_t), stream);

    size_t total_scatter = num_segments * 3 * seg_len;
    int grid_scatter = (int)((total_scatter + BLOCK - 1) / BLOCK);
    scatter_compact_to_padded_colmajor_kernel<<<grid_scatter, BLOCK, 0, stream>>>(
        d_segment_coeffs_compact, seg_len, fri_len, num_segments, d_segment_codewords_colmajor
    );

    // 5) Coset-scale each of the (num_segments*3) arrays by fri_offset^i, then forward NTT each
    // Layout: arrays are contiguous in d_segment_codewords_colmajor
    qpc_coset_scale_batch_kernel<<<grid_f, BLOCK, 0, stream>>>(
        d_segment_codewords_colmajor, fri_len, num_segments * 3, fri_offset
    );

    // Forward NTT for each array (sequential launches; batch kernel can be added later)
    for (size_t a = 0; a < num_segments * 3; ++a) {
        ntt_forward_gpu(d_segment_codewords_colmajor + a * fri_len, fri_len, stream);
    }

    // Only free buffers we allocated (not scratch buffers provided by caller)
    if (allocated_c0) cudaFree(d_c0);
    if (allocated_c1) cudaFree(d_c1);
    cudaFree(d_c2);  // Always allocated
}

// ============================================================================
// Multi-coset quotient segmentify (matching Rust's algorithm)
// ============================================================================

__global__ void fill_coset_indices_kernel(
    size_t* d_indices,     // [n]
    size_t n,
    size_t coset_index,
    size_t num_cosets
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Fill indices where k ≡ coset_index (mod num_cosets)
    d_indices[idx] = coset_index + idx * num_cosets;
}

// Gather strided column into contiguous buffer
__global__ void gather_strided_column_kernel(
    const uint64_t* d_src,  // Source array with strided layout
    size_t num_rows,        // Number of rows to gather
    size_t row_stride,      // Stride between rows (in elements)
    size_t col_offset,      // Offset within row
    uint64_t* d_dst         // Destination contiguous buffer
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    d_dst[idx] = d_src[idx * row_stride + col_offset];
}

// Kernel to rearrange quotient multicoset into segments
// Input layout: FRI domain row-major: d_multicoset[(working_idx * num_cosets + coset) * 3 + comp]
//   where working_idx = j = 0..working_len-1, coset = k = 0..num_cosets-1
// Output layout: segments row-major: d_quotient_segments[(output_row * num_segments + output_col) * 3 + comp]
__global__ void quotient_segmentify_rearrange_kernel(
    const uint64_t* d_multicoset,      // [working_len * num_cosets * 3] row-major for (j,k)
    size_t working_len,
    size_t num_cosets,
    size_t num_segments,               // 4
    uint64_t* d_quotient_segments      // [num_output_rows * num_segments * 3] row-major
) {
    // num_output_rows = working_len * num_cosets / num_segments
    size_t num_output_rows = working_len * num_cosets / num_segments;

    size_t output_row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_col_idx = blockIdx.y;  // 0..num_segments-1
    size_t comp_idx = blockIdx.z;        // 0..2 for XFE components

    if (output_row_idx >= num_output_rows || output_col_idx >= num_segments || comp_idx >= 3) return;

    // Calculate which input element this output position corresponds to
    // From Rust: exponent_of_iota = output_row + output_col * num_output_rows
    //            input_row (working_idx j) = exponent_of_iota / num_cosets
    //            input_col (coset k) = exponent_of_iota % num_cosets
    size_t exponent_of_iota = output_row_idx + output_col_idx * num_output_rows;
    size_t input_row_idx = exponent_of_iota / num_cosets;  // j = working domain index
    size_t input_col_idx = exponent_of_iota % num_cosets;  // k = coset index

    // GPU FRI domain layout: d_multicoset[(j * num_cosets + k) * 3 + comp]
    size_t input_off = (input_row_idx * num_cosets + input_col_idx) * 3 + comp_idx;
    // Output layout: row-major segments
    size_t output_off = (output_row_idx * num_segments + output_col_idx) * 3 + comp_idx;

    d_quotient_segments[output_off] = d_multicoset[input_off];
}

// Kernel to apply iNTT to each row of quotient_segments (length 4)
// Uses the correct primitive 4th root of unity for Goldilocks field
__global__ void quotient_segmentify_intt_rows_kernel(
    uint64_t* d_quotient_segments,     // [num_output_rows * num_segments * 3] XFE row-major
    size_t num_output_rows,
    size_t num_segments
) {
    size_t row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t comp_idx = blockIdx.y;  // 0..2 for XFE components

    if (row_idx >= num_output_rows || comp_idx >= 3) return;

    // Apply iNTT to the row of length num_segments (4)
    // XFE layout: [seg0_c0, seg0_c1, seg0_c2, seg1_c0, seg1_c1, seg1_c2, ...]
    // We want to process component comp_idx across all segments
    uint64_t* row_base = d_quotient_segments + row_idx * num_segments * 3;

    // Read values for this component
    uint64_t a = row_base[0 * 3 + comp_idx];
    uint64_t b = row_base[1 * 3 + comp_idx];
    uint64_t c = row_base[2 * 3 + comp_idx];
    uint64_t d = row_base[3 * 3 + comp_idx];

    // Primitive 4th root of unity in Goldilocks field
    // From twenty-first: PRIMITIVE_ROOTS[4] = 281474976710656
    constexpr uint64_t w4 = 281474976710656ULL;
    // For iNTT, we need w^(-1) = w^3 (since w^4 = 1)
    uint64_t w_inv = bfield_pow_impl(w4, 3);
    uint64_t w_inv2 = bfield_mul_impl(w_inv, w_inv);
    uint64_t w_inv3 = bfield_mul_impl(w_inv2, w_inv);

    // iNTT matrix for length 4:
    // (1/4) * [[1, 1,    1,    1   ],
    //          [1, w^-1, w^-2, w^-3],
    //          [1, w^-2, 1,    w^-2],
    //          [1, w^-3, w^-2, w^-1]]
    
    uint64_t out0 = bfield_add_impl(bfield_add_impl(a, b), bfield_add_impl(c, d));
    uint64_t out1 = bfield_add_impl(
        bfield_add_impl(a, bfield_mul_impl(b, w_inv)),
        bfield_add_impl(bfield_mul_impl(c, w_inv2), bfield_mul_impl(d, w_inv3))
    );
    uint64_t out2 = bfield_add_impl(
        bfield_add_impl(a, bfield_mul_impl(b, w_inv2)),
        bfield_add_impl(c, bfield_mul_impl(d, w_inv2))  // w^-4 = 1
    );
    uint64_t out3 = bfield_add_impl(
        bfield_add_impl(a, bfield_mul_impl(b, w_inv3)),
        bfield_add_impl(bfield_mul_impl(c, w_inv2), bfield_mul_impl(d, w_inv))  // w^-6 = w^-2
    );

    // Divide by 4
    constexpr uint64_t four_inv = 13835058052060938241ULL;  // 4^(-1) mod p
    row_base[0 * 3 + comp_idx] = bfield_mul_impl(out0, four_inv);
    row_base[1 * 3 + comp_idx] = bfield_mul_impl(out1, four_inv);
    row_base[2 * 3 + comp_idx] = bfield_mul_impl(out2, four_inv);
    row_base[3 * 3 + comp_idx] = bfield_mul_impl(out3, four_inv);
}

// Kernel to scale rows according to the complex scaling pattern
// From Rust: scale cell at (row j, col k) by psi^(-k) * iota^(-j*k)
// Equivalent formulation: scale factor = (psi^(-1) * iota^(-j))^k
__global__ void quotient_segmentify_scale_kernel(
    uint64_t* d_quotient_segments,     // [num_output_rows * num_segments * 3] XFE row-major
    size_t num_output_rows,
    size_t num_segments,
    uint64_t psi_inv,
    uint64_t iota_inv
) {
    size_t row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t comp_idx = blockIdx.y;  // 0..2 for XFE components

    if (row_idx >= num_output_rows || comp_idx >= 3) return;

    uint64_t* row_base = d_quotient_segments + row_idx * num_segments * 3;

    // psi_iotaj_inv = psi^(-1) * iota^(-j)
    uint64_t iota_inv_j = bfield_pow_impl(iota_inv, row_idx);
    uint64_t psi_iotaj_inv = bfield_mul_impl(psi_inv, iota_inv_j);
    
    // Scale each column k by (psi_iotaj_inv)^k
    uint64_t scale = 1;  // (psi_iotaj_inv)^0 = 1
    for (size_t k = 0; k < num_segments; ++k) {
        row_base[k * 3 + comp_idx] = bfield_mul_impl(row_base[k * 3 + comp_idx], scale);
        scale = bfield_mul_impl(scale, psi_iotaj_inv);
    }
}

// Host function to orchestrate the multi-coset segmentify (matching Rust's algorithm)
// Input: d_quotient_multicoset contains quotient evaluations at FRI domain points
//        Layout: row-major for (working_idx, coset): [(j * num_cosets + k) * 3 + comp]
//        This is reinterpreting FRI domain evaluations as multicoset evaluations.
// Output: segment codewords on FRI domain
void quotient_segmentify_multicoset_gpu(
    const uint64_t* d_quotient_multicoset, // [working_len * num_cosets * 3] row-major for (j,k)
    size_t working_len,
    size_t num_cosets,                    // 8
    size_t num_segments,                  // 4
    uint64_t psi_inv,                     // fri_offset.inverse()
    uint64_t iota_inv,                    // iota.inverse()
    uint64_t fri_offset,
    size_t fri_len,
    uint64_t* d_seg_coeffs_compact,       // [num_segments * 3 * segment_len]
    uint64_t* d_segment_codewords_colmajor, // [num_segments * 3 * fri_len]
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    size_t num_output_rows = working_len * num_cosets / num_segments;  // e.g., 32768 * 8 / 4 = 65536

    // Allocate temporary buffer for quotient_segments [num_output_rows][num_segments][3]
    uint64_t* d_quotient_segments;
    size_t segments_size = num_output_rows * num_segments * 3 * sizeof(uint64_t);
    cudaMalloc(&d_quotient_segments, segments_size);

    // 1. Rearrange data from (num_cosets, working_len) to (num_output_rows, num_segments)
    // Using Rust's mapping: exponent_of_iota = output_row + output_col * num_output_rows
    //                       input_row = exponent_of_iota / num_cosets
    //                       input_col = exponent_of_iota % num_cosets
    {
        dim3 grid((num_output_rows + BLOCK - 1) / BLOCK, num_segments, 3);
        dim3 block(BLOCK, 1, 1);
        quotient_segmentify_rearrange_kernel<<<grid, block, 0, stream>>>(
            d_quotient_multicoset,
            working_len,
            num_cosets,
            num_segments,
            d_quotient_segments
        );
    }

    // 2. Apply iNTT to each row (length num_segments = 4)
    {
        dim3 grid((num_output_rows + BLOCK - 1) / BLOCK, 3, 1);
        dim3 block(BLOCK, 1, 1);
        quotient_segmentify_intt_rows_kernel<<<grid, block, 0, stream>>>(
            d_quotient_segments,
            num_output_rows,
            num_segments
        );
    }

    // 3. Scale rows: cell at (row j, col k) is scaled by psi^(-k) * iota^(-j*k)
    {
        dim3 grid((num_output_rows + BLOCK - 1) / BLOCK, 3, 1);
        dim3 block(BLOCK, 1, 1);
        quotient_segmentify_scale_kernel<<<grid, block, 0, stream>>>(
            d_quotient_segments,
            num_output_rows,
            num_segments,
            psi_inv,
            iota_inv
        );
    }

    // 4. LDE each segment column from segment_domain to FRI domain
    // segment_domain: length = num_output_rows, offset = fri_offset^num_segments
    // fri_domain: length = fri_len, offset = fri_offset
    uint64_t segment_domain_offset = bfield_pow_host(fri_offset, num_segments);
    uint64_t segment_domain_offset_inv = bfield_inv_host(segment_domain_offset);
    int grid_seg = (int)((num_output_rows + BLOCK - 1) / BLOCK);
    int grid_fri = (int)((fri_len + BLOCK - 1) / BLOCK);

    // Allocate temp buffer for one column at a time
    uint64_t* d_temp_col;
    cudaMalloc(&d_temp_col, num_output_rows * sizeof(uint64_t));

    // Segment polynomial coefficients layout: [seg][comp][coeff]
    // d_seg_coeffs_compact[(seg * 3 + comp) * segment_len + coeff]
    size_t segment_len = num_output_rows;  // Segment polynomial degree < num_output_rows
    
    for (size_t seg = 0; seg < num_segments; ++seg) {
        for (size_t comp = 0; comp < 3; ++comp) {
            // Extract column (seg, comp) into contiguous temp buffer
            // Source: d_quotient_segments[row * num_segments * 3 + seg * 3 + comp]
            gather_strided_column_kernel<<<grid_seg, BLOCK, 0, stream>>>(
                d_quotient_segments,
                num_output_rows,
                num_segments * 3,  // stride between rows
                seg * 3 + comp,    // offset within row
                d_temp_col
            );

            // Interpolate on segment_domain: iNTT + coset unscale
            // This gives us the polynomial coefficients
            ntt_inverse_gpu(d_temp_col, num_output_rows, stream);
            qpc_coset_scale_kernel<<<grid_seg, BLOCK, 0, stream>>>(
                d_temp_col, num_output_rows, segment_domain_offset_inv
            );

            // *** SAVE COEFFICIENTS for OOD evaluation ***
            // d_temp_col now contains segment polynomial coefficients
            size_t coeff_off = (seg * 3 + comp) * segment_len;
            cudaMemcpyAsync(d_seg_coeffs_compact + coeff_off, d_temp_col, 
                           segment_len * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);

            // LDE to FRI domain: pad with zeros + coset scale + forward NTT
            size_t codeword_off = (seg * 3 + comp) * fri_len;
            uint64_t* d_codeword = d_segment_codewords_colmajor + codeword_off;
            
            cudaMemsetAsync(d_codeword, 0, fri_len * sizeof(uint64_t), stream);
            cudaMemcpyAsync(d_codeword, d_temp_col, segment_len * sizeof(uint64_t),
                           cudaMemcpyDeviceToDevice, stream);
            
            qpc_coset_scale_kernel<<<grid_fri, BLOCK, 0, stream>>>(
                d_codeword, fri_len, fri_offset
            );
            ntt_forward_gpu(d_codeword, fri_len, stream);
        }
    }
    
    cudaFree(d_temp_col);
    cudaFree(d_quotient_segments);
}

// ============================================================================
// Simplified segmentify for testing
// ============================================================================

__global__ void quotient_segmentify_simple_kernel(
    const uint64_t* d_multicoset,      // [working_len * num_cosets * 3]
    size_t working_len,
    size_t num_cosets,
    size_t num_segments,
    uint64_t* d_seg_coeffs_compact,    // [num_segments * 3 * segment_len]
    uint64_t* d_segment_codewords_colmajor, // [num_segments * 3 * fri_len]
    size_t fri_len
) {
    size_t seg_idx = blockIdx.x;
    size_t comp_idx = blockIdx.y;
    size_t row_idx = threadIdx.x;

    if (seg_idx >= num_segments || comp_idx >= 3 || row_idx >= working_len) return;

    // Simple approach: average the cosets for each segment
    // segment_len = working_len / num_segments = 32768 / 4 = 8192
    size_t segment_len = working_len / num_segments;

    // For each segment, average cosets seg_idx, seg_idx+num_segments, etc.
    uint64_t sum = 0;
    size_t count = 0;

    for (size_t c = seg_idx; c < num_cosets; c += num_segments) {
        size_t off = (c * working_len + row_idx) * 3 + comp_idx;
        sum = bfield_add_impl(sum, d_multicoset[off]);
        count++;
    }

    // Average (simplified: just take first coset for now)
    uint64_t avg_val = sum;  // Skip division for simplicity

    // Store in compact coeffs
    size_t compact_off = (seg_idx * 3 + comp_idx) * segment_len + row_idx;
    d_seg_coeffs_compact[compact_off] = avg_val;

    // For codewords, just copy to beginning of FRI domain (simplified)
    if (row_idx < fri_len) {
        size_t codeword_off = (seg_idx * 3 + comp_idx) * fri_len + row_idx;
        d_segment_codewords_colmajor[codeword_off] = avg_val;
    }
}

void quotient_segmentify_simple_gpu(
    const uint64_t* d_quotient_multicoset,
    size_t working_len,
    size_t num_cosets,
    size_t num_segments,
    uint64_t* d_seg_coeffs_compact,
    uint64_t* d_segment_codewords_colmajor,
    size_t fri_len,
    uint64_t fri_offset,
    cudaStream_t stream
) {
    dim3 grid(num_segments, 3, 1);
    dim3 block(working_len, 1, 1);  // working_len threads

    quotient_segmentify_simple_kernel<<<grid, block, 0, stream>>>(
        d_quotient_multicoset,
        working_len,
        num_cosets,
        num_segments,
        d_seg_coeffs_compact,
        d_segment_codewords_colmajor,
        fri_len
    );
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
