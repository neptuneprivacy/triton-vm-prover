/**
 * GPU-Accelerated U32 Table Fill
 * 
 * Fills the U32 table on GPU in parallel:
 * 1. Each thread handles one U32 entry's section
 * 2. Sections are computed independently (forward + backward pass)
 * 3. Results written directly to GPU table
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/u32_table_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/cuda_common.cuh"
#include "common/debug_control.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstdlib>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Opcodes for U32 instructions
static constexpr uint32_t SPLIT_OPCODE = 4;
static constexpr uint32_t LT_OPCODE = 6;
static constexpr uint32_t AND_OPCODE = 14;
static constexpr uint32_t XOR_OPCODE = 22;
static constexpr uint32_t POW_OPCODE = 30;

// Modular inverse of 2: (P+1)/2 in the Goldilocks field
static constexpr uint64_t INV_2 = 9223372034707292161ULL;

// U32 table column indices (relative to U32 table start)
static constexpr size_t COL_COPY_FLAG = 0;
static constexpr size_t COL_BITS = 1;
static constexpr size_t COL_BITS_MINUS_33_INV = 2;
static constexpr size_t COL_CI = 3;
static constexpr size_t COL_LHS = 4;
static constexpr size_t COL_LHS_INV = 5;
static constexpr size_t COL_RHS = 6;
static constexpr size_t COL_RHS_INV = 7;
static constexpr size_t COL_RESULT = 8;
static constexpr size_t COL_LOOKUP_MULT = 9;
static constexpr size_t U32_NUM_COLS = 10;

// ============================================================================
// Helper: Compute section size for a U32 entry
// Note: U32 table uses 32-bit operand values (masked)
// ============================================================================
__device__ __forceinline__ uint32_t compute_section_size(uint32_t opcode, uint64_t lhs, uint64_t rhs) {
    bool is_pow = (opcode == POW_OPCODE);
    // Mask to 32 bits as per Triton VM U32 table spec
    uint32_t lhs32 = static_cast<uint32_t>(lhs & 0xFFFFFFFFULL);
    uint32_t rhs32 = static_cast<uint32_t>(rhs & 0xFFFFFFFFULL);
    uint32_t val = is_pow ? rhs32 : (lhs32 | rhs32);
    
    uint32_t bits = 0;
    while (val > 0) {
        bits++;
        val >>= 1;
    }
    return (bits == 0) ? 1 : bits + 1;
}

// ============================================================================
// Kernel: Fill one U32 section per thread
// ============================================================================
__global__ void fill_u32_section_kernel(
    const uint32_t* __restrict__ d_opcodes,
    const uint64_t* __restrict__ d_lhs,
    const uint64_t* __restrict__ d_rhs,
    const uint64_t* __restrict__ d_multiplicities,
    const size_t* __restrict__ d_offsets,
    uint64_t* __restrict__ d_table,
    size_t u32_table_start,
    size_t table_width,
    size_t table_height,
    uint64_t minus_33_inv,
    size_t n_entries
) {
    size_t entry_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (entry_idx >= n_entries) return;
    
    uint32_t opcode = d_opcodes[entry_idx];
    uint64_t lhs = d_lhs[entry_idx];
    uint64_t rhs = d_rhs[entry_idx];
    uint64_t multiplicity = d_multiplicities[entry_idx];
    size_t section_offset = d_offsets[entry_idx];
    
    bool is_pow = (opcode == POW_OPCODE);
    bool is_lt = (opcode == LT_OPCODE);
    bool is_split = (opcode == SPLIT_OPCODE);
    
    // Local storage for section (max 34 rows)
    uint64_t lhs_vals[34], rhs_vals[34], bits_vals[34], results[34];
    
    // Forward pass: compute lhs, rhs, bits for each row
    lhs_vals[0] = lhs;
    rhs_vals[0] = rhs;
    bits_vals[0] = 0;
    
    uint32_t section_size = compute_section_size(opcode, lhs, rhs);
    
    for (uint32_t i = 1; i < section_size; ++i) {
        uint64_t prev_lhs = lhs_vals[i - 1];
        uint64_t prev_rhs = rhs_vals[i - 1];
        uint64_t lhs_lsb = prev_lhs & 1;
        uint64_t rhs_lsb = prev_rhs & 1;
        
        // Field division by 2: (val - lsb) * inv(2)
        // In BFieldElement: (val - lsb) / 2 = (val - lsb) * INV_2
        if (is_pow) {
            lhs_vals[i] = prev_lhs;  // Pow keeps lhs unchanged
        } else {
            uint64_t lhs_minus_lsb = bfield_sub_impl(prev_lhs, lhs_lsb);
            lhs_vals[i] = bfield_mul_impl(lhs_minus_lsb, INV_2);
        }
        uint64_t rhs_minus_lsb = bfield_sub_impl(prev_rhs, rhs_lsb);
        rhs_vals[i] = bfield_mul_impl(rhs_minus_lsb, INV_2);
        bits_vals[i] = i;
    }
    
    // Terminal row result
    uint32_t last = section_size - 1;
    if (is_split) {
        results[last] = 0;
    } else if (is_lt) {
        results[last] = (bits_vals[last] == 0) ? 0 : 2;
    } else if (is_pow) {
        results[last] = 1;
    } else {
        results[last] = 0;
    }
    
    // Backward pass: propagate results
    for (int i = static_cast<int>(last) - 1; i >= 0; --i) {
        // Get LSBs using the original (unreduced) values at this position
        // lhs_lsb = lhs_vals[i].value() % 2 in field
        // For small values (< P), lhs_vals[i] & 1 is correct
        // But for field elements, we need to consider they might be >= 2^63
        // In practice, lhs/rhs are u32 values so they're always small
        uint64_t lhs_lsb = lhs_vals[i] & 1;
        uint64_t rhs_lsb = rhs_vals[i] & 1;
        uint64_t next_res = results[i + 1];
        bool is_copy_flag_row = (i == 0);
        
        if (is_split) {
            results[i] = next_res;
        } else if (is_lt) {
            if (next_res == 0 || next_res == 1) {
                results[i] = next_res;
            } else if (next_res == 2 && lhs_lsb == 0 && rhs_lsb == 1) {
                results[i] = 1;
            } else if (next_res == 2 && lhs_lsb == 1 && rhs_lsb == 0) {
                results[i] = 0;
            } else if (next_res == 2 && is_copy_flag_row) {
                results[i] = 0;
            } else {
                results[i] = 2;
            }
        } else if (is_pow) {
            // Pow: result[i] = result[i+1]^2 if rhs_lsb == 0
            //                = result[i+1]^2 * lhs if rhs_lsb == 1
            uint64_t next_sq = bfield_mul_impl(next_res, next_res);
            if (rhs_lsb == 0) {
                results[i] = next_sq;
            } else {
                results[i] = bfield_mul_impl(next_sq, lhs_vals[i]);
            }
        } else {
            results[i] = next_res;
        }
    }
    
    // Write to table
    for (uint32_t i = 0; i < section_size; ++i) {
        size_t row = section_offset + i;
        if (row >= table_height) break;
        
        uint64_t cur_lhs = lhs_vals[i];
        uint64_t cur_rhs = rhs_vals[i];
        uint64_t bits = bits_vals[i];
        
        // Compute (bits - 33)^(-1)
        uint64_t bits_minus_33 = bfield_sub_impl(bits, 33);
        uint64_t bits_minus_33_inv_val = (bits == 33) ? 0 : bfield_inv_impl(bits_minus_33);
        
        // Compute inverses (return 0 if value is 0)
        uint64_t lhs_inv = (cur_lhs == 0) ? 0 : bfield_inv_impl(cur_lhs);
        uint64_t rhs_inv = (cur_rhs == 0) ? 0 : bfield_inv_impl(cur_rhs);
        
        // Write columns
        size_t base = row * table_width + u32_table_start;
        d_table[base + COL_COPY_FLAG] = (i == 0) ? 1 : 0;
        d_table[base + COL_BITS] = bits;
        d_table[base + COL_BITS_MINUS_33_INV] = bits_minus_33_inv_val;
        d_table[base + COL_CI] = static_cast<uint64_t>(opcode);
        d_table[base + COL_LHS] = cur_lhs;
        d_table[base + COL_LHS_INV] = lhs_inv;
        d_table[base + COL_RHS] = cur_rhs;
        d_table[base + COL_RHS_INV] = rhs_inv;
        d_table[base + COL_RESULT] = results[i];
        d_table[base + COL_LOOKUP_MULT] = (i == 0) ? multiplicity : 0;
    }
}

// ============================================================================
// Kernel: Pad U32 table
// ============================================================================
__global__ void pad_u32_table_kernel(
    uint64_t* __restrict__ d_table,
    size_t u32_table_start,
    size_t table_width,
    size_t start_row,
    size_t end_row,
    uint64_t padding_ci,
    uint64_t padding_lhs,
    uint64_t padding_lhs_inv,
    uint64_t padding_result,
    uint64_t minus_33_inv
) {
    size_t row = start_row + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= end_row) return;
    
    size_t base = row * table_width + u32_table_start;
    d_table[base + COL_COPY_FLAG] = 0;
    d_table[base + COL_BITS] = 0;
    d_table[base + COL_BITS_MINUS_33_INV] = minus_33_inv;
    d_table[base + COL_CI] = padding_ci;
    d_table[base + COL_LHS] = padding_lhs;
    d_table[base + COL_LHS_INV] = padding_lhs_inv;
    d_table[base + COL_RHS] = 0;
    d_table[base + COL_RHS_INV] = 0;
    d_table[base + COL_RESULT] = padding_result;
    d_table[base + COL_LOOKUP_MULT] = 0;
}

// ============================================================================
// Host Function
// ============================================================================

size_t gpu_fill_u32_table(
    uint64_t* d_table,
    const std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>>& entries,
    size_t u32_table_start,
    size_t table_width,
    size_t table_height,
    cudaStream_t stream
) {
    const size_t n_entries = entries.size();
    if (n_entries == 0) return 0;
    
    const bool profile = TRITON_PROFILE_ENABLED();
    auto t_start = std::chrono::high_resolution_clock::now();
    auto log_time = [&](const char* msg) {
        if (profile) {
            cudaStreamSynchronize(stream);
            auto now = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration<double, std::milli>(now - t_start).count();
            std::cout << "    GPU U32: " << msg << ": " << ms << " ms" << std::endl;
            t_start = now;
        }
    };
    
    // Compute (-33)^(-1) on CPU
    static constexpr uint64_t P = 18446744069414584321ULL;
    uint64_t minus_33 = P - 33;
    // Use Fermat's little theorem: a^(-1) = a^(p-2)
    auto pow_mod = [](uint64_t base, uint64_t exp) {
        uint64_t result = 1;
        while (exp > 0) {
            if (exp & 1) {
                __uint128_t prod = static_cast<__uint128_t>(result) * base;
                result = static_cast<uint64_t>(prod % P);
            }
            __uint128_t sq = static_cast<__uint128_t>(base) * base;
            base = static_cast<uint64_t>(sq % P);
            exp >>= 1;
        }
        return result;
    };
    uint64_t minus_33_inv = pow_mod(minus_33, P - 2);
    
    // Prepare entry data
    std::vector<uint32_t> opcodes(n_entries);
    std::vector<uint64_t> lhs(n_entries), rhs(n_entries), mults(n_entries);
    std::vector<size_t> offsets(n_entries + 1);
    
    offsets[0] = 0;
    for (size_t i = 0; i < n_entries; ++i) {
        opcodes[i] = std::get<0>(entries[i]);
        lhs[i] = std::get<1>(entries[i]);
        rhs[i] = std::get<2>(entries[i]);
        mults[i] = std::get<3>(entries[i]);
        
        // Compute section size (mask to 32 bits as per Triton VM spec)
        bool is_pow = (opcodes[i] == POW_OPCODE);
        uint32_t lhs32 = static_cast<uint32_t>(lhs[i] & 0xFFFFFFFFULL);
        uint32_t rhs32 = static_cast<uint32_t>(rhs[i] & 0xFFFFFFFFULL);
        uint32_t val = is_pow ? rhs32 : (lhs32 | rhs32);
        uint32_t bits = 0;
        while (val > 0) { bits++; val >>= 1; }
        uint32_t section_size = (bits == 0) ? 1 : bits + 1;
        offsets[i + 1] = offsets[i] + section_size;
    }
    log_time("Prepare entries");
    
    // Allocate device memory
    uint32_t *d_opcodes;
    uint64_t *d_lhs, *d_rhs, *d_mults;
    size_t *d_offsets;
    
    cudaMalloc(&d_opcodes, n_entries * sizeof(uint32_t));
    cudaMalloc(&d_lhs, n_entries * sizeof(uint64_t));
    cudaMalloc(&d_rhs, n_entries * sizeof(uint64_t));
    cudaMalloc(&d_mults, n_entries * sizeof(uint64_t));
    cudaMalloc(&d_offsets, (n_entries + 1) * sizeof(size_t));
    
    // Upload data
    cudaMemcpyAsync(d_opcodes, opcodes.data(), n_entries * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lhs, lhs.data(), n_entries * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rhs, rhs.data(), n_entries * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mults, mults.data(), n_entries * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_offsets, offsets.data(), (n_entries + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    log_time("Upload entries");
    
    // Launch fill kernel
    {
        size_t block = 256;
        size_t grid = (n_entries + block - 1) / block;
        fill_u32_section_kernel<<<grid, block, 0, stream>>>(
            d_opcodes, d_lhs, d_rhs, d_mults, d_offsets,
            d_table, u32_table_start, table_width, table_height,
            minus_33_inv, n_entries);
    }
    log_time("Fill sections");
    
    size_t total_rows = offsets.back();
    
    // Pad remaining rows
    if (total_rows < table_height) {
        uint64_t padding_ci = SPLIT_OPCODE;
        uint64_t padding_lhs = 0;
        uint64_t padding_lhs_inv = 0;
        uint64_t padding_result = 0;
        
        if (total_rows > 0) {
            // Get padding values based on last entry's terminal row values
            size_t last_entry = n_entries - 1;
            padding_ci = opcodes[last_entry];
            
            // Terminal row lhs is 0 for non-POW instructions
            // For POW, lhs is the original value (unchanged through forward pass)
            if (padding_ci == POW_OPCODE) {
                padding_lhs = lhs[last_entry];
                if (padding_lhs != 0) {
                    padding_lhs_inv = pow_mod(padding_lhs, P - 2);
                }
            }
            // padding_lhs stays 0 for SPLIT, LT, AND, XOR (terminal row has lhs = 0)
            // padding_lhs_inv stays 0 (inverse of 0 is 0)
            
            // Terminal result depends on instruction type
            if (padding_ci == LT_OPCODE) {
                // Lt terminal result is 2 (or 0 if bits was 0, but last row always has bits > 0 for multi-row sections)
                padding_result = 2;
            } else if (padding_ci == POW_OPCODE) {
                padding_result = 1;
            }
            // For SPLIT, AND, XOR: padding_result stays 0
        }
        
        size_t pad_count = table_height - total_rows;
        size_t block = 256;
        size_t grid = (pad_count + block - 1) / block;
        
        pad_u32_table_kernel<<<grid, block, 0, stream>>>(
            d_table, u32_table_start, table_width,
            total_rows, table_height,
            padding_ci, padding_lhs, padding_lhs_inv, padding_result, minus_33_inv);
    }
    log_time("Padding");
    
    // Cleanup
    cudaFree(d_opcodes);
    cudaFree(d_lhs);
    cudaFree(d_rhs);
    cudaFree(d_mults);
    cudaFree(d_offsets);
    
    return total_rows;
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

