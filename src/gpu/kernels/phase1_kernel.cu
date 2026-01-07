/**
 * GPU Phase 1 Kernels: Main Table Creation from AET
 * 
 * Implements GPU-accelerated table creation, sorting, and padding.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/phase1_kernel.cuh"
#include "gpu/kernels/table_fill_kernel.cuh"
#include "gpu/kernels/degree_lowering_main_kernel.cuh"
#include "gpu/cuda_common.cuh"

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <iostream>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Table column offsets (must match C++ definitions)
namespace TableOffsets {
    constexpr size_t PROGRAM_TABLE_START = 0;
    constexpr size_t PROGRAM_TABLE_COLS = 7;
    constexpr size_t PROCESSOR_TABLE_START = 7;
    constexpr size_t PROCESSOR_TABLE_COLS = 39;
    constexpr size_t OP_STACK_TABLE_START = 46;
    constexpr size_t OP_STACK_TABLE_COLS = 4;
    constexpr size_t RAM_TABLE_START = 50;
    constexpr size_t RAM_TABLE_COLS = 7;
    constexpr size_t JUMP_STACK_TABLE_START = 57;
    constexpr size_t JUMP_STACK_TABLE_COLS = 5;
    constexpr size_t HASH_TABLE_START = 62;
    constexpr size_t HASH_TABLE_COLS = 67;
    constexpr size_t CASCADE_TABLE_START = 129;
    constexpr size_t CASCADE_TABLE_COLS = 6;
    constexpr size_t LOOKUP_TABLE_START = 135;
    constexpr size_t LOOKUP_TABLE_COLS = 4;
    constexpr size_t U32_TABLE_START = 139;
    constexpr size_t U32_TABLE_COLS = 10;
    constexpr size_t DEGREE_LOWERING_START = 149;
    constexpr size_t DEGREE_LOWERING_COLS = 230;
    constexpr size_t TOTAL_COLS = 379;
}

// BFieldElement prime
constexpr uint64_t BFE_PRIME = 0xFFFFFFFF00000001ULL;

// ============================================================================
// Device Helper Functions
// ============================================================================

// Modular addition
__device__ __forceinline__ uint64_t bfe_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    // Check for overflow (sum < a or sum >= prime)
    if (sum < a || sum >= BFE_PRIME) {
        sum -= BFE_PRIME;
    }
    return sum;
}

// Modular subtraction
__device__ __forceinline__ uint64_t bfe_sub(uint64_t a, uint64_t b) {
    if (a >= b) return a - b;
    return BFE_PRIME - (b - a);
}

// Modular multiplication using 128-bit arithmetic
__device__ __forceinline__ uint64_t bfe_mul(uint64_t a, uint64_t b) {
    unsigned __int128 prod = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
    
    // Reduce mod (2^64 - 2^32 + 1)
    // hi*2^64 mod P = hi*(2^32 - 1) mod P
    uint64_t hi_lo = (uint32_t)hi;
    uint64_t hi_hi = hi >> 32;
    
    uint64_t result = lo;
    // Add hi_lo * 2^32
    uint64_t tmp = hi_lo << 32;
    result = bfe_add(result, tmp);
    // Subtract hi_lo (since 2^64 = 2^32 - 1 mod P)
    result = bfe_sub(result, hi_lo);
    // Add hi_hi * 2^64 mod P = hi_hi * (2^32 - 1)
    result = bfe_add(result, (hi_hi << 32) - hi_hi);
    
    // Final reduction
    if (result >= BFE_PRIME) result -= BFE_PRIME;
    return result;
}

// Extended GCD for inverse
__device__ uint64_t bfe_inverse(uint64_t a) {
    if (a == 0) return 0;
    
    int64_t t = 0, newt = 1;
    uint64_t r = BFE_PRIME, newr = a;
    
    while (newr != 0) {
        uint64_t q = r / newr;
        int64_t tmp_t = t - (int64_t)q * newt;
        t = newt;
        newt = tmp_t;
        uint64_t tmp_r = r - q * newr;
        r = newr;
        newr = tmp_r;
    }
    
    if (t < 0) t += BFE_PRIME;
    return (uint64_t)t;
}

// ============================================================================
// Sorting Kernels (using Thrust for simplicity and correctness)
// ============================================================================

// Key structure for sorting by two columns
struct TwoColumnKey {
    uint64_t key1;
    uint64_t key2;
    size_t original_idx;
};

struct TwoColumnComparator {
    __host__ __device__ bool operator()(const TwoColumnKey& a, const TwoColumnKey& b) const {
        if (a.key1 != b.key1) return a.key1 < b.key1;
        return a.key2 < b.key2;
    }
};

// Extract keys kernel
__global__ void extract_sort_keys_kernel(
    const uint64_t* d_trace,
    TwoColumnKey* d_keys,
    size_t num_rows,
    size_t num_cols,
    size_t key1_col,
    size_t key2_col
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    
    d_keys[idx].key1 = d_trace[idx * num_cols + key1_col];
    d_keys[idx].key2 = d_trace[idx * num_cols + key2_col];
    d_keys[idx].original_idx = idx;
}

// Reorder trace based on sorted keys
__global__ void reorder_trace_kernel(
    const uint64_t* d_trace_in,
    const TwoColumnKey* d_sorted_keys,
    uint64_t* d_trace_out,
    size_t num_rows,
    size_t num_cols
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    size_t src_row = d_sorted_keys[row].original_idx;
    for (size_t col = 0; col < num_cols; ++col) {
        d_trace_out[row * num_cols + col] = d_trace_in[src_row * num_cols + col];
    }
}

void gpu_sort_trace_by_two_keys(
    uint64_t* d_trace,
    size_t num_rows,
    size_t num_cols,
    size_t key1_col,
    size_t key2_col,
    cudaStream_t stream
) {
    if (num_rows <= 1) return;
    
    // Allocate keys array
    TwoColumnKey* d_keys;
    CUDA_CHECK(cudaMalloc(&d_keys, num_rows * sizeof(TwoColumnKey)));
    
    // Extract keys
    constexpr size_t BLOCK_SIZE = 256;
    size_t num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    extract_sort_keys_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_trace, d_keys, num_rows, num_cols, key1_col, key2_col);
    
    // Sort using Thrust
    thrust::device_ptr<TwoColumnKey> keys_ptr(d_keys);
    thrust::sort(thrust::cuda::par.on(stream), keys_ptr, keys_ptr + num_rows, TwoColumnComparator());
    
    // Allocate temp buffer for reordering
    uint64_t* d_trace_temp;
    CUDA_CHECK(cudaMalloc(&d_trace_temp, num_rows * num_cols * sizeof(uint64_t)));
    
    // Reorder trace
    reorder_trace_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_trace, d_keys, d_trace_temp, num_rows, num_cols);
    
    // Copy back
    CUDA_CHECK(cudaMemcpyAsync(d_trace, d_trace_temp, 
        num_rows * num_cols * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice, stream));
    
    CUDA_CHECK(cudaFree(d_trace_temp));
    CUDA_CHECK(cudaFree(d_keys));
}

// ============================================================================
// Table Filling Kernels
// ============================================================================

__global__ void fill_processor_table_kernel(
    const uint64_t* d_processor_trace,
    uint64_t* d_main_table,
    size_t processor_rows,
    size_t padded_height,
    size_t num_cols,
    size_t processor_table_start
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= processor_rows) return;
    
    // Copy all 39 columns from processor trace to main table
    for (size_t col = 0; col < TableOffsets::PROCESSOR_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + processor_table_start + col] = 
            d_processor_trace[row * TableOffsets::PROCESSOR_TABLE_COLS + col];
    }
}

__global__ void fill_op_stack_table_kernel(
    const uint64_t* d_sorted_op_stack,
    uint64_t* d_main_table,
    size_t op_stack_rows,
    size_t padded_height,
    size_t num_cols,
    size_t op_stack_table_start
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= op_stack_rows) return;
    
    // OpStack columns: CLK, IB1ShrinkStack, StackPointer, FirstUnderflowElement
    for (size_t col = 0; col < TableOffsets::OP_STACK_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + op_stack_table_start + col] = 
            d_sorted_op_stack[row * TableOffsets::OP_STACK_TABLE_COLS + col];
    }
}

__global__ void fill_ram_table_kernel(
    const uint64_t* d_sorted_ram,
    uint64_t* d_main_table,
    size_t ram_rows,
    size_t padded_height,
    size_t num_cols,
    size_t ram_table_start
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ram_rows) return;
    
    // RAM trace is already in main-table RAM layout (7 columns):
    //   [CLK, InstructionType, RamPointer, RamValue, InvRampDiff, Bezout0, Bezout1]
    // Copy directly into main table (matches CPU semantics exactly when host builds ram trace).
    #pragma unroll
    for (int c = 0; c < (int)TableOffsets::RAM_TABLE_COLS; ++c) {
        d_main_table[row * num_cols + ram_table_start + (size_t)c] =
            d_sorted_ram[row * TableOffsets::RAM_TABLE_COLS + (size_t)c];
    }
}

__global__ void fill_jump_stack_table_kernel(
    const uint64_t* d_sorted_jump_stack,
    uint64_t* d_main_table,
    size_t jump_stack_rows,
    size_t padded_height,
    size_t num_cols,
    size_t jump_stack_table_start
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= jump_stack_rows) return;
    
    for (size_t col = 0; col < TableOffsets::JUMP_STACK_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + jump_stack_table_start + col] = 
            d_sorted_jump_stack[row * TableOffsets::JUMP_STACK_TABLE_COLS + col];
    }
}

__global__ void fill_hash_table_kernel(
    const uint64_t* d_hash_trace,
    uint64_t* d_main_table,
    size_t program_hash_rows,
    size_t sponge_rows,
    size_t hash_rows,
    size_t padded_height,
    size_t num_cols,
    size_t hash_table_start
) {
    size_t total_rows = program_hash_rows + sponge_rows + hash_rows;
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    
    // Copy hash table columns
    for (size_t col = 0; col < TableOffsets::HASH_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + hash_table_start + col] = 
            d_hash_trace[row * TableOffsets::HASH_TABLE_COLS + col];
    }
    
    // Set Mode column (column 6 within hash table)
    // Mode is column 0 within the hash table (see `HashMainColumn::Mode`)
    constexpr size_t MODE_COL = 0;
    uint64_t mode;
    if (row < program_hash_rows) {
        mode = 1;  // ProgramHashing
    } else if (row < program_hash_rows + sponge_rows) {
        mode = 2;  // Sponge
    } else {
        mode = 3;  // Hash
    }
    d_main_table[row * num_cols + hash_table_start + MODE_COL] = mode;
}

__global__ void fill_cascade_table_kernel(
    const uint64_t* d_cascade_trace,
    uint64_t* d_main_table,
    size_t cascade_rows,
    size_t padded_height,
    size_t num_cols,
    size_t cascade_table_start
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= cascade_rows) return;
    
    for (size_t col = 0; col < TableOffsets::CASCADE_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + cascade_table_start + col] = 
            d_cascade_trace[row * TableOffsets::CASCADE_TABLE_COLS + col];
    }
}

__global__ void fill_lookup_table_kernel(
    const uint64_t* d_lookup_trace,
    uint64_t* d_main_table,
    size_t lookup_rows,
    size_t padded_height,
    size_t num_cols,
    size_t lookup_table_start
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= lookup_rows) return;
    
    for (size_t col = 0; col < TableOffsets::LOOKUP_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + lookup_table_start + col] = 
            d_lookup_trace[row * TableOffsets::LOOKUP_TABLE_COLS + col];
    }
}

__global__ void fill_u32_table_kernel(
    const uint64_t* d_u32_trace,
    uint64_t* d_main_table,
    size_t u32_rows,
    size_t padded_height,
    size_t num_cols,
    size_t u32_table_start
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= u32_rows) return;
    
    for (size_t col = 0; col < TableOffsets::U32_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + u32_table_start + col] = 
            d_u32_trace[row * TableOffsets::U32_TABLE_COLS + col];
    }
}

// ============================================================================
// Program Table Filling (special case - uses program instructions)
// ============================================================================

__global__ void fill_program_table_kernel(
    const uint64_t* d_program_trace,  // [rows][7]: Address, Instruction, LookupMult, IsHashInputPad, IsTablePad, IndexInChunk, MaxMinusIndexInChunkInv
    uint64_t* d_main_table,
    size_t program_rows,
    size_t num_cols
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= program_rows) return;
    
    for (size_t col = 0; col < TableOffsets::PROGRAM_TABLE_COLS; ++col) {
        d_main_table[row * num_cols + TableOffsets::PROGRAM_TABLE_START + col] = 
            d_program_trace[row * TableOffsets::PROGRAM_TABLE_COLS + col];
    }
}

// ============================================================================
// Clock Jump Difference Multiplicities
// ============================================================================

// Atomic add for BFieldElement (simplified - assumes no overflow for multiplicities)
__device__ void atomic_bfe_add(uint64_t* addr, uint64_t val) {
    atomicAdd((unsigned long long*)addr, val);
    // Note: For clock jump multiplicities, values are small enough that overflow is not a concern
    // A full implementation would need CAS loop for modular arithmetic
}

// Collect clock jump diffs and update multiplicities
__global__ void compute_clock_jump_diffs_op_stack_kernel(
    const uint64_t* d_sorted_op_stack,
    size_t op_stack_rows,
    uint64_t* d_main_table,
    size_t padded_height,
    size_t num_cols
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row == 0 || row >= op_stack_rows) return;
    
    // OpStack columns: CLK=0, IB1ShrinkStack=1, StackPointer=2, FirstUnderflowElement=3
    uint64_t curr_sp = d_sorted_op_stack[row * 4 + 2];
    uint64_t prev_sp = d_sorted_op_stack[(row - 1) * 4 + 2];
    
    if (curr_sp == prev_sp) {
        uint64_t curr_clk = d_sorted_op_stack[row * 4 + 0];
        uint64_t prev_clk = d_sorted_op_stack[(row - 1) * 4 + 0];
        uint64_t clk_diff = bfe_sub(curr_clk, prev_clk);
        
        // Increment multiplicity at processor table row = clk_diff
        if (clk_diff < padded_height) {
            size_t mult_col = TableOffsets::PROCESSOR_TABLE_START + 38;  // ClockJumpDifferenceLookupMultiplicity
            atomic_bfe_add(&d_main_table[clk_diff * num_cols + mult_col], 1);
        }
    }
}

__global__ void compute_clock_jump_diffs_ram_kernel(
    const uint64_t* d_sorted_ram,
    size_t ram_rows,
    uint64_t* d_main_table,
    size_t padded_height,
    size_t num_cols
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row == 0 || row >= ram_rows) return;

    // RAM trace columns: CLK=0, RamPointer=2
    uint64_t curr_ramp = d_sorted_ram[row * TableOffsets::RAM_TABLE_COLS + 2];
    uint64_t prev_ramp = d_sorted_ram[(row - 1) * TableOffsets::RAM_TABLE_COLS + 2];
    if (curr_ramp == prev_ramp) {
        uint64_t curr_clk = d_sorted_ram[row * TableOffsets::RAM_TABLE_COLS + 0];
        uint64_t prev_clk = d_sorted_ram[(row - 1) * TableOffsets::RAM_TABLE_COLS + 0];
        uint64_t clk_diff = bfe_sub(curr_clk, prev_clk);
        if (clk_diff < padded_height) {
            size_t mult_col = TableOffsets::PROCESSOR_TABLE_START + 38;  // ClockJumpDifferenceLookupMultiplicity
            atomic_bfe_add(&d_main_table[clk_diff * num_cols + mult_col], 1);
        }
    }
}

__global__ void compute_clock_jump_diffs_jump_stack_kernel(
    const uint64_t* d_sorted_jump_stack,
    size_t jump_stack_rows,
    uint64_t* d_main_table,
    size_t padded_height,
    size_t num_cols
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row == 0 || row >= jump_stack_rows) return;

    // JumpStack trace columns: CLK=0, JSP=2
    uint64_t curr_jsp = d_sorted_jump_stack[row * TableOffsets::JUMP_STACK_TABLE_COLS + 2];
    uint64_t prev_jsp = d_sorted_jump_stack[(row - 1) * TableOffsets::JUMP_STACK_TABLE_COLS + 2];
    if (curr_jsp == prev_jsp) {
        uint64_t curr_clk = d_sorted_jump_stack[row * TableOffsets::JUMP_STACK_TABLE_COLS + 0];
        uint64_t prev_clk = d_sorted_jump_stack[(row - 1) * TableOffsets::JUMP_STACK_TABLE_COLS + 0];
        uint64_t clk_diff = bfe_sub(curr_clk, prev_clk);
        if (clk_diff < padded_height) {
            size_t mult_col = TableOffsets::PROCESSOR_TABLE_START + 38;  // ClockJumpDifferenceLookupMultiplicity
            atomic_bfe_add(&d_main_table[clk_diff * num_cols + mult_col], 1);
        }
    }
}

// ============================================================================
// Host Functions
// ============================================================================

// Helper to flatten 2D trace to 1D
static std::vector<uint64_t> flatten_trace(const std::vector<std::vector<uint64_t>>& trace) {
    std::vector<uint64_t> flat;
    if (trace.empty()) return flat;
    size_t cols = trace[0].size();
    flat.reserve(trace.size() * cols);
    for (const auto& row : trace) {
        for (size_t c = 0; c < cols; ++c) {
            flat.push_back(c < row.size() ? row[c] : 0);
        }
    }
    return flat;
}

GpuAETData* gpu_upload_aet(
    const std::vector<std::vector<uint64_t>>& program_trace,
    const std::vector<std::vector<uint64_t>>& processor_trace,
    const std::vector<std::vector<uint64_t>>& op_stack_trace,
    const std::vector<std::vector<uint64_t>>& ram_trace,
    const std::vector<std::vector<uint64_t>>& jump_stack_trace,
    const std::vector<std::vector<uint64_t>>& program_hash_trace,
    const std::vector<std::vector<uint64_t>>& sponge_trace,
    const std::vector<std::vector<uint64_t>>& hash_trace,
    const std::vector<std::vector<uint64_t>>& cascade_trace,
    const std::vector<std::vector<uint64_t>>& lookup_trace,
    const std::vector<std::vector<uint64_t>>& u32_trace,
    cudaStream_t stream
) {
    GpuAETData* aet = new GpuAETData();

    // Program trace
    aet->program_rows = program_trace.size();
    if (aet->program_rows > 0) {
        auto flat = flatten_trace(program_trace);
        size_t size = flat.size() * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_program_trace, size));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_program_trace, flat.data(), size, cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_program_trace = nullptr;
    }
    
    // Processor trace
    aet->processor_rows = processor_trace.size();
    if (aet->processor_rows > 0) {
        auto flat = flatten_trace(processor_trace);
        size_t size = flat.size() * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_processor_trace, size));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_processor_trace, flat.data(), size, cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_processor_trace = nullptr;
    }
    
    // OpStack trace
    aet->op_stack_rows = op_stack_trace.size();
    if (aet->op_stack_rows > 0) {
        auto flat = flatten_trace(op_stack_trace);
        size_t size = flat.size() * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_op_stack_trace, size));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_op_stack_trace, flat.data(), size, cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_op_stack_trace = nullptr;
    }
    
    // RAM trace
    aet->ram_rows = ram_trace.size();
    if (aet->ram_rows > 0) {
        auto flat = flatten_trace(ram_trace);
        size_t size = flat.size() * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_ram_trace, size));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_ram_trace, flat.data(), size, cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_ram_trace = nullptr;
    }
    
    // JumpStack trace
    aet->jump_stack_rows = jump_stack_trace.size();
    if (aet->jump_stack_rows > 0) {
        auto flat = flatten_trace(jump_stack_trace);
        size_t size = flat.size() * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_jump_stack_trace, size));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_jump_stack_trace, flat.data(), size, cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_jump_stack_trace = nullptr;
    }
    
    // Combined hash traces
    aet->program_hash_rows = program_hash_trace.size();
    aet->sponge_rows = sponge_trace.size();
    aet->hash_rows = hash_trace.size();
    size_t total_hash_rows = aet->program_hash_rows + aet->sponge_rows + aet->hash_rows;
    if (total_hash_rows > 0) {
        std::vector<uint64_t> combined;
        combined.reserve(total_hash_rows * TableOffsets::HASH_TABLE_COLS);
        for (const auto& trace : {program_hash_trace, sponge_trace, hash_trace}) {
            for (const auto& row : trace) {
                for (size_t c = 0; c < TableOffsets::HASH_TABLE_COLS; ++c) {
                    combined.push_back(c < row.size() ? row[c] : 0);
                }
            }
        }
        CUDA_CHECK(cudaMalloc(&aet->d_hash_trace, combined.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_hash_trace, combined.data(), 
            combined.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_hash_trace = nullptr;
    }
    
    // Cascade trace
    aet->cascade_rows = cascade_trace.size();
    if (aet->cascade_rows > 0) {
        auto flat = flatten_trace(cascade_trace);
        CUDA_CHECK(cudaMalloc(&aet->d_cascade_trace, flat.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_cascade_trace, flat.data(), 
            flat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_cascade_trace = nullptr;
    }
    
    // Lookup trace
    aet->lookup_rows = lookup_trace.size();
    if (aet->lookup_rows > 0) {
        auto flat = flatten_trace(lookup_trace);
        CUDA_CHECK(cudaMalloc(&aet->d_lookup_trace, flat.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_lookup_trace, flat.data(), 
            flat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_lookup_trace = nullptr;
    }
    
    // U32 trace
    aet->u32_rows = u32_trace.size();
    if (aet->u32_rows > 0) {
        auto flat = flatten_trace(u32_trace);
        CUDA_CHECK(cudaMalloc(&aet->d_u32_trace, flat.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_u32_trace, flat.data(), 
            flat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    } else {
        aet->d_u32_trace = nullptr;
    }

    // Deprecated Bézout fields (RAM trace now carries bezout coeffs)
    aet->d_bezout_a = nullptr;
    aet->d_bezout_b = nullptr;
    aet->bezout_len = 0;
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return aet;
}

GpuAETData* gpu_upload_aet_flat(
    const uint64_t* h_program_trace, size_t program_rows,
    const uint64_t* h_processor_trace, size_t processor_rows,
    const uint64_t* h_op_stack_trace, size_t op_stack_rows,
    const uint64_t* h_ram_trace, size_t ram_rows,
    const uint64_t* h_jump_stack_trace, size_t jump_stack_rows,
    const uint64_t* h_hash_trace, size_t program_hash_rows, size_t sponge_rows, size_t hash_rows,
    const uint64_t* h_cascade_trace, size_t cascade_rows,
    const uint64_t* h_lookup_trace, size_t lookup_rows,
    const uint64_t* h_u32_trace, size_t u32_rows,
    cudaStream_t stream
) {
    GpuAETData* aet = new GpuAETData();
    aet->d_program_trace = nullptr; aet->program_rows = program_rows;
    aet->d_processor_trace = nullptr; aet->processor_rows = processor_rows;
    aet->d_op_stack_trace = nullptr; aet->op_stack_rows = op_stack_rows;
    aet->d_ram_trace = nullptr; aet->ram_rows = ram_rows;
    aet->d_jump_stack_trace = nullptr; aet->jump_stack_rows = jump_stack_rows;
    aet->d_hash_trace = nullptr; aet->program_hash_rows = program_hash_rows; aet->sponge_rows = sponge_rows; aet->hash_rows = hash_rows;
    aet->d_cascade_trace = nullptr; aet->cascade_rows = cascade_rows;
    aet->d_lookup_trace = nullptr; aet->lookup_rows = lookup_rows;
    aet->d_u32_trace = nullptr; aet->u32_rows = u32_rows;
    aet->d_unique_ramps = nullptr; aet->num_unique_ramps = 0;
    aet->d_bezout_a = nullptr; aet->d_bezout_b = nullptr; aet->bezout_len = 0;

    if (program_rows && h_program_trace) {
        size_t bytes = program_rows * TableOffsets::PROGRAM_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_program_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_program_trace, h_program_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (processor_rows && h_processor_trace) {
        size_t bytes = processor_rows * TableOffsets::PROCESSOR_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_processor_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_processor_trace, h_processor_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (op_stack_rows && h_op_stack_trace) {
        size_t bytes = op_stack_rows * TableOffsets::OP_STACK_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_op_stack_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_op_stack_trace, h_op_stack_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (ram_rows && h_ram_trace) {
        size_t bytes = ram_rows * TableOffsets::RAM_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_ram_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_ram_trace, h_ram_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (jump_stack_rows && h_jump_stack_trace) {
        size_t bytes = jump_stack_rows * TableOffsets::JUMP_STACK_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_jump_stack_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_jump_stack_trace, h_jump_stack_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    size_t total_hash_rows = program_hash_rows + sponge_rows + hash_rows;
    if (total_hash_rows && h_hash_trace) {
        size_t bytes = total_hash_rows * TableOffsets::HASH_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_hash_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_hash_trace, h_hash_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (cascade_rows && h_cascade_trace) {
        size_t bytes = cascade_rows * TableOffsets::CASCADE_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_cascade_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_cascade_trace, h_cascade_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (lookup_rows && h_lookup_trace) {
        size_t bytes = lookup_rows * TableOffsets::LOOKUP_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_lookup_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_lookup_trace, h_lookup_trace, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (u32_rows && h_u32_trace) {
        size_t bytes = u32_rows * TableOffsets::U32_TABLE_COLS * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&aet->d_u32_trace, bytes));
        CUDA_CHECK(cudaMemcpyAsync(aet->d_u32_trace, h_u32_trace, bytes, cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return aet;
}

void gpu_free_aet(GpuAETData* aet) {
    if (!aet) return;
    if (aet->d_program_trace) cudaFree(aet->d_program_trace);
    if (aet->d_processor_trace) cudaFree(aet->d_processor_trace);
    if (aet->d_op_stack_trace) cudaFree(aet->d_op_stack_trace);
    if (aet->d_ram_trace) cudaFree(aet->d_ram_trace);
    if (aet->d_jump_stack_trace) cudaFree(aet->d_jump_stack_trace);
    if (aet->d_hash_trace) cudaFree(aet->d_hash_trace);
    if (aet->d_cascade_trace) cudaFree(aet->d_cascade_trace);
    if (aet->d_lookup_trace) cudaFree(aet->d_lookup_trace);
    if (aet->d_u32_trace) cudaFree(aet->d_u32_trace);
    if (aet->d_bezout_a) cudaFree(aet->d_bezout_a);
    if (aet->d_bezout_b) cudaFree(aet->d_bezout_b);
    delete aet;
}

static void gpu_create_main_table_impl(
    const GpuAETData* aet,
    uint64_t* d_main_table,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream
) {
    constexpr size_t NUM_COLS = TableOffsets::TOTAL_COLS;
    constexpr size_t BLOCK_SIZE = 256;

    size_t table_size = padded_height * NUM_COLS * sizeof(uint64_t);
    CUDA_CHECK(cudaMemsetAsync(d_main_table, 0, table_size, stream));

    std::cout << "[GPU Phase1] Creating main table: " << padded_height << " x " << NUM_COLS << std::endl;

    // 1. Sort traces that need sorting
    std::cout << "[GPU Phase1] Sorting OpStack trace..." << std::endl;
    if (aet->op_stack_rows > 1) {
        gpu_sort_trace_by_two_keys(aet->d_op_stack_trace, aet->op_stack_rows,
            TableOffsets::OP_STACK_TABLE_COLS, 2, 0, stream);  // Sort by (StackPointer, CLK)
    }

    std::cout << "[GPU Phase1] Sorting RAM trace..." << std::endl;
    if (aet->ram_rows > 1) {
        gpu_sort_trace_by_two_keys(aet->d_ram_trace, aet->ram_rows,
            TableOffsets::RAM_TABLE_COLS, 2, 0, stream);  // Sort by (RamPointer, CLK)
    }

    std::cout << "[GPU Phase1] Sorting JumpStack trace..." << std::endl;
    if (aet->jump_stack_rows > 1) {
        gpu_sort_trace_by_two_keys(aet->d_jump_stack_trace, aet->jump_stack_rows,
            TableOffsets::JUMP_STACK_TABLE_COLS, 2, 0, stream);  // Sort by (JSP, CLK)
    }

    // 2. Fill all table columns
    std::cout << "[GPU Phase1] Filling table columns..." << std::endl;

    // Program table
    if (aet->program_rows > 0) {
        size_t blocks = (aet->program_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_program_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_program_trace, d_main_table, aet->program_rows, NUM_COLS);
    }

    // Processor table
    if (aet->processor_rows > 0) {
        size_t blocks = (aet->processor_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_processor_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_processor_trace, d_main_table, aet->processor_rows,
            padded_height, NUM_COLS, TableOffsets::PROCESSOR_TABLE_START);
    }

    // OpStack table
    if (aet->op_stack_rows > 0) {
        size_t blocks = (aet->op_stack_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_op_stack_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_op_stack_trace, d_main_table, aet->op_stack_rows,
            padded_height, NUM_COLS, TableOffsets::OP_STACK_TABLE_START);
    }

    // RAM table (precomputed inv + bezout included in RAM trace)
    if (aet->ram_rows > 0) {
        size_t blocks = (aet->ram_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_ram_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_ram_trace,
            d_main_table, aet->ram_rows, padded_height, NUM_COLS, TableOffsets::RAM_TABLE_START);
    }

    // JumpStack table
    if (aet->jump_stack_rows > 0) {
        size_t blocks = (aet->jump_stack_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_jump_stack_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_jump_stack_trace, d_main_table, aet->jump_stack_rows,
            padded_height, NUM_COLS, TableOffsets::JUMP_STACK_TABLE_START);
    }

    // Hash table
    size_t total_hash = aet->program_hash_rows + aet->sponge_rows + aet->hash_rows;
    if (total_hash > 0) {
        size_t blocks = (total_hash + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_hash_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_hash_trace, d_main_table, aet->program_hash_rows, aet->sponge_rows,
            aet->hash_rows, padded_height, NUM_COLS, TableOffsets::HASH_TABLE_START);
    }

    // Cascade table
    if (aet->cascade_rows > 0) {
        size_t blocks = (aet->cascade_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_cascade_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_cascade_trace, d_main_table, aet->cascade_rows,
            padded_height, NUM_COLS, TableOffsets::CASCADE_TABLE_START);
    }

    // Lookup table
    if (aet->lookup_rows > 0) {
        size_t blocks = (aet->lookup_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_lookup_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_lookup_trace, d_main_table, aet->lookup_rows,
            padded_height, NUM_COLS, TableOffsets::LOOKUP_TABLE_START);
    }

    // U32 table
    if (aet->u32_rows > 0) {
        size_t blocks = (aet->u32_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_u32_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_u32_trace, d_main_table, aet->u32_rows,
            padded_height, NUM_COLS, TableOffsets::U32_TABLE_START);
    }

    // 3. Compute clock jump multiplicities
    std::cout << "[GPU Phase1] Computing clock jump multiplicities..." << std::endl;
    if (aet->op_stack_rows > 1) {
        size_t blocks = (aet->op_stack_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_clock_jump_diffs_op_stack_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_op_stack_trace, aet->op_stack_rows, d_main_table, padded_height, NUM_COLS);
    }
    if (aet->ram_rows > 1) {
        size_t blocks = (aet->ram_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_clock_jump_diffs_ram_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_ram_trace, aet->ram_rows, d_main_table, padded_height, NUM_COLS);
    }
    if (aet->jump_stack_rows > 1) {
        size_t blocks = (aet->jump_stack_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_clock_jump_diffs_jump_stack_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            aet->d_jump_stack_trace, aet->jump_stack_rows, d_main_table, padded_height, NUM_COLS);
    }

    // 4. Pad tables
    std::cout << "[GPU Phase1] Padding tables..." << std::endl;
    gpu_pad_main_table(d_main_table, NUM_COLS, padded_height, table_lengths, stream);

    // 5. Compute degree-lowering columns (149-378)
    std::cout << "[GPU Phase1] Computing degree-lowering columns..." << std::endl;
    gpu_degree_lowering_main(d_main_table, padded_height, NUM_COLS, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[GPU Phase1] Main table creation complete" << std::endl;
}

uint64_t* gpu_create_main_table(
    const GpuAETData* aet,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream
) {
    constexpr size_t NUM_COLS = TableOffsets::TOTAL_COLS;
    uint64_t* d_main_table;
    size_t table_size = padded_height * NUM_COLS * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_main_table, table_size));
    gpu_create_main_table_impl(aet, d_main_table, padded_height, table_lengths, stream);
    return d_main_table;
}

void gpu_create_main_table_into(
    const GpuAETData* aet,
    uint64_t* d_main_table,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream
) {
    gpu_create_main_table_impl(aet, d_main_table, padded_height, table_lengths, stream);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

