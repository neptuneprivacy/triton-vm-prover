#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Forward declaration
struct AETData;

/**
 * GPU Phase 1: Create and pad main table from AET data
 * 
 * This performs all Phase 1 operations on GPU after CPU trace execution:
 * 1. Upload AET traces to GPU
 * 2. Sort traces (OpStack, RAM, JumpStack)
 * 3. Fill main table columns
 * 4. Pad tables
 * 5. Compute degree-lowering columns
 */

// ============================================================================
// AET Data Structure for GPU Upload
// ============================================================================

/**
 * Flattened AET data for GPU upload
 * All traces are stored as flat arrays for efficient transfer
 */
struct GpuAETData {
    // Program table trace: [rows][7]
    uint64_t* d_program_trace;
    size_t program_rows;

    // Processor trace: [num_rows][39 columns]
    uint64_t* d_processor_trace;
    size_t processor_rows;
    
    // OpStack trace: [num_rows][4 columns] - needs sorting
    uint64_t* d_op_stack_trace;
    size_t op_stack_rows;
    
    // RAM trace: [num_rows][7 columns] - needs sorting
    // Columns match main RAM table columns (CLK, InstrType, Ramp, RamVal, InvRampDiff, Bezout0, Bezout1)
    uint64_t* d_ram_trace;
    size_t ram_rows;
    
    // JumpStack trace: [num_rows][5 columns] - needs sorting
    uint64_t* d_jump_stack_trace;
    size_t jump_stack_rows;
    
    // Hash traces combined: [num_rows][67 columns]
    uint64_t* d_hash_trace;  // program_hash + sponge + hash traces concatenated
    size_t program_hash_rows;
    size_t sponge_rows;
    size_t hash_rows;
    
    // Cascade trace: [num_rows][6 columns]
    uint64_t* d_cascade_trace;
    size_t cascade_rows;
    
    // Lookup trace: [num_rows][4 columns]
    uint64_t* d_lookup_trace;
    size_t lookup_rows;
    
    // U32 trace: [num_rows][10 columns]
    uint64_t* d_u32_trace;
    size_t u32_rows;
    
    // Unique RAM pointers for Bézout computation
    uint64_t* d_unique_ramps;
    size_t num_unique_ramps;
    
    // (Deprecated) Bézout coefficients: no longer used by Phase 1 RAM fill (RAM trace carries them)
    uint64_t* d_bezout_a;
    uint64_t* d_bezout_b;
    size_t bezout_len;
};

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * GPU kernel to sort trace by (key1, key2) - radix sort based
 * Used for OpStack (StackPointer, CLK), RAM (RamPointer, CLK), JumpStack (JumpStackPointer, CLK)
 */
void gpu_sort_trace_by_two_keys(
    uint64_t* d_trace,           // Trace data [rows * cols]
    size_t num_rows,
    size_t num_cols,
    size_t key1_col,             // First key column index
    size_t key2_col,             // Second key column index
    cudaStream_t stream = 0
);

/**
 * GPU kernel to fill processor table columns from trace
 * Simple parallel copy - each thread handles one row
 */
__global__ void fill_processor_table_kernel(
    const uint64_t* d_processor_trace,  // [rows][39]
    uint64_t* d_main_table,              // [padded_height][379]
    size_t processor_rows,
    size_t padded_height,
    size_t num_cols,                     // 379
    size_t processor_table_start         // 7
);

/**
 * GPU kernel to fill OpStack table columns from sorted trace
 */
__global__ void fill_op_stack_table_kernel(
    const uint64_t* d_sorted_op_stack,   // [rows][4]
    uint64_t* d_main_table,              // [padded_height][379]
    size_t op_stack_rows,
    size_t padded_height,
    size_t num_cols,
    size_t op_stack_table_start          // 46
);

/**
 * GPU kernel to fill RAM table columns from sorted trace
 * Includes inverse computation and Bézout coefficients
 */
__global__ void fill_ram_table_kernel(
    const uint64_t* d_sorted_ram,        // [rows][7] already contains inv + bezout coeffs
    uint64_t* d_main_table,              // [padded_height][379]
    size_t ram_rows,
    size_t padded_height,
    size_t num_cols,
    size_t ram_table_start               // 50
);

/**
 * GPU kernel to fill JumpStack table columns from sorted trace
 */
__global__ void fill_jump_stack_table_kernel(
    const uint64_t* d_sorted_jump_stack, // [rows][5]
    uint64_t* d_main_table,              // [padded_height][379]
    size_t jump_stack_rows,
    size_t padded_height,
    size_t num_cols,
    size_t jump_stack_table_start        // 57
);

/**
 * GPU kernel to fill Hash table columns from combined traces
 */
__global__ void fill_hash_table_kernel(
    const uint64_t* d_hash_trace,        // Combined [rows][67]
    uint64_t* d_main_table,
    size_t program_hash_rows,
    size_t sponge_rows,
    size_t hash_rows,
    size_t padded_height,
    size_t num_cols,
    size_t hash_table_start              // 62
);

/**
 * GPU kernel to fill Cascade table columns
 */
__global__ void fill_cascade_table_kernel(
    const uint64_t* d_cascade_trace,
    uint64_t* d_main_table,
    size_t cascade_rows,
    size_t padded_height,
    size_t num_cols,
    size_t cascade_table_start           // 129
);

/**
 * GPU kernel to fill Lookup table columns
 */
__global__ void fill_lookup_table_kernel(
    const uint64_t* d_lookup_trace,
    uint64_t* d_main_table,
    size_t lookup_rows,
    size_t padded_height,
    size_t num_cols,
    size_t lookup_table_start            // 135
);

/**
 * GPU kernel to fill U32 table columns
 */
__global__ void fill_u32_table_kernel(
    const uint64_t* d_u32_trace,
    uint64_t* d_main_table,
    size_t u32_rows,
    size_t padded_height,
    size_t num_cols,
    size_t u32_table_start               // 139
);

/**
 * GPU kernel to compute clock jump difference multiplicities
 * For processor table ClockJumpDifferenceLookupMultiplicity column
 */
__global__ void compute_clock_jump_multiplicities_kernel(
    const uint64_t* d_sorted_op_stack,
    const uint64_t* d_sorted_ram,
    const uint64_t* d_sorted_jump_stack,
    size_t op_stack_rows,
    size_t ram_rows,
    size_t jump_stack_rows,
    uint64_t* d_main_table,
    size_t padded_height,
    size_t num_cols
);

// Specialized kernels matching CPU semantics for clock jump diffs:
// - RAM: record clk diffs when ramp stays constant
// - JumpStack: record clk diffs when jsp stays constant
__global__ void compute_clock_jump_diffs_ram_kernel(
    const uint64_t* d_sorted_ram,        // [rows][7], ramp at col 2, clk at col 0
    size_t ram_rows,
    uint64_t* d_main_table,
    size_t padded_height,
    size_t num_cols
);

__global__ void compute_clock_jump_diffs_jump_stack_kernel(
    const uint64_t* d_sorted_jump_stack, // [rows][5], jsp at col 2, clk at col 0
    size_t jump_stack_rows,
    uint64_t* d_main_table,
    size_t padded_height,
    size_t num_cols
);

/**
 * GPU kernel to compute degree-lowering main columns
 * Columns 149-378 (230 columns)
 */
__global__ void compute_degree_lowering_main_kernel(
    uint64_t* d_main_table,
    size_t padded_height,
    size_t num_cols
);

// ============================================================================
// Host Functions
// ============================================================================

/**
 * Allocate and upload AET data to GPU
 */
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
    cudaStream_t stream = 0
);

/**
 * Upload AET data to GPU from flat host buffers (preferred for performance).
 * All buffers are row-major contiguous arrays.
 */
GpuAETData* gpu_upload_aet_flat(
    const uint64_t* h_program_trace, size_t program_rows,        // [rows][7]
    const uint64_t* h_processor_trace, size_t processor_rows,    // [rows][39]
    const uint64_t* h_op_stack_trace, size_t op_stack_rows,      // [rows][4]
    const uint64_t* h_ram_trace, size_t ram_rows,                // [rows][7]
    const uint64_t* h_jump_stack_trace, size_t jump_stack_rows,  // [rows][5]
    const uint64_t* h_hash_trace, size_t program_hash_rows, size_t sponge_rows, size_t hash_rows, // [sum][67]
    const uint64_t* h_cascade_trace, size_t cascade_rows,        // [rows][6]
    const uint64_t* h_lookup_trace, size_t lookup_rows,          // [rows][4]
    const uint64_t* h_u32_trace, size_t u32_rows,                // [rows][10]
    cudaStream_t stream = 0
);

/**
 * Free GPU AET data
 */
void gpu_free_aet(GpuAETData* aet_data);

/**
 * Complete GPU Phase 1 pipeline
 * Creates and pads main table from AET data
 * 
 * @param aet_data GPU-resident AET data
 * @param padded_height Target padded height (power of 2)
 * @param table_lengths Lengths of each of the 9 tables
 * @param stream CUDA stream
 * @return Device pointer to padded main table [padded_height * 379]
 */
uint64_t* gpu_create_main_table(
    const GpuAETData* aet_data,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream = 0
);

/**
 * Same as gpu_create_main_table, but writes into an existing device/unified buffer.
 * The destination must have space for padded_height * 379 u64 elements.
 */
void gpu_create_main_table_into(
    const GpuAETData* aet_data,
    uint64_t* d_main_table,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream = 0
);

/**
 * Combined Phase 1: Upload AET, create table, pad, return GPU table
 */
uint64_t* gpu_phase1_from_aet(
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
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

