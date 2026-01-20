#include "field_arithmetic.cuh"
#include <cstdint>

// ============================================================================
// GPU Kernel: Extract Specific Rows from Row-Major Table
// ============================================================================
//
// This kernel extracts specific rows from a large GPU-resident table without
// downloading the entire table to CPU. This is used for FRI proof revelation
// where only ~80 rows out of millions are needed (~48KB instead of 80GB).
//
// Memory Layout:
// - Input table is row-major: [row0_col0, row0_col1, ..., row1_col0, ...]
// - Each row has `num_columns` elements
// - For BField: each element is 1 u64
// - For XField: each element is 3 u64s (stored consecutively)
//
// Thread Assignment:
// - One thread per output element
// - Global thread index determines which (row_idx, col_idx) to extract
//
// ============================================================================

extern "C" __global__ void extract_rows_bfield(
    const uint64_t* __restrict__ input_table,   // Input: full table on GPU
    const uint32_t* __restrict__ row_indices,   // Input: which rows to extract
    uint64_t* __restrict__ output,              // Output: extracted rows
    uint32_t num_rows_to_extract,               // How many rows to extract
    uint32_t num_columns                        // Columns per row
) {
    // Global thread index
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of output elements
    uint32_t total_output_elements = num_rows_to_extract * num_columns;

    if (tid >= total_output_elements) {
        return;  // Out of bounds
    }

    // Compute which output row and column this thread handles
    uint32_t output_row_idx = tid / num_columns;
    uint32_t col_idx = tid % num_columns;

    // Get the actual source row index
    uint32_t source_row_idx = row_indices[output_row_idx];

    // Compute source position in input table (row-major)
    uint64_t source_offset = (uint64_t)source_row_idx * num_columns + col_idx;

    // Copy element
    output[tid] = input_table[source_offset];
}

extern "C" __global__ void extract_rows_xfield(
    const uint64_t* __restrict__ input_table,   // Input: full table on GPU (3 u64s per XField)
    const uint32_t* __restrict__ row_indices,   // Input: which rows to extract
    uint64_t* __restrict__ output,              // Output: extracted rows
    uint32_t num_rows_to_extract,               // How many rows to extract
    uint32_t num_columns                        // Columns per row (XField count, not u64 count)
) {
    // Global thread index
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of output XField elements
    uint32_t total_output_xfields = num_rows_to_extract * num_columns;

    if (tid >= total_output_xfields) {
        return;  // Out of bounds
    }

    // Compute which output row and column this thread handles
    uint32_t output_row_idx = tid / num_columns;
    uint32_t col_idx = tid % num_columns;

    // Get the actual source row index
    uint32_t source_row_idx = row_indices[output_row_idx];

    // Compute source position in input table (row-major, 3 u64s per XField)
    // Source offset in terms of XField elements
    uint64_t source_xfield_offset = (uint64_t)source_row_idx * num_columns + col_idx;
    // Convert to u64 offset (3 u64s per XField)
    uint64_t source_u64_offset = source_xfield_offset * 3;
    uint64_t output_u64_offset = tid * 3;

    // Copy 3 u64s for this XField element
    output[output_u64_offset + 0] = input_table[source_u64_offset + 0];
    output[output_u64_offset + 1] = input_table[source_u64_offset + 1];
    output[output_u64_offset + 2] = input_table[source_u64_offset + 2];
}

// ============================================================================
// Optimization Notes:
// ============================================================================
//
// 1. Coalesced Memory Access:
//    - Consecutive threads access consecutive memory locations
//    - For BField: perfect coalescing (1 u64 per thread)
//    - For XField: good coalescing (3 u64s per thread, consecutive)
//
// 2. Performance Characteristics:
//    - Memory-bound kernel (pure copy operation)
//    - For ~80 rows × 379 columns (BField) = ~30K elements
//    - Bandwidth-limited, not compute-limited
//    - Expected: ~10ms for 48KB transfer vs 6s for 80GB
//
// 3. Thread Configuration:
//    - Recommended: 256 threads per block
//    - Grid size: (total_elements + 255) / 256 blocks
//
// 4. Future Optimizations (if needed):
//    - Shared memory staging (unlikely to help for such small transfers)
//    - Vector loads (e.g., float4) for better memory throughput
//    - Currently simple approach should be sufficient
//
// ============================================================================
