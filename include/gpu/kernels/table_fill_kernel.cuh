#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * GPU kernel to pad processor table rows
 * Copies last real row to all padding rows, updating CLK and IsPadding columns
 */
__global__ void pad_processor_table_kernel(
    uint64_t* d_table,           // Row-major table [num_rows * num_cols]
    size_t num_cols,             // 379 columns
    size_t processor_table_start, // Column offset for processor table
    size_t processor_table_cols,  // 39 columns
    size_t table_len,            // Original processor table length
    size_t padded_height,        // Target padded height
    size_t clk_col,              // CLK column index within processor table
    size_t is_padding_col,       // IsPadding column index within processor table
    size_t cjd_col               // ClockJumpDifferenceLookupMultiplicity column index
);

/**
 * GPU kernel to pad program table rows
 */
__global__ void pad_program_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t program_table_start,
    size_t program_len,
    size_t padded_height,
    size_t rate  // TIP5 rate = 10
);

/**
 * GPU kernel to pad op stack table rows
 */
__global__ void pad_op_stack_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t op_stack_table_start,
    size_t table_len,
    size_t padded_height
);

/**
 * GPU kernel to pad RAM table rows
 */
__global__ void pad_ram_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t ram_table_start,
    size_t table_len,
    size_t padded_height
);

/**
 * GPU kernel to pad jump stack table rows
 */
__global__ void pad_jump_stack_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t jump_stack_table_start,
    size_t table_len,
    size_t padded_height
);

/**
 * GPU kernel to pad hash table rows
 */
__global__ void pad_hash_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t hash_table_start,
    size_t hash_table_cols,
    size_t table_len,
    size_t padded_height
);

/**
 * GPU kernel to pad cascade table rows
 */
__global__ void pad_cascade_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t cascade_table_start,
    size_t table_len,
    size_t padded_height
);

/**
 * GPU kernel to pad lookup table rows
 */
__global__ void pad_lookup_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t lookup_table_start,
    size_t table_len,
    size_t padded_height
);

/**
 * GPU kernel to pad U32 table rows
 */
__global__ void pad_u32_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t u32_table_start,
    size_t table_len,
    size_t padded_height
);

/**
 * Host function to pad entire main table on GPU
 * Launches all padding kernels
 */
void gpu_pad_main_table(
    uint64_t* d_table,
    size_t num_cols,
    size_t padded_height,
    const size_t table_lengths[9],  // Lengths of each of the 9 tables
    cudaStream_t stream = 0
);

/**
 * Host function to upload unpadded table and pad on GPU
 * Returns padded table in GPU memory
 */
uint64_t* gpu_upload_and_pad_table(
    const uint64_t* h_unpadded_table,  // Host: unpadded table data
    size_t unpadded_rows,
    size_t num_cols,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

