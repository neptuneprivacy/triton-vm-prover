#pragma once

/**
 * Gather Kernel CUDA Declarations
 * 
 * GPU-accelerated gathering of values at specified indices.
 * Used for opening trace leaves at FRI query indices.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Gather BFieldElement rows from a table at specified indices
 * 
 * @param d_table Input table (num_rows * row_width uint64_t)
 * @param d_indices Query indices (num_indices elements)
 * @param d_output Output gathered values (num_indices * row_width uint64_t)
 * @param num_rows Number of rows in the table
 * @param row_width Number of BFieldElements per row
 * @param num_indices Number of indices to gather
 * @param stream CUDA stream
 */
void gather_bfield_rows_gpu(
    const uint64_t* d_table,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream = 0
);

/**
 * Gather XFieldElement rows from a table at specified indices
 * Each XFE is stored as 3 consecutive uint64_t values
 * 
 * @param d_table Input table (num_rows * row_width * 3 uint64_t)
 * @param d_indices Query indices
 * @param d_output Output (num_indices * row_width * 3 uint64_t)
 * @param num_rows Number of rows
 * @param row_width Number of XFieldElements per row
 * @param num_indices Number of indices to gather
 * @param stream CUDA stream
 */
void gather_xfield_rows_gpu(
    const uint64_t* d_table,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream = 0
);

/**
 * Gather BFieldElement column values at specified indices
 * 
 * @param d_column Input column (num_rows uint64_t)
 * @param d_indices Query indices
 * @param d_output Output (num_indices uint64_t)
 * @param num_rows Number of elements in column
 * @param num_indices Number of indices to gather
 * @param stream CUDA stream
 */
void gather_column_gpu(
    const uint64_t* d_column,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_indices,
    cudaStream_t stream = 0
);

/**
 * Gather XFieldElement column values at specified indices
 * 
 * @param d_column Input column (num_rows * 3 uint64_t)
 * @param d_indices Query indices
 * @param d_output Output (num_indices * 3 uint64_t)
 * @param num_rows Number of XFieldElements in column
 * @param num_indices Number of indices to gather
 * @param stream CUDA stream
 */
void gather_xfield_column_gpu(
    const uint64_t* d_column,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_indices,
    cudaStream_t stream = 0
);

/**
 * Gather BField rows from a column-major table (col * num_rows + row) into row-major output.
 */
void gather_bfield_rows_colmajor_gpu(
    const uint64_t* d_table_colmajor,
    const size_t* d_indices,
    uint64_t* d_output_rowmajor,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream = 0
);

/**
 * Gather XField rows from a component-major column table:
 *   input layout: (xfe_col*3 + comp)*num_rows + row
 *   output layout: (index_idx*row_width + xfe_col)*3 + comp
 */
void gather_xfield_rows_colmajor_gpu(
    const uint64_t* d_table_comp_colmajor,
    const size_t* d_indices,
    uint64_t* d_output_rowmajor,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream = 0
);

/**
 * Scatter values back to table at specified indices (inverse of gather)
 * 
 * @param d_values Source values to scatter
 * @param d_indices Target indices
 * @param d_table Destination table
 * @param num_rows Number of rows in table
 * @param row_width Number of elements per row
 * @param num_indices Number of indices
 * @param stream CUDA stream
 */
void scatter_bfield_gpu(
    const uint64_t* d_values,
    const size_t* d_indices,
    uint64_t* d_table,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

