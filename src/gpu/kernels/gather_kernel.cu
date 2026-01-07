/**
 * Gather Kernel CUDA Implementation
 * 
 * GPU-accelerated gathering of values at specified indices.
 * Used for opening trace leaves at FRI query indices.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Gather Kernels
// ============================================================================

/**
 * Gather BFieldElements from a table at specified indices
 * 
 * @param d_table Input table (num_rows * row_width BFieldElements)
 * @param d_indices Query indices (num_indices elements)
 * @param d_output Output gathered values (num_indices * row_width BFieldElements)
 * @param num_rows Number of rows in the table
 * @param row_width Number of BFieldElements per row
 * @param num_indices Number of indices to gather
 */
__global__ void gather_bfield_rows_kernel(
    const uint64_t* d_table,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t row_width,
    size_t num_indices
) {
    // Each thread handles one (index, column) pair
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = num_indices * row_width;
    
    if (tid >= total_elements) return;
    
    size_t index_idx = tid / row_width;
    size_t col_idx = tid % row_width;
    
    size_t row_idx = d_indices[index_idx];
    
    // Bounds check
    if (row_idx < num_rows) {
        d_output[tid] = d_table[row_idx * row_width + col_idx];
    } else {
        d_output[tid] = 0;  // Out of bounds
    }
}

/**
 * Gather XFieldElements from a table at specified indices
 * Each XFE is stored as 3 consecutive uint64_t values
 * 
 * @param d_table Input table (num_rows * row_width * 3 uint64_t)
 * @param d_indices Query indices
 * @param d_output Output gathered values (num_indices * row_width * 3 uint64_t)
 * @param num_rows Number of rows
 * @param row_width Number of XFieldElements per row
 * @param num_indices Number of indices to gather
 */
__global__ void gather_xfield_rows_kernel(
    const uint64_t* d_table,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t row_width,
    size_t num_indices
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = num_indices * row_width * 3;
    
    if (tid >= total_elements) return;
    
    size_t flat_idx = tid / 3;  // Which (index, column) pair
    size_t coeff_idx = tid % 3;  // Which coefficient of XFE
    
    size_t index_idx = flat_idx / row_width;
    size_t col_idx = flat_idx % row_width;
    
    size_t row_idx = d_indices[index_idx];
    
    if (row_idx < num_rows) {
        size_t src_offset = (row_idx * row_width + col_idx) * 3 + coeff_idx;
        d_output[tid] = d_table[src_offset];
    } else {
        d_output[tid] = 0;
    }
}

/**
 * Gather single column values at specified indices
 * 
 * @param d_column Input column (num_rows BFieldElements)
 * @param d_indices Query indices
 * @param d_output Output gathered values (num_indices BFieldElements)
 * @param num_rows Number of elements in column
 * @param num_indices Number of indices to gather
 */
__global__ void gather_column_kernel(
    const uint64_t* d_column,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_indices
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;
    
    size_t row_idx = d_indices[idx];
    
    if (row_idx < num_rows) {
        d_output[idx] = d_column[row_idx];
    } else {
        d_output[idx] = 0;
    }
}

/**
 * Gather XFE column values at specified indices
 */
__global__ void gather_xfield_column_kernel(
    const uint64_t* d_column,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_indices
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = num_indices * 3;
    
    if (tid >= total_elements) return;
    
    size_t index_idx = tid / 3;
    size_t coeff_idx = tid % 3;
    
    size_t row_idx = d_indices[index_idx];
    
    if (row_idx < num_rows) {
        d_output[tid] = d_column[row_idx * 3 + coeff_idx];
    } else {
        d_output[tid] = 0;
    }
}

/**
 * Gather from column-major BFE table into row-major output.
 */
__global__ void gather_bfield_rows_colmajor_kernel(
    const uint64_t* d_table,   // col * num_rows + row
    const size_t* d_indices,
    uint64_t* d_output,        // index_idx * row_width + col
    size_t num_rows,
    size_t row_width,
    size_t num_indices
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_indices * row_width;
    if (tid >= total) return;
    size_t index_idx = tid / row_width;
    size_t col_idx = tid % row_width;
    size_t row_idx = d_indices[index_idx];
    if (row_idx < num_rows) {
        d_output[tid] = d_table[col_idx * num_rows + row_idx];
    } else {
        d_output[tid] = 0;
    }
}

/**
 * Gather from component-major column table into row-major XFE output.
 */
__global__ void gather_xfield_rows_colmajor_kernel(
    const uint64_t* d_table,   // (xfe_col*3+comp)*num_rows + row
    const size_t* d_indices,
    uint64_t* d_output,        // (index_idx*row_width + xfe_col)*3 + comp
    size_t num_rows,
    size_t row_width,
    size_t num_indices
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_indices * row_width * 3;
    if (tid >= total) return;
    size_t flat = tid / 3;
    size_t comp = tid % 3;
    size_t index_idx = flat / row_width;
    size_t xfe_col = flat % row_width;
    size_t row_idx = d_indices[index_idx];
    if (row_idx < num_rows) {
        d_output[tid] = d_table[(xfe_col * 3 + comp) * num_rows + row_idx];
    } else {
        d_output[tid] = 0;
    }
}

/**
 * Scatter values back to table at specified indices
 * Inverse of gather operation
 */
__global__ void scatter_bfield_kernel(
    const uint64_t* d_values,
    const size_t* d_indices,
    uint64_t* d_table,
    size_t num_rows,
    size_t row_width,
    size_t num_indices
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = num_indices * row_width;
    
    if (tid >= total_elements) return;
    
    size_t index_idx = tid / row_width;
    size_t col_idx = tid % row_width;
    
    size_t row_idx = d_indices[index_idx];
    
    if (row_idx < num_rows) {
        d_table[row_idx * row_width + col_idx] = d_values[tid];
    }
}

// ============================================================================
// Host Interface
// ============================================================================

void gather_bfield_rows_gpu(
    const uint64_t* d_table,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream
) {
    size_t total_elements = num_indices * row_width;
    if (total_elements == 0) return;
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    gather_bfield_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        d_table, d_indices, d_output, num_rows, row_width, num_indices
    );
}

void gather_xfield_rows_gpu(
    const uint64_t* d_table,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream
) {
    size_t total_elements = num_indices * row_width * 3;
    if (total_elements == 0) return;
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    gather_xfield_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        d_table, d_indices, d_output, num_rows, row_width, num_indices
    );
}

void gather_column_gpu(
    const uint64_t* d_column,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_indices,
    cudaStream_t stream
) {
    if (num_indices == 0) return;
    
    int block_size = 256;
    int grid_size = (num_indices + block_size - 1) / block_size;
    
    gather_column_kernel<<<grid_size, block_size, 0, stream>>>(
        d_column, d_indices, d_output, num_rows, num_indices
    );
}

void gather_xfield_column_gpu(
    const uint64_t* d_column,
    const size_t* d_indices,
    uint64_t* d_output,
    size_t num_rows,
    size_t num_indices,
    cudaStream_t stream
) {
    size_t total_elements = num_indices * 3;
    if (total_elements == 0) return;
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    gather_xfield_column_kernel<<<grid_size, block_size, 0, stream>>>(
        d_column, d_indices, d_output, num_rows, num_indices
    );
}

void scatter_bfield_gpu(
    const uint64_t* d_values,
    const size_t* d_indices,
    uint64_t* d_table,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream
) {
    size_t total_elements = num_indices * row_width;
    if (total_elements == 0) return;
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    scatter_bfield_kernel<<<grid_size, block_size, 0, stream>>>(
        d_values, d_indices, d_table, num_rows, row_width, num_indices
    );
}

void gather_bfield_rows_colmajor_gpu(
    const uint64_t* d_table_colmajor,
    const size_t* d_indices,
    uint64_t* d_output_rowmajor,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream
) {
    size_t total = num_indices * row_width;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gather_bfield_rows_colmajor_kernel<<<grid_size, block_size, 0, stream>>>(
        d_table_colmajor, d_indices, d_output_rowmajor, num_rows, row_width, num_indices
    );
}

void gather_xfield_rows_colmajor_gpu(
    const uint64_t* d_table_comp_colmajor,
    const size_t* d_indices,
    uint64_t* d_output_rowmajor,
    size_t num_rows,
    size_t row_width,
    size_t num_indices,
    cudaStream_t stream
) {
    size_t total = num_indices * row_width * 3;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gather_xfield_rows_colmajor_kernel<<<grid_size, block_size, 0, stream>>>(
        d_table_comp_colmajor, d_indices, d_output_rowmajor, num_rows, row_width, num_indices
    );
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

