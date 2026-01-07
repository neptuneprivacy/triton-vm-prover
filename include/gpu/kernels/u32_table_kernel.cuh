#pragma once

/**
 * GPU-Accelerated U32 Table Fill
 * 
 * Fills the U32 table on GPU with parallel section computation.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Fill U32 table on GPU
 * 
 * @param d_table Device pointer to main table (row-major, uint64_t per element)
 * @param entries Vector of (opcode, lhs, rhs, multiplicity) tuples
 * @param u32_table_start Column offset of U32 table in main table
 * @param table_width Total width of main table
 * @param table_height Total height (padded) of main table
 * @param stream CUDA stream
 * @return Number of rows filled (for padding calculation)
 */
size_t gpu_fill_u32_table(
    uint64_t* d_table,
    const std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>>& entries,
    size_t u32_table_start,
    size_t table_width,
    size_t table_height,
    cudaStream_t stream = nullptr
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

