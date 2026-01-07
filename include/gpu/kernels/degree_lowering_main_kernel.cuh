#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Initialize constant memory for degree lowering
 */
void init_degree_lowering_constants();

/**
 * GPU-accelerated degree lowering for main table columns 149-378
 * 
 * This replaces the CPU implementation in degree_lowering_main_cpp.cpp
 * with a fully parallel GPU version.
 * 
 * @param d_table Device pointer to main table [num_rows * num_cols]
 * @param num_rows Number of rows (padded height)
 * @param num_cols Number of columns (379)
 * @param stream CUDA stream for async execution
 */
void gpu_degree_lowering_main(
    uint64_t* d_table,
    size_t num_rows,
    size_t num_cols,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

