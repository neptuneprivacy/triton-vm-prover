#pragma once

/**
 * Row Hashing CUDA Kernel Declarations
 * 
 * GPU-accelerated row hashing for Merkle tree leaf generation.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

/**
 * Hash BFieldElement table rows to digests
 * 
 * @param d_table Column-major table (num_cols * num_rows)
 * @param num_rows Number of rows
 * @param num_cols Number of columns
 * @param d_digests Output digests (num_rows * 5 elements)
 * @param stream CUDA stream
 */
void hash_bfield_rows_gpu(
    const uint64_t* d_table,
    size_t num_rows,
    size_t num_cols,
    uint64_t* d_digests,
    cudaStream_t stream = 0
);

/**
 * Hash XFieldElement table rows to digests
 * 
 * @param d_table Column-major XFE table (num_xfe_cols * 3 * num_rows)
 * @param num_rows Number of rows
 * @param num_xfe_cols Number of XFieldElement columns
 * @param d_digests Output digests (num_rows * 5 elements)
 * @param stream CUDA stream
 */
void hash_xfield_rows_gpu(
    const uint64_t* d_table,
    size_t num_rows,
    size_t num_xfe_cols,
    uint64_t* d_digests,
    cudaStream_t stream = 0
);

/**
 * Streaming row hashing (frugal mode):
 * Initialize per-row sponge states, absorb one RATE-sized chunk, and finalize.
 */
void row_sponge_init_gpu(
    uint64_t* d_states,   // [num_rows * 16]
    size_t num_rows,
    cudaStream_t stream = 0
);

void row_sponge_absorb_rate_gpu(
    const uint64_t* d_table,  // [batch_cols * num_rows] col-major
    size_t num_rows,
    size_t num_cols_total,
    size_t col_start,
    size_t batch_cols,
    uint64_t* d_states,       // [num_rows * 16]
    cudaStream_t stream = 0
);

void row_sponge_finalize_gpu(
    const uint64_t* d_states, // [num_rows * 16]
    size_t num_rows,
    uint64_t* d_digests,       // [num_rows * 5]
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

