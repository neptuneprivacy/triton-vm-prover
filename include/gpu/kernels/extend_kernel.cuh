#pragma once

/**
 * Auxiliary Table Extension CUDA Kernel Declarations
 * 
 * GPU-accelerated auxiliary table extension for STARK proof generation.
 * 
 * The aux table is computed from the main table + challenges:
 * - Running products (permutation arguments)
 * - Log derivatives (lookup arguments)
 * - Evaluation arguments
 * 
 * GPU Strategy:
 * - Parallel row compression
 * - Batch inverse computation
 * - Parallel prefix scans for accumulation
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Constants
constexpr size_t MASTER_AUX_NUM_COLUMNS = 88;
constexpr size_t NUM_CHALLENGES = 63;

/**
 * Extend main table to auxiliary table on GPU
 * 
 * This is the main entry point for aux table computation.
 * 
 * @param d_main_table Main table on GPU (num_rows * main_width u64s)
 * @param main_width Number of columns in main table (379)
 * @param num_rows Number of rows (padded height)
 * @param d_challenges Challenges on GPU (63 XFEs = 189 u64s)
 * @param d_aux_table Output aux table on GPU (num_rows * 88 * 3 u64s for XFEs)
 * @param stream CUDA stream
 */
void extend_aux_table_gpu(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux_table,
    cudaStream_t stream = 0
);

/**
 * Batch compute XFieldElement inverses
 * 
 * @param d_input Input XFEs (n * 3 u64s)
 * @param d_output Output inverse XFEs (n * 3 u64s)
 * @param n Number of XFEs
 * @param stream CUDA stream
 */
void xfe_batch_inverse_gpu(
    const uint64_t* d_input,
    uint64_t* d_output,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XFieldElement prefix product scan (inclusive)
 * 
 * Computes: result[i] = input[0] * input[1] * ... * input[i]
 * 
 * @param d_data XFE array (n * 3 u64s), modified in-place
 * @param n Number of XFEs
 * @param stream CUDA stream
 */
void xfe_prefix_product_gpu(
    uint64_t* d_data,
    size_t n,
    cudaStream_t stream = 0
);

/**
 * XFieldElement prefix sum scan (inclusive)
 * 
 * Computes: result[i] = input[0] + input[1] + ... + input[i]
 * 
 * @param d_data XFE array (n * 3 u64s), modified in-place
 * @param n Number of XFEs
 * @param stream CUDA stream
 */
void xfe_prefix_sum_gpu(
    uint64_t* d_data,
    size_t n,
    cudaStream_t stream = 0
);

// Backward compatibility
void aux_table_extend_device(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux_table,
    cudaStream_t stream = 0
);

/**
 * Full GPU aux table extension (all 9 sub-tables)
 * 
 * For true zero-copy proof generation. Computes all 88 aux columns on GPU.
 * 
 * @param d_main_table Main table on GPU [num_rows × main_width]
 * @param main_width Number of columns in main table (379)
 * @param num_rows Number of rows (padded height)
 * @param d_challenges Challenges on GPU [63 × 3 u64s]
 * @param d_aux_table Output aux table on GPU [num_rows × 88 × 3 u64s]
 * @param stream CUDA stream
 */
void extend_aux_table_full_gpu(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t aux_rng_seed_value,
    uint64_t* d_aux_table,
    const uint64_t* d_hash_limb_pairs,
    uint64_t* d_hash_cascade_diffs,
    uint64_t* d_hash_cascade_prefix,
    uint64_t* d_hash_cascade_inverses,
    uint8_t* d_hash_cascade_mask,
    cudaStream_t stream = 0
);

void extend_aux_table_degree_lowering_and_randomizer_gpu(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t aux_rng_seed_value,
    uint64_t* d_aux_table,
    cudaStream_t stream = 0
);

/**
 * Degree lowering only (aux cols 49..86).
 * Expects d_aux_table cols 0..48 already filled for all rows.
 * Does not touch other columns.
 */
void degree_lowering_only_gpu(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux_table,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

