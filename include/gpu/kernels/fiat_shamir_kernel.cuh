#pragma once

/**
 * Fiat-Shamir CUDA Kernel Declarations
 * 
 * GPU-accelerated Fiat-Shamir transform using Tip5 sponge.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// State and rate sizes matching Tip5
static constexpr int FS_STATE_SIZE = 16;
static constexpr int FS_RATE = 10;

/**
 * Initialize sponge state for variable-length absorption
 * Sets state to zeros (variable-length mode)
 */
void fs_init_sponge_gpu(uint64_t* d_state, cudaStream_t stream = 0);

/**
 * Absorb data into sponge (pad_and_absorb_all equivalent)
 * 
 * @param d_state Sponge state (16 uint64_t on device)
 * @param h_data Data to absorb (on host)
 * @param data_len Original data length
 * @param d_sbox_table S-box lookup table (on device)
 * @param d_mds_matrix MDS matrix first row (on device)
 * @param d_round_constants Round constants (on device)
 * @param stream CUDA stream
 */
void fs_absorb_gpu(
    uint64_t* d_state,
    const uint64_t* h_data,
    size_t data_len,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream = 0
);

/**
 * Absorb data into sponge from DEVICE memory (zero-copy friendly).
 *
 * This is the device equivalent of `fs_absorb_gpu` but avoids per-chunk H2D copies.
 */
void fs_absorb_device_gpu(
    uint64_t* d_state,
    const uint64_t* d_data,
    size_t data_len,
    cudaStream_t stream = 0
);

/**
 * Squeeze RATE (10) elements from sponge
 * 
 * @param d_state Sponge state
 * @param d_output Output buffer (RATE elements)
 * @param d_sbox_table S-box lookup table
 * @param d_mds_matrix MDS matrix first row
 * @param d_round_constants Round constants
 * @param stream CUDA stream
 */
void fs_squeeze_gpu(
    uint64_t* d_state,
    uint64_t* d_output,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream = 0
);

/**
 * Sample XFieldElements from sponge
 * Each XFieldElement uses 3 consecutive BFieldElements
 * 
 * @param d_state Sponge state
 * @param d_output Output buffer (3 * count elements for XFieldElements)
 * @param count Number of XFieldElements to sample
 * @param d_sbox_table S-box lookup table
 * @param d_mds_matrix MDS matrix first row
 * @param d_round_constants Round constants
 * @param stream CUDA stream
 */
void fs_sample_scalars_gpu(
    uint64_t* d_state,
    uint64_t* d_output,
    size_t count,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream = 0
);

/**
 * Sample XFieldElements from sponge entirely on GPU (device-only, no D2H).
 *
 * Output layout matches `fs_sample_scalars_gpu`: [count * 3] u64 (BField elements).
 */
void fs_sample_scalars_device_gpu(
    uint64_t* d_state,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream = 0
);

/**
 * Sample indices from sponge for FRI queries
 * 
 * @param d_state Sponge state
 * @param d_output Output buffer (count indices)
 * @param upper_bound Upper bound for indices (must be power of 2)
 * @param count Number of indices to sample
 * @param d_sbox_table S-box lookup table
 * @param d_mds_matrix MDS matrix first row
 * @param d_round_constants Round constants
 * @param stream CUDA stream
 */
void fs_sample_indices_gpu(
    uint64_t* d_state,
    size_t* d_output,
    size_t upper_bound,
    size_t count,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream = 0
);

/**
 * Sample indices from sponge entirely on GPU (device-only, no D2H).
 *
 * NOTE: uses rejection sampling (skips MAX_VALUE) like CPU ProofStream.
 */
void fs_sample_indices_device_gpu(
    uint64_t* d_state,
    size_t* d_output,
    size_t upper_bound,
    size_t count,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

