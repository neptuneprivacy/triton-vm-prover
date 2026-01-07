#pragma once

/**
 * Challenges CUDA Kernel Declarations
 *
 * Computes the 4 derived challenges (indices 59..62) from:
 * - sampled challenges (0..58)
 * - claim data (program digest, input, output)
 * - Tip5 lookup table (256 BFieldElements)
 *
 * This matches `Challenges::compute_derived_challenges` which uses EvalArg terminals.
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace triton_vm::gpu::kernels {

void compute_derived_challenges_gpu(
    uint64_t* d_challenges_xfe,           // [63 * 3] u64, XFE coefficients
    const uint64_t* d_program_digest_bfe, // [5] u64
    const uint64_t* d_input_bfe,          // [input_len] u64
    size_t input_len,
    const uint64_t* d_output_bfe,         // [output_len] u64
    size_t output_len,
    const uint64_t* d_lookup_table_bfe,   // [256] u64
    cudaStream_t stream = 0
);

} // namespace triton_vm::gpu::kernels

#endif // TRITON_CUDA_ENABLED


