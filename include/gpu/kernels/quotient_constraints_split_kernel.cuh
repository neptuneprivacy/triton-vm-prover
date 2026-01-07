#pragma once

/**
 * Split GPU Quotient Constraint Evaluation Kernels - Header
 *
 * Declares functions for split quotient constraint evaluation:
 * - Initial, Consistency, Terminal kernels (in quotient_constraints_split_kernel.cu)
 * - Transition kernels split into 4 parts (in quotient_transition_partN_kernel.cu)
 */

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace triton_vm::gpu::kernels {

/**
 * Compute initial, consistency, and terminal quotient contributions.
 * Each output buffer is quotient_len * 3 uint64_t values (XFieldElements).
 */
void compute_quotient_split_partial(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    const uint64_t* d_init_inv_bfe,
    const uint64_t* d_cons_inv_bfe,
    const uint64_t* d_term_inv_bfe,
    uint64_t* d_out_init,
    uint64_t* d_out_cons,
    uint64_t* d_out_term,
    cudaStream_t stream = nullptr
);

/**
 * Compute transition quotient contribution - part 0/4
 */
void launch_quotient_transition_part0(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    uint64_t* d_out,
    cudaStream_t stream = nullptr
);

/**
 * Compute transition quotient contribution - part 1/4
 */
void launch_quotient_transition_part1(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    uint64_t* d_out,
    cudaStream_t stream = nullptr
);

/**
 * Compute transition quotient contribution - part 2/4
 */
void launch_quotient_transition_part2(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    uint64_t* d_out,
    cudaStream_t stream = nullptr
);

/**
 * Compute transition quotient contribution - part 3/4
 */
void launch_quotient_transition_part3(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    uint64_t* d_out,
    cudaStream_t stream = nullptr
);

/**
 * Combine all quotient contributions:
 * out[row] = init[row] + cons[row] + tran[row] + term[row]
 */
void combine_quotient_results(
    const uint64_t* d_init,
    const uint64_t* d_cons,
    const uint64_t* d_tran,
    const uint64_t* d_term,
    size_t quotient_len,
    uint64_t* d_out,
    cudaStream_t stream = nullptr
);

/**
 * Add XFieldElements element-wise:
 * out[row] = out[row] + add[row]
 */
void add_xfield_arrays(
    uint64_t* d_out,      // in-place accumulator
    const uint64_t* d_add,
    size_t len,
    cudaStream_t stream = nullptr
);

/**
 * Multiply XFieldElements by BFieldElement scalars:
 * out[row] = in[row] * scalar[row]
 */
void scale_xfield_by_bfield(
    const uint64_t* d_in,
    const uint64_t* d_scalars_bfe,
    size_t len,
    uint64_t* d_out,
    cudaStream_t stream = nullptr
);

/**
 * FUSED: Sum 4 transition parts + scale by zerofier inverse in ONE kernel.
 * Replaces: memcpy + 3x add_xfield_arrays + scale_xfield_by_bfield + memcpy
 * out[row] = (part0[row] + part1[row] + part2[row] + part3[row]) * scale[row]
 */
void fused_sum4_scale_transition(
    const uint64_t* d_part0,
    const uint64_t* d_part1,
    const uint64_t* d_part2,
    const uint64_t* d_part3,
    const uint64_t* d_scale,  // BFieldElement scalars
    size_t len,
    uint64_t* d_out,
    cudaStream_t stream = nullptr
);

} // namespace triton_vm::gpu::kernels

#endif // TRITON_CUDA_ENABLED

