#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm::gpu::kernels {

/**
 * Compute quotient values (one XFieldElement per quotient-domain row).
 *
 * Inputs:
 * - d_main_lde: [quotient_len * main_width] BFEs (already downsampled to quotient domain)
 * - d_aux_lde:  [quotient_len * aux_width * 3] XFEs (already downsampled to quotient domain)
 * - d_challenges: [63 * 3] XFEs
 * - d_weights: [MASTER_AUX_NUM_CONSTRAINTS * 3] XFEs (quotient weights)
 * - d_init_inv/d_cons_inv/d_tran_inv/d_term_inv: [quotient_len * 3] XFEs
 *
 * Output:
 * - d_out_quotient_values: [quotient_len * 3] XFEs
 */
void compute_quotient_values_gpu(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance, // quotient_len / trace_len
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    const uint64_t* d_init_inv_bfe, // [quotient_len]
    const uint64_t* d_cons_inv_bfe,
    const uint64_t* d_tran_inv_bfe,
    const uint64_t* d_term_inv_bfe,
    uint64_t* d_out_quotient_values,
    cudaStream_t stream = 0
);

} // namespace triton_vm::gpu::kernels

#endif // TRITON_CUDA_ENABLED


