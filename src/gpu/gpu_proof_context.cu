/**
 * GPU Proof Context Implementation
 * 
 * Manages all GPU memory for zero-copy proof generation.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/gpu_proof_context.hpp"
#include "gpu/cuda_common.cuh"
#include "gpu/kernels/hash_table_constants.cuh"
#include "quotient/quotient.hpp"
#include "common/debug_control.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace triton_vm {
namespace gpu {

GpuProofContext::GpuProofContext(const Dimensions& dims) : dims_(dims) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    allocate_memory();
}

GpuProofContext::~GpuProofContext() {
    free_memory();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

namespace {

using namespace triton_vm::gpu::kernels;

__global__ void pack_hash_limb_pairs_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    uint64_t* __restrict__ d_packed_pairs
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) {
        return;
    }

    const size_t row_base = row * main_width + MAIN_HASH_START;

    // Columns relative to hash start
    constexpr size_t LK_IN_OFFSET = 3;
    constexpr size_t LK_OUT_OFFSET = 19;

    #pragma unroll
    for (int state = 0; state < HASH_NUM_STATES; ++state) {
        #pragma unroll
        for (int limb = 0; limb < HASH_LIMBS_PER_STATE; ++limb) {
            const int cascade_idx = state * HASH_LIMBS_PER_STATE + limb;
            const size_t hash_col_base = state * HASH_LIMBS_PER_STATE + limb;

            const uint64_t lk_in = d_main[row_base + LK_IN_OFFSET + hash_col_base];
            const uint64_t lk_out = d_main[row_base + LK_OUT_OFFSET + hash_col_base];

            // Packed layout is SoA by cascade id for coalesced reads in hash_prepare_diff_kernel:
            //   packed[(cascade_idx * num_rows + row) * 2 + {0,1}] = {lk_in, lk_out}
            const size_t out_idx = (static_cast<size_t>(cascade_idx) * num_rows + row) * 2;
            d_packed_pairs[out_idx + 0] = lk_in;
            d_packed_pairs[out_idx + 1] = lk_out;
        }
    }
}

} // namespace

void GpuProofContext::allocate_memory() {
    // Calculate sizes
    const size_t trace_height = dims_.padded_height;
    const size_t fri_length = dims_.fri_length;
    const size_t main_width = dims_.main_width;
    const size_t aux_width = dims_.aux_width;
    const size_t num_segments = dims_.num_quotient_segments;
    const size_t num_randomizers = dims_.num_trace_randomizers;
    
    // Merkle tree sizes (full tree = 2*leaves - 1, but we store just 2*leaves)
    const size_t main_merkle_leaves = fri_length;
    const size_t aux_merkle_leaves = fri_length;
    const size_t quot_merkle_leaves = fri_length;
    
    size_t total = 0;
    
    // Main table (trace domain): padded_height × main_width × sizeof(u64)
    size_t main_trace_size = trace_height * main_width * sizeof(uint64_t);
    CUDA_ALLOC(&d_main_trace_, main_trace_size);
    total += main_trace_size;
    
    // Main table (FRI domain): fri_length × main_width × sizeof(u64)
    size_t main_lde_size = fri_length * main_width * sizeof(uint64_t);
    CUDA_ALLOC(&d_main_lde_, main_lde_size);
    total += main_lde_size;
    
    // Aux table (trace domain): padded_height × aux_width × 3 (XFE) × sizeof(u64)
    size_t aux_trace_size = trace_height * aux_width * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_aux_trace_, aux_trace_size);
    total += aux_trace_size;
    
    // Aux table (FRI domain): fri_length × aux_width × 3 × sizeof(u64)
    size_t aux_lde_size = fri_length * aux_width * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_aux_lde_, aux_lde_size);
    total += aux_lde_size;

    // Trace randomizer coefficients (persist for entire proof)
    // Main: [main_width * num_randomizers] (BFE)
    size_t main_rand_size = main_width * num_randomizers * sizeof(uint64_t);
    CUDA_ALLOC(&d_main_randomizer_coeffs_, main_rand_size);
    total += main_rand_size;

    // Aux: [aux_width * num_randomizers * 3] (XFE)
    size_t aux_rand_size = aux_width * num_randomizers * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_aux_randomizer_coeffs_, aux_rand_size);
    total += aux_rand_size;
    
    // Quotient segments: fri_length × num_segments × 3 (XFE) × sizeof(u64)
    size_t quotient_size = fri_length * num_segments * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_quotient_segments_, quotient_size);
    total += quotient_size;
    
    // Quotient segment polynomial coefficients (compact):
    //   4 × (quotient_length/4) × 3 = quotient_length × 3
    size_t quotient_coeffs_size = dims_.quotient_length * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_quotient_segment_coeffs_, quotient_coeffs_size);
    total += quotient_coeffs_size;

    // Quotient codeword on quotient domain (debug buffer)
    size_t quotient_codeword_size = dims_.quotient_length * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_quotient_codeword_, quotient_codeword_size);
    total += quotient_codeword_size;

    // OOD points: 2 × XFE(3)
    size_t ood_points_size = 2 * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_ood_point_, 3 * sizeof(uint64_t));
    CUDA_ALLOC(&d_next_row_point_, 3 * sizeof(uint64_t));
    total += ood_points_size;
    
    // Merkle trees: 2 × num_leaves × 5 (Digest) × sizeof(u64)
    size_t main_merkle_size = 2 * main_merkle_leaves * 5 * sizeof(uint64_t);
    CUDA_ALLOC(&d_main_merkle_, main_merkle_size);
    total += main_merkle_size;
    
    size_t aux_merkle_size = 2 * aux_merkle_leaves * 5 * sizeof(uint64_t);
    CUDA_ALLOC(&d_aux_merkle_, aux_merkle_size);
    total += aux_merkle_size;
    
    size_t quot_merkle_size = 2 * quot_merkle_leaves * 5 * sizeof(uint64_t);
    CUDA_ALLOC(&d_quotient_merkle_, quot_merkle_size);
    total += quot_merkle_size;
    
    // FRI codewords and Merkle trees (sizes per round)
    // We allocate round 0 for the initial codeword (fri_length), then each subsequent round halves.
    size_t current_size = fri_length;
    for (size_t round = 0; round <= dims_.num_fri_rounds; ++round) {
        // Codeword: current_size × 3 (XFE) × sizeof(u64)
        uint64_t* d_codeword;
        size_t codeword_size = current_size * 3 * sizeof(uint64_t);
        CUDA_ALLOC(&d_codeword, codeword_size);
        d_fri_codewords_.push_back(d_codeword);
        total += codeword_size;
        
        // Merkle tree: 2 × current_size × 5 × sizeof(u64)
        uint64_t* d_merkle;
        size_t merkle_size = 2 * current_size * 5 * sizeof(uint64_t);
        CUDA_ALLOC(&d_merkle, merkle_size);
        d_fri_merkles_.push_back(d_merkle);
        total += merkle_size;

        if (current_size > 1) current_size /= 2;
    }
    
    // Challenges: MAX_CHALLENGES × 3 (XFE) × sizeof(u64)
    size_t challenges_size = MAX_CHALLENGES * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_challenges_, challenges_size);
    total += challenges_size;

    // Quotient weights: MASTER_AUX_NUM_CONSTRAINTS × 3 (XFE) × sizeof(u64)
    size_t quotient_weights_size = Quotient::MASTER_AUX_NUM_CONSTRAINTS * 3 * sizeof(uint64_t);
    CUDA_ALLOC(&d_quotient_weights_, quotient_weights_size);
    total += quotient_weights_size;

    // FRI query indices
    size_t fri_indices_size = NUM_FRI_QUERIES * sizeof(size_t);
    CUDA_ALLOC(&d_fri_query_indices_, fri_indices_size);
    total += fri_indices_size;
    
    // Sponge state: 16 × sizeof(u64)
    size_t sponge_size = 16 * sizeof(uint64_t);
    CUDA_ALLOC(&d_sponge_state_, sponge_size);
    total += sponge_size;
    
    // Proof buffer: estimate max size
    // Rough estimate: 10MB should be enough for most proofs
    proof_buffer_capacity_ = 10 * 1024 * 1024 / sizeof(uint64_t);
    size_t proof_buffer_size = proof_buffer_capacity_ * sizeof(uint64_t);
    CUDA_ALLOC(&d_proof_buffer_, proof_buffer_size);
    total += proof_buffer_size;
    
    // Scratch space: 2 buffers for transpose operations
    // scratch_a: main table transpose needs trace_height × main_width
    // scratch_b: aux table transpose needs trace_height × aux_width × 3 (XFE)
    // Use max of both for flexibility
    size_t scratch_a_elements = trace_height * main_width;
    size_t scratch_b_elements = trace_height * aux_width * 3;
    scratch_size_ = std::max(scratch_a_elements, scratch_b_elements);
    
    size_t scratch_a_bytes = scratch_a_elements * sizeof(uint64_t);
    size_t scratch_b_bytes = scratch_b_elements * sizeof(uint64_t);
    CUDA_ALLOC(&d_scratch_a_, scratch_a_bytes);
    CUDA_ALLOC(&d_scratch_b_, scratch_b_bytes);
    total += scratch_a_bytes + scratch_b_bytes;

    // Hash helpers
    size_t limb_pairs_size = dims_.padded_height * HASH_LIMB_PAIR_STRIDE * sizeof(uint64_t);
    // IMPORTANT: keep Hash helpers in device memory (NOT Unified Memory).
    // Unified Memory can page-migrate under multi-stream load and cause big slowdowns.
    CUDA_CHECK(cudaMalloc(&d_hash_limb_pairs_, limb_pairs_size));
    total += limb_pairs_size;

    size_t cascade_array_size = HASH_NUM_CASCADES * dims_.padded_height * 3 * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_hash_cascade_diffs_, cascade_array_size));
    CUDA_CHECK(cudaMalloc(&d_hash_cascade_prefix_, cascade_array_size));
    CUDA_CHECK(cudaMalloc(&d_hash_cascade_inverses_, cascade_array_size));
    total += cascade_array_size * 3;

    size_t cascade_mask_size = HASH_NUM_CASCADES * dims_.padded_height * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_hash_cascade_mask_, cascade_mask_size));
    total += cascade_mask_size;
    
    total_allocated_ = total;
    
    TRITON_PROFILE_COUT("[GPU] Allocated " << (total / (1024 * 1024)) << " MB for proof context" << std::endl);
}

void GpuProofContext::free_memory() {
    if (d_main_trace_) cudaFree(d_main_trace_);
    if (d_main_lde_) cudaFree(d_main_lde_);
    if (d_aux_trace_) cudaFree(d_aux_trace_);
    if (d_aux_lde_) cudaFree(d_aux_lde_);
    if (d_main_randomizer_coeffs_) cudaFree(d_main_randomizer_coeffs_);
    if (d_aux_randomizer_coeffs_) cudaFree(d_aux_randomizer_coeffs_);
    if (d_quotient_segments_) cudaFree(d_quotient_segments_);
    if (d_quotient_segment_coeffs_) cudaFree(d_quotient_segment_coeffs_);
    if (d_quotient_codeword_) cudaFree(d_quotient_codeword_);
    if (d_ood_point_) cudaFree(d_ood_point_);
    if (d_next_row_point_) cudaFree(d_next_row_point_);
    if (d_main_merkle_) cudaFree(d_main_merkle_);
    if (d_aux_merkle_) cudaFree(d_aux_merkle_);
    if (d_quotient_merkle_) cudaFree(d_quotient_merkle_);
    
    for (auto* ptr : d_fri_codewords_) {
        if (ptr) cudaFree(ptr);
    }
    d_fri_codewords_.clear();
    
    for (auto* ptr : d_fri_merkles_) {
        if (ptr) cudaFree(ptr);
    }
    d_fri_merkles_.clear();
    
    if (d_challenges_) cudaFree(d_challenges_);
    if (d_quotient_weights_) cudaFree(d_quotient_weights_);
    if (d_fri_query_indices_) cudaFree(d_fri_query_indices_);
    if (d_sponge_state_) cudaFree(d_sponge_state_);
    if (d_proof_buffer_) cudaFree(d_proof_buffer_);
    if (d_scratch_a_) cudaFree(d_scratch_a_);
    if (d_scratch_b_) cudaFree(d_scratch_b_);
    if (d_hash_limb_pairs_) cudaFree(d_hash_limb_pairs_);
    if (d_hash_cascade_diffs_) cudaFree(d_hash_cascade_diffs_);
    if (d_hash_cascade_prefix_) cudaFree(d_hash_cascade_prefix_);
    if (d_hash_cascade_inverses_) cudaFree(d_hash_cascade_inverses_);
    if (d_hash_cascade_mask_) cudaFree(d_hash_cascade_mask_);
}

void GpuProofContext::upload_main_table(const uint64_t* host_data, size_t num_elements) {
    TRITON_PROFILE_COUT("[GPU] Uploading main table: " << num_elements << " elements ("
              << (num_elements * sizeof(uint64_t) / (1024 * 1024)) << " MB)" << std::endl);
    
    CUDA_CHECK(cudaMemcpyAsync(
        d_main_trace_,
        host_data,
        num_elements * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        stream_
    ));

    build_hash_limb_cache();
}

void GpuProofContext::build_hash_limb_cache() {
    const size_t trace_len = dims_.padded_height;
    if (trace_len == 0) {
        return;
    }

    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid(static_cast<unsigned int>((trace_len + BLOCK - 1) / BLOCK));
    pack_hash_limb_pairs_kernel<<<grid, block, 0, stream_>>>(
        d_main_trace_, dims_.main_width, trace_len, d_hash_limb_pairs_
    );
    CUDA_CHECK(cudaGetLastError());
}

void GpuProofContext::upload_claim(
    const uint64_t* program_digest, size_t digest_len,
    const uint64_t* input, size_t input_len,
    const uint64_t* output, size_t output_len
) {
    // Claim data is small, upload to challenges buffer temporarily
    // This will be processed by step_initialize_fiat_shamir
    
    // For now, we'll handle this in the CPU and pass to GPU
    // The actual implementation would upload to a dedicated claim buffer
    (void)program_digest;
    (void)digest_len;
    (void)input;
    (void)input_len;
    (void)output;
    (void)output_len;
}

std::vector<uint64_t> GpuProofContext::download_proof() {
    TRITON_PROFILE_COUT("[GPU] Downloading proof: " << proof_size_ << " elements ("
              << (proof_size_ * sizeof(uint64_t) / 1024) << " KB)" << std::endl);
    
    std::vector<uint64_t> proof(proof_size_);
    
    CUDA_CHECK(cudaMemcpyAsync(
        proof.data(),
        d_proof_buffer_,
        proof_size_ * sizeof(uint64_t),
        cudaMemcpyDeviceToHost,
        stream_
    ));
    
    synchronize();
    
    return proof;
}

void GpuProofContext::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void GpuProofContext::print_memory_usage() const {
    std::cout << "\n[GPU] Memory Usage Summary:" << std::endl;
    std::cout << "  Padded height:     " << dims_.padded_height << std::endl;
    std::cout << "  FRI length:        " << dims_.fri_length << std::endl;
    std::cout << "  Main width:        " << dims_.main_width << std::endl;
    std::cout << "  Aux width:         " << dims_.aux_width << std::endl;
    std::cout << "  Total allocated:   " << (total_allocated_ / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Proof buffer used: " << proof_size_ << " elements" << std::endl;
}

} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

