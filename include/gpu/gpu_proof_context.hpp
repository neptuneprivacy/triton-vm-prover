#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"

namespace triton_vm {
namespace gpu {

/**
 * GPU-Resident Proof Generation Context
 * 
 * This class manages all GPU memory for proof generation.
 * The design goal is ZERO intermediate H2D/D2H copies:
 * 
 *   1. H2D: Load padded main table once at start
 *   2. GPU: All computation stays on GPU (LDE, Merkle, FRI, etc.)
 *   3. D2H: Only copy final small proof back (~100KB typically)
 * 
 * Memory Layout on GPU:
 *   - Main table (trace domain): padded_height × 379 BFieldElements
 *   - Main table (FRI domain):   fri_length × 379 BFieldElements  
 *   - Aux table (trace domain):  padded_height × 88 XFieldElements
 *   - Aux table (FRI domain):    fri_length × 88 XFieldElements
 *   - Quotient segments:         fri_length × 4 XFieldElements
 *   - Merkle trees:              Various sizes
 *   - FRI codewords:             Decreasing sizes per round
 *   - Proof items:               Accumulated during proving
 */
class GpuProofContext {
public:
    static constexpr size_t NUM_FRI_QUERIES = 80;
    // Table dimensions
    struct Dimensions {
        // Domain sizes
        size_t padded_height = 0;       // Trace domain size (power of 2)
        size_t quotient_length = 0;     // Quotient domain size (typically 4x padded_height)
        size_t fri_length = 0;          // FRI domain size (typically 8x padded_height)

        // Domain cosets (offset) and generators (root of unity)
        uint64_t trace_offset = 0;
        uint64_t trace_generator = 0;
        uint64_t quotient_offset = 0;
        uint64_t quotient_generator = 0;
        uint64_t fri_offset = 0;
        uint64_t fri_generator = 0;

        size_t main_width = 0;          // 379 for Triton VM
        size_t aux_width = 0;           // 88 for Triton VM
        size_t num_trace_randomizers = 0; // number of per-column trace randomizer coefficients
        size_t num_quotient_segments = 0; // NUM_QUOTIENT_SEGMENTS (typically 4)
        size_t num_fri_rounds = 0;      // log2(fri_length / last_polynomial_len)
        
        // LDE Frugal mode: trade compute for memory by not caching full LDE tables
        bool lde_frugal_mode = false;   // If true, use coset-based streaming LDE (default: disabled)
    };
    
    GpuProofContext(const Dimensions& dims);
    ~GpuProofContext();
    
    // No copy
    GpuProofContext(const GpuProofContext&) = delete;
    GpuProofContext& operator=(const GpuProofContext&) = delete;
    
    // =========================================================================
    // Frugal Mode: Memory Management
    // =========================================================================
    
    /**
     * Free main LDE buffer after Merkle commitment (frugal mode only)
     * Coefficients are kept in d_main_coeffs_ for on-demand evaluation
     */
    void free_main_lde_for_frugal();
    
    /**
     * Free aux LDE buffer after Merkle commitment (frugal mode only)
     * Coefficients are kept in d_aux_coeffs_ for on-demand evaluation
     */
    void free_aux_lde_for_frugal();
    
    // =========================================================================
    // Initial Data Upload (ONLY H2D transfer in entire proof generation)
    // =========================================================================
    
    /**
     * Upload padded main table from host to GPU
     * This is the ONLY large H2D transfer needed.
     */
    void upload_main_table(const uint64_t* host_data, size_t num_elements);
    
    /**
     * Upload claim data (small)
     */
    void upload_claim(const uint64_t* program_digest, size_t digest_len,
                     const uint64_t* input, size_t input_len,
                     const uint64_t* output, size_t output_len);
    
    // =========================================================================
    // GPU-Resident Data Accessors (NO copies - just return device pointers)
    // =========================================================================
    
    // Main table
    uint64_t* d_main_trace() { return d_main_trace_; }
    uint64_t* d_main_lde() { return d_main_lde_; }
    
    // Aux table
    uint64_t* d_aux_trace() { return d_aux_trace_; }       // XFE: 3 u64 per element
    uint64_t* d_aux_lde() { return d_aux_lde_; }

    // In frugal mode we do not cache full FRI-domain LDE tables. Instead we reuse these
    // working buffers for per-coset LDE evaluations (length = padded_height).
    // Layout matches the LDE kernels: column-major for main, component-major column-major for aux.
    
    uint64_t* d_working_main() { return d_working_main_; }  // padded_height × main_width (BFE), col-major
    uint64_t* d_working_aux() { return d_working_aux_; }    // padded_height × (aux_width*3) (BFE), comp-major col-major
    // Reserved for future coefficient caching (not used in current frugal implementation)
    uint64_t* d_main_coeffs() { return d_main_coeffs_; }
    uint64_t* d_aux_coeffs() { return d_aux_coeffs_; }
    size_t working_domain_len() const { return working_domain_len_; }
    bool is_frugal_mode() const { return dims_.lde_frugal_mode; }

    // Trace randomizer coefficients (used for randomized LDE and OOD evaluations)
    uint64_t* d_main_randomizer_coeffs() { return d_main_randomizer_coeffs_; } // [main_width * num_trace_randomizers]
    uint64_t* d_aux_randomizer_coeffs() { return d_aux_randomizer_coeffs_; }   // [aux_width * num_trace_randomizers * 3]
    
    // Quotient
    uint64_t* d_quotient_segments() { return d_quotient_segments_; }
    uint64_t* d_quotient_weights() { return d_quotient_weights_; } // [MASTER_AUX_NUM_CONSTRAINTS * 3]
    uint64_t* d_quotient_segment_coeffs() { return d_quotient_segment_coeffs_; } // [fri_length * 3]
    // Debug / cross-step validation: quotient codeword on quotient domain (XFE)
    uint64_t* d_quotient_codeword() { return d_quotient_codeword_; } // [quotient_length * 3]

    // OOD points (sampled once, used across DEEP/FRI)
    uint64_t* d_ood_point() { return d_ood_point_; }           // [3]
    uint64_t* d_next_row_point() { return d_next_row_point_; } // [3]
    
    // Merkle trees (digests: 5 u64 per digest)
    uint64_t* d_main_merkle() { return d_main_merkle_; }
    uint64_t* d_aux_merkle() { return d_aux_merkle_; }
    uint64_t* d_quotient_merkle() { return d_quotient_merkle_; }
    
    // FRI data (round 0 is initial codeword at fri_length; each subsequent round halves)
    uint64_t* d_fri_codeword(size_t round) { return d_fri_codewords_[round]; }
    uint64_t* d_fri_merkle(size_t round) { return d_fri_merkles_[round]; }

    // First-round FRI query indices (used for FriResponses and later trace openings)
    size_t* d_fri_query_indices() { return d_fri_query_indices_; } // [NUM_FRI_QUERIES]
    
    // Challenges (sampled on GPU from Fiat-Shamir)
    uint64_t* d_challenges() { return d_challenges_; }
    
    // Sponge state (for Fiat-Shamir, kept on GPU)
    uint64_t* d_sponge_state() { return d_sponge_state_; }
    
    // Proof accumulator (grows during proving)
    uint64_t* d_proof_buffer() { return d_proof_buffer_; }
    size_t proof_buffer_size() const { return proof_size_; }
    
    // =========================================================================
    // Scratch Space for Intermediate Computations
    // =========================================================================
    
    uint64_t* d_scratch_a() { return d_scratch_a_; }
    uint64_t* d_scratch_b() { return d_scratch_b_; }
    size_t scratch_size() const { return scratch_size_; }

    // Hash table helpers
    uint64_t* d_hash_limb_pairs() { return d_hash_limb_pairs_; }
    uint64_t* d_hash_cascade_diffs() { return d_hash_cascade_diffs_; }
    uint64_t* d_hash_cascade_prefix() { return d_hash_cascade_prefix_; }
    uint64_t* d_hash_cascade_inverses() { return d_hash_cascade_inverses_; }
    uint8_t* d_hash_cascade_mask() { return d_hash_cascade_mask_; }

    void build_hash_limb_cache();
    
    // =========================================================================
    // Final Proof Download (ONLY D2H transfer)
    // =========================================================================
    
    /**
     * Download final proof from GPU to host
     * This is the ONLY large D2H transfer needed.
     * Returns the proof as a vector of BFieldElements.
     */
    std::vector<uint64_t> download_proof();
    
    /**
     * Get proof size (in u64 elements) without downloading
     */
    size_t get_proof_size() const { return proof_size_; }
    
    /**
     * Set proof size (called by GpuStark after encoding)
     */
    void set_proof_size(size_t size) { proof_size_ = size; }
    
    // =========================================================================
    // Stream Management
    // =========================================================================
    
    cudaStream_t stream() { return stream_; }
    void synchronize();
    
    // =========================================================================
    // Memory Statistics
    // =========================================================================
    
    size_t total_gpu_memory_bytes() const { return total_allocated_; }
    void print_memory_usage() const;
    
private:
    Dimensions dims_;
    cudaStream_t stream_;
    
    // Main table storage
    uint64_t* d_main_trace_ = nullptr;      // padded_height × main_width
    uint64_t* d_main_lde_ = nullptr;        // fri_length × main_width (NULL in frugal mode)
    
    // Aux table storage (XFE = 3 × u64)
    uint64_t* d_aux_trace_ = nullptr;       // padded_height × aux_width × 3
    uint64_t* d_aux_lde_ = nullptr;         // fri_length × aux_width × 3 (NULL in frugal mode)
    
    // LDE Frugal mode: working domain buffers (only allocated in frugal mode)
    uint64_t* d_working_main_ = nullptr;    // working_domain_len × main_width
    uint64_t* d_working_aux_ = nullptr;     // working_domain_len × aux_width × 3
    uint64_t* d_main_coeffs_ = nullptr;     // padded_height × main_width (polynomial coefficients)
    uint64_t* d_aux_coeffs_ = nullptr;      // padded_height × aux_width × 3 (polynomial coefficients)
    size_t working_domain_len_ = 0;         // = randomized_trace_len / 2

    // Trace randomizer coefficients (persist for entire proof)
    uint64_t* d_main_randomizer_coeffs_ = nullptr; // main_width × num_trace_randomizers
    uint64_t* d_aux_randomizer_coeffs_ = nullptr;  // aux_width × num_trace_randomizers × 3
    
    // Quotient storage (XFE = 3 × u64)
    uint64_t* d_quotient_segments_ = nullptr;  // fri_length × num_quotient_segments × 3
    // Segment polynomials coefficients: num_quotient_segments × (fri_length/num_quotient_segments) × 3 = fri_length × 3
    uint64_t* d_quotient_segment_coeffs_ = nullptr;
    // Quotient codeword on quotient domain (for debug comparisons)
    uint64_t* d_quotient_codeword_ = nullptr; // quotient_length × 3

    // Out-of-domain points (XFE = 3 × u64)
    uint64_t* d_ood_point_ = nullptr;       // [3]
    uint64_t* d_next_row_point_ = nullptr;  // [3]
    
    // Merkle trees (Digest = 5 × u64)
    // Full tree size = 2 * num_leaves - 1
    uint64_t* d_main_merkle_ = nullptr;
    uint64_t* d_aux_merkle_ = nullptr;
    uint64_t* d_quotient_merkle_ = nullptr;
    
    // FRI data (variable sizes per round)
    std::vector<uint64_t*> d_fri_codewords_;
    std::vector<uint64_t*> d_fri_merkles_;

    // FRI query indices (device)
    size_t* d_fri_query_indices_ = nullptr;
    
    // Challenges (extension field elements)
    uint64_t* d_challenges_ = nullptr;
    // Needs to hold:
    // - 63 AIR challenges (59 sampled + 4 derived)
    // - MASTER_AUX_NUM_CONSTRAINTS quotient weights (sampled as XFieldElements)
    // - ~num_fri_rounds folding challenges + OOD point, etc.
    // Keep a generous buffer for now.
    static constexpr size_t MAX_CHALLENGES = 2048;

    // Quotient weights (MASTER_AUX_NUM_CONSTRAINTS XFieldElements) stored separately for clarity
    uint64_t* d_quotient_weights_ = nullptr; // [MASTER_AUX_NUM_CONSTRAINTS * 3]
    
    // Fiat-Shamir sponge state (16 u64 for Tip5)
    uint64_t* d_sponge_state_ = nullptr;
    
    // Proof buffer (grows during proving)
    uint64_t* d_proof_buffer_ = nullptr;
    size_t proof_buffer_capacity_ = 0;
    size_t proof_size_ = 0;
    
    // Scratch space for intermediate computations
    uint64_t* d_scratch_a_ = nullptr;
    uint64_t* d_scratch_b_ = nullptr;
    size_t scratch_size_ = 0;

    // Hash table packed data + cascade workspace
    uint64_t* d_hash_limb_pairs_ = nullptr;
    uint64_t* d_hash_cascade_diffs_ = nullptr;
    uint64_t* d_hash_cascade_prefix_ = nullptr;
    uint64_t* d_hash_cascade_inverses_ = nullptr;
    uint8_t* d_hash_cascade_mask_ = nullptr;
    
    size_t total_allocated_ = 0;
    
    void allocate_memory();
    void free_memory();
};

} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

