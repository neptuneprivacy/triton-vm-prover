#pragma once

#ifdef TRITON_CUDA_ENABLED

#include "gpu/gpu_proof_context.hpp"
#include "stark.hpp"
#include "proof_stream/proof_stream.hpp"
#include "stark/challenges.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <memory>

namespace triton_vm {
namespace gpu {

/**
 * GPU-Resident STARK Prover
 * 
 * This class performs the entire proof generation on GPU with minimal
 * host-device memory transfers:
 * 
 * Memory Transfer Summary:
 *   H2D (start):  padded_main_table (~10-100 MB depending on trace size)
 *   D2H (end):    final_proof (~100-500 KB typically)
 * 
 * ALL intermediate data stays on GPU:
 *   - LDE computations
 *   - Merkle tree construction
 *   - Fiat-Shamir challenge sampling
 *   - Quotient computation
 *   - FRI folding
 *   - Proof item encoding
 */
class GpuStark {
public:
    GpuStark();
    ~GpuStark();
    
    /**
     * Generate proof entirely on GPU
     * 
     * @param claim The claim (program digest, input, output)
     * @param main_table_data Padded main table data (host memory)
     * @param num_rows Number of rows (padded height)
     * @param num_cols Number of columns (379 for Triton VM)
     * @return Proof as vector of BFieldElements
     * 
     * This function:
     * 1. Uploads main_table_data to GPU (only H2D transfer)
     * 2. Performs all proof computation on GPU
     * 3. Downloads final proof (only D2H transfer)
     */
    Proof prove(
        const Claim& claim,
        const uint64_t* main_table_data,
        size_t num_rows,
        size_t num_cols,
        const uint64_t trace_domain_3[3],
        const uint64_t quotient_domain_3[3],
        const uint64_t fri_domain_3[3],
        const uint8_t randomness_seed[32],
        const std::vector<uint64_t>& main_randomizer_coeffs,
        const std::vector<uint64_t>& aux_randomizer_coeffs
    );
    
    /**
     * Generate proof with GPU Phase 1 (main table creation) + zero-copy pipeline.
     *
     * This path:
     * - Uploads AET-derived traces (smaller than full main table)
     * - Builds + pads the full main table directly into ctx_->d_main_trace()
     * - Skips the big H2D upload of the full main table
     *
     * NOTE: Hybrid CPU aux mode requires host main table; therefore this path forces GPU aux.
     */
    struct Phase1HostTraces {
        // All pointers are host pointers to contiguous row-major arrays.
        // Layout matches gpu_upload_aet_flat in phase1_kernel.cuh.
        const uint64_t* h_program_trace = nullptr;     size_t program_rows = 0;        // [rows][7]
        const uint64_t* h_processor_trace = nullptr;   size_t processor_rows = 0;      // [rows][39]
        const uint64_t* h_op_stack_trace = nullptr;    size_t op_stack_rows = 0;       // [rows][4]
        const uint64_t* h_ram_trace = nullptr;         size_t ram_rows = 0;            // [rows][7]
        const uint64_t* h_jump_stack_trace = nullptr;  size_t jump_stack_rows = 0;     // [rows][5]
        const uint64_t* h_hash_trace = nullptr;        size_t program_hash_rows = 0; size_t sponge_rows = 0; size_t hash_rows = 0; // [sum][67]
        const uint64_t* h_cascade_trace = nullptr;     size_t cascade_rows = 0;        // [rows][6]
        const uint64_t* h_lookup_trace = nullptr;      size_t lookup_rows = 0;         // [rows][4]
        const uint64_t* h_u32_trace = nullptr;         size_t u32_rows = 0;            // [rows][10]
        const size_t* table_lengths_9 = nullptr;                                       // [9]
    };

    Proof prove_with_gpu_phase1(
        const Claim& claim,
        const Phase1HostTraces& phase1,
        size_t padded_height,
        size_t num_cols,
        const uint64_t trace_domain_3[3],
        const uint64_t quotient_domain_3[3],
        const uint64_t fri_domain_3[3],
        const uint8_t randomness_seed[32],
        const std::vector<uint64_t>& main_randomizer_coeffs,
        const std::vector<uint64_t>& aux_randomizer_coeffs
    );
    
    /**
     * Generate proof with GPU padding
     * 
     * This version accepts UNPADDED table data and table lengths.
     * Padding is done on GPU after upload for maximum parallelism.
     * 
     * @param claim The claim
     * @param unpadded_table_data Unpadded main table data (host memory)
     * @param unpadded_rows Number of rows before padding
     * @param padded_height Target padded height (power of 2)
     * @param num_cols Number of columns (379)
     * @param table_lengths Lengths of each of the 9 tables
     * @param trace_domain_3 Trace domain [length, offset, generator]
     * @param quotient_domain_3 Quotient domain
     * @param fri_domain_3 FRI domain
     * @param randomness_seed 32-byte random seed
     * @param main_randomizer_coeffs Randomizer coefficients for main table
     * @param aux_randomizer_coeffs Randomizer coefficients for aux table
     * @return Proof
     */
    Proof prove_with_gpu_padding(
        const Claim& claim,
        const uint64_t* unpadded_table_data,
        size_t unpadded_rows,
        size_t padded_height,
        size_t num_cols,
        const size_t table_lengths[9],
        const uint64_t trace_domain_3[3],
        const uint64_t quotient_domain_3[3],
        const uint64_t fri_domain_3[3],
        const uint8_t randomness_seed[32],
        const std::vector<uint64_t>& main_randomizer_coeffs,
        const std::vector<uint64_t>& aux_randomizer_coeffs
    );
    
    /**
     * Get estimated GPU memory requirement (auto-selects frugal mode based on padded_height)
     */
    static size_t estimate_gpu_memory(size_t padded_height);
    
    /**
     * Get estimated GPU memory requirement with explicit mode selection
     * @param frugal_mode If true, estimate memory for coset-streaming mode (less memory, more compute)
     */
    static size_t estimate_gpu_memory_with_mode(size_t padded_height, bool frugal_mode);
    
    /**
     * Check if GPU has enough memory for given trace size
     */
    static bool check_gpu_memory(size_t padded_height);
    
private:
    std::unique_ptr<GpuProofContext> ctx_;
    // Keep a copy of the claim on host for small derived-challenge inputs.
    // (Still zero-copy for big tables; claim is tiny.)
    Claim claim_;
    // Store the prover randomness seed so GPU-side table construction can match Rust exactly
    // when running in deterministic/validation mode.
    ChaCha12Rng::Seed randomness_seed_{};
    // CPU-driven Fiat–Shamir transcript (gold standard matching Rust).
    // We only use it to sample challenges/indices and to enqueue FS-relevant proof items.
    ProofStream fs_cpu_;
    
    // Dimensions (default-initialized to prevent garbage values)
    GpuProofContext::Dimensions dims_{};
    
    // Tip5 tables on GPU
    uint16_t* d_sbox_table_ = nullptr;
    uint64_t* d_mds_matrix_ = nullptr;
    uint64_t* d_round_constants_ = nullptr;
    
    // Proof size tracker
    size_t proof_size_ = 0;
    
    // Host pointer to main table data (for hybrid CPU/GPU aux computation)
    const uint64_t* h_main_table_data_ = nullptr;
    
    // NOTE: Hybrid CPU aux operates directly on the flat `h_main_table_data_` buffer.
    
    // U32 entries for GPU table fill (optional)
    std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>> u32_entries_;
    
public:
    /**
     * Set U32 entries for GPU table fill
     * Call this before prove() to enable GPU U32 table fill with TRITON_GPU_U32=1
     * @param entries Vector of (opcode, lhs, rhs, multiplicity) tuples
     */
    void set_u32_entries(const std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>>& entries) {
        u32_entries_ = entries;
    }
    
private:
    // Initialize Tip5 lookup tables
    void init_tip5_tables();
    
    // =========================================================================
    // Proof Generation Steps (all on GPU)
    // =========================================================================
    
    /**
     * Step 1: Initialize Fiat-Shamir with claim
     * - Absorb claim into sponge state
     * - Store log2_padded_height as first proof item
     */
    void step_initialize_fiat_shamir(const Claim& claim);
    
    /**
     * Step 2: Main table LDE and Merkle commitment
     * - Extend main table from trace domain to FRI domain
     * - Build Merkle tree over extended table rows
     * - Absorb Merkle root into sponge
     */
    void step_main_table_commitment(const std::vector<uint64_t>& main_randomizer_coeffs);
    
    /**
     * Step 3: Sample extension challenges and extend aux table
     * - Sample challenges from sponge
     * - Compute auxiliary table from main table + challenges
     * - Extend aux table to FRI domain
     * - Build Merkle tree, absorb root
     */
    void step_aux_table_commitment(const std::vector<uint64_t>& aux_randomizer_coeffs);
    
    /**
     * Helper: Compute aux table on CPU (for hybrid mode)
     * Uses OpenMP to parallelize table extension
     */
    void compute_aux_table_cpu(
        const Challenges& challenges,
        const ChaCha12Rng::Seed& aux_seed,
        uint64_t* d_aux_trace,
        cudaStream_t stream
    );
    
    /**
     * Step 4: Quotient computation
     * - Sample quotient combination weights
     * - Evaluate AIR constraints at all FRI domain points
     * - Combine into quotient polynomial
     * - Split into segments, build Merkle tree
     */
    void step_quotient_commitment();
    void step_quotient_commitment_frugal();
    
    /**
     * Step 5: Out-of-domain sampling
     * - Sample out-of-domain point
     * - Evaluate main, aux, quotient at OOD point
     * - Add evaluations as proof items
     */
    void step_out_of_domain_evaluation();
    
    /**
     * Step 6: FRI protocol
     * - Compute DEEP polynomial
     * - FRI commit phase (folding + Merkle trees)
     * - FRI query phase (authentication paths)
     */
    void step_fri_protocol();
    void step_fri_protocol_frugal();
    
    /**
     * Step 7: Open trace at query indices
     * - Reveal main, aux, quotient rows at query indices
     * - Add authentication paths
     */
    void step_open_trace();
    void step_open_trace_frugal();
    
    /**
     * Step 8: Encode proof
     * - Serialize all proof items to BFieldElement encoding
     */
    void step_encode_proof();

    // CPU aux helper functions
    void compute_aux_table_cpu_extension_only(const Challenges& challenges, uint64_t* d_aux_trace);
    void apply_cpu_aux_randomizer(uint64_t* d_aux_trace, uint64_t aux_seed_value, size_t num_rows, size_t aux_width);

public:
    /**
     * Sample LDE rows for validation (public for cross-implementation testing)
     * Downloads specific rows from GPU LDE buffers
     */
    std::vector<uint64_t> sample_main_lde_row(size_t row_index, size_t num_cols);
    std::vector<std::vector<uint64_t>> sample_main_lde_rows(const std::vector<size_t>& row_indices, size_t num_cols);
    
    std::vector<std::string> sample_aux_lde_row(size_t row_index, size_t num_cols); // Returns XFE strings
    std::vector<std::vector<std::string>> sample_aux_lde_rows(const std::vector<size_t>& row_indices, size_t num_cols);
    
    /**
     * Download aux table trace from GPU (before LDE, after creation)
     * @param num_rows Number of rows
     * @param num_cols Number of columns (XFieldElements)
     * @return Vector of uint64_t values (row-major, 3 uint64_t per XFieldElement)
     */
    std::vector<uint64_t> download_aux_trace(size_t num_rows, size_t num_cols);
    
    std::vector<std::string> sample_quotient_lde_row(size_t row_index, size_t num_segments); // Returns XFE strings
    std::vector<std::vector<std::string>> sample_quotient_lde_rows(const std::vector<size_t>& row_indices, size_t num_segments);
    
    /**
     * Sample row digests (hashes) from the aux Merkle buffer
     * Used for debugging Merkle tree mismatches
     * @param row_indices Row indices to sample
     * @return Vector of digest strings (5 u64 values as hex per row)
     */
    std::vector<std::vector<uint64_t>> sample_aux_row_digests(const std::vector<size_t>& row_indices);
    
    /**
     * Sample row digests from the main Merkle buffer
     */
    std::vector<std::vector<uint64_t>> sample_main_row_digests(const std::vector<size_t>& row_indices);

    /**
     * Debug helper: download raw aux LDE row as BFieldElements (3 per XFE column).
     * Layout returned is [col0.c0, col0.c1, col0.c2, col1.c0, ...] where c0 is constant term.
     *
     * NOTE: This is intentionally slow (many tiny D2H copies) and should only be used for debugging.
     */
    std::vector<uint64_t> sample_aux_lde_row_bfes(size_t row_index, size_t num_xfe_cols);

    /**
     * Debug helper: sample arbitrary digests from the aux Merkle buffer by node index.
     *
     * Layout matches `merkle_tree_gpu`:
     * - leaves at indices [0, fri_length)
     * - first parent level at indices [fri_length, fri_length + fri_length/2)
     * - ...
     * - root at index (2*fri_length - 2)
     */
    std::vector<std::vector<uint64_t>> sample_aux_merkle_nodes(const std::vector<size_t>& node_indices);
};

// ============================================================================
// GPU Proof Generation Pipeline Functions (called by GpuStark)
// ============================================================================

namespace pipeline {

/**
 * Initialize sponge state with claim on GPU
 */
void gpu_init_fiat_shamir(
    uint64_t* d_sponge_state,
    const uint64_t* d_program_digest,
    const uint64_t* d_input,
    size_t input_len,
    const uint64_t* d_output,
    size_t output_len,
    uint32_t log2_padded_height,
    cudaStream_t stream
);

/**
 * Perform LDE on entire table (all columns in parallel)
 */
void gpu_lde_table(
    const uint64_t* d_trace_table,      // Input: trace domain table
    uint64_t* d_lde_table,              // Output: FRI domain table
    size_t trace_height,
    size_t lde_height,
    size_t num_columns,
    uint64_t trace_offset,
    uint64_t lde_offset,
    uint64_t* d_scratch,                // Scratch space for NTT
    cudaStream_t stream
);

/**
 * Hash table rows to digests for Merkle tree
 */
void gpu_hash_rows(
    const uint64_t* d_table,            // Table data (BFE or XFE)
    uint64_t* d_digests,                // Output: row digests
    size_t num_rows,
    size_t row_width,                   // Elements per row
    size_t element_size,                // 1 for BFE, 3 for XFE
    cudaStream_t stream
);

/**
 * Build Merkle tree from leaf digests
 */
void gpu_build_merkle_tree(
    uint64_t* d_tree,                   // In: leaves at bottom; Out: full tree
    size_t num_leaves,
    cudaStream_t stream
);

/**
 * Sample challenges from sponge state
 */
void gpu_sample_challenges(
    uint64_t* d_sponge_state,           // In/out: sponge state
    uint64_t* d_challenges,             // Output: sampled challenges
    size_t num_challenges,
    cudaStream_t stream
);

/**
 * Absorb Merkle root into sponge
 */
void gpu_absorb_merkle_root(
    uint64_t* d_sponge_state,
    const uint64_t* d_merkle_root,      // 5 elements (Digest)
    cudaStream_t stream
);

/**
 * Compute auxiliary table from main table and challenges
 */
void gpu_compute_aux_table(
    const uint64_t* d_main_table,
    const uint64_t* d_challenges,
    uint64_t* d_aux_table,
    size_t num_rows,
    size_t main_width,
    size_t aux_width,
    cudaStream_t stream
);

/**
 * Evaluate AIR constraints and compute quotient
 */
void gpu_compute_quotient(
    const uint64_t* d_main_lde,
    const uint64_t* d_aux_lde,
    const uint64_t* d_challenges,
    const uint64_t* d_quotient_weights,
    uint64_t* d_quotient_segments,
    size_t fri_length,
    size_t main_width,
    size_t aux_width,
    cudaStream_t stream
);

/**
 * Evaluate polynomial at out-of-domain point
 */
void gpu_evaluate_ood(
    const uint64_t* d_trace_table,
    const uint64_t* d_ood_point,        // XFE: 3 elements
    uint64_t* d_ood_row,                // Output: XFE row
    size_t trace_height,
    size_t row_width,
    uint64_t trace_offset,
    cudaStream_t stream
);

/**
 * FRI folding round
 */
void gpu_fri_fold(
    const uint64_t* d_codeword,
    uint64_t* d_folded,
    size_t codeword_len,
    const uint64_t* d_challenge,        // XFE: 3 elements
    cudaStream_t stream
);

/**
 * Sample query indices from sponge
 */
void gpu_sample_indices(
    uint64_t* d_sponge_state,
    uint64_t* d_indices,
    size_t num_indices,
    size_t domain_size,
    cudaStream_t stream
);

/**
 * Gather table rows at indices for proof
 */
void gpu_gather_rows(
    const uint64_t* d_table,
    const uint64_t* d_indices,
    uint64_t* d_gathered,
    size_t num_indices,
    size_t row_width,
    cudaStream_t stream
);

/**
 * Gather authentication paths from Merkle tree
 */
void gpu_gather_auth_paths(
    const uint64_t* d_merkle_tree,
    const uint64_t* d_indices,
    uint64_t* d_auth_paths,
    size_t num_indices,
    size_t tree_height,
    cudaStream_t stream
);

/**
 * Append proof item to proof buffer (on GPU)
 */
void gpu_append_proof_item(
    uint64_t* d_proof_buffer,
    size_t* proof_size,
    const uint64_t* d_item_data,
    size_t item_len,
    uint32_t discriminant,
    cudaStream_t stream
);

} // namespace pipeline

} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

