#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "hash/tip5.hpp"
#include "chacha12_rng.hpp"
#include <vector>
#include <variant>
#include <stdexcept>
#include <utility>

namespace triton_vm {

// Forward declaration
struct FriResponse;

/**
 * ProofItem - Variant type for items in the proof stream
 * Discriminants match Rust's BFieldCodec derive order exactly
 */
enum class ProofItemType {
    MerkleRoot = 0,
    OutOfDomainMainRow = 1,
    OutOfDomainAuxRow = 2,
    OutOfDomainQuotientSegments = 3,
    AuthenticationStructure = 4,
    MasterMainTableRows = 5,
    MasterAuxTableRows = 6,
    Log2PaddedHeight = 7,
    QuotientSegmentsElements = 8,
    FriCodeword = 9,
    FriPolynomial = 10,
    FriResponse = 11
};

/**
 * ProofItem - A single item in the proof stream
 */
struct ProofItem {
    ProofItemType type;
    
    // Possible payloads (using variant would be more type-safe but complex)
    Digest digest;                             // For MerkleRoot
    uint32_t log2_padded_height_value;         // For Log2PaddedHeight
    std::vector<XFieldElement> xfield_vec;     // For FriCodeword, rows, etc.
    std::vector<BFieldElement> bfield_vec;     // For encoding
    std::vector<Digest> digests;               // For AuthenticationStructure, FriResponse auth
    
    // Factory methods
    static ProofItem merkle_root(const Digest& root);
    static ProofItem make_log2_padded_height(uint32_t height);
    static ProofItem out_of_domain_main_row(const std::vector<XFieldElement>& row);
    static ProofItem out_of_domain_aux_row(const std::vector<XFieldElement>& row);
    static ProofItem out_of_domain_quotient_segments(const std::vector<XFieldElement>& segments);
    static ProofItem fri_codeword(const std::vector<XFieldElement>& codeword);
    static ProofItem fri_polynomial(const std::vector<XFieldElement>& polynomial);
    static ProofItem fri_response(const FriResponse& response);
    static ProofItem authentication_structure(const std::vector<Digest>& auth_path);
    static ProofItem master_main_table_rows(const std::vector<std::vector<BFieldElement>>& rows);
    static ProofItem master_aux_table_rows(const std::vector<std::vector<XFieldElement>>& rows);
    static ProofItem quotient_segments_elements(const std::vector<std::vector<XFieldElement>>& segments);
    static ProofItem decode(const std::vector<BFieldElement>& data);

    bool try_into_merkle_root(Digest& out) const;
    bool try_into_out_of_domain_main_row(std::vector<XFieldElement>& out) const;
    bool try_into_out_of_domain_aux_row(std::vector<XFieldElement>& out) const;
    bool try_into_out_of_domain_quotient_segments(std::vector<XFieldElement>& out) const;
    bool try_into_fri_response(FriResponse& out) const;
    bool try_into_master_main_table_rows(std::vector<std::vector<BFieldElement>>& out) const;
    bool try_into_master_aux_table_rows(std::vector<std::vector<XFieldElement>>& out) const;
    bool try_into_quotient_segments_elements(std::vector<std::vector<XFieldElement>>& out) const;
    bool try_into_fri_codeword(std::vector<XFieldElement>& out) const;
    bool try_into_fri_polynomial(std::vector<XFieldElement>& out) const;
    bool try_into_log2_padded_height(uint32_t& out) const;
    bool try_into_authentication_structure(std::vector<Digest>& out) const;
    
    // Should this item be included in Fiat-Shamir heuristic?
    bool include_in_fiat_shamir_heuristic() const;
    
    // Encode to BFieldElements for hashing
    std::vector<BFieldElement> encode() const;
};

/**
 * FriResponse - Matches Rust's FriResponse struct exactly
 */
struct FriResponse {
    std::vector<Digest> auth_structure;
    std::vector<XFieldElement> revealed_leaves;
};

/**
 * ProofStream - Fiat-Shamir transcript for non-interactive proofs
 * 
 * The proof stream maintains a Tip5 sponge that absorbs proof items
 * and produces random challenges using the Fiat-Shamir heuristic.
 */
class ProofStream {
public:
    ProofStream();
    static ProofStream decode(const std::vector<BFieldElement>& encoding);
    
    // Add item to proof stream (as prover)
    void enqueue(const ProofItem& item);
    
    // Get item from proof stream (as verifier)
    ProofItem dequeue();
    
    // Alter Fiat-Shamir state with additional data (e.g., claim)
    void alter_fiat_shamir_state_with(const std::vector<BFieldElement>& data);
    
    // Sample random scalars (XFieldElements) from sponge
    std::vector<XFieldElement> sample_scalars(size_t count);

    // Sample random scalars deterministically using ChaCha12Rng
    std::vector<XFieldElement> sample_scalars_deterministic(const ChaCha12Rng::Seed& seed, size_t count);

    // Sample random indices in range [0, upper_bound)
    std::vector<size_t> sample_indices(size_t upper_bound, size_t count);
    
    // Get the proof items
    const std::vector<ProofItem>& items() const { return items_; }
    
    // Encode entire proof stream (BFieldCodec-compatible)
    std::vector<BFieldElement> encode() const;
    
    /// Encode proof stream and serialize to file entirely in Rust (via FFI)
    /// This handles steps 12 and 13 entirely in Rust for 100% compatibility
    /// @param file_path Path to output proof file
    /// @throws std::runtime_error if encoding/serialization fails
    void encode_and_save_to_file(const std::string& file_path) const;
    
    // Get the sponge (for testing)
    const Tip5& sponge() const { return sponge_; }
    
    // Set the sponge state (for testing/verification)
    // Also sets first_varlen_absorption_ to false since we're reconstructing an already-used state
    void set_sponge_state(const Tip5& sponge) { 
        sponge_ = sponge; 
        first_varlen_absorption_ = false;  // State is already initialized and used
    }

private:
    std::vector<ProofItem> items_;
    size_t items_index_;
    Tip5 sponge_;
    bool first_varlen_absorption_;  // Track if this is the first variable-length absorption
    
    // Pad and absorb elements into sponge
    void pad_and_absorb_all(const std::vector<BFieldElement>& elements);
    
    // Squeeze elements from sponge
    std::array<BFieldElement, Tip5::RATE> squeeze();
};

} // namespace triton_vm
