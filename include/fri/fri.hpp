#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "table/master_table.hpp"
#include "merkle/merkle_tree.hpp"
#include <vector>
#include <memory>
#include <optional>

namespace triton_vm {

// Forward declarations
class ProofStream;
struct FriResponse;  // Defined in proof_stream.hpp

/**
 * FriRound - A single round in the FRI protocol
 */
struct FriRound {
    ArithmeticDomain domain;
    std::vector<XFieldElement> codeword;
    std::unique_ptr<MerkleTree> merkle_tree;
    
    FriRound(const ArithmeticDomain& domain, const std::vector<XFieldElement>& codeword);
    
    // Split and fold the codeword with the given challenge
    std::vector<XFieldElement> split_and_fold(const XFieldElement& folding_challenge) const;
    
    // Get the Merkle root
    Digest merkle_root() const;
};

/**
 * Fri - Fast Reed-Solomon IOP of Proximity
 * 
 * Main FRI protocol implementation for proving low-degreeness of polynomials.
 * Now integrates directly with ProofStream like Rust does.
 */
class Fri {
public:
    /**
     * Create a FRI instance with the given parameters.
     * 
     * @param domain The initial domain (FRI domain from prover domains)
     * @param expansion_factor The blowup factor (must be power of 2)
     * @param num_collinearity_checks Number of random queries for soundness
     */
    Fri(const ArithmeticDomain& domain, 
        size_t expansion_factor,
        size_t num_collinearity_checks);
    
    // Get the number of FRI rounds (matches Rust calculation with optimization)
    size_t num_rounds() const;
    
    // Get the first round max degree
    size_t first_round_max_degree() const;
    
    // Get the last round max degree
    size_t last_round_max_degree() const;
    
    // Get the domain
    const ArithmeticDomain& domain() const { return domain_; }
    
    // Get parameters
    size_t expansion_factor() const { return expansion_factor_; }
    size_t num_collinearity_checks() const { return num_collinearity_checks_; }
    
    /**
     * Generate a FRI proof, directly interacting with the proof stream.
     * This matches Rust's pattern where FRI commits and samples through the stream.
     * 
     * @param codeword The codeword to prove low-degreeness of
     * @param proof_stream The Fiat-Shamir transcript (for commits and challenges)
     * @return The revealed first-round indices for trace openings
     */
    std::vector<size_t> prove(
        const std::vector<XFieldElement>& codeword,
        ProofStream& proof_stream
    ) const;
    
    /**
     * Verify a FRI proof from the proof stream.
     * 
     * @param proof_stream The proof stream to read from
     * @return true if the proof is valid
     */
    bool verify(ProofStream& proof_stream) const;

    struct VerifyResult {
        std::vector<size_t> first_round_indices;
        std::vector<XFieldElement> first_round_values;
    };

    // Verify a FRI proof and return the first-round query indices and values (a-indices).
    // This is needed by the surrounding STARK verifier to open trace rows at those indices.
    std::optional<VerifyResult> verify_and_get_first_round(ProofStream& proof_stream) const;

#ifdef TRITON_CUDA_ENABLED
    /**
     * GPU-accelerated FRI prove.
     * Uses GPU for folding and Merkle tree operations.
     * 
     * @param codeword The codeword to prove low-degreeness of
     * @param proof_stream The Fiat-Shamir transcript
     * @return The revealed first-round indices for trace openings
     */
    std::vector<size_t> prove_gpu(
        const std::vector<XFieldElement>& codeword,
        ProofStream& proof_stream
    ) const;
#endif
    
    // Fold from index i in original domain to index in folded domain
    static size_t fold_index(size_t index, size_t domain_length);

private:
    ArithmeticDomain domain_;
    size_t expansion_factor_;
    size_t num_collinearity_checks_;
    
    // Compute the domain for a given round
    ArithmeticDomain round_domain(size_t round) const;
    
    // Compute b-indices for collinearity check
    std::vector<size_t> collinearity_check_b_indices(
        const std::vector<size_t>& a_indices,
        size_t domain_length) const;
};

/**
 * Batch inversion of BFieldElements
 */
std::vector<BFieldElement> batch_inverse(const std::vector<BFieldElement>& elements);

} // namespace triton_vm
