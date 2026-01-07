#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "table/master_table.hpp"
#include "table/table_commitment.hpp"
#include "fri/fri.hpp"
#include "proof_stream/proof_stream.hpp"
#include "stark/challenges.hpp"
#include "chacha12_rng.hpp"
#include <vector>
#include <cstdint>

namespace triton_vm {

/**
 * Claim - Public information about a computation
 */
struct Claim {
    Digest program_digest;
    uint32_t version;
    std::vector<BFieldElement> input;
    std::vector<BFieldElement> output;

    void save_to_file(const std::string& path) const;
};

/**
 * Proof serialization helpers.
 */
struct Proof {
    std::vector<BFieldElement> elements;
    
    size_t padded_height() const;

    void save_to_file(const std::string& path) const;
};

/**
 * SimpleAlgebraicExecutionTrace - Simplified witness for proof generation
 * 
 * NOTE: This is a simplified struct used only by Stark::prove().
 * The full AlgebraicExecutionTrace class is in vm/aet.hpp.
 * This struct exists to avoid including the full VM headers in stark.hpp.
 * 
 * IMPORTANT: Do NOT create a type alias named AlgebraicExecutionTrace here,
 * as it conflicts with the full class in vm/aet.hpp. Use SimpleAlgebraicExecutionTrace
 * directly or qualify the type when needed.
 */
struct SimpleAlgebraicExecutionTrace {
    size_t padded_height;
    size_t processor_trace_height;
    size_t processor_trace_width;
    std::vector<std::vector<BFieldElement>> processor_trace;
    // TODO: Add other trace tables
};

/**
 * Stark - Main STARK prover/verifier
 *
 * Implements the Triton VM STARK proof system.
 */
class Stark {
public:
    // The seed for all randomness used while proving.
    // By default, this is set to all zeros for deterministic results.
    // In production, this should be set to a uniformly random value and never reused,
    // as using the same randomness seed twice would violate the zero-knowledge property.
    ChaCha12Rng::Seed randomness_seed;

    // Default parameters: security_level=160, log2_fri_expansion=2
    static Stark default_stark();

    Stark(size_t security_level, size_t log2_fri_expansion_factor);

    // Default constructor using all-zero seed for deterministic results
    Stark() : Stark(160, 2) { randomness_seed.fill(0); }

    // Manual seed setting (for reproducible testing)
    Stark& set_randomness_seed(const ChaCha12Rng::Seed& seed) {
        randomness_seed = seed;
        return *this;
    }

    // Check if deterministic sampling should be used (seed is not all zeros)
    bool use_deterministic_sampling() const {
        for (uint8_t byte : randomness_seed) {
            if (byte != 0) return true;
        }
        return false;
    }

    // Prove correct execution
    Proof prove(const Claim& claim, const SimpleAlgebraicExecutionTrace& aet);
    
    // Prove correct execution with pre-created main table.
    // If proof_path is provided, proof encoding + serialization are handled entirely in Rust via FFI.
    Proof prove_with_table(const Claim& claim, MasterMainTable& main_table, ProofStream& proof_stream, const std::string& proof_path = "", const std::string& test_data_dir = "");
    
    // Verify a proof
    bool verify(const Claim& claim, const Proof& proof);
    
    // Accessors
    size_t security_level() const { return security_level_; }
    size_t fri_expansion_factor() const { return fri_expansion_factor_; }
    size_t num_trace_randomizers() const { return num_trace_randomizers_; }
    size_t num_collinearity_checks() const { return num_collinearity_checks_; }
    
    // Compute derived parameters
    size_t randomized_trace_len(size_t padded_height) const;
    int64_t max_degree(size_t padded_height) const;

    // Public helper functions for evaluation
    static std::vector<XFieldElement> evaluate_bfield_trace_at_point(
        const std::vector<std::vector<BFieldElement>>& trace_table,
        const ArithmeticDomain& trace_domain,
        const XFieldElement& point
    );

    static std::vector<XFieldElement> evaluate_xfield_trace_at_point(
        const std::vector<std::vector<XFieldElement>>& trace_table,
        const ArithmeticDomain& trace_domain,
        const XFieldElement& point
    );

    // Optimized: evaluate at two points, reusing polynomial coefficients
    static std::pair<std::vector<XFieldElement>, std::vector<XFieldElement>> evaluate_bfield_trace_at_two_points(
        const std::vector<std::vector<BFieldElement>>& trace_table,
        const ArithmeticDomain& trace_domain,
        const XFieldElement& point1,
        const XFieldElement& point2
    );

    static std::pair<std::vector<XFieldElement>, std::vector<XFieldElement>> evaluate_xfield_trace_at_two_points(
        const std::vector<std::vector<XFieldElement>>& trace_table,
        const ArithmeticDomain& trace_domain,
        const XFieldElement& point1,
        const XFieldElement& point2
    );

private:
    size_t security_level_;
    size_t fri_expansion_factor_;
    size_t num_trace_randomizers_;
    size_t num_collinearity_checks_;

    // Helper functions for STARK proving (implementation details)

    std::vector<XFieldElement> evaluate_quotient_at_point(
        const std::vector<std::vector<XFieldElement>>& quotient_segment_polynomials,
        const XFieldElement& point
    );

    std::vector<XFieldElement> extend_quotient_segment_to_fri_domain(
        const std::vector<XFieldElement>& segment_polynomial,
        const ArithmeticDomain& quotient_domain,
        const ArithmeticDomain& fri_domain
    );

    // Note: TableCommitment doesn't support quotient segments directly
    // In full implementation, would need a separate commitment mechanism for quotient
    // For now, this is a placeholder
};

} // namespace triton_vm

