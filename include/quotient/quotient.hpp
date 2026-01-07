#pragma once

#include <vector>
#include <memory>
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "table/master_table.hpp"
#include "stark/challenges.hpp"

namespace triton_vm {

/**
 * Quotient - STARK quotient polynomial computation
 *
 * Computes the quotient polynomial Q such that:
 *   AIR(ω^j) - AIR(zero) = Q(ω^j) * Z(ω^j)
 * where Z is the zerofier polynomial that vanishes on the constraint domain.
 */
class Quotient {
public:
    static constexpr size_t NUM_INITIAL_CONSTRAINTS = 81;
    static constexpr size_t NUM_CONSISTENCY_CONSTRAINTS = 94;
    static constexpr size_t NUM_TRANSITION_CONSTRAINTS = 398;
    static constexpr size_t NUM_TERMINAL_CONSTRAINTS = 23;
    static constexpr size_t MASTER_AUX_NUM_CONSTRAINTS =
        NUM_INITIAL_CONSTRAINTS + NUM_CONSISTENCY_CONSTRAINTS
        + NUM_TRANSITION_CONSTRAINTS + NUM_TERMINAL_CONSTRAINTS;
    static constexpr size_t NUM_QUOTIENT_SEGMENTS = 4;
    /**
     * Compute quotient polynomial from main and auxiliary tables
     *
     * @param main_table The main execution trace table
     * @param aux_table The auxiliary extension table
     * @param challenges Fiat-Shamir challenges
     * @return Quotient polynomial segments
     */
    static std::vector<std::vector<XFieldElement>> compute_quotient(
        const MasterMainTable& main_table,
        const MasterAuxTable& aux_table,
        const Challenges& challenges,
        const std::vector<XFieldElement>& quotient_weights,
        const ArithmeticDomain& fri_domain,
        std::vector<std::vector<XFieldElement>>* out_segment_polynomials = nullptr,
        std::vector<XFieldElement>* out_quotient_values = nullptr
    );

    /**
     * Segmentify quotient evaluations into polynomial segments (matches Rust segmentify)
     *
     * @param quotient_evaluations Quotient evaluations on quotient domain
     * @param trace_length Original trace length
     * @param num_segments Number of segments (NUM_QUOTIENT_SEGMENTS)
     * @param fri_domain FRI domain for NTT operations
     * @return Segment polynomials as coefficient vectors
     */
    static std::vector<std::vector<XFieldElement>> segmentify_quotient_evaluations(
        const std::vector<XFieldElement>& quotient_evaluations,
        size_t trace_length,
        size_t num_segments,
        const ArithmeticDomain& fri_domain
    );

    /**
     * Convert column-oriented quotient segments into row-oriented codewords.
     *
     * @param segments Quotient segments as produced by compute_quotient
     * @return Row-major representation suitable for Merkle commitments
     */
    static std::vector<std::vector<XFieldElement>> segments_to_rows(
        const std::vector<std::vector<XFieldElement>>& segments
    );

    /**
     * Evaluate AIR constraints at a given point
     *
     * @param point Evaluation point (XFieldElement)
     * @param main_table Main table data at this point
     * @param aux_table Auxiliary table data at this point
     * @param challenges Fiat-Shamir challenges
     * @return AIR constraint evaluation
     */
    static XFieldElement evaluate_air(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    /**
     * Compute zerofier inverses for quotient calculation
     * Made public for testing/verification purposes
     */
    static std::vector<BFieldElement> initial_zerofier_inverse(
        const ArithmeticDomain& quotient_domain);
    static std::vector<BFieldElement> consistency_zerofier_inverse(
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain);
    static std::vector<BFieldElement> transition_zerofier_inverse(
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain);
    static std::vector<BFieldElement> terminal_zerofier_inverse(
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain);

    // Constraint evaluation helpers generated from the Rust reference
    // Made public for testing/verification purposes
    static std::vector<XFieldElement> evaluate_initial_constraints(
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges);

    static std::vector<XFieldElement> evaluate_consistency_constraints(
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges);

    static std::vector<XFieldElement> evaluate_transition_constraints(
        const std::vector<BFieldElement>& current_main_row,
        const std::vector<XFieldElement>& current_aux_row,
        const std::vector<BFieldElement>& next_main_row,
        const std::vector<XFieldElement>& next_aux_row,
        const Challenges& challenges);

    static std::vector<XFieldElement> evaluate_terminal_constraints(
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges);

    // Verifier-only: Evaluate constraints when the *main row is over XFieldElement*
    // (out-of-domain main-row evaluations are XFieldElements).
    static std::vector<XFieldElement> evaluate_initial_constraints_xfe_main(
        const std::vector<XFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges);

    static std::vector<XFieldElement> evaluate_consistency_constraints_xfe_main(
        const std::vector<XFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges);

    static std::vector<XFieldElement> evaluate_transition_constraints_xfe_main(
        const std::vector<XFieldElement>& current_main_row,
        const std::vector<XFieldElement>& current_aux_row,
        const std::vector<XFieldElement>& next_main_row,
        const std::vector<XFieldElement>& next_aux_row,
        const Challenges& challenges);

    static std::vector<XFieldElement> evaluate_terminal_constraints_xfe_main(
        const std::vector<XFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges);

private:

    // Table-specific constraint evaluation functions
    static XFieldElement evaluate_program_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_processor_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_op_stack_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_ram_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_jump_stack_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_hash_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_cascade_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_lookup_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );

    static XFieldElement evaluate_u32_table_constraints(
        const XFieldElement& point,
        const std::vector<BFieldElement>& main_row,
        const std::vector<XFieldElement>& aux_row,
        const Challenges& challenges
    );
};

// Utility function for XFieldElement interpolation
std::vector<XFieldElement> interpolate_xfield_column(
    const std::vector<XFieldElement>& values,
    const ArithmeticDomain& domain);

} // namespace triton_vm