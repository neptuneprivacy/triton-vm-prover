#pragma once

#include "types/b_field_element.hpp"
#include "table/master_table.hpp"
#include <vector>

namespace triton_vm {

/**
 * Number Theoretic Transform (NTT) for Goldilocks field
 * 
 * Used for:
 * - Interpolation: evaluations → polynomial coefficients (inverse NTT)
 * - Evaluation: polynomial coefficients → evaluations (forward NTT)
 */
class NTT {
public:
    /**
     * Forward NTT: polynomial coefficients → evaluations
     * 
     * Given coefficients [c₀, c₁, ..., c_{n-1}], compute evaluations
     * [f(ω⁰), f(ω¹), ..., f(ω^{n-1})] where ω is the n-th root of unity.
     * 
     * @param coeffs Input coefficients (will be modified in-place)
     */
    static void forward(std::vector<BFieldElement>& coeffs);
    
    /**
     * Inverse NTT: evaluations → polynomial coefficients
     * 
     * Given evaluations [f(ω⁰), f(ω¹), ..., f(ω^{n-1})], compute
     * the polynomial coefficients [c₀, c₁, ..., c_{n-1}].
     * 
     * @param evals Input evaluations (will be modified in-place)
     */
    static void inverse(std::vector<BFieldElement>& evals);
    
    /**
     * Interpolate a single column from trace domain to polynomial coefficients
     * 
     * @param column Evaluations on trace domain
     * @return Polynomial coefficients
     */
    static std::vector<BFieldElement> interpolate(const std::vector<BFieldElement>& column);
    
    /**
     * Evaluate polynomial on a coset of the quotient domain
     * 
     * @param coeffs Polynomial coefficients
     * @param domain_length Length of evaluation domain
     * @param offset Coset offset (generator^k for coset k)
     * @return Evaluations on the coset
     */
    static std::vector<BFieldElement> evaluate_on_coset(
        const std::vector<BFieldElement>& coeffs,
        size_t domain_length,
        BFieldElement offset
    );

private:
    /**
     * Bit-reverse permutation for iterative NTT
     */
    static void bit_reverse_permutation(std::vector<BFieldElement>& data);
    
    /**
     * Core NTT butterfly operation
     * 
     * @param data Data to transform
     * @param omega Primitive root of unity
     * @param inverse True for inverse NTT
     */
    static void ntt_core(std::vector<BFieldElement>& data, BFieldElement omega, bool inverse);
};

/**
 * Low-Degree Extension (LDE) for a table
 * 
 * Extends evaluations from trace domain to quotient domain:
 * 1. Interpolate: trace evaluations → polynomial coefficients
 * 2. Evaluate: polynomial → quotient domain evaluations
 */
class LDE {
public:
    /**
     * Perform LDE on a single column
     * 
     * @param trace_column Column values on trace domain
     * @param trace_domain Trace domain parameters
     * @param quotient_domain Quotient domain parameters  
     * @return Column values on quotient domain
     */
    static std::vector<BFieldElement> extend_column(
        const std::vector<BFieldElement>& trace_column,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain
    );
    
    /**
     * Perform LDE on entire table
     * 
     * @param trace_table Input table on trace domain (rows × cols)
     * @param trace_domain Trace domain parameters
     * @param quotient_domain Quotient domain parameters
     * @return Extended table on quotient domain
     */
    static std::vector<std::vector<BFieldElement>> extend_table(
        const std::vector<std::vector<BFieldElement>>& trace_table,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain
    );
};

} // namespace triton_vm

