#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "table/master_table.hpp"
#include "polynomial/polynomial.hpp"
#include <vector>

namespace triton_vm {

/**
 * Randomized Low-Degree Extension
 * 
 * Implements LDE with trace randomizers for zero-knowledge:
 *   randomized_interpolant = interpolant + zerofier * randomizer
 */
class RandomizedLDE {
public:
    /**
     * Compute zerofier for a domain
     * 
     * Zerofier: x^n - offset^n where n = domain.length
     * Evaluates to zero on all points in the domain.
     */
    static BPolynomial compute_zerofier(
        const ArithmeticDomain& domain
    );
    
    /**
     * Multiply polynomial by zerofier (optimized)
     * 
     * For zerofier z(x) = x^n - offset^n:
     *   z(x) * p(x) = x^n * p(x) - offset^n * p(x)
     * 
     * This is more efficient than full polynomial multiplication.
     */
    static BPolynomial mul_zerofier_with(
        const ArithmeticDomain& domain,
        const BPolynomial& polynomial
    );
    
    /**
     * Perform randomized LDE on a column
     * 
     * Steps:
     *   1. Interpolate trace column -> interpolant
     *   2. Generate/load trace randomizer polynomial
     *   3. Compute: randomized_interpolant = interpolant + zerofier * randomizer
     *   4. Evaluate randomized_interpolant on quotient domain
     * 
     * @param trace_column Input trace values
     * @param trace_domain Trace domain parameters
     * @param quotient_domain Quotient domain parameters
     * @param randomizer_coeffs Randomizer polynomial coefficients
     * @return LDE values on quotient domain
     */
    static std::vector<BFieldElement> extend_column_with_randomizer(
        const std::vector<BFieldElement>& trace_column,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const std::vector<BFieldElement>& randomizer_coeffs
    );
    
    /**
     * Perform randomized LDE on an XFieldElement column
     * 
     * This handles XFieldElement interpolation and lifts BFieldElement randomizer
     * to XFieldElement (matching Rust's behavior).
     * 
     * @param trace_column Input trace values (XFieldElement)
     * @param trace_domain Trace domain parameters
     * @param quotient_domain Quotient domain parameters
     * @param randomizer_coeffs Randomizer polynomial coefficients (BFieldElement, will be lifted)
     * @return LDE values on quotient domain (XFieldElement)
     */
    static std::vector<XFieldElement> extend_xfield_column_with_randomizer(
        const std::vector<XFieldElement>& trace_column,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const std::vector<BFieldElement>& randomizer_coeffs
    );
    
    /**
     * Perform randomized LDE on an XFieldElement column with XFieldElement randomizer
     * 
     * This version accepts XFieldElement randomizer coefficients directly (for aux table).
     * 
     * @param trace_column Input trace values (XFieldElement)
     * @param trace_domain Trace domain parameters
     * @param quotient_domain Quotient domain parameters
     * @param randomizer_coeffs Randomizer polynomial coefficients (XFieldElement)
     * @return LDE values on quotient domain (XFieldElement)
     */
    static std::vector<XFieldElement> extend_xfield_column_with_xfield_randomizer(
        const std::vector<XFieldElement>& trace_column,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const std::vector<XFieldElement>& randomizer_coeffs
    );
    
    /**
     * Verify zerofier evaluates to zero on domain
     */
    static bool verify_zerofier(
        const ArithmeticDomain& domain,
        const BPolynomial& zerofier
    );
};

} // namespace triton_vm

