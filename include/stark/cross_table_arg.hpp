#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include <vector>

namespace triton_vm {

/**
 * CrossTableArg - Helper functions for cross-table arguments
 */

/**
 * EvalArg - Evaluation argument
 * 
 * Computes: initial * challenge^n + Σ symbols[i] * challenge^i
 * Uses Horner's method: fold(challenge * running + symbol)
 */
class EvalArg {
public:
    static XFieldElement default_initial() {
        return XFieldElement::one();
    }
    
    static XFieldElement compute_terminal(
        const std::vector<BFieldElement>& symbols,
        XFieldElement initial,
        XFieldElement challenge
    ) {
        XFieldElement result = initial;
        for (const auto& symbol : symbols) {
            result = challenge * result + XFieldElement(symbol);
        }
        return result;
    }
};

/**
 * PermArg - Permutation argument
 * 
 * Computes: initial * Π(challenge - symbols[i])
 */
class PermArg {
public:
    static XFieldElement default_initial() {
        return XFieldElement::one();
    }
    
    static XFieldElement compute_terminal(
        const std::vector<BFieldElement>& symbols,
        XFieldElement initial,
        XFieldElement challenge
    ) {
        XFieldElement result = initial;
        for (const auto& symbol : symbols) {
            result = result * (challenge - XFieldElement(symbol));
        }
        return result;
    }
};

/**
 * LookupArg - Lookup argument
 * 
 * Computes: initial + Σ 1/(challenge - symbols[i])
 */
class LookupArg {
public:
    static XFieldElement default_initial() {
        return XFieldElement::zero();
    }
    
    static XFieldElement compute_terminal(
        const std::vector<BFieldElement>& symbols,
        XFieldElement initial,
        XFieldElement challenge
    ) {
        XFieldElement result = initial;
        for (const auto& symbol : symbols) {
            XFieldElement symbol_xfe(symbol);
            XFieldElement diff = challenge - symbol_xfe;
            result = result + diff.inverse();
        }
        return result;
    }
};

} // namespace triton_vm

