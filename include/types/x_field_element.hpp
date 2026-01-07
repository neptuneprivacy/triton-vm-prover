#pragma once

#include "types/b_field_element.hpp"
#include <array>
#include <string>
#include <vector>

namespace triton_vm {

/**
 * XFieldElement - Extension Field Element
 * 
 * Element of the degree-3 extension of the Goldilocks field.
 * Represented as a polynomial a + b*X + c*X^2 where X^3 = X - 1.
 * 
 * This matches the Triton VM Rust implementation (twenty-first crate).
 */
class XFieldElement {
public:
    static constexpr size_t EXTENSION_DEGREE = 3;
    
    // Constructors
    constexpr XFieldElement() : coeffs_{BFieldElement::zero(), BFieldElement::zero(), BFieldElement::zero()} {}
    
    constexpr XFieldElement(BFieldElement c0, BFieldElement c1, BFieldElement c2)
        : coeffs_{c0, c1, c2} {}
    
    // Construct from a base field element (embed as constant polynomial)
    constexpr explicit XFieldElement(BFieldElement base)
        : coeffs_{base, BFieldElement::zero(), BFieldElement::zero()} {}
    
    // Factory methods
    static constexpr XFieldElement zero() { 
        return XFieldElement(); 
    }
    
    static constexpr XFieldElement one() { 
        return XFieldElement(BFieldElement::one(), BFieldElement::zero(), BFieldElement::zero()); 
    }
    
    // Accessors
    constexpr const std::array<BFieldElement, 3>& coefficients() const { return coeffs_; }
    constexpr BFieldElement coeff(size_t i) const { return coeffs_[i]; }
    
    // Arithmetic operations
    XFieldElement operator+(const XFieldElement& rhs) const;
    XFieldElement operator-(const XFieldElement& rhs) const;
    XFieldElement operator*(const XFieldElement& rhs) const;
    XFieldElement operator/(const XFieldElement& rhs) const;
    XFieldElement operator-() const;
    
    XFieldElement& operator+=(const XFieldElement& rhs);
    XFieldElement& operator-=(const XFieldElement& rhs);
    XFieldElement& operator*=(const XFieldElement& rhs);
    XFieldElement& operator/=(const XFieldElement& rhs);
    
    // Mixed arithmetic with BFieldElement
    XFieldElement operator+(const BFieldElement& rhs) const;
    XFieldElement operator-(const BFieldElement& rhs) const;
    XFieldElement operator*(const BFieldElement& rhs) const;
    XFieldElement operator/(const BFieldElement& rhs) const;
    
    // Comparison
    bool operator==(const XFieldElement& rhs) const;
    bool operator!=(const XFieldElement& rhs) const;
    
    // Field operations
    XFieldElement inverse() const;
    XFieldElement pow(uint64_t exp) const;
    bool is_zero() const;
    bool is_one() const;
    
    // String representation
    std::string to_string() const;

    // Batch inversion for efficiency
    static std::vector<XFieldElement> batch_inversion(const std::vector<XFieldElement>& elements);
    
    friend std::ostream& operator<<(std::ostream& os, const XFieldElement& elem);
    friend XFieldElement operator+(const BFieldElement& lhs, const XFieldElement& rhs);
    friend XFieldElement operator-(const BFieldElement& lhs, const XFieldElement& rhs);
    friend XFieldElement operator*(const BFieldElement& lhs, const XFieldElement& rhs);

private:
    std::array<BFieldElement, 3> coeffs_;
};

// Type alias for convenience
using XFE = XFieldElement;

} // namespace triton_vm

