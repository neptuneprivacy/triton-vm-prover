#include "types/x_field_element.hpp"
#include <sstream>
#include <stdexcept>
#include <vector>

namespace triton_vm {

XFieldElement XFieldElement::operator+(const XFieldElement& rhs) const {
    return XFieldElement(
        coeffs_[0] + rhs.coeffs_[0],
        coeffs_[1] + rhs.coeffs_[1],
        coeffs_[2] + rhs.coeffs_[2]
    );
}

XFieldElement XFieldElement::operator-(const XFieldElement& rhs) const {
    return XFieldElement(
        coeffs_[0] - rhs.coeffs_[0],
        coeffs_[1] - rhs.coeffs_[1],
        coeffs_[2] - rhs.coeffs_[2]
    );
}

XFieldElement XFieldElement::operator*(const XFieldElement& rhs) const {
    // Polynomial multiplication modulo X^3 - X + 1
    // Shah polynomial: X^3 - X + 1 = 0, so X^3 = X - 1
    // (a0 + a1*X + a2*X^2) * (b0 + b1*X + b2*X^2)
    
    const auto& a = coeffs_;
    const auto& b = rhs.coeffs_;
    
    // Compute products
    BFieldElement c0 = a[0] * b[0];
    BFieldElement c1 = a[0] * b[1] + a[1] * b[0];
    BFieldElement c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
    BFieldElement c3 = a[1] * b[2] + a[2] * b[1];
    BFieldElement c4 = a[2] * b[2];
    
    // Reduce using X^3 = X - 1
    // c3 * X^3 = c3 * (X - 1) = c3*X - c3
    // X^4 = X * X^3 = X * (X - 1) = X^2 - X
    // c4 * X^4 = c4 * (X^2 - X) = c4*X^2 - c4*X
    
    return XFieldElement(
        c0 - c3,           // constant term
        c1 + c3 - c4,      // X coefficient
        c2 + c4            // X^2 coefficient
    );
}

XFieldElement XFieldElement::operator-() const {
    return XFieldElement(-coeffs_[0], -coeffs_[1], -coeffs_[2]);
}

XFieldElement& XFieldElement::operator+=(const XFieldElement& rhs) {
    *this = *this + rhs;
    return *this;
}

XFieldElement& XFieldElement::operator-=(const XFieldElement& rhs) {
    *this = *this - rhs;
    return *this;
}

XFieldElement& XFieldElement::operator*=(const XFieldElement& rhs) {
    *this = *this * rhs;
    return *this;
}

XFieldElement& XFieldElement::operator/=(const XFieldElement& rhs) {
    *this = *this / rhs;
    return *this;
}

XFieldElement XFieldElement::operator+(const BFieldElement& rhs) const {
    return XFieldElement(coeffs_[0] + rhs, coeffs_[1], coeffs_[2]);
}

XFieldElement XFieldElement::operator-(const BFieldElement& rhs) const {
    return XFieldElement(coeffs_[0] - rhs, coeffs_[1], coeffs_[2]);
}

XFieldElement XFieldElement::operator*(const BFieldElement& rhs) const {
    return XFieldElement(
        coeffs_[0] * rhs,
        coeffs_[1] * rhs,
        coeffs_[2] * rhs
    );
}

XFieldElement XFieldElement::operator/(const BFieldElement& rhs) const {
    BFieldElement inv = rhs.inverse();
    return *this * inv;
}

bool XFieldElement::operator==(const XFieldElement& rhs) const {
    return coeffs_[0] == rhs.coeffs_[0] &&
           coeffs_[1] == rhs.coeffs_[1] &&
           coeffs_[2] == rhs.coeffs_[2];
}

bool XFieldElement::operator!=(const XFieldElement& rhs) const {
    return !(*this == rhs);
}

bool XFieldElement::is_zero() const {
    return coeffs_[0].is_zero() && coeffs_[1].is_zero() && coeffs_[2].is_zero();
}

bool XFieldElement::is_one() const {
    return coeffs_[0].is_one() && coeffs_[1].is_zero() && coeffs_[2].is_zero();
}

XFieldElement XFieldElement::inverse() const {
    if (is_zero()) {
        throw std::domain_error("Cannot invert zero");
    }
    
    // For extension field modulo X^3 - X + 1 (where X^3 = X - 1),
    // we use the adjugate/determinant formula.
    //
    // For element (a, b, c) representing a + b*X + c*X^2, the inverse is
    // computed by solving the linear system M * [d, e, f]^T = [1, 0, 0]^T
    // where M is the matrix representation of multiplication.
    //
    // Matrix M (from the multiplication expansion):
    // M = [  a,    -c,     -b   ]
    //     [  b,   a+c,    b-c   ]
    //     [  c,    b,     a+c   ]
    //
    // The inverse is given by (adjugate(M) / det(M)) * [1, 0, 0]^T
    // which is the first column of adjugate(M) divided by det(M).
    
    const BFieldElement& a = coeffs_[0];
    const BFieldElement& b = coeffs_[1];
    const BFieldElement& c = coeffs_[2];
    
    // Precompute common terms
    BFieldElement a2 = a * a;
    BFieldElement b2 = b * b;
    BFieldElement c2 = c * c;
    BFieldElement a3 = a2 * a;
    BFieldElement b3 = b2 * b;
    BFieldElement c3 = c2 * c;
    BFieldElement ab = a * b;
    BFieldElement ac = a * c;
    BFieldElement bc = b * c;
    BFieldElement abc = a * b * c;
    
    // Determinant: det = a³ + 2a²c + ac² - ab² + 3abc + c³ - b³ + bc²
    BFieldElement det = a3 - b3 + c3 
                      + BFieldElement(3) * abc 
                      + BFieldElement(2) * a2 * c 
                      + ac * c 
                      + b * c2
                      - a * b2;
    
    // First column of adjugate matrix (which is what we multiply [1,0,0]^T by):
    // adj00 = (a+c)² - b² + bc = a² + 2ac + c² - b² + bc
    // adj10 = -(−ab − c²) = ab + c²  -- wait, this is wrong. Let me recalculate.
    // Actually: C₀₁ = -ab - c², so adj₁₀ = C₀₁ = -ab - c²
    // And: C₀₂ = b² - ca - c², so adj₂₀ = C₀₂ = b² - ac - c²
    
    BFieldElement a_plus_c = a + c;
    BFieldElement adj0 = a_plus_c * a_plus_c - b2 + bc;  // (a+c)² - b² + bc
    BFieldElement adj1 = -(ab + c2);                      // -ab - c²
    BFieldElement adj2 = b2 - ac - c2;                    // b² - ac - c²
    
    // Inverse coefficients
    BFieldElement det_inv = det.inverse();
    
    return XFieldElement(
        adj0 * det_inv,
        adj1 * det_inv,
        adj2 * det_inv
    );
}

XFieldElement XFieldElement::operator/(const XFieldElement& rhs) const {
    return *this * rhs.inverse();
}

XFieldElement XFieldElement::pow(uint64_t exp) const {
    XFieldElement result = XFieldElement::one();
    XFieldElement base = *this;
    
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    return result;
}

std::string XFieldElement::to_string() const {
    std::ostringstream oss;
    oss << "(" << coeffs_[0].value() 
        << ", " << coeffs_[1].value() 
        << ", " << coeffs_[2].value() << ")";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const XFieldElement& elem) {
    return os << elem.to_string();
}

std::vector<XFieldElement> XFieldElement::batch_inversion(const std::vector<XFieldElement>& elements) {
    if (elements.empty()) {
        return {};
    }

    size_t n = elements.size();
    std::vector<XFieldElement> result(n);

    // Compute running products
    std::vector<XFieldElement> products(n);
    products[0] = elements[0];
    for (size_t i = 1; i < n; ++i) {
        products[i] = products[i - 1] * elements[i];
    }

    // Compute inverse of final product
    XFieldElement product_inverse = products.back().inverse();

    // Compute individual inverses by traversing backwards
    for (size_t i = n; i > 0; --i) {
        size_t idx = i - 1;
        XFieldElement prefix = (idx == 0) ? XFieldElement::one() : products[idx - 1];
        result[idx] = product_inverse * prefix;
        product_inverse = product_inverse * elements[idx];
    }

    return result;
}

XFieldElement operator+(const BFieldElement& lhs, const XFieldElement& rhs) {
    return rhs + lhs;
}

XFieldElement operator-(const BFieldElement& lhs, const XFieldElement& rhs) {
    return XFieldElement(lhs) - rhs;
}

XFieldElement operator*(const BFieldElement& lhs, const XFieldElement& rhs) {
    return rhs * lhs;
}

} // namespace triton_vm

