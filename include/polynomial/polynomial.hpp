#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include <vector>
#include <algorithm>

namespace triton_vm {

/**
 * Polynomial representation and operations
 * 
 * A polynomial is represented as a vector of coefficients:
 *   p(x) = c₀ + c₁x + c₂x² + ... + c_{n-1}x^{n-1}
 */
template<typename Field>
class Polynomial {
public:
    Polynomial() = default;
    explicit Polynomial(const std::vector<Field>& coefficients) : coeffs_(coefficients) {}
    explicit Polynomial(std::vector<Field>&& coefficients) : coeffs_(std::move(coefficients)) {}
    
    // Get coefficients, removing trailing zeros (matching Rust's Polynomial::coefficients())
    std::vector<Field> coefficients() const {
        if (coeffs_.empty()) {
            return {};
        }
        
        // Find the last non-zero coefficient (Rust uses rposition)
        size_t last_nonzero = 0;
        Field zero = Field::zero();
        for (int i = static_cast<int>(coeffs_.size()) - 1; i >= 0; --i) {
            if (!(coeffs_[i] == zero)) {
                last_nonzero = static_cast<size_t>(i) + 1;
                break;
            }
        }
        
        return std::vector<Field>(coeffs_.begin(), coeffs_.begin() + last_nonzero);
    }
    
    // Get raw coefficients (for internal use when we need the full vector including trailing zeros)
    const std::vector<Field>& raw_coefficients() const { return coeffs_; }
    std::vector<Field>& raw_coefficients() { return coeffs_; }
    
    size_t degree() const {
        // Find highest non-zero coefficient
        Field zero = Field::zero();
        for (int i = static_cast<int>(coeffs_.size()) - 1; i >= 0; i--) {
            if (!(coeffs_[i] == zero)) {
                return static_cast<size_t>(i);
            }
        }
        return 0;  // Zero polynomial
    }
    
    size_t size() const { return coeffs_.size(); }
    void resize(size_t n, Field value = Field::zero()) {
        coeffs_.resize(n, value);
    }
    
    Field& operator[](size_t i) { return coeffs_[i]; }
    const Field& operator[](size_t i) const { return coeffs_[i]; }
    
    // Evaluate polynomial at point x
    Field evaluate(const Field& x) const {
        if (coeffs_.empty()) {
            return Field::zero();  // Use zero() static method
        }
        
        // Horner's method for efficient evaluation
        Field result = coeffs_.back();
        for (int i = static_cast<int>(coeffs_.size()) - 2; i >= 0; i--) {
            result = result * x + coeffs_[i];
        }
        return result;
    }
    
    // Evaluate BFieldElement polynomial at XFieldElement point
    // This allows evaluating Polynomial<BFieldElement> at XFieldElement, matching Rust's behavior
    template<typename EvalType>
    EvalType evaluate_at_extension(const EvalType& x) const {
        static_assert(std::is_same_v<Field, BFieldElement>, 
                     "evaluate_at_extension only works for Polynomial<BFieldElement>");
        if (coeffs_.empty()) {
            return EvalType::zero();
        }
        
        // Horner's method: result = result * x + c
        // where c is BFieldElement (lifted to EvalType) and x is EvalType
        EvalType result = EvalType(coeffs_.back());
        for (int i = static_cast<int>(coeffs_.size()) - 2; i >= 0; i--) {
            result = result * x + EvalType(coeffs_[i]);
        }
        return result;
    }
    
    // Add two polynomials
    Polynomial operator+(const Polynomial& other) const {
        size_t max_size = std::max(coeffs_.size(), other.coeffs_.size());
        std::vector<Field> result(max_size, Field());
        
        for (size_t i = 0; i < coeffs_.size(); i++) {
            result[i] = result[i] + coeffs_[i];
        }
        
        for (size_t i = 0; i < other.coeffs_.size(); i++) {
            result[i] = result[i] + other.coeffs_[i];
        }
        
        return Polynomial(result);
    }
    
    // Subtract two polynomials
    Polynomial operator-(const Polynomial& other) const {
        size_t max_size = std::max(coeffs_.size(), other.coeffs_.size());
        std::vector<Field> result(max_size, Field());
        
        for (size_t i = 0; i < coeffs_.size(); i++) {
            result[i] = result[i] + coeffs_[i];
        }
        
        for (size_t i = 0; i < other.coeffs_.size(); i++) {
            result[i] = result[i] - other.coeffs_[i];
        }
        
        return Polynomial(result);
    }
    
    // Multiply polynomial by scalar
    Polynomial operator*(const Field& scalar) const {
        std::vector<Field> result;
        result.reserve(coeffs_.size());
        
        for (const auto& coeff : coeffs_) {
            result.push_back(coeff * scalar);
        }
        
        return Polynomial(result);
    }
    
    // Multiply two polynomials
    Polynomial operator*(const Polynomial& other) const {
        size_t result_size = coeffs_.size() + other.coeffs_.size() - 1;
        std::vector<Field> result(result_size, Field::zero());
        
        for (size_t i = 0; i < coeffs_.size(); i++) {
            for (size_t j = 0; j < other.coeffs_.size(); j++) {
                result[i + j] = result[i + j] + (coeffs_[i] * other.coeffs_[j]);
            }
        }
        
        return Polynomial(result);
    }
    
    // Shift coefficients (multiply by x^n)
    Polynomial shift_coefficients(size_t n) const {
        std::vector<Field> result;
        result.reserve(n + coeffs_.size());
        // Add n zeros at the beginning
        for (size_t i = 0; i < n; i++) {
            result.push_back(Field());
        }
        result.insert(result.end(), coeffs_.begin(), coeffs_.end());
        return Polynomial(result);
    }
    
    static Polynomial x_to_the(size_t n) {
        std::vector<Field> coeffs(n + 1, Field::zero());
        coeffs[n] = Field::one();
        return Polynomial(coeffs);
    }
    
    static Polynomial from_constant(Field constant) {
        return Polynomial(std::vector<Field>{constant});
    }
    
private:
    std::vector<Field> coeffs_;
};

using BPolynomial = Polynomial<BFieldElement>;
using XPolynomial = Polynomial<XFieldElement>;

} // namespace triton_vm

