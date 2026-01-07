#include "lde/lde_randomized.hpp"
#include "ntt/ntt.hpp"
#include "polynomial/polynomial.hpp"
#include "quotient/quotient.hpp"
#include <stdexcept>
#include <algorithm>

namespace triton_vm {

BPolynomial RandomizedLDE::compute_zerofier(const ArithmeticDomain& domain) {
    // Zerofier: x^n - offset^n where n = domain.length
    
    if (domain.length == 0) {
        throw std::invalid_argument("Domain length must be > 0");
    }
    
    // If offset is zero, zerofier is x^n
    if (domain.offset.value() == 0) {
        return BPolynomial::x_to_the(domain.length);
    }
    
    // Compute offset^n
    BFieldElement offset_pow_n = domain.offset.pow(domain.length);
    
    // Create x^n polynomial
    BPolynomial x_to_n = BPolynomial::x_to_the(domain.length);
    
    // Create constant polynomial -offset^n
    BPolynomial minus_offset_n = BPolynomial::from_constant(BFieldElement(0) - offset_pow_n);
    
    // Zerofier = x^n - offset^n
    BPolynomial zerofier = x_to_n + minus_offset_n;
    
    return zerofier;
}

BPolynomial RandomizedLDE::mul_zerofier_with(
    const ArithmeticDomain& domain,
    const BPolynomial& polynomial
) {
    // Optimized multiplication: z(x) * p(x) where z(x) = x^n - offset^n
    // Result = x^n * p(x) - offset^n * p(x)
    // 
    // Rust implementation:
    //   polynomial.clone().shift_coefficients(self.length)
    //     - polynomial.scalar_mul(self.offset.mod_pow(self.length as u64))
    
    // Shift polynomial coefficients by n (multiply by x^n)
    BPolynomial shifted = polynomial.shift_coefficients(domain.length);
    
    // Compute offset^n
    BFieldElement offset_pow_n = domain.offset.pow(domain.length);
    
    // Multiply polynomial by offset^n (scalar multiplication)
    BPolynomial scaled = polynomial * offset_pow_n;
    
    // Need to align sizes for subtraction
    size_t max_size = std::max(shifted.size(), scaled.size());
    shifted.resize(max_size, BFieldElement(0));
    scaled.resize(max_size, BFieldElement(0));
    
    // Compute: shifted - scaled
    std::vector<BFieldElement> result_coeffs(max_size);
    for (size_t i = 0; i < max_size; i++) {
        result_coeffs[i] = shifted[i] - scaled[i];
    }
    
    return BPolynomial(result_coeffs);
}

std::vector<BFieldElement> RandomizedLDE::extend_column_with_randomizer(
    const std::vector<BFieldElement>& trace_column,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& target_domain,
    const std::vector<BFieldElement>& randomizer_coeffs
) {
    // Step 1: Interpolate trace column
    // Rust uses: trace_domain.interpolate(column) which calls fast_coset_interpolate(offset, values)
    // fast_coset_interpolate does: intt(values), then scale by offset.inverse()
    std::vector<BFieldElement> interpolant_coeffs = trace_column;
    NTT::inverse(interpolant_coeffs);
    
    // Scale by offset.inverse() (matching Rust's fast_coset_interpolate)
    BFieldElement offset_inv = trace_domain.offset.inverse();
    BFieldElement scale = BFieldElement::one();
    for (size_t i = 0; i < interpolant_coeffs.size(); ++i) {
        interpolant_coeffs[i] = interpolant_coeffs[i] * scale;
        scale = scale * offset_inv;
    }
    
    // Step 2: Create randomizer polynomial
    BPolynomial randomizer_poly(randomizer_coeffs);
    
    // Step 3: Compute zerofier
    BPolynomial zerofier = compute_zerofier(trace_domain);
    
    // Step 4: Multiply zerofier * randomizer
    BPolynomial zerofier_times_randomizer = mul_zerofier_with(trace_domain, randomizer_poly);
    
    // Step 5: Create interpolant polynomial
    BPolynomial interpolant_poly(interpolant_coeffs);
    
    // Step 6: Add: randomized_interpolant = interpolant + zerofier * randomizer
    // Need to align sizes
    size_t max_size = std::max(interpolant_poly.size(), zerofier_times_randomizer.size());
    interpolant_poly.resize(max_size, BFieldElement(0));
    zerofier_times_randomizer.resize(max_size, BFieldElement(0));
    
    BPolynomial randomized_interpolant = interpolant_poly + zerofier_times_randomizer;
    
    // Step 7: Evaluate on target domain with offset
    // Rust's evaluation_domain.evaluate() handles chunking when polynomial degree > domain length
    std::vector<BFieldElement> coeffs = randomized_interpolant.raw_coefficients();
    
    // Match Rust's ArithmeticDomain::evaluate() chunking logic
    if (coeffs.size() <= target_domain.length) {
        // Simple case: polynomial fits in one chunk
        coeffs.resize(target_domain.length, BFieldElement(0));
        return NTT::evaluate_on_coset(coeffs, target_domain.length, target_domain.offset);
    } else {
        // Chunking case: polynomial is larger than domain length
        std::vector<BFieldElement> result(target_domain.length, BFieldElement::zero());
        
        // Process first chunk
        size_t first_chunk_size = std::min(target_domain.length, coeffs.size());
        std::vector<BFieldElement> first_chunk(target_domain.length, BFieldElement::zero());
        for (size_t i = 0; i < first_chunk_size; ++i) {
            first_chunk[i] = coeffs[i];
        }
        result = NTT::evaluate_on_coset(first_chunk, target_domain.length, target_domain.offset);
        
        // Process remaining chunks
        for (size_t chunk_start = target_domain.length; chunk_start < coeffs.size(); chunk_start += target_domain.length) {
            size_t chunk_size = std::min(target_domain.length, coeffs.size() - chunk_start);
            std::vector<BFieldElement> chunk(target_domain.length, BFieldElement::zero());
            for (size_t i = 0; i < chunk_size; ++i) {
                chunk[i] = coeffs[chunk_start + i];
            }
            
            std::vector<BFieldElement> chunk_eval = NTT::evaluate_on_coset(
                chunk, target_domain.length, target_domain.offset);
            
            size_t chunk_index = chunk_start / target_domain.length;
            uint64_t coefficient_index = static_cast<uint64_t>(chunk_index) * static_cast<uint64_t>(target_domain.length);
            BFieldElement scaled_offset = target_domain.offset.pow(coefficient_index);
            
            for (size_t i = 0; i < target_domain.length; ++i) {
                result[i] = result[i] + (chunk_eval[i] * scaled_offset);
            }
        }
        
        return result;
    }
}

bool RandomizedLDE::verify_zerofier(
    const ArithmeticDomain& domain,
    const BPolynomial& zerofier
) {
    // Generate domain values
    std::vector<BFieldElement> domain_values;
    BFieldElement x = domain.offset;
    for (size_t i = 0; i < domain.length; i++) {
        domain_values.push_back(x);
        x = x * domain.generator;
    }
    
    // Verify zerofier evaluates to zero at all domain points
    for (const auto& domain_point : domain_values) {
        BFieldElement eval = zerofier.evaluate(domain_point);
        if (eval.value() != 0) {
            return false;
        }
    }
    
    return true;
}

std::vector<XFieldElement> RandomizedLDE::extend_xfield_column_with_randomizer(
    const std::vector<XFieldElement>& trace_column,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain,
    const std::vector<BFieldElement>& randomizer_coeffs
) {
    // Step 1: Interpolate XFieldElement column as a whole (matching Rust)
    // Component-wise interpolation with coset handling (from quotient.cpp logic)
    const size_t n = trace_column.size();
    std::vector<BFieldElement> component0(n);
    std::vector<BFieldElement> component1(n);
    std::vector<BFieldElement> component2(n);
    for (size_t i = 0; i < n; ++i) {
        component0[i] = trace_column[i].coeff(0);
        component1[i] = trace_column[i].coeff(1);
        component2[i] = trace_column[i].coeff(2);
    }
    
    auto coeff0 = NTT::interpolate(component0);
    auto coeff1 = NTT::interpolate(component1);
    auto coeff2 = NTT::interpolate(component2);
    
    // Apply coset interpolation scaling (matching interpolate_xfield_column)
    // Note: This scaling might be incorrect - Rust's fast_coset_interpolate does this internally
    std::vector<XFieldElement> interpolant_coeffs(n);
    BFieldElement offset_inv = trace_domain.offset.inverse();
    BFieldElement scale = BFieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        interpolant_coeffs[i] = XFieldElement(coeff0[i], coeff1[i], coeff2[i]) * scale;
        scale *= offset_inv;
    }
    
    // Store interpolant for debugging (accessible via static variable or return)
    // For now, we'll compute and compare in the test
    
    // Step 2: Lift BFieldElement randomizer to XFieldElement
    // In Rust: BFieldElement polynomial gets lifted to XFieldElement when multiplied with zerofier
    // Lift each coefficient: b -> XFieldElement(b, 0, 0)
    std::vector<XFieldElement> lifted_randomizer_coeffs;
    lifted_randomizer_coeffs.reserve(randomizer_coeffs.size());
    for (const auto& bfe : randomizer_coeffs) {
        lifted_randomizer_coeffs.push_back(XFieldElement(bfe));
    }
    XPolynomial lifted_randomizer_poly(lifted_randomizer_coeffs);
    
    // Step 3: Compute zerofier * lifted_randomizer
    // Optimized: z(x) * p(x) where z(x) = x^n - offset^n
    // Rust: polynomial.clone().shift_coefficients(self.length)
    //       - polynomial.scalar_mul(self.offset.mod_pow(self.length as u64))
    XPolynomial shifted_randomizer = lifted_randomizer_poly.shift_coefficients(trace_domain.length);
    BFieldElement offset_pow_n = trace_domain.offset.pow(trace_domain.length);
    
    // Scalar multiply: multiply each XFieldElement coefficient by BFieldElement scalar
    // This matches Rust's scalar_mul behavior
    std::vector<XFieldElement> scaled_coeffs;
    scaled_coeffs.reserve(lifted_randomizer_poly.size());
    for (size_t i = 0; i < lifted_randomizer_poly.size(); i++) {
        scaled_coeffs.push_back(lifted_randomizer_poly[i] * offset_pow_n);
    }
    XPolynomial scaled_randomizer(scaled_coeffs);
    
    // Align sizes
    size_t max_size = std::max(shifted_randomizer.size(), scaled_randomizer.size());
    shifted_randomizer.resize(max_size, XFieldElement());
    scaled_randomizer.resize(max_size, XFieldElement());
    
    // Compute: shifted - scaled
    std::vector<XFieldElement> zerofier_times_randomizer_coeffs(max_size);
    for (size_t i = 0; i < max_size; i++) {
        zerofier_times_randomizer_coeffs[i] = shifted_randomizer[i] - scaled_randomizer[i];
    }
    XPolynomial zerofier_times_randomizer(zerofier_times_randomizer_coeffs);
    
    // Step 4: Create interpolant polynomial
    XPolynomial interpolant_poly(interpolant_coeffs);
    
    // Step 5: Add: randomized_interpolant = interpolant + zerofier * randomizer
    size_t max_poly_size = std::max(interpolant_poly.size(), zerofier_times_randomizer.size());
    interpolant_poly.resize(max_poly_size, XFieldElement());
    zerofier_times_randomizer.resize(max_poly_size, XFieldElement());
    
    XPolynomial randomized_interpolant = interpolant_poly + zerofier_times_randomizer;
    
    // Step 6: Evaluate on target domain (parameter named quotient_domain but it's actually the target/evaluation domain)
    std::vector<XFieldElement> coeffs = randomized_interpolant.coefficients();

    // Extend to target domain length
    if (coeffs.size() < quotient_domain.length) {
        coeffs.resize(quotient_domain.length, XFieldElement());
    } else if (coeffs.size() > quotient_domain.length) {
        coeffs.resize(quotient_domain.length);
    }
    
    // Evaluate component-wise (NTT works on BFieldElement)
    std::vector<BFieldElement> eval_coeff0, eval_coeff1, eval_coeff2;
    eval_coeff0.reserve(coeffs.size());
    eval_coeff1.reserve(coeffs.size());
    eval_coeff2.reserve(coeffs.size());
    for (const auto& xfe : coeffs) {
        eval_coeff0.push_back(xfe.coeff(0));
        eval_coeff1.push_back(xfe.coeff(1));
        eval_coeff2.push_back(xfe.coeff(2));
    }

    // Use target_domain (passed as quotient_domain parameter) for evaluation
    std::vector<BFieldElement> eval0 = NTT::evaluate_on_coset(eval_coeff0, quotient_domain.length, quotient_domain.offset);
    std::vector<BFieldElement> eval1 = NTT::evaluate_on_coset(eval_coeff1, quotient_domain.length, quotient_domain.offset);
    std::vector<BFieldElement> eval2 = NTT::evaluate_on_coset(eval_coeff2, quotient_domain.length, quotient_domain.offset);
    
    std::vector<XFieldElement> result(quotient_domain.length);
    for (size_t i = 0; i < quotient_domain.length; i++) {
        result[i] = XFieldElement(eval0[i], eval1[i], eval2[i]);
    }
    
    return result;
}

std::vector<XFieldElement> RandomizedLDE::extend_xfield_column_with_xfield_randomizer(
    const std::vector<XFieldElement>& trace_column,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain,
    const std::vector<XFieldElement>& randomizer_coeffs
) {
    // Same as extend_xfield_column_with_randomizer, but randomizer is already XFieldElement
    // Step 1: Interpolate XFieldElement column
    const size_t n = trace_column.size();
    std::vector<BFieldElement> component0(n);
    std::vector<BFieldElement> component1(n);
    std::vector<BFieldElement> component2(n);
    for (size_t i = 0; i < n; ++i) {
        component0[i] = trace_column[i].coeff(0);
        component1[i] = trace_column[i].coeff(1);
        component2[i] = trace_column[i].coeff(2);
    }
    
    auto coeff0 = NTT::interpolate(component0);
    auto coeff1 = NTT::interpolate(component1);
    auto coeff2 = NTT::interpolate(component2);
    
    // Apply coset interpolation scaling
    std::vector<XFieldElement> interpolant_coeffs(n);
    BFieldElement offset_inv = trace_domain.offset.inverse();
    BFieldElement scale = BFieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        interpolant_coeffs[i] = XFieldElement(coeff0[i], coeff1[i], coeff2[i]) * scale;
        scale *= offset_inv;
    }
    
    // Step 2: Use XFieldElement randomizer directly (no lifting needed)
    XPolynomial randomizer_poly(randomizer_coeffs);
    
    // Step 3: Compute zerofier * randomizer
    XPolynomial shifted_randomizer = randomizer_poly.shift_coefficients(trace_domain.length);
    BFieldElement offset_pow_n = trace_domain.offset.pow(trace_domain.length);
    
    // Scalar multiply: multiply each XFieldElement coefficient by BFieldElement scalar
    std::vector<XFieldElement> scaled_coeffs;
    scaled_coeffs.reserve(randomizer_poly.size());
    for (size_t i = 0; i < randomizer_poly.size(); i++) {
        scaled_coeffs.push_back(randomizer_poly[i] * offset_pow_n);
    }
    XPolynomial scaled_randomizer(scaled_coeffs);
    
    // Align sizes
    size_t max_size = std::max(shifted_randomizer.size(), scaled_randomizer.size());
    shifted_randomizer.resize(max_size, XFieldElement());
    scaled_randomizer.resize(max_size, XFieldElement());
    
    // Compute: shifted - scaled
    std::vector<XFieldElement> zerofier_times_randomizer_coeffs(max_size);
    for (size_t i = 0; i < max_size; i++) {
        zerofier_times_randomizer_coeffs[i] = shifted_randomizer[i] - scaled_randomizer[i];
    }
    XPolynomial zerofier_times_randomizer(zerofier_times_randomizer_coeffs);
    
    // Step 4: Create interpolant polynomial
    XPolynomial interpolant_poly(interpolant_coeffs);
    
    // Step 5: Add: randomized_interpolant = interpolant + zerofier * randomizer
    size_t max_poly_size = std::max(interpolant_poly.size(), zerofier_times_randomizer.size());
    interpolant_poly.resize(max_poly_size, XFieldElement());
    zerofier_times_randomizer.resize(max_poly_size, XFieldElement());
    
    XPolynomial randomized_interpolant = interpolant_poly + zerofier_times_randomizer;
    
    // Step 6: Evaluate on quotient domain
    std::vector<XFieldElement> coeffs = randomized_interpolant.coefficients();
    
    if (coeffs.size() < quotient_domain.length) {
        coeffs.resize(quotient_domain.length, XFieldElement());
    } else if (coeffs.size() > quotient_domain.length) {
        coeffs.resize(quotient_domain.length);
    }
    
    // Evaluate component-wise
    std::vector<BFieldElement> eval_coeff0, eval_coeff1, eval_coeff2;
    eval_coeff0.reserve(coeffs.size());
    eval_coeff1.reserve(coeffs.size());
    eval_coeff2.reserve(coeffs.size());
    for (const auto& xfe : coeffs) {
        eval_coeff0.push_back(xfe.coeff(0));
        eval_coeff1.push_back(xfe.coeff(1));
        eval_coeff2.push_back(xfe.coeff(2));
    }

    // Use target_domain (passed as quotient_domain parameter) for evaluation
    std::vector<BFieldElement> eval0 = NTT::evaluate_on_coset(eval_coeff0, quotient_domain.length, quotient_domain.offset);
    std::vector<BFieldElement> eval1 = NTT::evaluate_on_coset(eval_coeff1, quotient_domain.length, quotient_domain.offset);
    std::vector<BFieldElement> eval2 = NTT::evaluate_on_coset(eval_coeff2, quotient_domain.length, quotient_domain.offset);
    
    std::vector<XFieldElement> result(quotient_domain.length);
    for (size_t i = 0; i < quotient_domain.length; i++) {
        result[i] = XFieldElement(eval0[i], eval1[i], eval2[i]);
    }
    
    return result;
}

} // namespace triton_vm

