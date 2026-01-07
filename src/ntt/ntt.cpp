#include "ntt/ntt.hpp"
#include <cmath>
#include <stdexcept>
#include <omp.h>

namespace triton_vm {

void NTT::bit_reverse_permutation(std::vector<BFieldElement>& data) {
    size_t n = data.size();
    size_t log_n = 0;
    while ((1ULL << log_n) < n) log_n++;
    
    for (size_t i = 0; i < n; i++) {
        size_t rev = 0;
        for (size_t j = 0; j < log_n; j++) {
            if (i & (1ULL << j)) {
                rev |= (1ULL << (log_n - 1 - j));
            }
        }
        if (i < rev) {
            std::swap(data[i], data[rev]);
        }
    }
}

void NTT::ntt_core(std::vector<BFieldElement>& data, BFieldElement omega, bool inverse) {
    size_t n = data.size();
    
    // Bit-reverse permutation
    bit_reverse_permutation(data);
    
    // Iterative Cooley-Tukey NTT
    for (size_t len = 2; len <= n; len *= 2) {
        // omega_len = omega^(n/len) - the len-th root of unity
        size_t step = n / len;
        BFieldElement omega_len = omega.pow(step);
        
        for (size_t i = 0; i < n; i += len) {
            BFieldElement w = BFieldElement::one();
            for (size_t j = 0; j < len / 2; j++) {
                BFieldElement u = data[i + j];
                BFieldElement v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w = w * omega_len;
            }
        }
    }
    
    // For inverse NTT, divide by n
    if (inverse) {
        BFieldElement n_inv = BFieldElement(n).inverse();
        for (size_t i = 0; i < n; i++) {
            data[i] = data[i] * n_inv;
        }
    }
}

void NTT::forward(std::vector<BFieldElement>& coeffs) {
    size_t n = coeffs.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("NTT size must be a power of 2");
    }
    
    // Get primitive n-th root of unity
    size_t log_n = 0;
    while ((1ULL << log_n) < n) log_n++;
    BFieldElement omega = BFieldElement::primitive_root_of_unity(log_n);
    
    ntt_core(coeffs, omega, false);
}

void NTT::inverse(std::vector<BFieldElement>& evals) {
    size_t n = evals.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("NTT size must be a power of 2");
    }
    
    // Get primitive n-th root of unity and its inverse
    size_t log_n = 0;
    while ((1ULL << log_n) < n) log_n++;
    BFieldElement omega = BFieldElement::primitive_root_of_unity(log_n);
    BFieldElement omega_inv = omega.inverse();
    
    ntt_core(evals, omega_inv, true);
}

std::vector<BFieldElement> NTT::interpolate(const std::vector<BFieldElement>& column) {
    std::vector<BFieldElement> coeffs = column;
    inverse(coeffs);
    return coeffs;
}

std::vector<BFieldElement> NTT::evaluate_on_coset(
    const std::vector<BFieldElement>& coeffs,
    size_t domain_length,
    BFieldElement offset
) {
    // Extend coefficients with zeros to domain_length
    std::vector<BFieldElement> extended(domain_length, BFieldElement::zero());
    for (size_t i = 0; i < coeffs.size() && i < domain_length; i++) {
        extended[i] = coeffs[i];
    }
    
    // Scale coefficients by offset powers: c_i -> c_i * offset^i
    // This shifts the evaluation from {ω^j} to {offset * ω^j}
    BFieldElement scale = BFieldElement::one();
    for (size_t i = 0; i < domain_length; i++) {
        extended[i] = extended[i] * scale;
        scale = scale * offset;
    }
    
    // Forward NTT to get evaluations
    forward(extended);
    
    return extended;
}

// LDE Implementation

std::vector<BFieldElement> LDE::extend_column(
    const std::vector<BFieldElement>& trace_column,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain
) {
    // Step 1: Interpolate - get polynomial coefficients from trace evaluations
    std::vector<BFieldElement> coeffs = NTT::interpolate(trace_column);
    
    // Step 2: Evaluate on quotient domain (with offset)
    return NTT::evaluate_on_coset(coeffs, quotient_domain.length, quotient_domain.offset);
}

std::vector<std::vector<BFieldElement>> LDE::extend_table(
    const std::vector<std::vector<BFieldElement>>& trace_table,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain
) {
    if (trace_table.empty()) {
        return {};
    }
    
    size_t num_rows = trace_table.size();
    size_t num_cols = trace_table[0].size();
    size_t output_rows = quotient_domain.length;
    
    // Initialize output table
    std::vector<std::vector<BFieldElement>> lde_table(output_rows);
    for (size_t r = 0; r < output_rows; r++) {
        lde_table[r].resize(num_cols);
    }
    
    // Process each column
    for (size_t c = 0; c < num_cols; c++) {
        // Extract column from trace table
        std::vector<BFieldElement> trace_column(num_rows);
        for (size_t r = 0; r < num_rows; r++) {
            trace_column[r] = trace_table[r][c];
        }
        
        // Extend column
        std::vector<BFieldElement> lde_column = extend_column(
            trace_column, trace_domain, quotient_domain
        );
        
        // Copy to output table
        for (size_t r = 0; r < output_rows; r++) {
            lde_table[r][c] = lde_column[r];
        }
    }
    
    return lde_table;
}

} // namespace triton_vm
