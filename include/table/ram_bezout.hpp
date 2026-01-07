#pragma once

#include "types/b_field_element.hpp"
#include <utility>
#include <vector>

namespace triton_vm {

/**
 * Compute RAM Bézout coefficient polynomials (a,b) for the RAM table.
 *
 * This matches the Rust logic used by Triton VM:
 * - rp(x) = Π (x - r) for r in unique_ramps
 * - fd(x) = rp'(x)
 * - b(x) interpolates (r_i, 1/fd(r_i))
 * - a(x) = (1 - fd*b) / rp   (exact division)
 *
 * NOTE: This implementation is performance-oriented (OpenMP + optional GPU poly mul).
 */
std::pair<std::vector<BFieldElement>, std::vector<BFieldElement>>
compute_ram_bezout_coefficients(const std::vector<BFieldElement>& unique_ramps);

} // namespace triton_vm


