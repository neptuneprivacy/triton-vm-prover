#pragma once

#include "types/b_field_element.hpp"
#include <cstdint>

namespace triton_vm {

/**
 * RamTableCall - Mirrors Rust `RamTableCall { clk, ram_pointer, ram_value, is_write }`.
 *
 * This is the raw memory access event recorded during VM execution.
 */
struct RamTableCall {
    uint32_t clk;
    BFieldElement ram_pointer;
    BFieldElement ram_value;
    bool is_write;
};

} // namespace triton_vm


