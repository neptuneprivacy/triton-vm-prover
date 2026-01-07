#pragma once

#include "types/b_field_element.hpp"
#include <vector>
#include <array>

namespace triton_vm {

/**
 * Pad all tables in the main table according to Rust implementation
 * 
 * @param main_table The main table to pad (will be modified)
 * @param table_lengths Array of 9 table lengths: [Program, Processor, OpStack, Ram, JumpStack, Hash, Cascade, Lookup, U32]
 * @param padded_height Target padded height (must be power of 2)
 */
void pad_all_tables(
    std::vector<std::vector<BFieldElement>>& main_table,
    const std::array<size_t, 9>& table_lengths,
    size_t padded_height
);

} // namespace triton_vm

