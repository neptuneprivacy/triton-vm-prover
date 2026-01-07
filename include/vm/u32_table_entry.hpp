#pragma once

#include "types/b_field_element.hpp"
#include <cstdint>
#include <cstddef>

namespace triton_vm {

/**
 * U32TableEntry - Key for the U32 table multiset.
 *
 * Mirrors Rust `U32TableEntry { instruction, left_operand, right_operand }`.
 * The operands are BFieldElements but are intended to represent u32 values.
 */
struct U32TableEntry {
    uint32_t instruction_opcode;
    BFieldElement left_operand;
    BFieldElement right_operand;

    bool operator==(const U32TableEntry& other) const {
        return instruction_opcode == other.instruction_opcode &&
               left_operand.value() == other.left_operand.value() &&
               right_operand.value() == other.right_operand.value();
    }

    bool operator<(const U32TableEntry& other) const {
        if (instruction_opcode != other.instruction_opcode) return instruction_opcode < other.instruction_opcode;
        if (left_operand.value() != other.left_operand.value()) return left_operand.value() < other.left_operand.value();
        return right_operand.value() < other.right_operand.value();
    }
};

struct U32TableEntryHash {
    size_t operator()(const U32TableEntry& e) const noexcept {
        // Simple hash combine
        size_t h = static_cast<size_t>(e.instruction_opcode);
        auto mix = [&](uint64_t x) {
            size_t k = static_cast<size_t>(x ^ (x >> 33));
            h ^= k + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(e.left_operand.value());
        mix(e.right_operand.value());
        return h;
    }
};

} // namespace triton_vm


