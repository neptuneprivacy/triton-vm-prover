#pragma once

#include "types/b_field_element.hpp"
#include <vector>

namespace triton_vm {

/**
 * UnderflowIO - Mirrors Rust `isa::op_stack::UnderflowIO`.
 * It records a read or write to the op-stack underflow memory and carries the
 * payload `first_underflow_element` at the time of the IO.
 */
struct UnderflowIO {
    enum class Kind { Read, Write };
    Kind kind;
    BFieldElement payload;

    bool shrinks_stack() const { return kind == Kind::Read; }
    bool grows_stack() const { return kind == Kind::Write; }

    static UnderflowIO read(BFieldElement p) { return UnderflowIO{Kind::Read, p}; }
    static UnderflowIO write(BFieldElement p) { return UnderflowIO{Kind::Write, p}; }

    bool is_dual_to(const UnderflowIO& other) const {
        return payload == other.payload &&
               ((kind == Kind::Read && other.kind == Kind::Write) ||
                (kind == Kind::Write && other.kind == Kind::Read));
    }

    static void canonicalize_sequence(std::vector<UnderflowIO>& seq) {
        // Remove adjacent dual pairs until none remain (matches Rust).
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                if (seq[i].is_dual_to(seq[i + 1])) {
                    seq.erase(seq.begin() + static_cast<long>(i), seq.begin() + static_cast<long>(i + 2));
                    changed = true;
                    break;
                }
            }
        }
    }

    static bool is_uniform_sequence(const std::vector<UnderflowIO>& seq) {
        if (seq.empty()) return true;
        Kind k = seq.front().kind;
        for (const auto& io : seq) if (io.kind != k) return false;
        return true;
    }
};

} // namespace triton_vm


