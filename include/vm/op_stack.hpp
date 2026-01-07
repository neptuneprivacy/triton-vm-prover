#pragma once

#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include "vm/underflow_io.hpp"
#include <vector>
#include <stdexcept>

namespace triton_vm {

/**
 * OpStack - Operational stack for VM execution
 * 
 * Implements a stack of BFieldElements with operations like push, pop, dup, swap, etc.
 */
class OpStack {
public:
    static constexpr size_t NUM_REGISTERS = 16;

    OpStack() : OpStack(Digest::zero()) {}

    explicit OpStack(const Digest& program_digest) {
        // Rust: stack has at least 16 elements; bottom-most Digest::LEN elements equal program digest (reversed).
        stack_.assign(NUM_REGISTERS, BFieldElement::zero());
        for (size_t i = 0; i < Digest::LEN; ++i) {
            stack_[i] = program_digest[Digest::LEN - 1 - i];
        }
    }

    /**
     * Push a value onto the stack
     */
    void push(BFieldElement value);
    
    /**
     * Pop n words from the stack
     */
    std::vector<BFieldElement> pop(size_t n);
    
    /**
     * Pop a single word
     */
    BFieldElement pop();
    
    /**
     * Peek at the top element without popping
     */
    BFieldElement peek() const;
    
    /**
     * Peek at element at depth (0 = top)
     */
    BFieldElement peek_at(size_t depth) const;
    
    /**
     * Duplicate element at depth
     */
    void dup(size_t depth);
    
    /**
     * Swap top element with element at depth
     */
    void swap(size_t depth);
    
    /**
     * Pick element at depth to top
     */
    void pick(size_t depth);
    
    /**
     * Place top element at depth
     */
    void place(size_t depth);
    
    /**
     * Get current stack size
     */
    size_t size() const { return stack_.size(); }
    
    /**
     * Check if stack is empty
     */
    bool empty() const { return stack_.empty(); }
    
    /**
     * Get all stack elements (for debugging/tracing)
     */
    const std::vector<BFieldElement>& elements() const { return stack_; }

    // Rust-compatible helpers
    BFieldElement pointer() const { return BFieldElement(static_cast<uint64_t>(stack_.size())); }

    BFieldElement first_underflow_element() const {
        // Matches Rust (`triton-isa::op_stack::OpStack::first_underflow_element`):
        // Return the element just below the register window, or 0 if underflow memory is empty.
        if (stack_.size() <= NUM_REGISTERS) return BFieldElement::zero();
        const size_t top_of_stack_index = stack_.size() - 1;
        const size_t underflow_start = top_of_stack_index - NUM_REGISTERS; // == len - 17
        return stack_[underflow_start];
    }

    void start_recording_underflow_io_sequence() { underflow_io_sequence_.clear(); }
    std::vector<UnderflowIO> stop_recording_underflow_io_sequence() {
        std::vector<UnderflowIO> out = std::move(underflow_io_sequence_);
        underflow_io_sequence_.clear();
        return out;
    }
    
    /**
     * Clear the stack
     */
    void clear() { stack_.clear(); }
    
private:
    std::vector<BFieldElement> stack_;
    std::vector<UnderflowIO> underflow_io_sequence_;
    
    void check_size(size_t required) const {
        if (stack_.size() < required) {
            throw std::runtime_error("OpStack underflow: required " + 
                                    std::to_string(required) + 
                                    ", have " + std::to_string(stack_.size()));
        }
    }

    void check_can_pop(size_t n) const {
        check_size(n);
        if (stack_.size() - n < NUM_REGISTERS) {
            throw std::runtime_error("OpStack underflow: would go below 16 registers");
        }
    }

    void record_underflow_io(const UnderflowIO& io) { underflow_io_sequence_.push_back(io); }
};

} // namespace triton_vm

