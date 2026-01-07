#include "vm/op_stack.hpp"
#include <algorithm>

namespace triton_vm {

void OpStack::push(BFieldElement value) {
    stack_.push_back(value);
    // Rust: record Write(payload) after push
    record_underflow_io(UnderflowIO::write(first_underflow_element()));
}

std::vector<BFieldElement> OpStack::pop(size_t n) {
    check_can_pop(n);
    
    std::vector<BFieldElement> result;
    result.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        // Rust: record Read(payload) before pop
        record_underflow_io(UnderflowIO::read(first_underflow_element()));
        result.push_back(stack_.back());
        stack_.pop_back();
    }
    
    // Reverse to maintain order (top element first)
    std::reverse(result.begin(), result.end());
    return result;
}

BFieldElement OpStack::pop() {
    check_can_pop(1);
    // Rust: record Read(payload) before pop
    record_underflow_io(UnderflowIO::read(first_underflow_element()));
    BFieldElement value = stack_.back();
    stack_.pop_back();
    return value;
}

BFieldElement OpStack::peek() const {
    check_size(1);
    return stack_.back();
}

BFieldElement OpStack::peek_at(size_t depth) const {
    check_size(depth + 1);
    return stack_[stack_.size() - 1 - depth];
}

void OpStack::dup(size_t depth) {
    check_size(depth + 1);
    BFieldElement value = stack_[stack_.size() - 1 - depth];
    // Rust `OpStack::dup` increases stack length, so it records a Write underflow IO.
    push(value);
}

void OpStack::swap(size_t depth) {
    if (depth == 0) return; // No-op
    check_size(depth + 1);
    
    size_t top_idx = stack_.size() - 1;
    size_t target_idx = stack_.size() - 1 - depth;
    
    std::swap(stack_[top_idx], stack_[target_idx]);
}

void OpStack::pick(size_t depth) {
    check_size(depth + 1);
    BFieldElement value = stack_[stack_.size() - 1 - depth];
    // Picking duplicates an element onto the top (length increases): record Write.
    push(value);
}

void OpStack::place(size_t depth) {
    if (depth == 0) return; // No-op
    check_size(depth + 1);
    
    // Removing top element shrinks stack length: record Read (matches Rust).
    BFieldElement value = pop();
    
    size_t target_idx = stack_.size() - depth;
    stack_[target_idx] = value;
}

} // namespace triton_vm

