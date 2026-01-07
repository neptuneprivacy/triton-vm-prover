#pragma once

#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include "hash/tip5.hpp"
#include "vm/op_stack.hpp"
#include "vm/program.hpp"
#include "vm/processor_columns.hpp"
#include "vm/co_processor_call.hpp"
#include "table/extend_helpers.hpp"
#include <vector>
#include <deque>
#include <map>
#include <optional>
#include <memory>
#include <array>
#include <cstdint>

namespace triton_vm {

// Forward declarations
class Program;
struct CoProcessorCall;

/**
 * VMState - Tracks the execution state of the VM
 * 
 * Contains all registers, memory, stacks, and state needed for execution.
 */
class VMState {
public:
    /**
     * Create a new VMState for the given program
     */
    VMState(
        const Program& program,
        const std::vector<BFieldElement>& public_input,
        const std::vector<BFieldElement>& secret_input = {}
    );
    
    /**
     * Execute one step (instruction)
     * Returns co-processor calls made during this step
     */
    std::vector<CoProcessorCall> step();
    
    /**
     * Execute one step, appending co-processor calls to existing vector
     * (Optimization: avoids allocation per step)
     */
    void step_into(std::vector<CoProcessorCall>& out_calls);
    
    /**
     * Run until halt
     */
    void run();
    
    // State accessors
    const Program& program() const { return program_; }
    size_t instruction_pointer() const { return instruction_pointer_; }
    uint32_t cycle_count() const { return cycle_count_; }
    bool halting() const { return halting_; }
    const std::vector<BFieldElement>& public_output() const { return public_output_; }
    
    // Debug accessor for public input size
    size_t public_input_size() const { return public_input_.size(); }
    
    // Memory accessors
    OpStack& op_stack() { return *op_stack_; }
    const OpStack& op_stack() const { return *op_stack_; }
    const std::vector<std::pair<BFieldElement, BFieldElement>>& jump_stack() const { return jump_stack_; }
    const std::map<BFieldElement, BFieldElement>& ram() const { return ram_; }
    
    // Sponge state
    const std::optional<Tip5>& sponge() const { return sponge_; }
    
    /**
     * Extract processor state as a row (39 columns)
     * This is the main method for creating processor trace rows
     */
    std::vector<BFieldElement> to_processor_row() const;
    
    /**
     * Fill a pre-allocated processor row (optimized - no allocation)
     */
    void fill_processor_row(std::vector<BFieldElement>& row) const;
    
    /**
     * Fill a FLAT processor row (zero-copy, direct write to contiguous memory)
     * row_ptr must point to at least PROCESSOR_WIDTH BFieldElements
     */
    void fill_processor_row_flat(BFieldElement* row_ptr) const;
    
    /**
     * Fill a FLAT processor row with pre-cached instruction (fastest path)
     * Avoids redundant current_instruction() lookup
     */
    void fill_processor_row_flat_cached(BFieldElement* row_ptr, const TritonInstruction& instr) const;
    
    /**
     * Derive helper variables with cached instruction (fast path)
     */
    std::array<BFieldElement, 6> derive_helper_variables_fast(const TritonInstruction& instr) const;
    
    /**
     * Step with pre-cached instruction (avoids redundant current_instruction() call)
     */
    void step_into_cached(const TritonInstruction& instr, std::vector<CoProcessorCall>& out_calls);
    
    /**
     * Get current instruction
     */
    std::optional<TritonInstruction> current_instruction() const;
    
private:
    const Program& program_;
    
    /// Public input (read via read_io)
    std::deque<BFieldElement> public_input_;
    
    /// Public output (written via write_io)
    std::vector<BFieldElement> public_output_;
    
    /// Secret input (read via divine)
    std::deque<BFieldElement> secret_individual_tokens_;
    
    /// Secret digests (for merkle_step)
    std::deque<Digest> secret_digests_;
    
    /// Random-access memory
    std::map<BFieldElement, BFieldElement> ram_;
    
    /// RAM calls (for trace)
    /// TODO: Will be properly implemented when RAM table is added
    /// For now, using void* to avoid incomplete type issues
    // std::vector<RamTableCall> ram_calls_;  // Commented out until RamTableCall is fully defined
    
    /// Op stack
    std::unique_ptr<OpStack> op_stack_;
    
    /// Jump stack: (return address, stack height)
    /// Each entry is (origin, destination) where:
    /// - origin = return address
    /// - destination = stack height at call time
    std::vector<std::pair<BFieldElement, BFieldElement>> jump_stack_;
    
    /// Cycle count
    uint32_t cycle_count_ = 0;
    
    /// Instruction pointer
    size_t instruction_pointer_ = 0;
    
    /// Sponge state (optional, created on sponge_init)
    std::optional<Tip5> sponge_;
    
    /// Halt flag
    bool halting_ = false;
    
    /**
     * Execute a single instruction
     * Returns empty vector for now - co-processor calls will be implemented later
     */
    std::vector<CoProcessorCall> execute_instruction(const TritonInstruction& instr);
    
    /**
     * Execute a single instruction, appending co-processor calls to existing vector
     * (Optimization: zero allocation)
     */
    void execute_instruction_into(const TritonInstruction& instr, std::vector<CoProcessorCall>& out_calls);
    
    /**
     * Get next instruction or argument (NIA)
     * - Argument of current instruction if it has one, or
     * - Opcode of next instruction otherwise
     */
    BFieldElement next_instruction_or_argument() const;
    
    /**
     * Get next instruction (skipping arguments)
     */
    std::optional<TritonInstruction> next_instruction() const;
    
    /**
     * Derive helper variables (6 values) based on current instruction
     */
    std::array<BFieldElement, 6> derive_helper_variables() const;
    
    /**
     * Jump stack pointer (length of jump stack)
     */
    BFieldElement jump_stack_pointer() const;
    
    /**
     * Jump stack origin (return address from top of jump stack)
     */
    BFieldElement jump_stack_origin() const;
    
    /**
     * Jump stack destination (stack height from top of jump stack)
     */
    BFieldElement jump_stack_destination() const;
};

} // namespace triton_vm

