#pragma once

#include "types/b_field_element.hpp"
#include "vm/u32_table_entry.hpp"
#include "vm/ram_table_call.hpp"
#include "vm/underflow_io.hpp"
#include <vector>
#include <array>
#include <memory>

namespace triton_vm {

// Forward declaration
class Tip5;

/**
 * CoProcessorCall - Represents a call to a co-processor during execution
 * 
 * Matches Rust's CoProcessorCall enum variants.
 */
struct CoProcessorCall {
    enum class Type {
        SpongeStateReset,
        Tip5Trace,
        U32,
        OpStack,
        Ram
    };
    
    Type type;
    
    // For Tip5Trace: instruction opcode and trace
    uint32_t instruction_opcode;
    std::shared_ptr<std::vector<std::array<BFieldElement, 16>>> tip5_trace; // PermutationTrace

    // For U32: entry payload
    std::shared_ptr<U32TableEntry> u32_entry;

    // For RAM: call payload
    std::shared_ptr<RamTableCall> ram_call;

    // For OpStack: one OpStackTableEntry encoded directly
    struct OpStackTableEntryPayload {
        uint32_t clk;
        BFieldElement op_stack_pointer;
        UnderflowIO underflow_io;
    };
    std::shared_ptr<OpStackTableEntryPayload> op_stack_entry;
    
    // Default constructor
    CoProcessorCall()
        : type(Type::SpongeStateReset)
        , instruction_opcode(0)
        , tip5_trace(nullptr)
        , u32_entry(nullptr)
        , ram_call(nullptr) {}
    
    // Constructor for SpongeStateReset
    static CoProcessorCall sponge_state_reset() {
        CoProcessorCall call;
        call.type = Type::SpongeStateReset;
        return call;
    }
    
    // Constructor for Tip5Trace
    static CoProcessorCall make_tip5_trace(uint32_t opcode, std::vector<std::array<BFieldElement, 16>> trace) {
        CoProcessorCall call;
        call.type = Type::Tip5Trace;
        call.instruction_opcode = opcode;
        call.tip5_trace = std::make_shared<std::vector<std::array<BFieldElement, 16>>>(std::move(trace));
        return call;
    }

    // Constructor for U32
    static CoProcessorCall make_u32(U32TableEntry entry) {
        CoProcessorCall call;
        call.type = Type::U32;
        call.u32_entry = std::make_shared<U32TableEntry>(std::move(entry));
        return call;
    }

    // Constructor for RAM
    static CoProcessorCall make_ram(RamTableCall call_payload) {
        CoProcessorCall call;
        call.type = Type::Ram;
        call.ram_call = std::make_shared<RamTableCall>(std::move(call_payload));
        return call;
    }

    static CoProcessorCall make_op_stack(uint32_t clk, BFieldElement sp, UnderflowIO io) {
        CoProcessorCall call;
        call.type = Type::OpStack;
        call.op_stack_entry = std::make_shared<OpStackTableEntryPayload>(OpStackTableEntryPayload{clk, sp, io});
        return call;
    }
};

} // namespace triton_vm

