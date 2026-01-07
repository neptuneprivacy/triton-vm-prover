#include "vm/vm.hpp"
#include "vm/vm_state.hpp"
#include "vm/aet.hpp"
#include "vm/program.hpp"
#include "table/extend_helpers.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>

namespace triton_vm {

VM::TraceResult VM::trace_execution(
    const Program& program,
    const std::vector<BFieldElement>& public_input,
    const std::vector<BFieldElement>& secret_input
) {
    VMState state(program, public_input, secret_input);
    return trace_execution_of_state(std::move(state));
}

VM::TraceResult VM::trace_execution_of_state(VMState state) {
    // Get program from state
    const Program& program = state.program();
    
    // AET needs the program bwords (opcode + optional arg) for program table + program hashing trace.
    AlgebraicExecutionTrace aet(program.to_bwords());
    
    // OPTIMIZATION: Cache debug flag once outside the loop
    static const bool debug_trace = (std::getenv("TVM_DEBUG_TRACE") != nullptr);
    
    // OPTIMIZATION: Pre-reserve trace capacity based on program size estimate
    // Typical programs run ~100 cycles per bword on average
    const size_t estimated_rows = program.len_bwords() * 100;
    aet.reserve_processor_trace(estimated_rows);
    aet.reserve_tables(estimated_rows); // Reserve space for all sub-tables
    
    if (debug_trace) {
        std::cout << "DEBUG trace_execution: Starting trace, initial IP=" << state.instruction_pointer() 
                  << ", halting=" << state.halting() << ", input_size=" << state.public_input_size() << std::endl;
    }
    
    // OPTIMIZED HOT LOOP - minimal overhead per iteration
    // Pre-allocate vector for reuse (avoids allocation each iteration)
    std::vector<CoProcessorCall> co_processor_calls;
    co_processor_calls.reserve(16); // Most instructions produce 0-4 calls
    
    // OPTIMIZATION: Cache instruction lookup once per iteration (avoid redundant lookups)
    auto instr_opt = state.current_instruction();
    
    while (!state.halting()) {
        // Record current state (BEFORE instruction executes) with cached instruction
        if (instr_opt.has_value()) {
            aet.record_state_cached(state, instr_opt.value());
        } else {
            aet.record_state(state);
        }
        
        // Execute one step - reuse vector to avoid allocation
        co_processor_calls.clear();
        if (instr_opt.has_value()) {
            // Use cached instruction to avoid redundant lookup in step_into
            state.step_into_cached(instr_opt.value(), co_processor_calls);
        } else {
            state.step_into(co_processor_calls);
        }
        
        // Record co-processor calls
        for (const auto& call : co_processor_calls) {
            aet.record_co_processor_call(call);
        }
        
        // Cache next instruction for next iteration (if not halting)
        if (!state.halting()) {
            instr_opt = state.current_instruction();
        }
    }
    
    if (debug_trace) {
        std::cout << "DEBUG trace_execution: Finished, total trace_rows=" << aet.processor_trace_height() 
                  << ", halting=" << state.halting() << std::endl;
    }
    
    return TraceResult(std::move(aet), state.public_output());
}

VM::TraceResult VM::trace_execution_from_rust_ffi(
    const std::vector<BFieldElement>& program_bwords,
    const uint64_t* processor_trace_data,
    size_t processor_trace_rows,
    size_t processor_trace_cols,
    const uint32_t* instruction_multiplicities,
    size_t instruction_multiplicities_len,
    const std::vector<BFieldElement>& output,
    const uint64_t* op_stack_trace_data,
    size_t op_stack_trace_rows,
    size_t op_stack_trace_cols,
    const uint64_t* ram_trace_data,
    size_t ram_trace_rows,
    size_t ram_trace_cols,
    const uint64_t* program_hash_trace_data,
    size_t program_hash_trace_rows,
    size_t program_hash_trace_cols,
    const uint64_t* hash_trace_data,
    size_t hash_trace_rows,
    size_t hash_trace_cols,
    const uint64_t* sponge_trace_data,
    size_t sponge_trace_rows,
    size_t sponge_trace_cols,
    const uint64_t* u32_entries_data,
    size_t u32_entries_len,
    const uint64_t* cascade_multiplicities_data,
    size_t cascade_multiplicities_len,
    const uint64_t* lookup_multiplicities_256,
    const size_t* table_lengths_9
) {
    // Create AET with program bwords
    AlgebraicExecutionTrace aet(program_bwords);
    
    // Copy processor trace from flat data (row-major)
    if (processor_trace_cols != AlgebraicExecutionTrace::PROCESSOR_WIDTH) {
        throw std::runtime_error("Processor trace width mismatch: expected " + 
                                std::to_string(AlgebraicExecutionTrace::PROCESSOR_WIDTH) + 
                                ", got " + std::to_string(processor_trace_cols));
    }
    
    // Load AET data from Rust FFI (now includes all traces)
    aet.load_from_rust_ffi(
        processor_trace_data,
        processor_trace_rows,
        instruction_multiplicities,
        instruction_multiplicities_len,
        op_stack_trace_data,
        op_stack_trace_rows,
        op_stack_trace_cols,
        ram_trace_data,
        ram_trace_rows,
        ram_trace_cols,
        program_hash_trace_data,
        program_hash_trace_rows,
        program_hash_trace_cols,
        hash_trace_data,
        hash_trace_rows,
        hash_trace_cols,
        sponge_trace_data,
        sponge_trace_rows,
        sponge_trace_cols,
        u32_entries_data,
        u32_entries_len,
        cascade_multiplicities_data,
        cascade_multiplicities_len,
        lookup_multiplicities_256,
        table_lengths_9
    );
    
    return TraceResult(std::move(aet), output);
}

std::vector<BFieldElement> VM::run(
    const Program& program,
    const std::vector<BFieldElement>& public_input,
    const std::vector<BFieldElement>& secret_input
) {
    VMState state(program, public_input, secret_input);
    state.run();
    return state.public_output();
}

} // namespace triton_vm

