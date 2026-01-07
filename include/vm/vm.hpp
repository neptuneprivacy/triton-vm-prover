#pragma once

#include "vm/vm_state.hpp"
#include "vm/aet.hpp"
#include "vm/program.hpp"
#include <vector>

namespace triton_vm {

// Forward declarations
class Program;

/**
 * VM - Triton Virtual Machine
 * 
 * Executes TASM programs and generates execution traces.
 */
class VM {
public:
    /**
     * Result of trace execution
     */
    struct TraceResult {
        AlgebraicExecutionTrace aet;
        std::vector<BFieldElement> output;
        
        // TraceResult cannot be default-constructed because AET requires a Program
        // Use the factory method trace_execution() instead
        TraceResult() = delete;
        TraceResult(AlgebraicExecutionTrace a, std::vector<BFieldElement> o)
            : aet(std::move(a)), output(std::move(o)) {}
    };
    
    /**
     * Trace the execution of a program
     * 
     * Executes the program and records all intermediate states needed for proof generation.
     * 
     * @param program The TASM program to execute
     * @param public_input Public input values
     * @param secret_input Secret input values (optional, for non-deterministic execution)
     * @return TraceResult containing the AET and program output
     */
    static TraceResult trace_execution(
        const Program& program,
        const std::vector<BFieldElement>& public_input,
        const std::vector<BFieldElement>& secret_input = {}
    );
    
    /**
     * Trace execution from an existing VMState
     * 
     * Useful for resuming execution or starting from a specific state.
     */
    static TraceResult trace_execution_of_state(VMState state);
    
    /**
     * Create TraceResult from Rust FFI data (faster trace execution via Rust)
     * 
     * @param program_bwords Program bwords from Rust
     * @param processor_trace_data Flat processor trace data (row-major)
     * @param processor_trace_rows Number of rows
     * @param processor_trace_cols Number of columns (should be 39)
     * @param instruction_multiplicities Instruction multiplicities from Rust
     * @param instruction_multiplicities_len Length of multiplicities array
     * @param output Public output from Rust
     * @param table_lengths_9 Optional: table lengths from Rust [program, processor, op_stack, ram, jump_stack, hash, cascade, lookup, u32]
     *                        If provided, used to override height calculations since co-processor traces are not loaded
     * @return TraceResult containing the AET and program output
     */
    static TraceResult trace_execution_from_rust_ffi(
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
        const size_t* table_lengths_9 = nullptr
    );
    
    /**
     * Run a program without tracing (faster, but no proof generation)
     * 
     * @param program The TASM program to execute
     * @param public_input Public input values
     * @param secret_input Secret input values (optional)
     * @return Program output
     */
    static std::vector<BFieldElement> run(
        const Program& program,
        const std::vector<BFieldElement>& public_input,
        const std::vector<BFieldElement>& secret_input = {}
    );
};

} // namespace triton_vm

