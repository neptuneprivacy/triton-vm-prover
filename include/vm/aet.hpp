#pragma once

#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include "vm/co_processor_call.hpp"
#include "vm/u32_table_entry.hpp"
#include "table/extend_helpers.hpp"
#include <vector>
#include <map>
#include <unordered_map>
#include <array>
#include <optional>

namespace triton_vm {

// Forward declarations
class VMState;
struct OpStackTableEntry;
struct RamTableCall;
struct U32TableEntry;

// Forward declaration - full definition in co_processor_call.hpp
struct CoProcessorCall;

/**
 * AlgebraicExecutionTrace (AET) - Primary witness for proof generation
 * 
 * Records every intermediate state of the processor and all co-processors,
 * alongside additional witness information such as instruction execution counts.
 */
class AlgebraicExecutionTrace {
public:
    static constexpr size_t LOOKUP_TABLE_HEIGHT = 256; // 2^8
    
    // Table widths (from Rust implementation)
    static constexpr size_t PROCESSOR_WIDTH = 39;
    static constexpr size_t OP_STACK_WIDTH = 4;
    static constexpr size_t RAM_WIDTH = 7;
    static constexpr size_t HASH_WIDTH = 67;
    
    /**
     * Create a new AET for a program encoded as bwords (opcode + optional arg).
     * This matches Rust `Program::to_bwords()` / `program.len_bwords()`.
     */
    explicit AlgebraicExecutionTrace(std::vector<BFieldElement> program_bwords = {});
    
    // Move constructor and assignment for TraceResult
    AlgebraicExecutionTrace(AlgebraicExecutionTrace&&) noexcept = default;
    AlgebraicExecutionTrace& operator=(AlgebraicExecutionTrace&&) noexcept = default;
    
    // Copy constructor and assignment - needed for some use cases
    AlgebraicExecutionTrace(const AlgebraicExecutionTrace&) = default;
    AlgebraicExecutionTrace& operator=(const AlgebraicExecutionTrace&) = default;
    
    /**
     * Pre-reserve capacity for processor trace
     */
    void reserve_processor_trace(size_t capacity);
    
    /**
     * Pre-reserve capacity for all sub-tables (optimization)
     */
    void reserve_tables(size_t estimated_rows);
    
    /**
     * Load AET data from Rust FFI (for faster trace execution)
     * 
     * @param processor_trace_data Flat processor trace (row-major)
     * @param processor_trace_rows Number of processor trace rows
     * @param instruction_multiplicities Instruction multiplicities array
     * @param instruction_multiplicities_len Length of multiplicities array
     * @param op_stack_trace_data Flat op_stack trace (row-major)
     * @param op_stack_trace_rows Number of op_stack rows
     * @param op_stack_trace_cols Number of op_stack cols (4)
     * @param ram_trace_data Flat ram trace (row-major)
     * @param ram_trace_rows Number of ram rows
     * @param ram_trace_cols Number of ram cols (7)
     * @param program_hash_trace_data Flat program_hash trace (row-major)
     * @param program_hash_trace_rows Number of program_hash rows
     * @param program_hash_trace_cols Number of program_hash cols (67)
     * @param hash_trace_data Flat hash trace (row-major)
     * @param hash_trace_rows Number of hash rows
     * @param hash_trace_cols Number of hash cols (67)
     * @param sponge_trace_data Flat sponge trace (row-major)
     * @param sponge_trace_rows Number of sponge rows
     * @param sponge_trace_cols Number of sponge cols (67)
     * @param u32_entries_data Flat u32 entries [instruction, op1, op2, mult] per entry
     * @param u32_entries_len Number of u32 entries
     * @param cascade_multiplicities_data Flat cascade multiplicities [limb, mult] pairs
     * @param cascade_multiplicities_len Number of cascade entries
     * @param lookup_multiplicities_256 Lookup multiplicities array (256 u64)
     * @param table_lengths_9 Table lengths from Rust [program, processor, op_stack, ram, jump_stack, hash, cascade, lookup, u32]
     */
    void load_from_rust_ffi(
        const uint64_t* processor_trace_data,
        size_t processor_trace_rows,
        const uint32_t* instruction_multiplicities,
        size_t instruction_multiplicities_len,
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
     * Record the current VM state
     */
    void record_state(const VMState& state);
    
    /**
     * Record the current VM state with pre-cached instruction (faster)
     */
    void record_state_cached(const VMState& state, const TritonInstruction& instr);
    
    /**
     * Record a co-processor call
     */
    void record_co_processor_call(const CoProcessorCall& call);
    
    /**
     * Get the padded height (next power of 2)
     */
    size_t padded_height() const;
    
    /**
     * Get the height of a specific table
     */
    size_t height_of_table(size_t table_id) const;
    
    /**
     * Get the height of the longest table
     */
    size_t height() const;
    
    // Accessors
    const std::vector<BFieldElement>& program_bwords() const { return program_bwords_; }
    size_t program_length() const { return program_bwords_.size(); }
    const std::vector<uint32_t>& instruction_multiplicities() const { return instruction_multiplicities_; }
    const std::vector<std::vector<BFieldElement>>& processor_trace() const;
    
    // Direct flat buffer access (for GPU upload, avoids conversion)
    const BFieldElement* processor_trace_flat_data() const { return processor_trace_flat_.data(); }
    size_t processor_trace_flat_size() const { return processor_trace_rows_ * PROCESSOR_WIDTH; }
    const std::vector<std::vector<BFieldElement>>& op_stack_underflow_trace() const { return op_stack_underflow_trace_; }
    const std::vector<std::vector<BFieldElement>>& ram_trace() const { return ram_trace_; }
    const std::vector<std::vector<BFieldElement>>& program_hash_trace() const { return program_hash_trace_; }
    const std::vector<std::vector<BFieldElement>>& hash_trace() const { return hash_trace_; }
    const std::vector<std::vector<BFieldElement>>& sponge_trace() const { return sponge_trace_; }
    const std::vector<std::pair<uint16_t, uint64_t>>& cascade_table_lookup_multiplicities() const { return cascade_table_lookup_multiplicities_; }
    const std::array<uint64_t, LOOKUP_TABLE_HEIGHT>& lookup_table_lookup_multiplicities() const { return lookup_table_lookup_multiplicities_; }
    const std::vector<std::pair<U32TableEntry, uint64_t>>& u32_entries() const { return u32_entries_; }
    
    size_t processor_trace_height() const { return processor_trace_rows_; }
    size_t processor_trace_width() const { return PROCESSOR_WIDTH; }
    
    /**
     * Override height calculation using Rust table lengths
     * This is needed when loading from Rust FFI which doesn't return all traces
     */
    void set_rust_table_lengths(const size_t* table_lengths_9);
    
private:
    // Cached table lengths from Rust (when using Rust FFI)
    mutable std::optional<std::array<size_t, 9>> rust_table_lengths_;
    /// Number of times each instruction has been executed
    std::vector<uint32_t> instruction_multiplicities_;
    
    // Program encoded as bwords (opcode + optional arg). This is sufficient for:
    // - program table filling
    // - program hashing trace (program attestation)
    std::vector<BFieldElement> program_bwords_;
    
    /// Processor state after each instruction
    /// OPTIMIZATION: Flat buffer for cache-friendly access (row-major layout)
    std::vector<BFieldElement> processor_trace_flat_;
    size_t processor_trace_rows_ = 0;
    size_t processor_trace_capacity_ = 0;
    
    /// Legacy accessor (for compatibility with existing code)
    mutable std::vector<std::vector<BFieldElement>> processor_trace_legacy_;
    
    /// Op stack underflow trace
    std::vector<std::vector<BFieldElement>> op_stack_underflow_trace_;
    
    /// RAM trace
    std::vector<std::vector<BFieldElement>> ram_trace_;
    
    /// Program hash trace
    std::vector<std::vector<BFieldElement>> program_hash_trace_;
    
    /// Hash trace (for hash instruction)
    std::vector<std::vector<BFieldElement>> hash_trace_;
    
    /// Sponge trace (for sponge instructions)
    std::vector<std::vector<BFieldElement>> sponge_trace_;

    /// U32 entries (instruction + operands) with multiplicities
    // Rust uses IndexMap to preserve insertion order. We track insertion order explicitly.
    std::vector<std::pair<U32TableEntry, uint64_t>> u32_entries_;
    std::unordered_map<U32TableEntry, size_t, U32TableEntryHash> u32_entries_index_;
    
    /// Cascade table lookup multiplicities
    // Rust uses IndexMap to preserve insertion order. We track insertion order explicitly.
    std::vector<std::pair<uint16_t, uint64_t>> cascade_table_lookup_multiplicities_;
    std::unordered_map<uint16_t, size_t> cascade_table_lookup_index_;
    
    /// Lookup table lookup multiplicities
    std::array<uint64_t, LOOKUP_TABLE_HEIGHT> lookup_table_lookup_multiplicities_{};
    
    /**
     * Fill program hash trace (called during construction)
     */
    void fill_program_hash_trace();
};

} // namespace triton_vm

