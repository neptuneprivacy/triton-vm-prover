#pragma once

#include "table/extend_helpers.hpp"
#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include <string>
#include <vector>
#include <optional>

namespace triton_vm {

/**
 * Program - Represents a TASM program
 * 
 * Wraps TritonProgram and provides interface for VM execution.
 */
class Program {
public:
    /**
     * Create program from TASM code string
     */
    static Program from_code(const std::string& code);
    
    /**
     * Create program from TASM file
     */
    static Program from_file(const std::string& filepath);
    
    /**
     * Create program from bwords array (used when loading from Rust FFI)
     */
    static Program from_bwords(const uint64_t* bwords_data, size_t bwords_len);
    
    /**
     * Get program digest (hash of the program)
     */
    Digest hash() const;
    
    /**
     * Get number of instructions (in bwords)
     */
    size_t len_bwords() const;

    /**
     * Get program as a sequence of BFieldElements ("bwords"):
     * each instruction encoded as opcode, followed by its argument (if any).
     *
     * Matches Rust `Program::to_bwords()`.
     */
    std::vector<BFieldElement> to_bwords() const;
    
    /**
     * Get instruction at index
     */
    std::optional<TritonInstruction> instruction_at(size_t index) const;
    
    /**
     * Find label address
     */
    std::optional<size_t> find_label(const std::string& name) const;
    
    /**
     * Get the underlying TritonProgram
     */
    const TritonProgram& triton_program() const { return program_; }
    
    // Copy constructor and assignment - needed for AET
    // Program is copyable since TritonProgram contains standard containers
    Program(const Program&) = default;
    Program& operator=(const Program&) = default;
    Program(Program&&) = default;
    Program& operator=(Program&&) = default;
    
private:
    TritonProgram program_;

    // Cached bword representation (opcode + optional arg), used for hashing and execution.
    // This is the canonical address space for IP/NIA in Triton VM.
    std::vector<BFieldElement> bwords_;

    // Label addresses in bword space (IP values).
    std::unordered_map<std::string, size_t> label_bword_map_;
    
    Program() = default;
    
    /**
     * Parse TASM code into instructions
     */
    void parse_code(const std::string& code);
};

} // namespace triton_vm

