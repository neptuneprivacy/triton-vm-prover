#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "stark/challenges.hpp"
#include "stark/cross_table_arg.hpp"
#include <vector>
#include <optional>
#include <string>
#include <unordered_map>
#include <array>

namespace triton_vm {

// Lightweight view over a row-major main table stored as raw u64 BFEs.
// Intended for hybrid CPU aux: avoid pre-converting the full table into `BFieldElement` objects.
struct MainTableFlatView {
    const uint64_t* data = nullptr;   // [num_rows * num_cols]
    size_t num_rows = 0;
    size_t num_cols = 0;

    // Cast before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    inline BFieldElement at(size_t r, size_t c) const { 
        return BFieldElement(data[static_cast<size_t>(r) * static_cast<size_t>(num_cols) + static_cast<size_t>(c)]); 
    }
    inline uint64_t raw(size_t r, size_t c) const { 
        return data[static_cast<size_t>(r) * static_cast<size_t>(num_cols) + static_cast<size_t>(c)]; 
    }

    struct RowProxy {
        const MainTableFlatView* parent;
        size_t r;
        inline BFieldElement operator[](size_t c) const { return parent->at(r, c); }
    };

    inline RowProxy operator[](size_t r) const { return RowProxy{this, r}; }
};

// Column constants (matching Rust)
namespace TableColumnOffsets {
    // Main table column offsets
    constexpr size_t PROGRAM_TABLE_START = 0;
    constexpr size_t PROGRAM_TABLE_COLS = 7;
    constexpr size_t PROCESSOR_TABLE_START = 7;
    constexpr size_t PROCESSOR_TABLE_COLS = 39;
    constexpr size_t OP_STACK_TABLE_START = 46;
    constexpr size_t OP_STACK_TABLE_COLS = 4;
    constexpr size_t RAM_TABLE_START = 50;
    constexpr size_t RAM_TABLE_COLS = 7;
    constexpr size_t JUMP_STACK_TABLE_START = 57;
    constexpr size_t JUMP_STACK_TABLE_COLS = 5;
    constexpr size_t HASH_TABLE_START = 62;
    constexpr size_t HASH_TABLE_COLS = 67;
    constexpr size_t CASCADE_TABLE_START = 129;
    constexpr size_t CASCADE_TABLE_COLS = 6;
    constexpr size_t LOOKUP_TABLE_START = 135;
    constexpr size_t LOOKUP_TABLE_COLS = 4;
    constexpr size_t U32_TABLE_START = 139;
    constexpr size_t U32_TABLE_COLS = 10;
    
    // Aux table column offsets
    // Sequential layout matching Rust air crate
    constexpr size_t AUX_PROGRAM_TABLE_START = 0;
    constexpr size_t AUX_PROGRAM_TABLE_COLS = 3;
    constexpr size_t AUX_PROCESSOR_TABLE_START = 3;  // 0 + 3
    constexpr size_t AUX_PROCESSOR_TABLE_COLS = 11;
    constexpr size_t AUX_OP_STACK_TABLE_START = 14;  // 3 + 11
    constexpr size_t AUX_OP_STACK_TABLE_COLS = 2;
    constexpr size_t AUX_RAM_TABLE_START = 16;  // 14 + 2
    constexpr size_t AUX_RAM_TABLE_COLS = 6;
    constexpr size_t AUX_JUMP_STACK_TABLE_START = 22;  // 16 + 6
    constexpr size_t AUX_JUMP_STACK_TABLE_COLS = 2;
    constexpr size_t AUX_HASH_TABLE_START = 24;  // 22 + 2
    constexpr size_t AUX_HASH_TABLE_COLS = 20;  // HashAuxColumn: 4 eval args + 16 CASCADE log derivs
    constexpr size_t AUX_CASCADE_TABLE_START = 44;  // 24 + 20
    constexpr size_t AUX_CASCADE_TABLE_COLS = 2;
    constexpr size_t AUX_LOOKUP_TABLE_START = 46;  // 44 + 2
    constexpr size_t AUX_LOOKUP_TABLE_COLS = 2;
    constexpr size_t AUX_U32_TABLE_START = 48;  // 46 + 2
    constexpr size_t AUX_U32_TABLE_COLS = 1;
}

// Program table main column indices
namespace ProgramMainColumn {
    constexpr size_t Address = 0;
    constexpr size_t Instruction = 1;
    constexpr size_t LookupMultiplicity = 2;
    constexpr size_t IndexInChunk = 3;
    constexpr size_t MaxMinusIndexInChunkInv = 4;
    constexpr size_t IsHashInputPadding = 5;
    constexpr size_t IsTablePadding = 6;
}

// OpStack table main column indices (match Rust OpStackMainColumn)
namespace OpStackMainColumn {
    constexpr size_t CLK = 0;
    constexpr size_t IB1ShrinkStack = 1;
    constexpr size_t StackPointer = 2;
    constexpr size_t FirstUnderflowElement = 3;
}

// RAM table main column indices (match Rust RamMainColumn)
namespace RamMainColumn {
    constexpr size_t CLK = 0;
    constexpr size_t InstructionType = 1;
    constexpr size_t RamPointer = 2;
    constexpr size_t RamValue = 3;
    constexpr size_t InverseOfRampDifference = 4;
    constexpr size_t BezoutCoefficientPolynomialCoefficient0 = 5;
    constexpr size_t BezoutCoefficientPolynomialCoefficient1 = 6;
}

// JumpStack table main column indices (match Rust JumpStackMainColumn)
namespace JumpStackMainColumn {
    constexpr size_t CLK = 0;
    constexpr size_t CI = 1;
    constexpr size_t JSP = 2;
    constexpr size_t JSO = 3;
    constexpr size_t JSD = 4;
}

// Hash table main column indices (match Rust HashMainColumn)
namespace HashMainColumn {
    constexpr size_t Mode = 0;
    constexpr size_t CI = 1;
    constexpr size_t RoundNumber = 2;

    constexpr size_t State0HighestLkIn = 3;
    constexpr size_t State0MidHighLkIn = 4;
    constexpr size_t State0MidLowLkIn = 5;
    constexpr size_t State0LowestLkIn = 6;
    constexpr size_t State1HighestLkIn = 7;
    constexpr size_t State1MidHighLkIn = 8;
    constexpr size_t State1MidLowLkIn = 9;
    constexpr size_t State1LowestLkIn = 10;
    constexpr size_t State2HighestLkIn = 11;
    constexpr size_t State2MidHighLkIn = 12;
    constexpr size_t State2MidLowLkIn = 13;
    constexpr size_t State2LowestLkIn = 14;
    constexpr size_t State3HighestLkIn = 15;
    constexpr size_t State3MidHighLkIn = 16;
    constexpr size_t State3MidLowLkIn = 17;
    constexpr size_t State3LowestLkIn = 18;

    constexpr size_t State0HighestLkOut = 19;
    constexpr size_t State0MidHighLkOut = 20;
    constexpr size_t State0MidLowLkOut = 21;
    constexpr size_t State0LowestLkOut = 22;
    constexpr size_t State1HighestLkOut = 23;
    constexpr size_t State1MidHighLkOut = 24;
    constexpr size_t State1MidLowLkOut = 25;
    constexpr size_t State1LowestLkOut = 26;
    constexpr size_t State2HighestLkOut = 27;
    constexpr size_t State2MidHighLkOut = 28;
    constexpr size_t State2MidLowLkOut = 29;
    constexpr size_t State2LowestLkOut = 30;
    constexpr size_t State3HighestLkOut = 31;
    constexpr size_t State3MidHighLkOut = 32;
    constexpr size_t State3MidLowLkOut = 33;
    constexpr size_t State3LowestLkOut = 34;

    constexpr size_t State4 = 35;
    constexpr size_t State5 = 36;
    constexpr size_t State6 = 37;
    constexpr size_t State7 = 38;
    constexpr size_t State8 = 39;
    constexpr size_t State9 = 40;
    constexpr size_t State10 = 41;
    constexpr size_t State11 = 42;
    constexpr size_t State12 = 43;
    constexpr size_t State13 = 44;
    constexpr size_t State14 = 45;
    constexpr size_t State15 = 46;

    constexpr size_t State0Inv = 47;
    constexpr size_t State1Inv = 48;
    constexpr size_t State2Inv = 49;
    constexpr size_t State3Inv = 50;

    constexpr size_t Constant0 = 51;
    constexpr size_t Constant1 = 52;
    constexpr size_t Constant2 = 53;
    constexpr size_t Constant3 = 54;
    constexpr size_t Constant4 = 55;
    constexpr size_t Constant5 = 56;
    constexpr size_t Constant6 = 57;
    constexpr size_t Constant7 = 58;
    constexpr size_t Constant8 = 59;
    constexpr size_t Constant9 = 60;
    constexpr size_t Constant10 = 61;
    constexpr size_t Constant11 = 62;
    constexpr size_t Constant12 = 63;
    constexpr size_t Constant13 = 64;
    constexpr size_t Constant14 = 65;
    constexpr size_t Constant15 = 66;
}

// Hash table modes (match Rust HashTableMode discriminants)
namespace HashTableMode {
    constexpr uint64_t Pad = 0;
    constexpr uint64_t ProgramHashing = 1;
    constexpr uint64_t Sponge = 2;
    constexpr uint64_t Hash = 3;
}

// Cascade table main column indices (match Rust CascadeMainColumn)
namespace CascadeMainColumn {
    constexpr size_t IsPadding = 0;
    constexpr size_t LookInHi = 1;
    constexpr size_t LookInLo = 2;
    constexpr size_t LookOutHi = 3;
    constexpr size_t LookOutLo = 4;
    constexpr size_t LookupMultiplicity = 5;
}

// Lookup table main column indices (match Rust LookupMainColumn)
namespace LookupMainColumn {
    constexpr size_t IsPadding = 0;
    constexpr size_t LookIn = 1;
    constexpr size_t LookOut = 2;
    constexpr size_t LookupMultiplicity = 3;
}

// U32 table main column indices (match Rust U32MainColumn)
namespace U32MainColumn {
    constexpr size_t CopyFlag = 0;
    constexpr size_t Bits = 1;
    constexpr size_t BitsMinus33Inv = 2;
    constexpr size_t CI = 3;
    constexpr size_t LHS = 4;
    constexpr size_t LhsInv = 5;
    constexpr size_t RHS = 6;
    constexpr size_t RhsInv = 7;
    constexpr size_t Result = 8;
    constexpr size_t LookupMultiplicity = 9;
}

// U32 table aux column indices (relative to AUX_U32_TABLE_START)
namespace U32AuxColumn {
    constexpr size_t LookupServerLogDerivative = 0;
}

// Program table aux column indices (relative to AUX_PROGRAM_TABLE_START)
namespace ProgramAuxColumn {
    constexpr size_t InstructionLookupServerLogDerivative = 0;
    constexpr size_t PrepareChunkRunningEvaluation = 1;
    constexpr size_t SendChunkRunningEvaluation = 2;
}

// Helper function: Get main table row for a specific table
std::vector<BFieldElement> get_main_table_row(
    const std::vector<std::vector<BFieldElement>>& main_table,
    size_t row_idx,
    size_t start_col,
    size_t num_cols
);

// Helper function: Set aux table row for a specific table
void set_aux_table_row(
    std::vector<std::vector<XFieldElement>>& aux_table,
    size_t row_idx,
    size_t start_col,
    const std::vector<XFieldElement>& values
);

// Extend ProgramTable
void extend_program_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_program_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend OpStackTable
void extend_op_stack_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_op_stack_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend JumpStackTable
void extend_jump_stack_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_jump_stack_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend LookupTable
void extend_lookup_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_lookup_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend HashTable (simplified - only ReceiveChunkRunningEvaluation)
void extend_hash_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_hash_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend CascadeTable
void extend_cascade_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_cascade_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend U32Table
void extend_u32_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_u32_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend RamTable
void extend_ram_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_ram_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// Extend ProcessorTable (most complex - 11 aux columns)
void extend_processor_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

void extend_processor_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
);

// CRITICAL: Instruction Infrastructure for LDE Extended Purposes
// ============================================================================

// Supporting types for instruction processing
enum class NumberOfWords {
    N1 = 1,
    N2 = 2,
    N3 = 3,
    N4 = 4,
    N5 = 5
};

enum class OpStackElement {
    ST0 = 0, ST1 = 1, ST2 = 2, ST3 = 3, ST4 = 4, ST5 = 5,
    ST6 = 6, ST7 = 7, ST8 = 8, ST9 = 9, ST10 = 10, ST11 = 11,
    ST12 = 12, ST13 = 13, ST14 = 14, ST15 = 15
};

// Instruction bit system for constraint circuits
enum class InstructionBit {
    IB0 = 0, IB1 = 1, IB2 = 2, IB3 = 3, IB4 = 4, IB5 = 5, IB6 = 6
};

// Instruction bucket system for categorization
enum class InstructionBucket {
    HasArg,
    ShrinksStack,
    IsU32
};

// Labelled instruction system for program parsing
struct InstructionLabel {
    std::string name;
};

struct TypeHint {
    std::string variable_name;
    std::optional<std::string> type_name;
    size_t starting_index;
};

enum class AssertionContext {
    ID
};

struct AssertionContextData {
    AssertionContext context;
    size_t id;
};

// Labelled instructions with metadata

// Error handling for instruction operations
class InstructionError : public std::runtime_error {
public:
    enum class Type {
        InvalidOpcode,
        IllegalArgument,
        OutOfRange
    };

private:
    Type type_;
    uint32_t opcode_;
    BFieldElement argument_;

public:
    InstructionError(Type type, uint32_t opcode = 0, BFieldElement arg = BFieldElement::zero())
        : std::runtime_error(make_message(type, opcode, arg)), type_(type), opcode_(opcode), argument_(arg) {}

    Type get_type() const { return type_; }
    uint32_t opcode() const { return opcode_; }
    BFieldElement argument() const { return argument_; }

private:
    static std::string make_message(Type type, uint32_t opcode, BFieldElement arg) {
        switch (type) {
            case Type::InvalidOpcode:
                return "Invalid opcode: " + std::to_string(opcode);
            case Type::IllegalArgument:
                return "Illegal argument " + arg.to_string() + " for instruction with opcode " + std::to_string(opcode);
            case Type::OutOfRange:
                return "Argument out of range: " + arg.to_string();
        }
        return "Unknown instruction error";
    }
};

// Complete AnInstruction enum with all Triton VM operations
enum class AnInstruction {
    // OpStack manipulation
    Push, Pop, Divine, Pick, Place, Dup, Swap,
    // Control flow
    Halt, Nop, Skiz, Call, Return, Recurse, RecurseOrReturn, Assert,
    // Memory access
    ReadMem, WriteMem,
    // Hashing-related
    Hash, AssertVector, SpongeInit, SpongeAbsorb, SpongeAbsorbMem, SpongeSqueeze,
    // Base field arithmetic on stack
    Add, AddI, Mul, Invert, Eq,
    // Bitwise arithmetic on stack
    Split, Lt, And, Xor, Log2Floor, Pow, DivMod, PopCount,
    // Extension field arithmetic on stack
    XxAdd, XxMul, XInvert, XbMul,
    // Read/write
    ReadIo, WriteIo,
    // Many-in-One
    MerkleStep, MerkleStepMem, XxDotStep, XbDotStep,
};

// Triton instruction with arguments - CRITICAL for LDE extended purposes
struct TritonInstruction {
    AnInstruction type;
    BFieldElement bfield_arg;  // For Push, Call, AddI
    NumberOfWords num_words_arg;  // For Pop, Divine, ReadMem, etc.
    OpStackElement op_stack_arg;  // For Pick, Place, Dup, Swap

    // MOST CRITICAL METHODS FOR LDE EXTENDED PURPOSE - INLINE IMPLEMENTATIONS
    std::string name() const {
        switch (type) {
            case AnInstruction::Pop: return "pop";
            case AnInstruction::Push: return "push";
            case AnInstruction::Divine: return "divine";
            case AnInstruction::Pick: return "pick";
            case AnInstruction::Place: return "place";
            case AnInstruction::Dup: return "dup";
            case AnInstruction::Swap: return "swap";
            case AnInstruction::Halt: return "halt";
            case AnInstruction::Nop: return "nop";
            case AnInstruction::Skiz: return "skiz";
            case AnInstruction::Call: return "call";
            case AnInstruction::Return: return "return";
            case AnInstruction::Recurse: return "recurse";
            case AnInstruction::RecurseOrReturn: return "recurse_or_return";
            case AnInstruction::Assert: return "assert";
            case AnInstruction::ReadMem: return "read_mem";
            case AnInstruction::WriteMem: return "write_mem";
            case AnInstruction::Hash: return "hash";
            case AnInstruction::AssertVector: return "assert_vector";
            case AnInstruction::SpongeInit: return "sponge_init";
            case AnInstruction::SpongeAbsorb: return "sponge_absorb";
            case AnInstruction::SpongeAbsorbMem: return "sponge_absorb_mem";
            case AnInstruction::SpongeSqueeze: return "sponge_squeeze";
            case AnInstruction::Add: return "add";
            case AnInstruction::AddI: return "addi";
            case AnInstruction::Mul: return "mul";
            case AnInstruction::Invert: return "invert";
            case AnInstruction::Eq: return "eq";
            case AnInstruction::Split: return "split";
            case AnInstruction::Lt: return "lt";
            case AnInstruction::And: return "and";
            case AnInstruction::Xor: return "xor";
            case AnInstruction::Log2Floor: return "log_2_floor";
            case AnInstruction::Pow: return "pow";
            case AnInstruction::DivMod: return "div_mod";
            case AnInstruction::PopCount: return "pop_count";
            case AnInstruction::XxAdd: return "xx_add";
            case AnInstruction::XxMul: return "xx_mul";
            case AnInstruction::XInvert: return "x_invert";
            case AnInstruction::XbMul: return "xb_mul";
            case AnInstruction::ReadIo: return "read_io";
            case AnInstruction::WriteIo: return "write_io";
            case AnInstruction::MerkleStep: return "merkle_step";
            case AnInstruction::MerkleStepMem: return "merkle_step_mem";
            case AnInstruction::XxDotStep: return "xx_dot_step";
            case AnInstruction::XbDotStep: return "xb_dot_step";
            default: return "unknown";
        }
    }

    int32_t op_stack_size_influence() const {
        switch (type) {
            case AnInstruction::Pop: return -static_cast<int32_t>(num_words_arg);
            case AnInstruction::Push: return 1;
            case AnInstruction::Divine: return static_cast<int32_t>(num_words_arg);
            case AnInstruction::Pick: return 0;
            case AnInstruction::Place: return 0;
            case AnInstruction::Dup: return 1;
            case AnInstruction::Swap: return 0;
            case AnInstruction::Halt: return 0;
            case AnInstruction::Nop: return 0;
            case AnInstruction::Skiz: return -1;
            case AnInstruction::Call: return 0;
            case AnInstruction::Return: return 0;
            case AnInstruction::Recurse: return 0;
            case AnInstruction::RecurseOrReturn: return 0;
            case AnInstruction::Assert: return -1;
            case AnInstruction::ReadMem: return static_cast<int32_t>(num_words_arg);
            case AnInstruction::WriteMem: return -static_cast<int32_t>(num_words_arg);
            case AnInstruction::Hash: return -5;
            case AnInstruction::AssertVector: return -5;
            case AnInstruction::SpongeInit: return 0;
            case AnInstruction::SpongeAbsorb: return -10;
            case AnInstruction::SpongeAbsorbMem: return 0;
            case AnInstruction::SpongeSqueeze: return 10;
            case AnInstruction::Add: return -1;
            case AnInstruction::AddI: return 0;
            case AnInstruction::Mul: return -1;
            case AnInstruction::Invert: return 0;
            case AnInstruction::Eq: return -1;
            case AnInstruction::Split: return 1;
            case AnInstruction::Lt: return -1;
            case AnInstruction::And: return -1;
            case AnInstruction::Xor: return -1;
            case AnInstruction::Log2Floor: return 0;
            case AnInstruction::Pow: return -1;
            case AnInstruction::DivMod: return 0;
            case AnInstruction::PopCount: return 0;
            case AnInstruction::XxAdd: return -3;
            case AnInstruction::XxMul: return -3;
            case AnInstruction::XInvert: return 0;
            case AnInstruction::XbMul: return -1;
            case AnInstruction::ReadIo: return static_cast<int32_t>(num_words_arg);
            case AnInstruction::WriteIo: return -static_cast<int32_t>(num_words_arg);
            case AnInstruction::MerkleStep: return 0;
            case AnInstruction::MerkleStepMem: return 0;
            case AnInstruction::XxDotStep: return 0;
            case AnInstruction::XbDotStep: return 0;
            default: return 0;
        }
    }

    std::optional<BFieldElement> arg() const {
        switch (type) {
            case AnInstruction::Push: return bfield_arg;
            case AnInstruction::Call: return bfield_arg;
            case AnInstruction::AddI: return bfield_arg;
            case AnInstruction::Pop: return BFieldElement(static_cast<uint64_t>(num_words_arg));
            case AnInstruction::Divine: return BFieldElement(static_cast<uint64_t>(num_words_arg));
            case AnInstruction::Pick: return BFieldElement(static_cast<uint64_t>(op_stack_arg));
            case AnInstruction::Place: return BFieldElement(static_cast<uint64_t>(op_stack_arg));
            case AnInstruction::Dup: return BFieldElement(static_cast<uint64_t>(op_stack_arg));
            case AnInstruction::Swap: return BFieldElement(static_cast<uint64_t>(op_stack_arg));
            case AnInstruction::ReadMem: return BFieldElement(static_cast<uint64_t>(num_words_arg));
            case AnInstruction::WriteMem: return BFieldElement(static_cast<uint64_t>(num_words_arg));
            case AnInstruction::ReadIo: return BFieldElement(static_cast<uint64_t>(num_words_arg));
            case AnInstruction::WriteIo: return BFieldElement(static_cast<uint64_t>(num_words_arg));
            default: return std::nullopt;
        }
    }

    bool is_u32_instruction() const {
        switch (type) {
            case AnInstruction::Split:
            case AnInstruction::Lt:
            case AnInstruction::And:
            case AnInstruction::Xor:
            case AnInstruction::Log2Floor:
            case AnInstruction::Pow:
            case AnInstruction::DivMod:
            case AnInstruction::PopCount:
            case AnInstruction::MerkleStep:
            case AnInstruction::MerkleStepMem:
                return true;
            default:
                return false;
        }
    }

    size_t size() const {
        switch (type) {
            case AnInstruction::Pop:
            case AnInstruction::Push:
            case AnInstruction::Divine:
            case AnInstruction::Pick:
            case AnInstruction::Place:
            case AnInstruction::Dup:
            case AnInstruction::Swap:
            case AnInstruction::Call:
            case AnInstruction::ReadMem:
            case AnInstruction::WriteMem:
            case AnInstruction::AddI:
            case AnInstruction::ReadIo:
            case AnInstruction::WriteIo:
                return 2;
            default:
                return 1;
        }
    }

    BFieldElement ib(InstructionBit bit) const {
        uint32_t opcode_val = opcode();
        size_t bit_index = static_cast<size_t>(bit);
        uint32_t bit_value = (opcode_val >> bit_index) & 1;
        return BFieldElement(bit_value);
    }

    uint32_t opcode() const {
        switch (type) {
            case AnInstruction::Pop: return 3;
            case AnInstruction::Push: return 1;
            case AnInstruction::Divine: return 9;
            case AnInstruction::Pick: return 17;
            case AnInstruction::Place: return 25;
            case AnInstruction::Dup: return 33;
            case AnInstruction::Swap: return 41;
            case AnInstruction::Halt: return 0;
            case AnInstruction::Nop: return 8;
            case AnInstruction::Skiz: return 2;
            case AnInstruction::Call: return 49;
            case AnInstruction::Return: return 16;
            case AnInstruction::Recurse: return 24;
            case AnInstruction::RecurseOrReturn: return 32;
            case AnInstruction::Assert: return 10;
            case AnInstruction::ReadMem: return 57;
            case AnInstruction::WriteMem: return 11;
            case AnInstruction::Hash: return 18;
            case AnInstruction::AssertVector: return 26;
            case AnInstruction::SpongeInit: return 40;
            case AnInstruction::SpongeAbsorb: return 34;
            case AnInstruction::SpongeAbsorbMem: return 48;
            case AnInstruction::SpongeSqueeze: return 56;
            case AnInstruction::Add: return 42;
            case AnInstruction::AddI: return 65;
            case AnInstruction::Mul: return 50;
            case AnInstruction::Invert: return 64;
            case AnInstruction::Eq: return 58;
            case AnInstruction::Split: return 4;
            case AnInstruction::Lt: return 6;
            case AnInstruction::And: return 14;
            case AnInstruction::Xor: return 22;
            case AnInstruction::Log2Floor: return 12;
            case AnInstruction::Pow: return 30;
            case AnInstruction::DivMod: return 20;
            case AnInstruction::PopCount: return 28;
            case AnInstruction::XxAdd: return 66;
            case AnInstruction::XxMul: return 74;
            case AnInstruction::XInvert: return 72;
            case AnInstruction::XbMul: return 82;
            case AnInstruction::ReadIo: return 73;
            case AnInstruction::WriteIo: return 19;
            case AnInstruction::MerkleStep: return 36;
            case AnInstruction::MerkleStepMem: return 44;
            case AnInstruction::XxDotStep: return 80;
            case AnInstruction::XbDotStep: return 88;
            default: return 0;
        }
    }

    // Instruction creation and decoding
    static std::optional<TritonInstruction> from_opcode(uint32_t opcode, BFieldElement nia) {
        std::optional<TritonInstruction> instr_opt;
        switch (opcode) {
            case 0: instr_opt = TritonInstruction{AnInstruction::Halt}; break;
            case 1: instr_opt = TritonInstruction{AnInstruction::Push}; break;
            case 2: instr_opt = TritonInstruction{AnInstruction::Skiz}; break;
            case 3: instr_opt = TritonInstruction{AnInstruction::Pop, {}, NumberOfWords::N1}; break;
            case 4: instr_opt = TritonInstruction{AnInstruction::Split}; break;
            case 6: instr_opt = TritonInstruction{AnInstruction::Lt}; break;
            case 8: instr_opt = TritonInstruction{AnInstruction::Nop}; break;
            case 9: instr_opt = TritonInstruction{AnInstruction::Divine, {}, NumberOfWords::N1}; break;
            case 10: instr_opt = TritonInstruction{AnInstruction::Assert}; break;
            case 11: instr_opt = TritonInstruction{AnInstruction::WriteMem, {}, NumberOfWords::N1}; break;
            case 12: instr_opt = TritonInstruction{AnInstruction::Log2Floor}; break;
            case 14: instr_opt = TritonInstruction{AnInstruction::And}; break;
            case 16: instr_opt = TritonInstruction{AnInstruction::Return}; break;
            case 17: instr_opt = TritonInstruction{AnInstruction::Pick, {}, {}, OpStackElement::ST0}; break;
            case 18: instr_opt = TritonInstruction{AnInstruction::Hash}; break;
            case 19: instr_opt = TritonInstruction{AnInstruction::WriteIo, {}, NumberOfWords::N1}; break;
            case 20: instr_opt = TritonInstruction{AnInstruction::DivMod}; break;
            case 22: instr_opt = TritonInstruction{AnInstruction::Xor}; break;
            case 24: instr_opt = TritonInstruction{AnInstruction::Recurse}; break;
            case 25: instr_opt = TritonInstruction{AnInstruction::Place, {}, {}, OpStackElement::ST0}; break;
            case 26: instr_opt = TritonInstruction{AnInstruction::AssertVector}; break;
            case 28: instr_opt = TritonInstruction{AnInstruction::PopCount}; break;
            case 30: instr_opt = TritonInstruction{AnInstruction::Pow}; break;
            case 32: instr_opt = TritonInstruction{AnInstruction::RecurseOrReturn}; break;
            case 33: instr_opt = TritonInstruction{AnInstruction::Dup, {}, {}, OpStackElement::ST0}; break;
            case 34: instr_opt = TritonInstruction{AnInstruction::SpongeAbsorb}; break;
            case 36: instr_opt = TritonInstruction{AnInstruction::MerkleStep}; break;
            case 40: instr_opt = TritonInstruction{AnInstruction::SpongeInit}; break;
            case 41: instr_opt = TritonInstruction{AnInstruction::Swap, {}, {}, OpStackElement::ST0}; break;
            case 42: instr_opt = TritonInstruction{AnInstruction::Add}; break;
            case 44: instr_opt = TritonInstruction{AnInstruction::MerkleStepMem}; break;
            case 48: instr_opt = TritonInstruction{AnInstruction::SpongeAbsorbMem}; break;
            case 49: instr_opt = TritonInstruction{AnInstruction::Call}; break;
            case 50: instr_opt = TritonInstruction{AnInstruction::Mul}; break;
            case 56: instr_opt = TritonInstruction{AnInstruction::SpongeSqueeze}; break;
            case 57: instr_opt = TritonInstruction{AnInstruction::ReadMem, {}, NumberOfWords::N1}; break;
            case 58: instr_opt = TritonInstruction{AnInstruction::Eq}; break;
            case 64: instr_opt = TritonInstruction{AnInstruction::Invert}; break;
            case 65: instr_opt = TritonInstruction{AnInstruction::AddI}; break;
            case 66: instr_opt = TritonInstruction{AnInstruction::XxAdd}; break;
            case 72: instr_opt = TritonInstruction{AnInstruction::XInvert}; break;
            case 73: instr_opt = TritonInstruction{AnInstruction::ReadIo, {}, NumberOfWords::N1}; break;
            case 74: instr_opt = TritonInstruction{AnInstruction::XxMul}; break;
            case 80: instr_opt = TritonInstruction{AnInstruction::XxDotStep}; break;
            case 82: instr_opt = TritonInstruction{AnInstruction::XbMul}; break;
            case 88: instr_opt = TritonInstruction{AnInstruction::XbDotStep}; break;
            default: return std::nullopt;
        }
        if (!instr_opt.has_value()) {
            return std::nullopt;
        }
        if (instr_opt->arg().has_value()) {
            return instr_opt->change_arg(nia);
        }
        return instr_opt;
    }

    // Change the argument of the instruction, if it has one
    std::optional<TritonInstruction> change_arg(BFieldElement new_arg) const {
        NumberOfWords num_words;
        OpStackElement op_stack_elem;

        // Try to convert new_arg to the appropriate type
        auto try_num_words = [new_arg]() -> std::optional<NumberOfWords> {
            uint64_t val = new_arg.value();
            switch (val) {
                case 1: return NumberOfWords::N1;
                case 2: return NumberOfWords::N2;
                case 3: return NumberOfWords::N3;
                case 4: return NumberOfWords::N4;
                case 5: return NumberOfWords::N5;
                default: return std::nullopt;
            }
        };

        auto try_op_stack_elem = [new_arg]() -> std::optional<OpStackElement> {
            uint64_t val = new_arg.value();
            if (val <= 15) {
                return static_cast<OpStackElement>(val);
            }
            return std::nullopt;
        };

        switch (type) {
            case AnInstruction::Pop:
                if (auto nw = try_num_words()) {
                    return TritonInstruction{AnInstruction::Pop, {}, *nw};
                }
                break;
            case AnInstruction::Push:
                return TritonInstruction{AnInstruction::Push, new_arg};
            case AnInstruction::Divine:
                if (auto nw = try_num_words()) {
                    return TritonInstruction{AnInstruction::Divine, {}, *nw};
                }
                break;
            case AnInstruction::Pick:
                if (auto ose = try_op_stack_elem()) {
                    return TritonInstruction{AnInstruction::Pick, {}, {}, *ose};
                }
                break;
            case AnInstruction::Place:
                if (auto ose = try_op_stack_elem()) {
                    return TritonInstruction{AnInstruction::Place, {}, {}, *ose};
                }
                break;
            case AnInstruction::Dup:
                if (auto ose = try_op_stack_elem()) {
                    return TritonInstruction{AnInstruction::Dup, {}, {}, *ose};
                }
                break;
            case AnInstruction::Swap:
                if (auto ose = try_op_stack_elem()) {
                    return TritonInstruction{AnInstruction::Swap, {}, {}, *ose};
                }
                break;
            case AnInstruction::Call:
                return TritonInstruction{AnInstruction::Call, new_arg};
            case AnInstruction::ReadMem:
                if (auto nw = try_num_words()) {
                    return TritonInstruction{AnInstruction::ReadMem, {}, *nw};
                }
                break;
            case AnInstruction::WriteMem:
                if (auto nw = try_num_words()) {
                    return TritonInstruction{AnInstruction::WriteMem, {}, *nw};
                }
                break;
            case AnInstruction::AddI:
                return TritonInstruction{AnInstruction::AddI, new_arg};
            case AnInstruction::ReadIo:
                if (auto nw = try_num_words()) {
                    return TritonInstruction{AnInstruction::ReadIo, {}, *nw};
                }
                break;
            case AnInstruction::WriteIo:
                if (auto nw = try_num_words()) {
                    return TritonInstruction{AnInstruction::WriteIo, {}, *nw};
                }
                break;
            default:
                break; // Instruction doesn't have an argument
        }

        return std::nullopt; // Illegal argument for this instruction
    }

    // Instruction bucket flag system
    uint32_t flag_set() const {
        uint32_t flags = 0;
        if (arg().has_value()) flags |= (1 << 0); // HasArg
        if (op_stack_size_influence() < 0) flags |= (1 << 1); // ShrinksStack
        if (is_u32_instruction()) flags |= (1 << 2); // IsU32
        return flags;
    }

    // Alternative opcode computation using flag system
    uint32_t computed_opcode() const {
        uint32_t flag_set_val = flag_set();
        uint32_t base_opcode = opcode();

        // Find index within flag set
        uint32_t index_within_flag_set = 0;
        for (uint32_t test_opcode = 0; test_opcode < base_opcode; test_opcode++) {
            auto test_instr_opt = from_opcode(test_opcode, BFieldElement::zero());
            if (test_instr_opt && test_instr_opt->flag_set() == flag_set_val) {
                index_within_flag_set++;
            }
        }

        return index_within_flag_set * 8 + flag_set_val; // 2^3 = 8 for 3 buckets
    }
};

// Labelled instructions with metadata
class LabelledInstruction {
public:
    enum class Type {
        Instruction,
        Label,
        Breakpoint,
        TypeHint,
        AssertionContext
    };

private:
    Type type_;
    TritonInstruction instruction_;
    std::string label_;

public:
    // Constructors
    static LabelledInstruction make_instruction(TritonInstruction instr) {
        LabelledInstruction li;
        li.type_ = Type::Instruction;
        li.instruction_ = instr;
        return li;
    }

    static LabelledInstruction make_label(std::string name) {
        LabelledInstruction li;
        li.type_ = Type::Label;
        li.label_ = name;
        return li;
    }

    static LabelledInstruction make_breakpoint() {
        LabelledInstruction li;
        li.type_ = Type::Breakpoint;
        return li;
    }

    // Accessors
    Type get_type() const { return type_; }
    const TritonInstruction& instruction() const { return instruction_; }
    const std::string& label() const { return label_; }

    // Op stack size influence (delegates to instruction if it's an instruction)
    int32_t op_stack_size_influence() const {
        if (type_ == Type::Instruction) {
            return instruction_.op_stack_size_influence();
        }
        return 0;
    }

    // String representation
    std::string to_string() const {
        switch (type_) {
            case Type::Instruction:
                return instruction_.name();
            case Type::Label:
                return label_ + ":";
            case Type::Breakpoint:
                return "break";
            default:
                return "";
        }
        return "";
    }
};

// Basic program representation
class TritonProgram {
private:
    std::vector<LabelledInstruction> instructions_;
    std::unordered_map<std::string, size_t> labels_;

public:
    // Add an instruction
    void add_instruction(TritonInstruction instr) {
        instructions_.push_back(LabelledInstruction::make_instruction(instr));
    }

    // Add a label
    void add_label(const std::string& name) {
        labels_[name] = instructions_.size();
        instructions_.push_back(LabelledInstruction::make_label(name));
    }

    // Add a breakpoint
    void add_breakpoint() {
        instructions_.push_back(LabelledInstruction::make_breakpoint());
    }

    // Get instructions
    const std::vector<LabelledInstruction>& instructions() const {
        return instructions_;
    }

    // Find label address
    std::optional<size_t> find_label(const std::string& name) const {
        auto it = labels_.find(name);
        if (it != labels_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Get program size (number of actual instructions, excluding labels/metadata)
    size_t size() const {
        size_t count = 0;
        for (const auto& li : instructions_) {
            if (li.get_type() == LabelledInstruction::Type::Instruction) {
                count++;
            }
        }
        return count;
    }

    // Get instruction at index (skipping labels/metadata)
    std::optional<TritonInstruction> instruction_at(size_t index) const {
        size_t current_index = 0;
        for (const auto& li : instructions_) {
            if (li.get_type() == LabelledInstruction::Type::Instruction) {
                if (current_index == index) {
                    return li.instruction();
                }
                current_index++;
            }
        }
        return std::nullopt;
    }
};

// Instruction decoding function
inline std::optional<TritonInstruction> decode_instruction(BFieldElement ci, BFieldElement nia) {
    uint64_t opcode_u64 = ci.value();
    if (opcode_u64 > UINT32_MAX) {
        return std::nullopt;
    }
    uint32_t opcode = static_cast<uint32_t>(opcode_u64);
    return TritonInstruction::from_opcode(opcode, nia);
}

// Global instruction constants
namespace InstructionConstants {
    constexpr size_t INSTRUCTION_COUNT = 46;
}

} // namespace triton_vm

