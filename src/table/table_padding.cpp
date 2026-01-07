#include "table/master_table.hpp"
#include "table/extend_helpers.hpp"
#include "hash/tip5.hpp"
#include "vm/processor_columns.hpp"
#include "parallel/thread_coordination.h"
#include <algorithm>
#include <cassert>
#include <array>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <future>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef TVM_USE_TBB
#include <tbb/parallel_invoke.h>
#endif
#ifdef TVM_USE_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

namespace triton_vm {

using namespace TableColumnOffsets;

// Constants from Rust
constexpr BFieldElement PADDING_VALUE(2);
constexpr BFieldElement PADDING_INDICATOR(2);
constexpr size_t TIP5_RATE = Tip5::RATE; // 10

// Forward declarations
namespace {
    void pad_program_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t program_len,
        size_t padded_height
    );
    
    void pad_processor_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
    
    void pad_op_stack_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
    
    void pad_ram_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
    
    void pad_jump_stack_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
    
    void pad_hash_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
    
    void pad_cascade_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
    
    void pad_lookup_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
    
    void pad_u32_table(
        std::vector<std::vector<BFieldElement>>& main_table,
        size_t table_len,
        size_t padded_height
    );
}

// Program table padding
namespace {
void pad_program_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t program_len,
    size_t padded_height
) {
    using namespace ProgramMainColumn;
    
    // Precompute inverses for MaxMinusIndexInChunkInv (only 10 possible values for TIP5_RATE=10)
    std::array<BFieldElement, TIP5_RATE> inverses;
    for (size_t idx = 0; idx < TIP5_RATE; ++idx) {
        size_t max_minus = TIP5_RATE - 1 - idx;
        BFieldElement val(max_minus);
        inverses[idx] = val.is_zero() ? BFieldElement::zero() : val.inverse();
    }
    
    const size_t col_addr = PROGRAM_TABLE_START + Address;
    const size_t col_idx_chunk = PROGRAM_TABLE_START + IndexInChunk;
    const size_t col_inv = PROGRAM_TABLE_START + MaxMinusIndexInChunkInv;
    const size_t col_hash_pad = PROGRAM_TABLE_START + IsHashInputPadding;
    const size_t col_table_pad = PROGRAM_TABLE_START + IsTablePadding;
    
    // Single parallel loop for all columns
    #pragma omp parallel for schedule(static)
    for (size_t i = program_len; i < padded_height; ++i) {
        size_t index_in_chunk = i % TIP5_RATE;
        main_table[i][col_addr] = BFieldElement(static_cast<uint64_t>(i));
        main_table[i][col_idx_chunk] = BFieldElement(static_cast<uint64_t>(index_in_chunk));
        main_table[i][col_inv] = inverses[index_in_chunk];
        main_table[i][col_hash_pad] = BFieldElement::one();
        main_table[i][col_table_pad] = BFieldElement::one();
    }
}

// Processor table padding
void pad_processor_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    assert(table_len > 0 && "Processor Table must have at least one row");

    const size_t last_row_idx = table_len - 1;
    const size_t is_padding_col =
        PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::IsPadding);
    const size_t cjd_col =
        PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::ClockJumpDifferenceLookupMultiplicity);
    const size_t clk_col =
        PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::CLK);

    // Cache the template row for parallel copy
    std::array<BFieldElement, PROCESSOR_TABLE_COLS> template_row;
    for (size_t c = 0; c < PROCESSOR_TABLE_COLS; ++c) {
        template_row[c] = main_table[last_row_idx][PROCESSOR_TABLE_START + c];
    }

    // Fill padding section (rows >= table_len) for *processor columns only* - parallelized
    // OPTIMIZED: Use element-wise copy with compiler-friendly loop (compiler can vectorize)
    #pragma omp parallel for schedule(static)
    for (size_t i = table_len; i < padded_height; ++i) {
        // Copy template row (compiler will optimize this)
        for (size_t c = 0; c < PROCESSOR_TABLE_COLS; ++c) {
            main_table[i][PROCESSOR_TABLE_START + c] = template_row[c];
        }
        // Override specific columns
        main_table[i][is_padding_col] = BFieldElement::one();
        main_table[i][cjd_col] = BFieldElement::zero();
        main_table[i][clk_col] = BFieldElement(static_cast<uint64_t>(i));
    }

    // Rust special-case: Jump Stack Table has no padding indicator; clock jump diffs are looked up
    // in its padding section and are always 1. Increase multiplicity at clock value 1 accordingly.
    if (padded_height > 1) {
        const size_t num_padding_rows = padded_height - table_len;
        main_table[1][cjd_col] =
            main_table[1][cjd_col] + BFieldElement(static_cast<uint64_t>(num_padding_rows));
    }
}

// OpStack table padding
void pad_op_stack_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    using namespace OpStackMainColumn;

    const size_t last_row_idx = (table_len > 0) ? (table_len - 1) : 0;
    std::array<BFieldElement, OP_STACK_TABLE_COLS> padding_row{};
    padding_row.fill(BFieldElement::zero());

    if (table_len > 0) {
        for (size_t c = 0; c < OP_STACK_TABLE_COLS; ++c) {
            padding_row[c] = main_table[last_row_idx][OP_STACK_TABLE_START + c];
        }
    } else {
        // Rust: if table_len == 0, set StackPointer to OpStackElement::COUNT (16)
        padding_row[StackPointer] = BFieldElement(16);
    }

    // Rust: set IB1ShrinkStack to PADDING_VALUE (2)
    padding_row[IB1ShrinkStack] = PADDING_VALUE;

    // OPTIMIZED: Parallelized padding loop with compiler-friendly copy
    #pragma omp parallel for schedule(static)
    for (size_t i = table_len; i < padded_height; ++i) {
        // Element-wise copy (compiler will optimize/vectorize)
        for (size_t c = 0; c < OP_STACK_TABLE_COLS; ++c) {
            main_table[i][OP_STACK_TABLE_START + c] = padding_row[c];
        }
    }
}

// Ram table padding
void pad_ram_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    using namespace RamMainColumn;
    
    size_t last_row_idx = (table_len > 0) ? table_len - 1 : 0;
    std::array<BFieldElement, RAM_TABLE_COLS> padding_row{};
    padding_row.fill(BFieldElement::zero());
    
    if (table_len > 0) {
        // Copy last row
        for (size_t c = 0; c < RAM_TABLE_COLS; ++c) {
            padding_row[c] = main_table[last_row_idx][RAM_TABLE_START + c];
        }
    } else {
        // If table is empty, set BezoutCoefficientPolynomialCoefficient1 = 1
        padding_row[BezoutCoefficientPolynomialCoefficient1] = BFieldElement::one();
    }
    
    // Set InstructionType to PADDING_INDICATOR (2)
    padding_row[InstructionType] = PADDING_INDICATOR;
    
    // OPTIMIZED: Fill all padding rows - parallelized with compiler-friendly copy
    #pragma omp parallel for schedule(static)
    for (size_t i = table_len; i < padded_height; ++i) {
        // Element-wise copy (compiler will optimize/vectorize)
        for (size_t c = 0; c < RAM_TABLE_COLS; ++c) {
            main_table[i][RAM_TABLE_START + c] = padding_row[c];
        }
    }
}

// JumpStack table padding (most complex - involves row movement)
void pad_jump_stack_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    assert(table_len > 0 && "Jump Stack Table must have at least 1 row");
    
    // Find row with CLK = table_len - 1 (max clock before padding)
    size_t max_clk_before_padding = table_len - 1;
    size_t max_clk_row_idx = SIZE_MAX;
    
    size_t clk_col = JUMP_STACK_TABLE_START; // CLK is first column in jump stack table
    
    for (size_t i = 0; i < table_len; ++i) {
        if (main_table[i][clk_col].value() == max_clk_before_padding) {
            max_clk_row_idx = i;
            break;
        }
    }
    
    assert(max_clk_row_idx != SIZE_MAX && "Jump Stack Table must contain row with max clock cycle");
    
    size_t num_padding_rows = padded_height - table_len;
    size_t rows_to_move_start = max_clk_row_idx + 1;
    size_t rows_to_move_end = table_len;
    size_t num_rows_to_move = rows_to_move_end - rows_to_move_start;
    size_t rows_to_move_dest_start = rows_to_move_start + num_padding_rows;
    
    // Move rows below max_clk_row to the end (if any exist)
    if (num_rows_to_move > 0) {
        // Create temporary storage for rows to move
        std::vector<std::vector<BFieldElement>> rows_to_move;
        rows_to_move.reserve(num_rows_to_move);
        
        for (size_t i = rows_to_move_start; i < rows_to_move_end; ++i) {
            std::vector<BFieldElement> row(JUMP_STACK_TABLE_COLS);
            for (size_t c = 0; c < JUMP_STACK_TABLE_COLS; ++c) {
                row[c] = main_table[i][JUMP_STACK_TABLE_START + c];
            }
            rows_to_move.push_back(row);
        }
        
        // Copy to destination
        for (size_t i = 0; i < num_rows_to_move; ++i) {
            size_t dest_row = rows_to_move_dest_start + i;
            for (size_t c = 0; c < JUMP_STACK_TABLE_COLS; ++c) {
                main_table[dest_row][JUMP_STACK_TABLE_START + c] = rows_to_move[i][c];
            }
        }
    }
    
    // Fill padding section with copies of max_clk_row
    std::vector<BFieldElement> padding_template(JUMP_STACK_TABLE_COLS);
    for (size_t c = 0; c < JUMP_STACK_TABLE_COLS; ++c) {
        padding_template[c] = main_table[max_clk_row_idx][JUMP_STACK_TABLE_START + c];
    }
    
    size_t padding_section_start = rows_to_move_start;
    size_t padding_section_end = padding_section_start + num_padding_rows;
    
    for (size_t i = padding_section_start; i < padding_section_end; ++i) {
        for (size_t c = 0; c < JUMP_STACK_TABLE_COLS; ++c) {
            main_table[i][JUMP_STACK_TABLE_START + c] = padding_template[c];
        }
    }

    // Rust: CLK keeps increasing by 1 also in the padding section.
    for (size_t k = 0; k < num_padding_rows; ++k) {
        const size_t row_idx = padding_section_start + k;
        main_table[row_idx][clk_col] = BFieldElement(static_cast<uint64_t>(table_len + k));
    }
}

// Hash table padding
void pad_hash_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    using namespace HashMainColumn;

    // Rust: inverse_or_zero_of_highest_2_limbs(0) = (2^32 - 1)^(-1)
    const BFieldElement inv_2p32_minus_1 = BFieldElement((1ULL << 32) - 1).inverse();
    const BFieldElement pad_mode(HashTableMode::Pad);
    const BFieldElement hash_opcode(18); // Instruction::Hash opcode

    // Pre-cache round constants
    std::array<BFieldElement, Tip5::STATE_SIZE> round_consts;
    for (size_t k = 0; k < Tip5::STATE_SIZE; ++k) {
        round_consts[k] = Tip5::ROUND_CONSTANTS[k];
    }

    // OPTIMIZED: Parallelized padding loop with better cache locality
    // Pre-compute all column offsets to avoid repeated calculations
    const size_t col_state0_inv = HASH_TABLE_START + State0Inv;
    const size_t col_state1_inv = HASH_TABLE_START + State1Inv;
    const size_t col_state2_inv = HASH_TABLE_START + State2Inv;
    const size_t col_state3_inv = HASH_TABLE_START + State3Inv;
    const size_t col_mode = HASH_TABLE_START + Mode;
    const size_t col_ci = HASH_TABLE_START + CI;
    const size_t col_const0 = HASH_TABLE_START + Constant0;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = table_len; i < padded_height; ++i) {
        // State inverses
        main_table[i][col_state0_inv] = inv_2p32_minus_1;
        main_table[i][col_state1_inv] = inv_2p32_minus_1;
        main_table[i][col_state2_inv] = inv_2p32_minus_1;
        main_table[i][col_state3_inv] = inv_2p32_minus_1;

        // Round 0 constants (16 constants) - unroll small loop for better performance
        for (size_t k = 0; k < Tip5::STATE_SIZE; ++k) {
            main_table[i][col_const0 + k] = round_consts[k];
        }

        // Mode + CI
        main_table[i][col_mode] = pad_mode;
        main_table[i][col_ci] = hash_opcode;
    }
}

// Cascade table padding
void pad_cascade_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    using namespace CascadeMainColumn;
    const BFieldElement one = BFieldElement::one();
    #pragma omp parallel for schedule(static)
    for (size_t i = table_len; i < padded_height; ++i) {
        main_table[i][CASCADE_TABLE_START + IsPadding] = one;
    }
}

// Lookup table padding
void pad_lookup_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    using namespace LookupMainColumn;
    const BFieldElement one = BFieldElement::one();
    #pragma omp parallel for schedule(static)
    for (size_t i = table_len; i < padded_height; ++i) {
        main_table[i][LOOKUP_TABLE_START + IsPadding] = one;
    }
}

// U32 table padding
void pad_u32_table(
    std::vector<std::vector<BFieldElement>>& main_table,
    size_t table_len,
    size_t padded_height
) {
    using namespace U32MainColumn;

    // Debug: Check rows 181-186 before padding
    const bool debug_u32_pad = (std::getenv("TVM_DEBUG_U32_PAD") != nullptr);
    if (debug_u32_pad) {
        std::cerr << "[DBG U32_PAD] table_len=" << table_len << ", padded_height=" << padded_height << std::endl;
        for (size_t r = 180; r <= 186 && r < main_table.size(); ++r) {
            if ((U32_TABLE_START + CI) < main_table[r].size()) {
                uint64_t ci_val = main_table[r][U32_TABLE_START + CI].value();
                uint64_t result_val = main_table[r][U32_TABLE_START + Result].value();
                std::cerr << "[DBG U32_PAD] Row " << r << ": CI=" << ci_val << ", Result=" << result_val << std::endl;
            }
        }
    }

    std::array<BFieldElement, U32_TABLE_COLS> padding_row{};
    padding_row.fill(BFieldElement::zero());

    const uint32_t split_opcode = TritonInstruction{AnInstruction::Split}.opcode();
    const uint32_t lt_opcode = TritonInstruction{AnInstruction::Lt}.opcode();

    padding_row[CI] = BFieldElement(static_cast<uint64_t>(split_opcode));
    padding_row[BitsMinus33Inv] = BFieldElement(BFieldElement::MODULUS - 33).inverse(); // (-33)^(-1)

    // Find the actual end of U32 table by scanning for the last row with CI set
    // This is more reliable than using table_len, which might be incorrect
    // Scan from the end of the table backwards to find the last row with U32 data
    size_t actual_u32_end = table_len;  // Default to table_len
    size_t last_filled_row = 0;
    bool found_filled = false;
    
    // Scan backwards from min(padded_height, main_table.size()) to find last row with CI set
    size_t scan_end = std::min(padded_height, main_table.size());
    for (size_t r = scan_end; r > 0; --r) {
        size_t check_row = r - 1;
        if (check_row < main_table.size() && 
            (U32_TABLE_START + CI) < main_table[check_row].size()) {
            BFieldElement ci_val = main_table[check_row][U32_TABLE_START + CI];
            if (!ci_val.is_zero()) {
                // Found the last row with U32 data
                actual_u32_end = r;  // r is one past the last filled row (check_row = r - 1)
                last_filled_row = check_row;
                found_filled = true;
                break;  // Found it, no need to continue
            }
        }
    }

    // Use the actual end if found, otherwise fall back to table_len
    size_t u32_table_end = found_filled ? actual_u32_end : table_len;

    if (found_filled) {
        padding_row[CI] = main_table[last_filled_row][U32_TABLE_START + CI];
        padding_row[LHS] = main_table[last_filled_row][U32_TABLE_START + LHS];
        padding_row[LhsInv] = main_table[last_filled_row][U32_TABLE_START + LhsInv];
        padding_row[Result] = main_table[last_filled_row][U32_TABLE_START + Result];

        if (padding_row[CI] == BFieldElement(static_cast<uint64_t>(lt_opcode))) {
            padding_row[Result] = BFieldElement(2);
        }
    } else if (table_len > 0) {
        // Fallback to original logic if no filled row found
        const size_t last_row = table_len - 1;
        if (last_row < main_table.size() && (U32_TABLE_START + CI) < main_table[last_row].size()) {
            padding_row[CI] = main_table[last_row][U32_TABLE_START + CI];
            padding_row[LHS] = main_table[last_row][U32_TABLE_START + LHS];
            padding_row[LhsInv] = main_table[last_row][U32_TABLE_START + LhsInv];
            padding_row[Result] = main_table[last_row][U32_TABLE_START + Result];

            if (padding_row[CI] == BFieldElement(static_cast<uint64_t>(lt_opcode))) {
                padding_row[Result] = BFieldElement(2);
            }
        }
    }

    // Parallelized padding loop - only pad rows that are truly empty
    // Use u32_table_end (actual end found by scanning) instead of table_len
    // CRITICAL: Also check if CI or Result is set - if so, the row already has U32 data and should NOT be overwritten
    #pragma omp parallel for schedule(static)
    for (size_t i = u32_table_end; i < padded_height; ++i) {
        if (i >= main_table.size()) continue;
        if ((U32_TABLE_START + CI) >= main_table[i].size()) continue;
        if ((U32_TABLE_START + Result) >= main_table[i].size()) continue;
        
        // Double-check: if row already has U32 data by checking CI or Result column, skip it
        // This is a safety check in case u32_table_end was computed incorrectly
        // We check both CI and Result because rows padded in from_aet might have Result set but CI might be from padding
        BFieldElement existing_ci = main_table[i][U32_TABLE_START + CI];
        BFieldElement existing_result = main_table[i][U32_TABLE_START + Result];
        
        // If CI is set, definitely skip (row has actual U32 data)
        // If Result is non-zero, also skip (row was padded in from_aet and should be preserved)
        if (!existing_ci.is_zero() || !existing_result.is_zero()) {
            // Row already has U32 data, don't overwrite
            continue;
        }
        
        // Row is empty (both CI and Result are zero), apply padding
        for (size_t c = 0; c < U32_TABLE_COLS; ++c) {
            if ((U32_TABLE_START + c) < main_table[i].size()) {
                main_table[i][U32_TABLE_START + c] = padding_row[c];
            }
        }
    }
    
    // Debug: Check rows 181-186 after padding
    if (debug_u32_pad) {
        std::cerr << "[DBG U32_PAD] After padding:" << std::endl;
        std::cerr << "[DBG U32_PAD] u32_table_end=" << u32_table_end << ", found_filled=" << found_filled << std::endl;
        for (size_t r = 180; r <= 186 && r < main_table.size(); ++r) {
            if ((U32_TABLE_START + CI) < main_table[r].size()) {
                uint64_t ci_val = main_table[r][U32_TABLE_START + CI].value();
                uint64_t result_val = main_table[r][U32_TABLE_START + Result].value();
                std::cerr << "[DBG U32_PAD] Row " << r << ": CI=" << ci_val << ", Result=" << result_val << std::endl;
            }
        }
    }
}

} // anonymous namespace

// Main padding function - calls all table-specific pad functions
void pad_all_tables(
    std::vector<std::vector<BFieldElement>>& main_table,
    const std::array<size_t, 9>& table_lengths,
    size_t padded_height
) {
    const bool profile = (std::getenv("TVM_PROFILE_PAD") != nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Table lengths: [Program, Processor, OpStack, Ram, JumpStack, Hash, Cascade, Lookup, U32]
    // Run all padding functions in parallel - they write to different column ranges
    // JumpStack is run separately since it modifies rows that other tables might read
    
    // Phase 1: Run JumpStack first (has row movement that could conflict)
    pad_jump_stack_table(main_table, table_lengths[4], padded_height);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "      JumpStack: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
    
    // Phase 2: Run all other tables in parallel (no conflicts - different column ranges)
    // Use TBB parallel_invoke for better thread management and work stealing
    // Falls back to std::thread if TBB is not available
#ifdef TVM_USE_TBB
    // TBB parallel_invoke: Better thread management, automatic work stealing
    tbb::parallel_invoke(
        [&]() { pad_program_table(main_table, table_lengths[0], padded_height); },
        [&]() { pad_processor_table(main_table, table_lengths[1], padded_height); },
        [&]() { pad_op_stack_table(main_table, table_lengths[2], padded_height); },
        [&]() { pad_ram_table(main_table, table_lengths[3], padded_height); },
        [&]() { pad_hash_table(main_table, table_lengths[5], padded_height); },
        [&]() { pad_cascade_table(main_table, table_lengths[6], padded_height); },
        [&]() { pad_lookup_table(main_table, table_lengths[7], padded_height); },
        [&]() { pad_u32_table(main_table, table_lengths[8], padded_height); }
    );
#else
    // Fallback to std::thread if TBB is not available
    std::vector<std::thread> threads;
    threads.reserve(8);

    threads.emplace_back([&]() { pad_program_table(main_table, table_lengths[0], padded_height); });
    threads.emplace_back([&]() { pad_processor_table(main_table, table_lengths[1], padded_height); });
    threads.emplace_back([&]() { pad_op_stack_table(main_table, table_lengths[2], padded_height); });
    threads.emplace_back([&]() { pad_ram_table(main_table, table_lengths[3], padded_height); });
    threads.emplace_back([&]() { pad_hash_table(main_table, table_lengths[5], padded_height); });
    threads.emplace_back([&]() { pad_cascade_table(main_table, table_lengths[6], padded_height); });
    threads.emplace_back([&]() { pad_lookup_table(main_table, table_lengths[7], padded_height); });
    threads.emplace_back([&]() { pad_u32_table(main_table, table_lengths[8], padded_height); });

    for (auto& t : threads) {
        t.join();
    }
#endif
    
    auto t2 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "      All others (parallel): " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
}

} // namespace triton_vm

