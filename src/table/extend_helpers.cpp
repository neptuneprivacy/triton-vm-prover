#include "table/extend_helpers.hpp"
#include "stark/challenges.hpp"
#include "hash/tip5.hpp"
#include <stdexcept>
#include <map>
#include <iostream>
#include <algorithm>
#include <optional>
#include <array>
#include <cstdlib>
#include <thread>
#include <vector>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TVM_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#endif

// SIMD support: AVX2 enabled via CMake compile flags
// The compiler will auto-vectorize unrolled loops when AVX2 is available
// No need for explicit intrinsics - rely on compiler optimization

namespace triton_vm {

using namespace TableColumnOffsets;
// NOTE: Do NOT `using namespace ProgramMainColumn/U32MainColumn` at file scope.
// Several tables reuse names like `LookupMultiplicity`, which becomes ambiguous when
// multiple column namespaces are in scope. Keep these `using` directives inside functions.

std::vector<BFieldElement> get_main_table_row(
    const std::vector<std::vector<BFieldElement>>& main_table,
    size_t row_idx,
    size_t start_col,
    size_t num_cols
) {
    if (row_idx >= main_table.size()) {
        throw std::out_of_range("Row index out of range");
    }
    if (start_col + num_cols > main_table[row_idx].size()) {
        throw std::out_of_range("Column range out of bounds");
    }
    
    std::vector<BFieldElement> result;
    result.reserve(num_cols);
    for (size_t i = 0; i < num_cols; i++) {
        result.push_back(main_table[row_idx][start_col + i]);
    }
    return result;
}

void set_aux_table_row(
    std::vector<std::vector<XFieldElement>>& aux_table,
    size_t row_idx,
    size_t start_col,
    const std::vector<XFieldElement>& values
) {
    if (row_idx >= aux_table.size()) {
        throw std::out_of_range("Row index out of range");
    }
    if (start_col + values.size() > aux_table[row_idx].size()) {
        throw std::out_of_range("Column range out of bounds");
    }
    
    for (size_t i = 0; i < values.size(); i++) {
        aux_table[row_idx][start_col + i] = values[i];
    }
}

static BFieldElement opcode_to_b_field(AnInstruction instr) {
    TritonInstruction placeholder{instr, BFieldElement::zero(), NumberOfWords::N1, OpStackElement::ST0};
    return BFieldElement(placeholder.opcode());
}

void extend_program_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ProgramMainColumn;
    using namespace ProgramAuxColumn;
    using namespace ChallengeId;
    
    size_t main_start = PROGRAM_TABLE_START;
    size_t aux_start = AUX_PROGRAM_TABLE_START;

    auto instruction_lookup_log_derivative_summand =
        [&](const std::vector<BFieldElement>& row,
            const std::vector<BFieldElement>& next_row) -> XFieldElement {
            XFieldElement compressed_row = XFieldElement(row[Address]) * challenges[ProgramAddressWeight]
                + XFieldElement(row[Instruction]) * challenges[ProgramInstructionWeight]
                + XFieldElement(next_row[Instruction]) * challenges[ProgramNextInstructionWeight];
            XFieldElement diff = challenges[InstructionLookupIndeterminate] - compressed_row;
            return diff.inverse() * XFieldElement(row[LookupMultiplicity]);
        };

    auto update_instruction_lookup_log_derivative =
        [&](const std::vector<BFieldElement>& row,
            const std::vector<BFieldElement>& next_row,
            const XFieldElement& current) -> XFieldElement {
            if (row[IsHashInputPadding].is_one()) {
                return current;
            }
            return current + instruction_lookup_log_derivative_summand(row, next_row);
        };

    auto update_prepare_chunk_running_evaluation =
        [&](const std::vector<BFieldElement>& row,
            const XFieldElement& current) -> XFieldElement {
            XFieldElement eval = current;
            if (row[IndexInChunk].is_zero()) {
                eval = EvalArg::default_initial();
            }
            return eval * challenges[ProgramAttestationPrepareChunkIndeterminate]
                + XFieldElement(row[Instruction]);
        };

    auto update_send_chunk_running_evaluation =
        [&](const std::vector<BFieldElement>& row,
            const XFieldElement& current,
            const XFieldElement& prepare_eval) -> XFieldElement {
            bool is_table_padding_row = row[IsTablePadding].is_one();
            uint64_t max_index_in_chunk = Tip5::RATE - 1;
            bool needs_update =
                !is_table_padding_row && row[IndexInChunk].value() == max_index_in_chunk;
            if (!needs_update) {
                return current;
            }
            return current * challenges[ProgramAttestationSendChunkIndeterminate] + prepare_eval;
        };

    XFieldElement instruction_lookup_log_derivative = LookupArg::default_initial();
    XFieldElement prepare_chunk_running_evaluation = EvalArg::default_initial();
    XFieldElement send_chunk_running_evaluation = EvalArg::default_initial();
    
    for (size_t idx = 0; idx + 1 < num_rows; ++idx) {
        auto row = get_main_table_row(main_table, idx, main_start, PROGRAM_TABLE_COLS);
        auto next_row = get_main_table_row(main_table, idx + 1, main_start, PROGRAM_TABLE_COLS);
        
        aux_table[idx][aux_start + InstructionLookupServerLogDerivative] = instruction_lookup_log_derivative;
        
        instruction_lookup_log_derivative =
            update_instruction_lookup_log_derivative(row, next_row, instruction_lookup_log_derivative);

        prepare_chunk_running_evaluation = 
            update_prepare_chunk_running_evaluation(row, prepare_chunk_running_evaluation);
        aux_table[idx][aux_start + PrepareChunkRunningEvaluation] = prepare_chunk_running_evaluation;
        
            send_chunk_running_evaluation = 
            update_send_chunk_running_evaluation(row, send_chunk_running_evaluation, prepare_chunk_running_evaluation);
        aux_table[idx][aux_start + SendChunkRunningEvaluation] = send_chunk_running_evaluation;
    }
    
    // Handle last row (padding row guaranteed to exist)
    size_t last_idx = num_rows - 1;
    auto last_row = get_main_table_row(main_table, last_idx, main_start, PROGRAM_TABLE_COLS);
    
    prepare_chunk_running_evaluation = 
        update_prepare_chunk_running_evaluation(last_row, prepare_chunk_running_evaluation);
        send_chunk_running_evaluation = 
        update_send_chunk_running_evaluation(last_row, send_chunk_running_evaluation, prepare_chunk_running_evaluation);

    aux_table[last_idx][aux_start + InstructionLookupServerLogDerivative] = instruction_lookup_log_derivative;
    aux_table[last_idx][aux_start + PrepareChunkRunningEvaluation] = prepare_chunk_running_evaluation;
    aux_table[last_idx][aux_start + SendChunkRunningEvaluation] = send_chunk_running_evaluation;
}

void extend_program_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ProgramMainColumn;
    using namespace ProgramAuxColumn;
    using namespace ChallengeId;

    const size_t main_start = PROGRAM_TABLE_START;
    const size_t aux_start = AUX_PROGRAM_TABLE_START;

    auto at = [&](size_t r, size_t rel) -> BFieldElement {
        return main_table.at(r, main_start + rel);
    };

    auto instruction_lookup_log_derivative_summand =
        [&](size_t idx, size_t next_idx) -> XFieldElement {
            XFieldElement compressed_row =
                XFieldElement(at(idx, Address)) * challenges[ProgramAddressWeight]
                + XFieldElement(at(idx, Instruction)) * challenges[ProgramInstructionWeight]
                + XFieldElement(at(next_idx, Instruction)) * challenges[ProgramNextInstructionWeight];
            XFieldElement diff = challenges[InstructionLookupIndeterminate] - compressed_row;
            return diff.inverse() * XFieldElement(at(idx, LookupMultiplicity));
        };

    auto update_instruction_lookup_log_derivative =
        [&](size_t idx, size_t next_idx, const XFieldElement& current) -> XFieldElement {
            if (at(idx, IsHashInputPadding).is_one()) return current;
            return current + instruction_lookup_log_derivative_summand(idx, next_idx);
        };

    auto update_prepare_chunk_running_evaluation =
        [&](size_t idx, const XFieldElement& current) -> XFieldElement {
            XFieldElement eval = current;
            if (at(idx, IndexInChunk).is_zero()) {
                eval = EvalArg::default_initial();
            }
            return eval * challenges[ProgramAttestationPrepareChunkIndeterminate]
                + XFieldElement(at(idx, Instruction));
        };

    auto update_send_chunk_running_evaluation =
        [&](size_t idx, const XFieldElement& current, const XFieldElement& prepare_eval) -> XFieldElement {
            bool is_table_padding_row = at(idx, IsTablePadding).is_one();
            uint64_t max_index_in_chunk = Tip5::RATE - 1;
            bool needs_update = !is_table_padding_row && at(idx, IndexInChunk).value() == max_index_in_chunk;
            if (!needs_update) return current;
            return current * challenges[ProgramAttestationSendChunkIndeterminate] + prepare_eval;
        };

    XFieldElement instruction_lookup_log_derivative = LookupArg::default_initial();
    XFieldElement prepare_chunk_running_evaluation = EvalArg::default_initial();
    XFieldElement send_chunk_running_evaluation = EvalArg::default_initial();

    for (size_t idx = 0; idx + 1 < num_rows; ++idx) {
        aux_table[idx][aux_start + InstructionLookupServerLogDerivative] = instruction_lookup_log_derivative;
        instruction_lookup_log_derivative =
            update_instruction_lookup_log_derivative(idx, idx + 1, instruction_lookup_log_derivative);

        prepare_chunk_running_evaluation =
            update_prepare_chunk_running_evaluation(idx, prepare_chunk_running_evaluation);
        aux_table[idx][aux_start + PrepareChunkRunningEvaluation] = prepare_chunk_running_evaluation;

        send_chunk_running_evaluation =
            update_send_chunk_running_evaluation(idx, send_chunk_running_evaluation, prepare_chunk_running_evaluation);
        aux_table[idx][aux_start + SendChunkRunningEvaluation] = send_chunk_running_evaluation;
    }

    // Handle last row
    size_t last_idx = num_rows - 1;
    prepare_chunk_running_evaluation =
        update_prepare_chunk_running_evaluation(last_idx, prepare_chunk_running_evaluation);
    send_chunk_running_evaluation =
        update_send_chunk_running_evaluation(last_idx, send_chunk_running_evaluation, prepare_chunk_running_evaluation);

    aux_table[last_idx][aux_start + InstructionLookupServerLogDerivative] = instruction_lookup_log_derivative;
    aux_table[last_idx][aux_start + PrepareChunkRunningEvaluation] = prepare_chunk_running_evaluation;
    aux_table[last_idx][aux_start + SendChunkRunningEvaluation] = send_chunk_running_evaluation;
}

// OpStackMainColumn indices are declared in `include/table/extend_helpers.hpp`.

// OpStack table aux column indices (relative to AUX_OP_STACK_TABLE_START)
namespace OpStackAuxColumn {
    constexpr size_t RunningProductPermArg = 0;
    constexpr size_t ClockJumpDifferenceLookupLogDerivative = 1;
    // Note: Only the first 2 columns are actually computed (matching Rust).
    // The remaining 8 columns are left as zeros for compatibility with test data structure.
}

// =============================================================================
// OPTIMIZED extend_op_stack_table
// =============================================================================
// Optimizations:
// 1. Direct array access instead of get_main_table_row copies
// 2. Fixed array cache for common inverse values (0-99)
// =============================================================================

void extend_op_stack_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace OpStackMainColumn;
    using namespace OpStackAuxColumn;

    const size_t main_start = OP_STACK_TABLE_START;
    const size_t aux_start = AUX_OP_STACK_TABLE_START;
    constexpr BFieldElement PADDING_VALUE(2);

    // Cache challenges
    const XFieldElement perm_arg_indeterminate = challenges[OpStackIndeterminate];
    const XFieldElement clk_weight = challenges[OpStackClkWeight];
    const XFieldElement ib1_weight = challenges[OpStackIb1Weight];
    const XFieldElement ptr_weight = challenges[OpStackPointerWeight];
    const XFieldElement underflow_weight = challenges[OpStackFirstUnderflowElementWeight];
    const XFieldElement cjd_indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];

    // Pre-compute inverses for common values 0-99
    std::array<XFieldElement, 100> cached_inv;
    for (uint64_t i = 0; i < 100; i++) {
        cached_inv[i] = (cjd_indeterminate - XFieldElement(BFieldElement(i))).inverse();
    }

    XFieldElement running_product = PermArg::default_initial();
    XFieldElement cjd_log_deriv = LookupArg::default_initial();
    size_t padding_start = num_rows;

    // First pass: running product (find padding boundary)
    for (size_t idx = 0; idx < num_rows; idx++) {
        const auto& row = main_table[idx];
        
        if (row[main_start + IB1ShrinkStack] != PADDING_VALUE) {
            XFieldElement compressed = XFieldElement(row[main_start + CLK]) * clk_weight
                + XFieldElement(row[main_start + IB1ShrinkStack]) * ib1_weight
                + XFieldElement(row[main_start + StackPointer]) * ptr_weight
                + XFieldElement(row[main_start + FirstUnderflowElement]) * underflow_weight;
            running_product = running_product * (perm_arg_indeterminate - compressed);
        } else if (padding_start == num_rows) {
            padding_start = idx;
        }
        aux_table[idx][aux_start + RunningProductPermArg] = running_product;
    }

    // Second pass: CJD log derivative (up to padding)
    aux_table[0][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    for (size_t idx = 1; idx < padding_start; idx++) {
        const auto& curr = main_table[idx];
        const auto& prev = main_table[idx - 1];
        
        if (prev[main_start + StackPointer] == curr[main_start + StackPointer]) {
            uint64_t diff = (curr[main_start + CLK] - prev[main_start + CLK]).value();
            XFieldElement inv = (diff < 100) ? cached_inv[diff]
                : (cjd_indeterminate - XFieldElement(BFieldElement(diff))).inverse();
            cjd_log_deriv += inv;
        }
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    }
    
    // Fill padding with last value
    for (size_t idx = padding_start; idx < num_rows; idx++) {
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    }
}

void extend_op_stack_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace OpStackMainColumn;
    using namespace OpStackAuxColumn;

    const size_t main_start = OP_STACK_TABLE_START;
    const size_t aux_start = AUX_OP_STACK_TABLE_START;
    constexpr BFieldElement PADDING_VALUE(2);

    auto at = [&](size_t r, size_t rel) -> BFieldElement {
        return main_table.at(r, main_start + rel);
    };

    // Cache challenges
    const XFieldElement perm_arg_indeterminate = challenges[OpStackIndeterminate];
    const XFieldElement clk_weight = challenges[OpStackClkWeight];
    const XFieldElement ib1_weight = challenges[OpStackIb1Weight];
    const XFieldElement ptr_weight = challenges[OpStackPointerWeight];
    const XFieldElement underflow_weight = challenges[OpStackFirstUnderflowElementWeight];
    const XFieldElement cjd_indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];

    // Pre-compute inverses for common values 0-99
    std::array<XFieldElement, 100> cached_inv;
    for (uint64_t i = 0; i < 100; i++) {
        cached_inv[i] = (cjd_indeterminate - XFieldElement(BFieldElement(i))).inverse();
    }

    XFieldElement running_product = PermArg::default_initial();
    XFieldElement cjd_log_deriv = LookupArg::default_initial();
    size_t padding_start = num_rows;

    // First pass: running product (find padding boundary)
    for (size_t idx = 0; idx < num_rows; idx++) {
        if (at(idx, IB1ShrinkStack) != PADDING_VALUE) {
            XFieldElement compressed = XFieldElement(at(idx, CLK)) * clk_weight
                + XFieldElement(at(idx, IB1ShrinkStack)) * ib1_weight
                + XFieldElement(at(idx, StackPointer)) * ptr_weight
                + XFieldElement(at(idx, FirstUnderflowElement)) * underflow_weight;
            running_product = running_product * (perm_arg_indeterminate - compressed);
        } else if (padding_start == num_rows) {
            padding_start = idx;
        }
        aux_table[idx][aux_start + RunningProductPermArg] = running_product;
    }

    // Second pass: CJD log derivative (up to padding)
    aux_table[0][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    for (size_t idx = 1; idx < padding_start; idx++) {
        if (at(idx - 1, StackPointer) == at(idx, StackPointer)) {
            uint64_t diff = (at(idx, CLK) - at(idx - 1, CLK)).value();
            XFieldElement inv = (diff < 100) ? cached_inv[diff]
                : (cjd_indeterminate - XFieldElement(BFieldElement(diff))).inverse();
            cjd_log_deriv += inv;
        }
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    }

    // Fill padding with last value
    for (size_t idx = padding_start; idx < num_rows; idx++) {
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    }
}

// JumpStackMainColumn indices are declared in `include/table/extend_helpers.hpp`.

// JumpStack table aux column indices (relative to AUX_JUMP_STACK_TABLE_START)
namespace JumpStackAuxColumn {
    constexpr size_t RunningProductPermArg = 0;
    constexpr size_t ClockJumpDifferenceLookupLogDerivative = 1;
}

// Keep direct array access but use cached inverses (faster for small diffs)
void extend_jump_stack_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace JumpStackMainColumn;
    using namespace JumpStackAuxColumn;
    
    const size_t main_start = JUMP_STACK_TABLE_START;
    const size_t aux_start = AUX_JUMP_STACK_TABLE_START;
    
    // Cache challenges
    const XFieldElement perm_arg_indeterminate = challenges[JumpStackIndeterminate];
    const XFieldElement clk_weight = challenges[JumpStackClkWeight];
    const XFieldElement ci_weight = challenges[JumpStackCiWeight];
    const XFieldElement jsp_weight = challenges[JumpStackJspWeight];
    const XFieldElement jso_weight = challenges[JumpStackJsoWeight];
    const XFieldElement jsd_weight = challenges[JumpStackJsdWeight];
    const XFieldElement cjd_indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];
    
    // Pre-compute inverses for common values 0-99
    std::array<XFieldElement, 100> cached_inv;
    for (uint64_t i = 0; i < 100; i++) {
        cached_inv[i] = (cjd_indeterminate - XFieldElement(BFieldElement(i))).inverse();
    }
    
    XFieldElement running_product = PermArg::default_initial();
    XFieldElement cjd_log_deriv = LookupArg::default_initial();
    
    for (size_t idx = 0; idx < num_rows; idx++) {
        const auto& row = main_table[idx];
        
        // RunningProductPermArg - direct array access
        XFieldElement compressed = XFieldElement(row[main_start + CLK]) * clk_weight
            + XFieldElement(row[main_start + CI]) * ci_weight
            + XFieldElement(row[main_start + JSP]) * jsp_weight
            + XFieldElement(row[main_start + JSO]) * jso_weight
            + XFieldElement(row[main_start + JSD]) * jsd_weight;
        running_product = running_product * (perm_arg_indeterminate - compressed);
        aux_table[idx][aux_start + RunningProductPermArg] = running_product;
        
        // CJD log derivative
        if (idx > 0) {
            const auto& prev = main_table[idx - 1];
            if (prev[main_start + JSP] == row[main_start + JSP]) {
                uint64_t diff = (row[main_start + CLK] - prev[main_start + CLK]).value();
                XFieldElement inv = (diff < 100) ? cached_inv[diff] 
                    : (cjd_indeterminate - XFieldElement(BFieldElement(diff))).inverse();
                cjd_log_deriv += inv;
            }
        }
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    }
}

void extend_jump_stack_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace JumpStackMainColumn;
    using namespace JumpStackAuxColumn;

    const size_t main_start = JUMP_STACK_TABLE_START;
    const size_t aux_start = AUX_JUMP_STACK_TABLE_START;

    auto at = [&](size_t r, size_t rel) -> BFieldElement {
        return main_table.at(r, main_start + rel);
    };

    // Cache challenges
    const XFieldElement perm_arg_indeterminate = challenges[JumpStackIndeterminate];
    const XFieldElement clk_weight = challenges[JumpStackClkWeight];
    const XFieldElement ci_weight = challenges[JumpStackCiWeight];
    const XFieldElement jsp_weight = challenges[JumpStackJspWeight];
    const XFieldElement jso_weight = challenges[JumpStackJsoWeight];
    const XFieldElement jsd_weight = challenges[JumpStackJsdWeight];
    const XFieldElement cjd_indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];

    // Pre-compute inverses for common values 0-99
    std::array<XFieldElement, 100> cached_inv;
    for (uint64_t i = 0; i < 100; i++) {
        cached_inv[i] = (cjd_indeterminate - XFieldElement(BFieldElement(i))).inverse();
    }

    XFieldElement running_product = PermArg::default_initial();
    XFieldElement cjd_log_deriv = LookupArg::default_initial();

    for (size_t idx = 0; idx < num_rows; idx++) {
        XFieldElement compressed = XFieldElement(at(idx, CLK)) * clk_weight
            + XFieldElement(at(idx, CI)) * ci_weight
            + XFieldElement(at(idx, JSP)) * jsp_weight
            + XFieldElement(at(idx, JSO)) * jso_weight
            + XFieldElement(at(idx, JSD)) * jsd_weight;
        running_product = running_product * (perm_arg_indeterminate - compressed);
        aux_table[idx][aux_start + RunningProductPermArg] = running_product;

        if (idx > 0) {
            if (at(idx - 1, JSP) == at(idx, JSP)) {
                uint64_t diff = (at(idx, CLK) - at(idx - 1, CLK)).value();
                XFieldElement inv = (diff < 100) ? cached_inv[diff]
                    : (cjd_indeterminate - XFieldElement(BFieldElement(diff))).inverse();
                cjd_log_deriv += inv;
            }
        }
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_deriv;
    }
}

// Lookup table main column indices
// LookupMainColumn indices are declared in `include/table/extend_helpers.hpp`.

// Lookup table aux column indices (relative to AUX_LOOKUP_TABLE_START)
namespace LookupAuxColumn {
    constexpr size_t CascadeRunningSumLogDerivative = 0;
    constexpr size_t PublicRunningEvaluation = 1;
}

void extend_lookup_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    
    size_t main_start = LOOKUP_TABLE_START;
    size_t aux_start = AUX_LOOKUP_TABLE_START;
    
    XFieldElement cascade_running_sum = LookupArg::default_initial();
    XFieldElement cascade_lookup_indeterminate = challenges[CascadeLookupIndeterminate];
    
    XFieldElement public_running_eval = EvalArg::default_initial();
    XFieldElement lookup_public_indeterminate = challenges[LookupTablePublicIndeterminate];
    
    for (size_t idx = 0; idx < num_rows; idx++) {
        auto row = get_main_table_row(main_table, idx, main_start, LOOKUP_TABLE_COLS);
        bool is_padding = row[LookupMainColumn::IsPadding].is_one();
        if (is_padding) {
            for (size_t fill_idx = idx; fill_idx < num_rows; ++fill_idx) {
                aux_table[fill_idx][aux_start + LookupAuxColumn::CascadeRunningSumLogDerivative] = cascade_running_sum;
                aux_table[fill_idx][aux_start + LookupAuxColumn::PublicRunningEvaluation] = public_running_eval;
            }
            break;
        }

        XFieldElement compressed_row =
            XFieldElement(row[LookupMainColumn::LookIn]) * challenges[LookupTableInputWeight] +
            XFieldElement(row[LookupMainColumn::LookOut]) * challenges[LookupTableOutputWeight];
        XFieldElement diff = cascade_lookup_indeterminate - compressed_row;
        cascade_running_sum = cascade_running_sum + diff.inverse() * row[LookupMainColumn::LookupMultiplicity];

        public_running_eval = public_running_eval * lookup_public_indeterminate + XFieldElement(row[LookupMainColumn::LookOut]);

        aux_table[idx][aux_start + LookupAuxColumn::CascadeRunningSumLogDerivative] = cascade_running_sum;
        aux_table[idx][aux_start + LookupAuxColumn::PublicRunningEvaluation] = public_running_eval;
    }
}

void extend_lookup_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace LookupMainColumn;
    using namespace LookupAuxColumn;

    const size_t main_start = LOOKUP_TABLE_START;
    const size_t aux_start = AUX_LOOKUP_TABLE_START;

    auto at = [&](size_t r, size_t rel) -> BFieldElement { return main_table.at(r, main_start + rel); };

    XFieldElement cascade_running_sum = LookupArg::default_initial();
    XFieldElement cascade_lookup_indeterminate = challenges[CascadeLookupIndeterminate];

    XFieldElement public_running_eval = EvalArg::default_initial();
    XFieldElement lookup_public_indeterminate = challenges[LookupTablePublicIndeterminate];

    for (size_t idx = 0; idx < num_rows; idx++) {
        bool is_padding = at(idx, IsPadding).is_one();
        if (is_padding) {
            for (size_t fill_idx = idx; fill_idx < num_rows; ++fill_idx) {
                aux_table[fill_idx][aux_start + CascadeRunningSumLogDerivative] = cascade_running_sum;
                aux_table[fill_idx][aux_start + PublicRunningEvaluation] = public_running_eval;
            }
            break;
        }

        XFieldElement compressed_row =
            XFieldElement(at(idx, LookIn)) * challenges[LookupTableInputWeight] +
            XFieldElement(at(idx, LookOut)) * challenges[LookupTableOutputWeight];
        XFieldElement diff = cascade_lookup_indeterminate - compressed_row;
        cascade_running_sum = cascade_running_sum + diff.inverse() * XFieldElement(at(idx, LookupMultiplicity));

        public_running_eval = public_running_eval * lookup_public_indeterminate + XFieldElement(at(idx, LookOut));

        aux_table[idx][aux_start + CascadeRunningSumLogDerivative] = cascade_running_sum;
        aux_table[idx][aux_start + PublicRunningEvaluation] = public_running_eval;
    }
}

// Hash table main column indices (relative to HASH_TABLE_START)
// HashMainColumn indices are declared in `include/table/extend_helpers.hpp`.

// Hash table aux column indices (relative to AUX_HASH_TABLE_START)
namespace HashAuxColumn {
    constexpr size_t ReceiveChunkRunningEvaluation = 0;
    constexpr size_t HashInputRunningEvaluation = 1;
    constexpr size_t HashDigestRunningEvaluation = 2;
    constexpr size_t SpongeRunningEvaluation = 3;
    constexpr size_t CascadeState0HighestClientLogDerivative = 4;
    constexpr size_t CascadeState0MidHighClientLogDerivative = 5;
    constexpr size_t CascadeState0MidLowClientLogDerivative = 6;
    constexpr size_t CascadeState0LowestClientLogDerivative = 7;
    constexpr size_t CascadeState1HighestClientLogDerivative = 8;
    constexpr size_t CascadeState1MidHighClientLogDerivative = 9;
    constexpr size_t CascadeState1MidLowClientLogDerivative = 10;
    constexpr size_t CascadeState1LowestClientLogDerivative = 11;
    constexpr size_t CascadeState2HighestClientLogDerivative = 12;
    constexpr size_t CascadeState2MidHighClientLogDerivative = 13;
    constexpr size_t CascadeState2MidLowClientLogDerivative = 14;
    constexpr size_t CascadeState2LowestClientLogDerivative = 15;
    constexpr size_t CascadeState3HighestClientLogDerivative = 16;
    constexpr size_t CascadeState3MidHighClientLogDerivative = 17;
    constexpr size_t CascadeState3MidLowClientLogDerivative = 18;
    constexpr size_t CascadeState3LowestClientLogDerivative = 19;
}

// Montgomery modulus R = 2^64 mod p (from Rust: MONTGOMERY_MODULUS)
// Computed as: (1_u128 << 64) % BFieldElement::MODULUS
// P = 2^64 - 2^32 + 1 = 18446744069414584321
// R = (2^64) % P = 18446744073709551616 % 18446744069414584321 = 4294967295
static const BFieldElement MONTGOMERY_MODULUS(4294967295ULL);

// =============================================================================
// OPTIMIZED extend_hash_table
// =============================================================================
// Optimizations:
// 1. CROSS-ROW BATCH INVERSION for cascade log derivatives (16 batch inversions total vs R per-row inversions)
// 2. PARALLEL cascade processing using TBB (16 cascades processed in parallel)
// 3. PARALLEL threads for independent column groups
// 4. Direct array access instead of get_main_table_row copies
// =============================================================================

void extend_hash_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace HashAuxColumn;
    using namespace HashMainColumn;
    
    const size_t main_start = HASH_TABLE_START;
    const size_t aux_start = AUX_HASH_TABLE_START;
    
    // Cache challenge values
    const XFieldElement ci_weight = challenges[HashCIWeight];
    const XFieldElement hash_input_eval_indeterminate = challenges[HashInputIndeterminate];
    const XFieldElement hash_digest_eval_indeterminate = challenges[HashDigestIndeterminate];
    const XFieldElement sponge_eval_indeterminate = challenges[SpongeIndeterminate];
    const XFieldElement cascade_indeterminate = challenges[HashCascadeLookupIndeterminate];
    const XFieldElement cascade_look_in_weight = challenges[HashCascadeLookInWeight];
    const XFieldElement cascade_look_out_weight = challenges[HashCascadeLookOutWeight];
    const XFieldElement send_chunk_indeterminate = challenges[ProgramAttestationSendChunkIndeterminate];
    const XFieldElement prepare_chunk_indeterminate = challenges[ProgramAttestationPrepareChunkIndeterminate];

    std::array<XFieldElement, Tip5::RATE> state_weights;
    for (size_t i = 0; i < Tip5::RATE; ++i) {
        state_weights[i] = challenges[StackWeight0 + i];
    }

    const BFieldElement montgomery_modulus_inverse = MONTGOMERY_MODULUS.inverse();
    const BFieldElement two_pow_16(1ULL << 16);
    const BFieldElement two_pow_32(1ULL << 32);
    const BFieldElement two_pow_48(1ULL << 48);
    const BFieldElement last_round_value(static_cast<uint64_t>(Tip5::NUM_ROUNDS));
    const BFieldElement sponge_init_opcode = opcode_to_b_field(AnInstruction::SpongeInit);

    // Column indices for cascade log derivatives (16 pairs of in/out)
    constexpr std::array<std::pair<size_t, size_t>, 16> CASCADE_COLS = {{
        {State0HighestLkIn, State0HighestLkOut}, {State0MidHighLkIn, State0MidHighLkOut},
        {State0MidLowLkIn, State0MidLowLkOut}, {State0LowestLkIn, State0LowestLkOut},
        {State1HighestLkIn, State1HighestLkOut}, {State1MidHighLkIn, State1MidHighLkOut},
        {State1MidLowLkIn, State1MidLowLkOut}, {State1LowestLkIn, State1LowestLkOut},
        {State2HighestLkIn, State2HighestLkOut}, {State2MidHighLkIn, State2MidHighLkOut},
        {State2MidLowLkIn, State2MidLowLkOut}, {State2LowestLkIn, State2LowestLkOut},
        {State3HighestLkIn, State3HighestLkOut}, {State3MidHighLkIn, State3MidHighLkOut},
        {State3MidLowLkIn, State3MidLowLkOut}, {State3LowestLkIn, State3LowestLkOut}
    }};
    
    constexpr std::array<size_t, 16> CASCADE_AUX_COLS = {
        CascadeState0HighestClientLogDerivative, CascadeState0MidHighClientLogDerivative,
        CascadeState0MidLowClientLogDerivative, CascadeState0LowestClientLogDerivative,
        CascadeState1HighestClientLogDerivative, CascadeState1MidHighClientLogDerivative,
        CascadeState1MidLowClientLogDerivative, CascadeState1LowestClientLogDerivative,
        CascadeState2HighestClientLogDerivative, CascadeState2MidHighClientLogDerivative,
        CascadeState2MidLowClientLogDerivative, CascadeState2LowestClientLogDerivative,
        CascadeState3HighestClientLogDerivative, CascadeState3MidHighClientLogDerivative,
        CascadeState3MidLowClientLogDerivative, CascadeState3LowestClientLogDerivative
    };

    // ==========================================================================
    // PASS 1: Compute cascade log derivatives with CROSS-ROW batch inversion (optimized)
    // ==========================================================================
    std::vector<std::thread> threads;

    // ---- Thread Group 1: Cascade log derivatives (16 columns) ----
    // OPTIMIZATION: Process all contributing rows for each cascade, then batch invert all at once
    // This reduces inversions from R (number of contributing rows) to 16 (number of cascades)
    threads.emplace_back([&]() {
        // Pre-compute which rows contribute (to avoid repeated checks)
        std::vector<bool> row_contributes(num_rows, false);
        for (size_t idx = 0; idx < num_rows; idx++) {
            const auto& row = main_table[idx];
            BFieldElement mode = row[main_start + Mode];
            BFieldElement round_number = row[main_start + RoundNumber];
            BFieldElement current_instruction = row[main_start + CI];

            bool in_pad_mode = mode == BFieldElement(HashTableMode::Pad);
            bool in_last_round = round_number == last_round_value;
            bool is_sponge_init = current_instruction == sponge_init_opcode;

            row_contributes[idx] = (!in_pad_mode && !in_last_round && !is_sponge_init);
        }

        // Collect contributing row indices (for efficient iteration)
        std::vector<size_t> contributing_rows;
        contributing_rows.reserve(num_rows);
        for (size_t idx = 0; idx < num_rows; idx++) {
            if (row_contributes[idx]) {
                contributing_rows.push_back(idx);
            }
        }

#ifdef TVM_USE_TBB
        // OPTIMIZATION: Process all 16 cascades in parallel using TBB
        tbb::parallel_for(size_t(0), size_t(16), [&](size_t c) {
            const auto& [lk_in, lk_out] = CASCADE_COLS[c];
            const size_t aux_col = aux_start + CASCADE_AUX_COLS[c];

            // Step 1: Collect all compressed values for this cascade from contributing rows
            std::vector<XFieldElement> compressed_values;
            compressed_values.reserve(contributing_rows.size());

            for (size_t contrib_idx : contributing_rows) {
                const auto& row = main_table[contrib_idx];
                XFieldElement compressed = cascade_indeterminate
                    - cascade_look_in_weight * XFieldElement(row[main_start + lk_in])
                    - cascade_look_out_weight * XFieldElement(row[main_start + lk_out]);
                compressed_values.push_back(compressed);
            }

            // Step 2: Batch invert all compressed values at once (Montgomery's trick)
            std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(compressed_values);

            // Step 3: Compute prefix sum (log derivative) and write to aux table
            XFieldElement running_sum = XFieldElement::zero();
            size_t inv_idx = 0;

            for (size_t idx = 0; idx < num_rows; idx++) {
                if (row_contributes[idx] && inv_idx < inverses.size()) {
                    // This row contributes, add its inverse to the running sum
                    running_sum += inverses[inv_idx];
                    inv_idx++;
                }
                // Write running sum for this row (even if it doesn't contribute, it gets the current sum)
                aux_table[idx][aux_col] = running_sum;
            }
        });
#else
        // Fallback: Sequential processing if TBB not available
        for (size_t c = 0; c < 16; ++c) {
            const auto& [lk_in, lk_out] = CASCADE_COLS[c];
            const size_t aux_col = aux_start + CASCADE_AUX_COLS[c];

            // Step 1: Collect all compressed values for this cascade
            std::vector<XFieldElement> compressed_values;
            compressed_values.reserve(contributing_rows.size());

            for (size_t contrib_idx : contributing_rows) {
                const auto& row = main_table[contrib_idx];
                XFieldElement compressed = cascade_indeterminate
                    - cascade_look_in_weight * XFieldElement(row[main_start + lk_in])
                    - cascade_look_out_weight * XFieldElement(row[main_start + lk_out]);
                compressed_values.push_back(compressed);
            }

            // Step 2: Batch invert all compressed values at once
            std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(compressed_values);

            // Step 3: Compute prefix sum and write to aux table
            XFieldElement running_sum = XFieldElement::zero();
            size_t inv_idx = 0;

            for (size_t idx = 0; idx < num_rows; idx++) {
                if (row_contributes[idx] && inv_idx < inverses.size()) {
                    running_sum += inverses[inv_idx];
                    inv_idx++;
                }
                aux_table[idx][aux_col] = running_sum;
            }
        }
#endif
    });
    
    // ---- Thread Group 2: receive_chunk, hash_input, hash_digest, sponge ----
    threads.emplace_back([&]() {
        XFieldElement receive_chunk = EvalArg::default_initial();
        XFieldElement hash_input = EvalArg::default_initial();
        XFieldElement hash_digest = EvalArg::default_initial();
        XFieldElement sponge = EvalArg::default_initial();
        
        auto re_compose = [&](const std::vector<BFieldElement>& row, size_t h, size_t mh, size_t ml, size_t l) {
            return (row[main_start + h] * two_pow_48 + row[main_start + mh] * two_pow_32 +
                    row[main_start + ml] * two_pow_16 + row[main_start + l]) * montgomery_modulus_inverse;
        };
        
        auto get_regs = [&](const std::vector<BFieldElement>& row) -> std::array<BFieldElement, 10> {
            return {
                re_compose(row, State0HighestLkIn, State0MidHighLkIn, State0MidLowLkIn, State0LowestLkIn),
                re_compose(row, State1HighestLkIn, State1MidHighLkIn, State1MidLowLkIn, State1LowestLkIn),
                re_compose(row, State2HighestLkIn, State2MidHighLkIn, State2MidLowLkIn, State2LowestLkIn),
                re_compose(row, State3HighestLkIn, State3MidHighLkIn, State3MidLowLkIn, State3LowestLkIn),
                row[main_start + State4], row[main_start + State5], row[main_start + State6],
                row[main_start + State7], row[main_start + State8], row[main_start + State9]
            };
        };
        
        auto compress = [&](const std::array<BFieldElement, 10>& regs) -> XFieldElement {
            XFieldElement acc = XFieldElement::zero();
            for (size_t i = 0; i < 10; ++i) acc += state_weights[i] * XFieldElement(regs[i]);
            return acc;
        };
    
    for (size_t idx = 0; idx < num_rows; idx++) {
            const auto& row = main_table[idx];
            BFieldElement mode = row[main_start + Mode];
            BFieldElement round_number = row[main_start + RoundNumber];
            BFieldElement current_instruction = row[main_start + CI];
            
            bool in_program_hashing = mode == BFieldElement(HashTableMode::ProgramHashing);
            bool in_sponge = mode == BFieldElement(HashTableMode::Sponge);
            bool in_hash = mode == BFieldElement(HashTableMode::Hash);
        bool in_round_0 = round_number.is_zero();
        bool in_last_round = round_number == last_round_value;
            bool is_sponge_init = current_instruction == sponge_init_opcode;
            
            if (in_program_hashing && in_round_0) {
                auto regs = get_regs(row);
            std::vector<BFieldElement> regs_vec(regs.begin(), regs.end());
                XFieldElement compressed_chunk = EvalArg::compute_terminal(
                    regs_vec, EvalArg::default_initial(), prepare_chunk_indeterminate);
                receive_chunk = receive_chunk * send_chunk_indeterminate + compressed_chunk;
            }
            
            if (in_sponge && in_round_0) {
                if (is_sponge_init) {
                    sponge = sponge * sponge_eval_indeterminate + ci_weight * XFieldElement(current_instruction);
                } else {
                    auto regs = get_regs(row);
                    sponge = sponge * sponge_eval_indeterminate + ci_weight * XFieldElement(current_instruction) + compress(regs);
                }
            }
            
            if (in_hash && in_round_0) {
                auto regs = get_regs(row);
                hash_input = hash_input * hash_input_eval_indeterminate + compress(regs);
            }
            
            if (in_hash && in_last_round) {
                auto regs = get_regs(row);
                XFieldElement digest = XFieldElement::zero();
                for (size_t j = 0; j < Digest::LEN; ++j) digest += state_weights[j] * XFieldElement(regs[j]);
                hash_digest = hash_digest * hash_digest_eval_indeterminate + digest;
            }
            
            aux_table[idx][aux_start + ReceiveChunkRunningEvaluation] = receive_chunk;
            aux_table[idx][aux_start + HashInputRunningEvaluation] = hash_input;
            aux_table[idx][aux_start + HashDigestRunningEvaluation] = hash_digest;
            aux_table[idx][aux_start + SpongeRunningEvaluation] = sponge;
        }
    });
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
}

void extend_hash_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace HashAuxColumn;
    using namespace HashMainColumn;

    const size_t main_start = HASH_TABLE_START;
    const size_t aux_start = AUX_HASH_TABLE_START;

    // Cache challenge values
    const XFieldElement ci_weight = challenges[HashCIWeight];
    const XFieldElement hash_input_eval_indeterminate = challenges[HashInputIndeterminate];
    const XFieldElement hash_digest_eval_indeterminate = challenges[HashDigestIndeterminate];
    const XFieldElement sponge_eval_indeterminate = challenges[SpongeIndeterminate];
    const XFieldElement cascade_indeterminate = challenges[HashCascadeLookupIndeterminate];
    const XFieldElement cascade_look_in_weight = challenges[HashCascadeLookInWeight];
    const XFieldElement cascade_look_out_weight = challenges[HashCascadeLookOutWeight];
    const XFieldElement send_chunk_indeterminate = challenges[ProgramAttestationSendChunkIndeterminate];
    const XFieldElement prepare_chunk_indeterminate = challenges[ProgramAttestationPrepareChunkIndeterminate];

    std::array<XFieldElement, Tip5::RATE> state_weights;
    for (size_t i = 0; i < Tip5::RATE; ++i) {
        state_weights[i] = challenges[StackWeight0 + i];
    }

    const BFieldElement montgomery_modulus_inverse = MONTGOMERY_MODULUS.inverse();
    const BFieldElement two_pow_16(1ULL << 16);
    const BFieldElement two_pow_32(1ULL << 32);
    const BFieldElement two_pow_48(1ULL << 48);
    const BFieldElement last_round_value(static_cast<uint64_t>(Tip5::NUM_ROUNDS));
    const BFieldElement sponge_init_opcode = opcode_to_b_field(AnInstruction::SpongeInit);

    constexpr std::array<std::pair<size_t, size_t>, 16> CASCADE_COLS = {{
        {State0HighestLkIn, State0HighestLkOut}, {State0MidHighLkIn, State0MidHighLkOut},
        {State0MidLowLkIn, State0MidLowLkOut}, {State0LowestLkIn, State0LowestLkOut},
        {State1HighestLkIn, State1HighestLkOut}, {State1MidHighLkIn, State1MidHighLkOut},
        {State1MidLowLkIn, State1MidLowLkOut}, {State1LowestLkIn, State1LowestLkOut},
        {State2HighestLkIn, State2HighestLkOut}, {State2MidHighLkIn, State2MidHighLkOut},
        {State2MidLowLkIn, State2MidLowLkOut}, {State2LowestLkIn, State2LowestLkOut},
        {State3HighestLkIn, State3HighestLkOut}, {State3MidHighLkIn, State3MidHighLkOut},
        {State3MidLowLkIn, State3MidLowLkOut}, {State3LowestLkIn, State3LowestLkOut}
    }};

    constexpr std::array<size_t, 16> CASCADE_AUX_COLS = {
        CascadeState0HighestClientLogDerivative, CascadeState0MidHighClientLogDerivative,
        CascadeState0MidLowClientLogDerivative, CascadeState0LowestClientLogDerivative,
        CascadeState1HighestClientLogDerivative, CascadeState1MidHighClientLogDerivative,
        CascadeState1MidLowClientLogDerivative, CascadeState1LowestClientLogDerivative,
        CascadeState2HighestClientLogDerivative, CascadeState2MidHighClientLogDerivative,
        CascadeState2MidLowClientLogDerivative, CascadeState2LowestClientLogDerivative,
        CascadeState3HighestClientLogDerivative, CascadeState3MidHighClientLogDerivative,
        CascadeState3MidLowClientLogDerivative, CascadeState3LowestClientLogDerivative
    };

    // PASS 1: compute cascade log derivatives with CROSS-ROW batch inversion (optimized)
    std::vector<std::thread> threads;

    // Thread 1: cascade log derivatives (16 columns) - CROSS-ROW BATCH INVERSION
    // OPTIMIZATION: Process all contributing rows for each cascade, then batch invert all at once
    // This reduces inversions from R (number of contributing rows) to 16 (number of cascades)
    threads.emplace_back([&]() {
        static int cpu_aux_tag_enabled = -1;
        if (cpu_aux_tag_enabled == -1) {
            const char* env = std::getenv("TRITON_AUX_CPU_TAG");
            cpu_aux_tag_enabled = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        }

        // Pre-compute which rows contribute (to avoid repeated checks)
        std::vector<bool> row_contributes(num_rows, false);
        for (size_t idx = 0; idx < num_rows; idx++) {
            const auto row = main_table[idx];
            BFieldElement mode = row[main_start + Mode];
            BFieldElement round_number = row[main_start + RoundNumber];
            BFieldElement current_instruction = row[main_start + CI];

            bool in_pad_mode = mode == BFieldElement(HashTableMode::Pad);
            bool in_last_round = round_number == last_round_value;
            bool is_sponge_init = current_instruction == sponge_init_opcode;

            row_contributes[idx] = (!in_pad_mode && !in_last_round && !is_sponge_init);
        }

        // Collect contributing row indices (for efficient iteration)
        std::vector<size_t> contributing_rows;
        contributing_rows.reserve(num_rows); // Reserve max possible size
        for (size_t idx = 0; idx < num_rows; idx++) {
            if (row_contributes[idx]) {
                contributing_rows.push_back(idx);
            }
        }

#ifdef TVM_USE_TBB
        // OPTIMIZATION: Process all 16 cascades in parallel using TBB
        tbb::parallel_for(size_t(0), size_t(16), [&](size_t c) {
            const auto& [lk_in, lk_out] = CASCADE_COLS[c];
            const size_t aux_col = aux_start + CASCADE_AUX_COLS[c];

            // Step 1: Collect all compressed values for this cascade from contributing rows
            std::vector<XFieldElement> compressed_values;
            compressed_values.reserve(contributing_rows.size());

            for (size_t contrib_idx : contributing_rows) {
                const auto row = main_table[contrib_idx];
                XFieldElement compressed = cascade_indeterminate
                    - cascade_look_in_weight * XFieldElement(row[main_start + lk_in])
                    - cascade_look_out_weight * XFieldElement(row[main_start + lk_out]);
                compressed_values.push_back(compressed);
            }

            // Step 2: Batch invert all compressed values at once (Montgomery's trick)
            std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(compressed_values);

            // Step 3: Compute prefix sum (log derivative) and write to aux table
            XFieldElement running_sum = XFieldElement::zero();
            size_t inv_idx = 0;

            for (size_t idx = 0; idx < num_rows; idx++) {
                if (row_contributes[idx] && inv_idx < inverses.size()) {
                    // This row contributes, add its inverse to the running sum
                    running_sum += inverses[inv_idx];
                    inv_idx++;
                }
                // Write running sum for this row (even if it doesn't contribute, it gets the current sum)
                aux_table[idx][aux_col] = running_sum;
            }
        });
#else
        // Fallback: Sequential processing if TBB not available
        for (size_t c = 0; c < 16; ++c) {
            const auto& [lk_in, lk_out] = CASCADE_COLS[c];
            const size_t aux_col = aux_start + CASCADE_AUX_COLS[c];

            // Step 1: Collect all compressed values for this cascade
            std::vector<XFieldElement> compressed_values;
            compressed_values.reserve(contributing_rows.size());

            for (size_t contrib_idx : contributing_rows) {
                const auto row = main_table[contrib_idx];
                XFieldElement compressed = cascade_indeterminate
                    - cascade_look_in_weight * XFieldElement(row[main_start + lk_in])
                    - cascade_look_out_weight * XFieldElement(row[main_start + lk_out]);
                compressed_values.push_back(compressed);
            }

            // Step 2: Batch invert all compressed values at once
            std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(compressed_values);

            // Step 3: Compute prefix sum and write to aux table
            XFieldElement running_sum = XFieldElement::zero();
            size_t inv_idx = 0;

            for (size_t idx = 0; idx < num_rows; idx++) {
                if (row_contributes[idx] && inv_idx < inverses.size()) {
                    running_sum += inverses[inv_idx];
                    inv_idx++;
                }
                aux_table[idx][aux_col] = running_sum;
            }
        }
#endif
    });

    // Thread 2: receive_chunk, hash_input, hash_digest, sponge
    threads.emplace_back([&]() {
        XFieldElement receive_chunk = EvalArg::default_initial();
        XFieldElement hash_input = EvalArg::default_initial();
        XFieldElement hash_digest = EvalArg::default_initial();
        XFieldElement sponge = EvalArg::default_initial();

        // OPTIMIZATION: Pre-extract weight coefficients ONCE (not per compress call)
        // This eliminates redundant extraction in the hot loop
        alignas(64) BFieldElement weights_c0[10], weights_c1[10], weights_c2[10];
        for (size_t i = 0; i < 10; ++i) {
            weights_c0[i] = state_weights[i].coeff(0);
            weights_c1[i] = state_weights[i].coeff(1);
            weights_c2[i] = state_weights[i].coeff(2);
        }

        // Pre-compute mode constants to avoid repeated conversions
        const BFieldElement mode_program_hashing = BFieldElement(HashTableMode::ProgramHashing);
        const BFieldElement mode_sponge = BFieldElement(HashTableMode::Sponge);
        const BFieldElement mode_hash = BFieldElement(HashTableMode::Hash);

        // OPTIMIZED: Inline re_compose to avoid lambda overhead
        auto re_compose_optimized = [&](const MainTableFlatView::RowProxy& row, size_t h, size_t mh, size_t ml, size_t l) -> BFieldElement {
            // Direct access through row proxy (already optimized)
            return (row[main_start + h] * two_pow_48 + row[main_start + mh] * two_pow_32 +
                    row[main_start + ml] * two_pow_16 + row[main_start + l]) * montgomery_modulus_inverse;
        };

        auto get_regs_optimized = [&](const MainTableFlatView::RowProxy& row) -> std::array<BFieldElement, 10> {
            return {
                re_compose_optimized(row, State0HighestLkIn, State0MidHighLkIn, State0MidLowLkIn, State0LowestLkIn),
                re_compose_optimized(row, State1HighestLkIn, State1MidHighLkIn, State1MidLowLkIn, State1LowestLkIn),
                re_compose_optimized(row, State2HighestLkIn, State2MidHighLkIn, State2MidLowLkIn, State2LowestLkIn),
                re_compose_optimized(row, State3HighestLkIn, State3MidHighLkIn, State3MidLowLkIn, State3LowestLkIn),
                row[main_start + State4], row[main_start + State5], row[main_start + State6],
                row[main_start + State7], row[main_start + State8], row[main_start + State9]
            };
        };

        // OPTIMIZED: SIMD-accelerated compress function using pre-extracted weights
        // This version uses the pre-extracted weights to avoid redundant extraction
        auto compress_optimized = [&](const std::array<BFieldElement, 10>& regs) -> XFieldElement {
            // Accumulate coefficients separately (allows compiler auto-vectorization)
            // Unroll operations for better compiler optimization with AVX2
            BFieldElement acc_c0 = BFieldElement::zero();
            BFieldElement acc_c1 = BFieldElement::zero();
            BFieldElement acc_c2 = BFieldElement::zero();
            
            // Unroll loop for compiler auto-vectorization (works with AVX2)
            // Compiler can vectorize these independent operations
            acc_c0 = acc_c0 + weights_c0[0] * regs[0] + weights_c0[1] * regs[1] +
                     weights_c0[2] * regs[2] + weights_c0[3] * regs[3] +
                     weights_c0[4] * regs[4] + weights_c0[5] * regs[5] +
                     weights_c0[6] * regs[6] + weights_c0[7] * regs[7] +
                     weights_c0[8] * regs[8] + weights_c0[9] * regs[9];
            
            acc_c1 = acc_c1 + weights_c1[0] * regs[0] + weights_c1[1] * regs[1] +
                     weights_c1[2] * regs[2] + weights_c1[3] * regs[3] +
                     weights_c1[4] * regs[4] + weights_c1[5] * regs[5] +
                     weights_c1[6] * regs[6] + weights_c1[7] * regs[7] +
                     weights_c1[8] * regs[8] + weights_c1[9] * regs[9];
            
            acc_c2 = acc_c2 + weights_c2[0] * regs[0] + weights_c2[1] * regs[1] +
                     weights_c2[2] * regs[2] + weights_c2[3] * regs[3] +
                     weights_c2[4] * regs[4] + weights_c2[5] * regs[5] +
                     weights_c2[6] * regs[6] + weights_c2[7] * regs[7] +
                     weights_c2[8] * regs[8] + weights_c2[9] * regs[9];
            
            return XFieldElement(acc_c0, acc_c1, acc_c2);
        };
        
        // OPTIMIZED: Inline EvalArg::compute_terminal to avoid vector allocation
        auto compute_terminal_inline = [&](const std::array<BFieldElement, 10>& symbols, 
                                           const XFieldElement& initial, 
                                           const XFieldElement& challenge) -> XFieldElement {
            XFieldElement result = initial;
            // Unroll loop for better performance
            result = challenge * result + XFieldElement(symbols[0]);
            result = challenge * result + XFieldElement(symbols[1]);
            result = challenge * result + XFieldElement(symbols[2]);
            result = challenge * result + XFieldElement(symbols[3]);
            result = challenge * result + XFieldElement(symbols[4]);
            result = challenge * result + XFieldElement(symbols[5]);
            result = challenge * result + XFieldElement(symbols[6]);
            result = challenge * result + XFieldElement(symbols[7]);
            result = challenge * result + XFieldElement(symbols[8]);
            result = challenge * result + XFieldElement(symbols[9]);
            return result;
        };

        // OPTIMIZATION: Cache frequently accessed aux table column indices
        const size_t aux_receive_chunk = aux_start + ReceiveChunkRunningEvaluation;
        const size_t aux_hash_input = aux_start + HashInputRunningEvaluation;
        const size_t aux_hash_digest = aux_start + HashDigestRunningEvaluation;
        const size_t aux_sponge = aux_start + SpongeRunningEvaluation;

        for (size_t idx = 0; idx < num_rows; idx++) {
            const auto row = main_table[idx];
            
            // OPTIMIZATION: Cache row accesses to reduce redundant lookups
            const BFieldElement mode = row[main_start + Mode];
            const BFieldElement round_number = row[main_start + RoundNumber];
            const BFieldElement current_instruction = row[main_start + CI];

            // OPTIMIZATION: Use direct comparisons instead of creating temporary BFieldElements
            const bool in_program_hashing = (mode == mode_program_hashing);
            const bool in_sponge = (mode == mode_sponge);
            const bool in_hash = (mode == mode_hash);
            const bool in_round_0 = round_number.is_zero();
            const bool in_last_round = (round_number == last_round_value);
            const bool is_sponge_init = (current_instruction == sponge_init_opcode);

            // OPTIMIZATION: Determine if we need regs computation early to avoid optional overhead
            const bool needs_regs = (in_program_hashing && in_round_0) || 
                                   (in_sponge && in_round_0 && !is_sponge_init) ||
                                   (in_hash && in_round_0) ||
                                   (in_hash && in_last_round);
            
            // Compute regs only if needed (avoid optional overhead)
            std::array<BFieldElement, 10> regs;
            if (needs_regs) {
                regs = get_regs_optimized(row);
            }

            if (in_program_hashing && in_round_0) {
                XFieldElement compressed_chunk = compute_terminal_inline(
                    regs, EvalArg::default_initial(), prepare_chunk_indeterminate);
                receive_chunk = receive_chunk * send_chunk_indeterminate + compressed_chunk;
            }

            if (in_sponge && in_round_0) {
                if (is_sponge_init) {
                    sponge = sponge * sponge_eval_indeterminate + ci_weight * XFieldElement(current_instruction);
                } else {
                    sponge = sponge * sponge_eval_indeterminate + ci_weight * XFieldElement(current_instruction) + compress_optimized(regs);
                }
            }

            if (in_hash && in_round_0) {
                hash_input = hash_input * hash_input_eval_indeterminate + compress_optimized(regs);
            }

            if (in_hash && in_last_round) {
                // OPTIMIZED: Unroll digest computation (Digest::LEN = 5)
                // Unroll for better compiler optimization and potential SIMD
                XFieldElement digest = state_weights[0] * XFieldElement(regs[0]) +
                                      state_weights[1] * XFieldElement(regs[1]) +
                                      state_weights[2] * XFieldElement(regs[2]) +
                                      state_weights[3] * XFieldElement(regs[3]) +
                                      state_weights[4] * XFieldElement(regs[4]);
                hash_digest = hash_digest * hash_digest_eval_indeterminate + digest;
            }

            // OPTIMIZATION: Use cached column indices
            aux_table[idx][aux_receive_chunk] = receive_chunk;
            aux_table[idx][aux_hash_input] = hash_input;
            aux_table[idx][aux_hash_digest] = hash_digest;
            aux_table[idx][aux_sponge] = sponge;
        }
    });

    for (auto& t : threads) t.join();
}

// Cascade table main column indices
// CascadeMainColumn indices are declared in `include/table/extend_helpers.hpp`.

// Cascade table aux column indices (relative to AUX_CASCADE_TABLE_START)
namespace CascadeAuxColumn {
    constexpr size_t HashTableServerLogDerivative = 0;
    constexpr size_t LookupTableClientLogDerivative = 1;
}

// =============================================================================
// OPTIMIZED extend_cascade_table
// =============================================================================
// Optimizations:
// 1. BATCH INVERSION for all 3 inversions per row
// 2. Direct array access instead of get_main_table_row copies
// =============================================================================

void extend_cascade_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace CascadeMainColumn;
    using namespace CascadeAuxColumn;
    
    const size_t main_start = CASCADE_TABLE_START;
    const size_t aux_start = AUX_CASCADE_TABLE_START;
    
    // Cache challenges
    const XFieldElement hash_indeterminate = challenges[HashCascadeLookupIndeterminate];
    const XFieldElement hash_input_weight = challenges[HashCascadeLookInWeight];
    const XFieldElement hash_output_weight = challenges[HashCascadeLookOutWeight];
    const XFieldElement lookup_indeterminate = challenges[CascadeLookupIndeterminate];
    const XFieldElement lookup_input_weight = challenges[LookupTableInputWeight];
    const XFieldElement lookup_output_weight = challenges[LookupTableOutputWeight];
    const BFieldElement two_pow_8(1ULL << 8);
    
    // ==========================================================================
    // PASS 1: Collect all values to invert (3 per non-padding row)
    // ==========================================================================
    std::vector<XFieldElement> to_invert;
    std::vector<size_t> active_rows;
    std::vector<BFieldElement> multiplicities;
    to_invert.reserve(num_rows * 3);
    active_rows.reserve(num_rows);
    multiplicities.reserve(num_rows);
    
    for (size_t idx = 0; idx < num_rows; idx++) {
        const auto& row = main_table[idx];
        if (!row[main_start + IsPadding].is_one()) {
            active_rows.push_back(idx);
            multiplicities.push_back(row[main_start + CascadeMainColumn::LookupMultiplicity]);
            
            BFieldElement look_in_hi = row[main_start + LookInHi];
            BFieldElement look_in_lo = row[main_start + LookInLo];
            BFieldElement look_out_hi = row[main_start + LookOutHi];
            BFieldElement look_out_lo = row[main_start + LookOutLo];
            
            // Hash diff
            XFieldElement look_in = XFieldElement(two_pow_8 * look_in_hi + look_in_lo);
            XFieldElement look_out = XFieldElement(two_pow_8 * look_out_hi + look_out_lo);
            to_invert.push_back(hash_indeterminate - hash_input_weight * look_in - hash_output_weight * look_out);
            
            // Lookup diffs (lo and hi)
            to_invert.push_back(lookup_indeterminate - lookup_input_weight * XFieldElement(look_in_lo) - lookup_output_weight * XFieldElement(look_out_lo));
            to_invert.push_back(lookup_indeterminate - lookup_input_weight * XFieldElement(look_in_hi) - lookup_output_weight * XFieldElement(look_out_hi));
        }
    }
    
    // BATCH INVERSION
    std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
    
    // ==========================================================================
    // PASS 2: Apply inverses
    // ==========================================================================
    XFieldElement hash_log_deriv = LookupArg::default_initial();
    XFieldElement lookup_log_deriv = LookupArg::default_initial();
    size_t active_idx = 0;
    size_t inv_idx = 0;
    
    for (size_t idx = 0; idx < num_rows; idx++) {
        if (active_idx < active_rows.size() && active_rows[active_idx] == idx) {
            hash_log_deriv += inverses[inv_idx] * multiplicities[active_idx];
            lookup_log_deriv += inverses[inv_idx + 1] + inverses[inv_idx + 2];
            inv_idx += 3;
            active_idx++;
        }
        aux_table[idx][aux_start + HashTableServerLogDerivative] = hash_log_deriv;
        aux_table[idx][aux_start + LookupTableClientLogDerivative] = lookup_log_deriv;
    }
}

// U32MainColumn and RamMainColumn indices are declared in `include/table/extend_helpers.hpp`.

// Processor table main column indices
namespace ProcessorMainColumn {
    constexpr size_t CLK = 0;
    constexpr size_t IsPadding = 1;
    constexpr size_t IP = 2;
    constexpr size_t CI = 3;
    constexpr size_t NIA = 4;
    constexpr size_t IB0 = 5;
    constexpr size_t IB1 = 6;
    constexpr size_t IB2 = 7;
    constexpr size_t IB3 = 8;
    constexpr size_t IB4 = 9;
    constexpr size_t IB5 = 10;
    constexpr size_t IB6 = 11;
    constexpr size_t JSP = 12;
    constexpr size_t JSO = 13;
    constexpr size_t JSD = 14;
    constexpr size_t ST0 = 15;
    constexpr size_t ST1 = 16;
    constexpr size_t ST2 = 17;
    constexpr size_t ST3 = 18;
    constexpr size_t ST4 = 19;
    constexpr size_t ST5 = 20;
    constexpr size_t ST6 = 21;
    constexpr size_t ST7 = 22;
    constexpr size_t ST8 = 23;
    constexpr size_t ST9 = 24;
    constexpr size_t ST10 = 25;
    constexpr size_t ST11 = 26;
    constexpr size_t ST12 = 27;
    constexpr size_t ST13 = 28;
    constexpr size_t ST14 = 29;
    constexpr size_t ST15 = 30;
    constexpr size_t OpStackPointer = 31;
    constexpr size_t HV0 = 32;
    constexpr size_t HV1 = 33;
    constexpr size_t HV2 = 34;
    constexpr size_t HV3 = 35;
    constexpr size_t HV4 = 36;
    constexpr size_t HV5 = 37;
    constexpr size_t ClockJumpDifferenceLookupMultiplicity = 38;
}

// Processor table aux column indices (relative to AUX_PROCESSOR_TABLE_START)
namespace ProcessorAuxColumn {
    constexpr size_t InputTableEvalArg = 0;
    constexpr size_t OutputTableEvalArg = 1;
    constexpr size_t InstructionLookupClientLogDerivative = 2;
    constexpr size_t OpStackTablePermArg = 3;
    constexpr size_t RamTablePermArg = 4;
    constexpr size_t JumpStackTablePermArg = 5;
    constexpr size_t HashInputEvalArg = 6;
    constexpr size_t HashDigestEvalArg = 7;
    constexpr size_t SpongeEvalArg = 8;
    constexpr size_t U32LookupClientLogDerivative = 9;
    constexpr size_t ClockJumpDifferenceLookupClientLogDerivative = 10;
}

// Ram table aux column indices (relative to AUX_RAM_TABLE_START)
namespace RamAuxColumn {
    constexpr size_t RunningProductOfRAMP = 0;
    constexpr size_t FormalDerivative = 1;
    constexpr size_t BezoutCoefficient0 = 2;
    constexpr size_t BezoutCoefficient1 = 3;
    constexpr size_t RunningProductPermArg = 4;
    constexpr size_t ClockJumpDifferenceLookupLogDerivative = 5;
}

// =============================================================================
// OPTIMIZED extend_processor_table
// =============================================================================
// Optimizations:
// 1. Pre-decode all instructions ONCE (was decoded 8+ times per row before)
// 2. Run independent column computations in PARALLEL threads
// 3. Cache frequently-used challenge values
// 4. Use direct reference access instead of copying rows
// =============================================================================

void extend_processor_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ProcessorMainColumn;
    using namespace ProcessorAuxColumn;
    
    const size_t main_start = PROCESSOR_TABLE_START;
    const size_t aux_start = AUX_PROCESSOR_TABLE_START;
    
    // ==========================================================================
    // OPTIMIZATION 1: Pre-decode ALL instructions ONCE (parallelized)
    // ==========================================================================
    std::vector<std::optional<TritonInstruction>> decoded_instructions(num_rows);
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < num_rows; ++idx) {
        uint32_t opcode = static_cast<uint32_t>(main_table[idx][main_start + CI].value());
        BFieldElement nia = main_table[idx][main_start + NIA];
        decoded_instructions[idx] = TritonInstruction::from_opcode(opcode, nia);
    }
    
    // ==========================================================================
    // OPTIMIZATION 2: Cache frequently-used challenge values
    // ==========================================================================
    const XFieldElement op_stack_indeterminate = challenges[ChallengeId::OpStackIndeterminate];
    const XFieldElement op_stack_clk_weight = challenges[ChallengeId::OpStackClkWeight];
    const XFieldElement op_stack_ib1_weight = challenges[ChallengeId::OpStackIb1Weight];
    const XFieldElement op_stack_pointer_weight = challenges[ChallengeId::OpStackPointerWeight];
    const XFieldElement op_stack_underflow_weight = challenges[ChallengeId::OpStackFirstUnderflowElementWeight];
    
    const XFieldElement ram_indeterminate = challenges[ChallengeId::RamIndeterminate];
    const XFieldElement ram_clk_weight = challenges[ChallengeId::RamClkWeight];
    const XFieldElement ram_type_weight = challenges[ChallengeId::RamInstructionTypeWeight];
    const XFieldElement ram_pointer_weight = challenges[ChallengeId::RamPointerWeight];
    const XFieldElement ram_value_weight = challenges[ChallengeId::RamValueWeight];
    
    const XFieldElement jump_stack_indeterminate = challenges[ChallengeId::JumpStackIndeterminate];
    const XFieldElement jump_stack_clk_weight = challenges[ChallengeId::JumpStackClkWeight];
    const XFieldElement jump_stack_ci_weight = challenges[ChallengeId::JumpStackCiWeight];
    const XFieldElement jump_stack_jsp_weight = challenges[ChallengeId::JumpStackJspWeight];
    const XFieldElement jump_stack_jso_weight = challenges[ChallengeId::JumpStackJsoWeight];
    const XFieldElement jump_stack_jsd_weight = challenges[ChallengeId::JumpStackJsdWeight];
    
    const XFieldElement hash_input_indeterminate = challenges[ChallengeId::HashInputIndeterminate];
    const XFieldElement hash_digest_indeterminate = challenges[ChallengeId::HashDigestIndeterminate];
    const XFieldElement sponge_indeterminate = challenges[ChallengeId::SpongeIndeterminate];
    const XFieldElement hash_ci_weight = challenges[ChallengeId::HashCIWeight];
    
    const XFieldElement standard_input_indeterminate = challenges[ChallengeId::StandardInputIndeterminate];
    const XFieldElement standard_output_indeterminate = challenges[ChallengeId::StandardOutputIndeterminate];
    
    const XFieldElement u32_indeterminate = challenges[ChallengeId::U32Indeterminate];
    const XFieldElement u32_lhs_weight = challenges[ChallengeId::U32LhsWeight];
    const XFieldElement u32_rhs_weight = challenges[ChallengeId::U32RhsWeight];
    const XFieldElement u32_ci_weight = challenges[ChallengeId::U32CiWeight];
    const XFieldElement u32_result_weight = challenges[ChallengeId::U32ResultWeight];
    
    const XFieldElement cjd_indeterminate = challenges[ChallengeId::ClockJumpDifferenceLookupIndeterminate];
    
    const XFieldElement instr_lookup_indeterminate = challenges[ChallengeId::InstructionLookupIndeterminate];
    const XFieldElement program_addr_weight = challenges[ChallengeId::ProgramAddressWeight];
    const XFieldElement program_instr_weight = challenges[ChallengeId::ProgramInstructionWeight];
    const XFieldElement program_nia_weight = challenges[ChallengeId::ProgramNextInstructionWeight];
    
    std::array<XFieldElement, 10> hash_state_weights;
    for (size_t i = 0; i < 10; ++i) {
        hash_state_weights[i] = challenges[ChallengeId::StackWeight0 + i];
    }
    
    // Constants
    const BFieldElement ram_type_write = BFieldElement::zero();
    const BFieldElement ram_type_read = BFieldElement::one();
    constexpr size_t NUM_OP_STACK_REGISTERS = 16;
    constexpr std::array<size_t, NUM_OP_STACK_REGISTERS> OP_STACK_COLUMNS = {
        ProcessorMainColumn::ST0, ProcessorMainColumn::ST1, ProcessorMainColumn::ST2, ProcessorMainColumn::ST3,
        ProcessorMainColumn::ST4, ProcessorMainColumn::ST5, ProcessorMainColumn::ST6, ProcessorMainColumn::ST7,
        ProcessorMainColumn::ST8, ProcessorMainColumn::ST9, ProcessorMainColumn::ST10, ProcessorMainColumn::ST11,
        ProcessorMainColumn::ST12, ProcessorMainColumn::ST13, ProcessorMainColumn::ST14, ProcessorMainColumn::ST15
    };
    constexpr std::array<size_t, 10> ST_INDICES = {ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9};
    constexpr std::array<size_t, 10> MERKLE_LEFT = {ST0, ST1, ST2, ST3, ST4, HV0, HV1, HV2, HV3, HV4};
    constexpr std::array<size_t, 10> MERKLE_RIGHT = {HV0, HV1, HV2, HV3, HV4, ST0, ST1, ST2, ST3, ST4};
    constexpr std::array<size_t, 4> ABSORB_MEM_STACK = {ST1, ST2, ST3, ST4};
    constexpr std::array<size_t, 6> HV_INDICES = {HV0, HV1, HV2, HV3, HV4, HV5};
    const BFieldElement sponge_absorb_opcode = opcode_to_b_field(AnInstruction::SpongeAbsorb);
    
    // ==========================================================================
    // OPTIMIZATION 3: Run column computations in PARALLEL threads
    // ==========================================================================
    std::vector<std::thread> threads;
    
    // ---- Thread 1: Columns 0-1 (Input/Output Table Eval Args) ----
    threads.emplace_back([&]() {
        XFieldElement input_eval = EvalArg::default_initial();
        XFieldElement output_eval = EvalArg::default_initial();
        aux_table[0][aux_start + InputTableEvalArg] = input_eval;
        aux_table[0][aux_start + OutputTableEvalArg] = output_eval;

        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                size_t num_words = 0;
                switch (instr.num_words_arg) {
                    case NumberOfWords::N1: num_words = 1; break;
                    case NumberOfWords::N2: num_words = 2; break;
                    case NumberOfWords::N3: num_words = 3; break;
                    case NumberOfWords::N4: num_words = 4; break;
                    case NumberOfWords::N5: num_words = 5; break;
                    default: num_words = 0; break;
                }
                // OPTIMIZED: Unroll loops for better compiler optimization (max 5 iterations)
                if (instr.type == AnInstruction::ReadIo && num_words > 0) {
                    // Unroll for common cases (1-5 words)
                    if (num_words == 1) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else if (num_words == 2) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else if (num_words == 3) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST2]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else if (num_words == 4) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST3]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST2]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else { // num_words == 5
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST4]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST3]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST2]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    }
                } else if (instr.type == AnInstruction::WriteIo && num_words > 0) {
                    // Unroll for common cases (1-5 words)
                    if (num_words == 1) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                    } else if (num_words == 2) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                    } else if (num_words == 3) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST2]);
                    } else if (num_words == 4) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST2]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST3]);
                    } else { // num_words == 5
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST2]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST3]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST4]);
                    }
                }
            }
            aux_table[idx][aux_start + InputTableEvalArg] = input_eval;
            aux_table[idx][aux_start + OutputTableEvalArg] = output_eval;
        }
    });

    // ---- Thread 2: Column 2 (InstructionLookupClientLogDerivative) ----
    threads.emplace_back([&]() {
    std::vector<XFieldElement> to_invert;
        to_invert.reserve(num_rows);
    for (size_t idx = 0; idx < num_rows; idx++) {
            if (main_table[idx][main_start + IsPadding].is_one()) break;

            BFieldElement ip = main_table[idx][main_start + IP];
            BFieldElement ci = main_table[idx][main_start + CI];
            BFieldElement nia = main_table[idx][main_start + NIA];

            XFieldElement compressed = XFieldElement(ip) * program_addr_weight
                + XFieldElement(ci) * program_instr_weight
                + XFieldElement(nia) * program_nia_weight;
            to_invert.push_back(instr_lookup_indeterminate - compressed);
        }
        
        std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
        XFieldElement log_deriv = LookupArg::default_initial();
        size_t inv_idx = 0;
        for (size_t idx = 0; idx < num_rows; idx++) {
            if (!main_table[idx][main_start + IsPadding].is_one() && inv_idx < inverses.size()) {
                log_deriv = log_deriv + inverses[inv_idx++];
            }
            aux_table[idx][aux_start + InstructionLookupClientLogDerivative] = log_deriv;
        }
    });
    
    // ---- Thread 3: Column 5 (JumpStackTablePermArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_product = PermArg::default_initial();
        for (size_t idx = 0; idx < num_rows; idx++) {
            BFieldElement clk = main_table[idx][main_start + CLK];
            BFieldElement ci = main_table[idx][main_start + CI];
            BFieldElement jsp = main_table[idx][main_start + JSP];
            BFieldElement jso = main_table[idx][main_start + JSO];
            BFieldElement jsd = main_table[idx][main_start + JSD];

            XFieldElement compressed = XFieldElement(clk) * jump_stack_clk_weight
                + XFieldElement(ci) * jump_stack_ci_weight
                + XFieldElement(jsp) * jump_stack_jsp_weight
                + XFieldElement(jso) * jump_stack_jso_weight
                + XFieldElement(jsd) * jump_stack_jsd_weight;
            
            running_product = running_product * (jump_stack_indeterminate - compressed);
            aux_table[idx][aux_start + JumpStackTablePermArg] = running_product;
        }
    });
    
    // ---- Thread 4: Column 3 (OpStackTablePermArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_product = PermArg::default_initial();
        aux_table[0][aux_start + OpStackTablePermArg] = running_product;
        
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_curr = main_table[idx];
            const auto& row_prev = main_table[idx - 1];
            
            if (row_curr[main_start + IsPadding].is_one()) {
                aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (!instr_opt.has_value()) {
                aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr = *instr_opt;
            int32_t influence = instr.op_stack_size_influence();
            if (influence == 0) {
                aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
                continue;
            }
            
            const auto& shorter_row = (influence > 0) ? row_prev : row_curr;
            size_t op_stack_delta = static_cast<size_t>(std::abs(influence));
            XFieldElement factor = XFieldElement::one();
            
            BFieldElement clk = row_prev[main_start + CLK];
            BFieldElement ib1 = row_prev[main_start + IB1];
            BFieldElement osp = shorter_row[main_start + OpStackPointer];
            
            for (size_t offset = 0; offset < op_stack_delta; ++offset) {
                size_t stack_idx = NUM_OP_STACK_REGISTERS - 1 - offset;
                BFieldElement underflow = shorter_row[main_start + OP_STACK_COLUMNS[stack_idx]];
                BFieldElement ptr = osp + BFieldElement(static_cast<uint64_t>(offset));
                
                XFieldElement compressed = XFieldElement(clk) * op_stack_clk_weight
                    + XFieldElement(ib1) * op_stack_ib1_weight
                    + XFieldElement(ptr) * op_stack_pointer_weight
                    + XFieldElement(underflow) * op_stack_underflow_weight;
                factor = factor * (op_stack_indeterminate - compressed);
            }
            running_product = running_product * factor;
            aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
        }
    });
    
    // ---- Thread 5: Column 4 (RamTablePermArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_product = PermArg::default_initial();
        aux_table[0][aux_start + RamTablePermArg] = running_product;
        
    for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_curr = main_table[idx];
            const auto& row_prev = main_table[idx - 1];
            
            if (row_curr[main_start + IsPadding].is_one()) {
                aux_table[idx][aux_start + RamTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (!instr_opt.has_value()) {
                aux_table[idx][aux_start + RamTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr = *instr_opt;
            BFieldElement instruction_type;
            std::vector<std::pair<BFieldElement, BFieldElement>> accesses;
            
            switch (instr.type) {
                case AnInstruction::ReadMem:
                case AnInstruction::WriteMem: {
                    instruction_type = (instr.type == AnInstruction::WriteMem) ? ram_type_write : ram_type_read;
                    const auto& longer_row = (instr.type == AnInstruction::ReadMem) ? row_curr : row_prev;
                    size_t op_stack_delta = static_cast<size_t>(std::abs(instr.op_stack_size_influence()));
                    BFieldElement ram_ptr = longer_row[main_start + ST0];
                    for (size_t off = 0; off < op_stack_delta; ++off) {
                        size_t val_col = OP_STACK_COLUMNS[off + 1];
                        BFieldElement ptr = ram_ptr + BFieldElement(static_cast<uint64_t>(off));
                        if (instr.type == AnInstruction::ReadMem) ptr = ptr + BFieldElement::one();
                        accesses.emplace_back(ptr, longer_row[main_start + val_col]);
                    }
                    break;
                }
                case AnInstruction::SpongeAbsorbMem: {
                    instruction_type = ram_type_read;
                    BFieldElement mp = row_prev[main_start + ST0];
                    accesses = {
                        {mp, row_curr[main_start + ST1]}, {mp + BFieldElement(1), row_curr[main_start + ST2]},
                        {mp + BFieldElement(2), row_curr[main_start + ST3]}, {mp + BFieldElement(3), row_curr[main_start + ST4]},
                        {mp + BFieldElement(4), row_prev[main_start + HV0]}, {mp + BFieldElement(5), row_prev[main_start + HV1]},
                        {mp + BFieldElement(6), row_prev[main_start + HV2]}, {mp + BFieldElement(7), row_prev[main_start + HV3]},
                        {mp + BFieldElement(8), row_prev[main_start + HV4]}, {mp + BFieldElement(9), row_prev[main_start + HV5]}
                    };
                    break;
                }
                case AnInstruction::MerkleStepMem: {
                    instruction_type = ram_type_read;
                    BFieldElement mp = row_prev[main_start + ST7];
                    accesses = {
                        {mp, row_prev[main_start + HV0]}, {mp + BFieldElement(1), row_prev[main_start + HV1]},
                        {mp + BFieldElement(2), row_prev[main_start + HV2]}, {mp + BFieldElement(3), row_prev[main_start + HV3]},
                        {mp + BFieldElement(4), row_prev[main_start + HV4]}
                    };
                    break;
                }
                case AnInstruction::XxDotStep: {
                    instruction_type = ram_type_read;
                    BFieldElement rhs = row_prev[main_start + ST0], lhs = row_prev[main_start + ST1];
                    accesses = {
                        {rhs, row_prev[main_start + HV0]}, {rhs + BFieldElement(1), row_prev[main_start + HV1]},
                        {rhs + BFieldElement(2), row_prev[main_start + HV2]}, {lhs, row_prev[main_start + HV3]},
                        {lhs + BFieldElement(1), row_prev[main_start + HV4]}, {lhs + BFieldElement(2), row_prev[main_start + HV5]}
                    };
                    break;
                }
                case AnInstruction::XbDotStep: {
                    instruction_type = ram_type_read;
                    BFieldElement rhs = row_prev[main_start + ST0], lhs = row_prev[main_start + ST1];
                    accesses = {
                        {rhs, row_prev[main_start + HV0]}, {lhs, row_prev[main_start + HV1]},
                        {lhs + BFieldElement(1), row_prev[main_start + HV2]}, {lhs + BFieldElement(2), row_prev[main_start + HV3]}
                    };
                    break;
                }
                default:
                    aux_table[idx][aux_start + RamTablePermArg] = running_product;
                    continue;
            }
            
            if (!accesses.empty()) {
                BFieldElement clk = row_prev[main_start + CLK];
                XFieldElement factor = XFieldElement::one();
                for (const auto& [ptr, val] : accesses) {
                    XFieldElement compressed = XFieldElement(clk) * ram_clk_weight
                        + XFieldElement(instruction_type) * ram_type_weight
                        + XFieldElement(ptr) * ram_pointer_weight
                        + XFieldElement(val) * ram_value_weight;
                    factor = factor * (ram_indeterminate - compressed);
                }
                running_product = running_product * factor;
            }
            aux_table[idx][aux_start + RamTablePermArg] = running_product;
        }
    });
    
    // ---- Thread 6: Column 6 (HashInputEvalArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_eval = EvalArg::default_initial();
        for (size_t idx = 0; idx < num_rows; idx++) {
            const auto& row = main_table[idx];
            const auto& instr_opt = decoded_instructions[idx];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                const std::array<size_t, 10>* cols = nullptr;
                if (instr.type == AnInstruction::Hash) {
                    cols = &ST_INDICES;
                } else if (instr.type == AnInstruction::MerkleStep || instr.type == AnInstruction::MerkleStepMem) {
                    bool is_left = (row[main_start + ST5].value() % 2ULL) == 0ULL;
                    cols = is_left ? &MERKLE_LEFT : &MERKLE_RIGHT;
                }
                if (cols) {
                    XFieldElement compressed = XFieldElement::zero();
                    for (size_t j = 0; j < 10; ++j) {
                        compressed += hash_state_weights[j] * XFieldElement(row[main_start + (*cols)[j]]);
                    }
                    running_eval = running_eval * hash_input_indeterminate + compressed;
                }
            }
            aux_table[idx][aux_start + HashInputEvalArg] = running_eval;
        }
    });
    
    // ---- Thread 7: Column 7 (HashDigestEvalArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_eval = EvalArg::default_initial();
        aux_table[0][aux_start + HashDigestEvalArg] = running_eval;
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                if (instr.type == AnInstruction::Hash || instr.type == AnInstruction::MerkleStep || instr.type == AnInstruction::MerkleStepMem) {
                    XFieldElement compressed = XFieldElement::zero();
                    for (size_t j = 0; j < 5; ++j) {
                        compressed += hash_state_weights[j] * XFieldElement(main_table[idx][main_start + ST0 + j]);
                    }
                    running_eval = running_eval * hash_digest_indeterminate + compressed;
                }
            }
            aux_table[idx][aux_start + HashDigestEvalArg] = running_eval;
        }
    });
    
    // ---- Thread 8: Column 8 (SpongeEvalArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_eval = EvalArg::default_initial();
        aux_table[0][aux_start + SpongeEvalArg] = running_eval;
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_prev = main_table[idx - 1];
            const auto& row_curr = main_table[idx];
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                if (instr.type == AnInstruction::SpongeInit) {
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(row_prev[main_start + CI]);
                } else if (instr.type == AnInstruction::SpongeAbsorb) {
                    // OPTIMIZED: Unroll hash state compression (10 operations)
                    XFieldElement compressed = hash_state_weights[0] * XFieldElement(row_prev[main_start + ST_INDICES[0]]) +
                                             hash_state_weights[1] * XFieldElement(row_prev[main_start + ST_INDICES[1]]) +
                                             hash_state_weights[2] * XFieldElement(row_prev[main_start + ST_INDICES[2]]) +
                                             hash_state_weights[3] * XFieldElement(row_prev[main_start + ST_INDICES[3]]) +
                                             hash_state_weights[4] * XFieldElement(row_prev[main_start + ST_INDICES[4]]) +
                                             hash_state_weights[5] * XFieldElement(row_prev[main_start + ST_INDICES[5]]) +
                                             hash_state_weights[6] * XFieldElement(row_prev[main_start + ST_INDICES[6]]) +
                                             hash_state_weights[7] * XFieldElement(row_prev[main_start + ST_INDICES[7]]) +
                                             hash_state_weights[8] * XFieldElement(row_prev[main_start + ST_INDICES[8]]) +
                                             hash_state_weights[9] * XFieldElement(row_prev[main_start + ST_INDICES[9]]);
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(row_prev[main_start + CI]) + compressed;
                } else if (instr.type == AnInstruction::SpongeAbsorbMem) {
                    // OPTIMIZED: Unroll absorb mem compression (4 + 6 operations)
                    XFieldElement compressed = hash_state_weights[0] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[0]]) +
                                             hash_state_weights[1] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[1]]) +
                                             hash_state_weights[2] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[2]]) +
                                             hash_state_weights[3] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[3]]) +
                                             hash_state_weights[4] * XFieldElement(row_prev[main_start + HV_INDICES[0]]) +
                                             hash_state_weights[5] * XFieldElement(row_prev[main_start + HV_INDICES[1]]) +
                                             hash_state_weights[6] * XFieldElement(row_prev[main_start + HV_INDICES[2]]) +
                                             hash_state_weights[7] * XFieldElement(row_prev[main_start + HV_INDICES[3]]) +
                                             hash_state_weights[8] * XFieldElement(row_prev[main_start + HV_INDICES[4]]) +
                                             hash_state_weights[9] * XFieldElement(row_prev[main_start + HV_INDICES[5]]);
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(sponge_absorb_opcode) + compressed;
                } else if (instr.type == AnInstruction::SpongeSqueeze) {
                    // OPTIMIZED: Unroll squeeze compression (10 operations)
                    XFieldElement compressed = hash_state_weights[0] * XFieldElement(row_curr[main_start + ST_INDICES[0]]) +
                                             hash_state_weights[1] * XFieldElement(row_curr[main_start + ST_INDICES[1]]) +
                                             hash_state_weights[2] * XFieldElement(row_curr[main_start + ST_INDICES[2]]) +
                                             hash_state_weights[3] * XFieldElement(row_curr[main_start + ST_INDICES[3]]) +
                                             hash_state_weights[4] * XFieldElement(row_curr[main_start + ST_INDICES[4]]) +
                                             hash_state_weights[5] * XFieldElement(row_curr[main_start + ST_INDICES[5]]) +
                                             hash_state_weights[6] * XFieldElement(row_curr[main_start + ST_INDICES[6]]) +
                                             hash_state_weights[7] * XFieldElement(row_curr[main_start + ST_INDICES[7]]) +
                                             hash_state_weights[8] * XFieldElement(row_curr[main_start + ST_INDICES[8]]) +
                                             hash_state_weights[9] * XFieldElement(row_curr[main_start + ST_INDICES[9]]);
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(row_prev[main_start + CI]) + compressed;
                }
            }
            aux_table[idx][aux_start + SpongeEvalArg] = running_eval;
        }
    });

    // ---- Thread 9: Column 9 (U32LookupClientLogDerivative) ----
    threads.emplace_back([&]() {
        std::vector<XFieldElement> to_invert;
        to_invert.reserve(num_rows);
        std::vector<size_t> inverse_counts(num_rows, 0);  // Track how many inverses per row
        
        const BFieldElement and_opcode = opcode_to_b_field(AnInstruction::And);
        const BFieldElement lt_opcode = opcode_to_b_field(AnInstruction::Lt);
        const BFieldElement split_opcode = opcode_to_b_field(AnInstruction::Split);
        
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_prev = main_table[idx - 1];
            const auto& row_curr = main_table[idx];
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (!instr_opt.has_value()) continue;
            const auto& instr = *instr_opt;
            
            BFieldElement prev_ci = row_prev[main_start + CI];
            auto push = [&](const XFieldElement& term) {
                to_invert.push_back(u32_indeterminate - term);
                inverse_counts[idx]++;
            };

            if (instr.type == AnInstruction::Split) {
                push(XFieldElement(row_curr[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_curr[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(prev_ci) * u32_ci_weight);
            } else if (instr.type == AnInstruction::Lt || instr.type == AnInstruction::And || instr.type == AnInstruction::Pow) {
                push(XFieldElement(row_prev[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_prev[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(prev_ci) * u32_ci_weight
                    + XFieldElement(row_curr[main_start + ST0]) * u32_result_weight);
            } else if (instr.type == AnInstruction::Xor) {
                BFieldElement st0p = row_prev[main_start + ST0], st1p = row_prev[main_start + ST1], st0c = row_curr[main_start + ST0];
                BFieldElement from_xor = (st0p + st1p - st0c) / BFieldElement(2);
                push(XFieldElement(st0p) * u32_lhs_weight + XFieldElement(st1p) * u32_rhs_weight
                    + XFieldElement(and_opcode) * u32_ci_weight + XFieldElement(from_xor) * u32_result_weight);
            } else if (instr.type == AnInstruction::Log2Floor || instr.type == AnInstruction::PopCount) {
                push(XFieldElement(row_prev[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(prev_ci) * u32_ci_weight
                    + XFieldElement(row_curr[main_start + ST0]) * u32_result_weight);
            } else if (instr.type == AnInstruction::DivMod) {
                push(XFieldElement(row_curr[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_prev[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(lt_opcode) * u32_ci_weight
                    + XFieldElement(BFieldElement(1)) * u32_result_weight);
                push(XFieldElement(row_prev[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_curr[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(split_opcode) * u32_ci_weight);
            } else if (instr.type == AnInstruction::MerkleStep || instr.type == AnInstruction::MerkleStepMem) {
                push(XFieldElement(row_prev[main_start + ST5]) * u32_lhs_weight
                    + XFieldElement(row_curr[main_start + ST5]) * u32_rhs_weight
                    + XFieldElement(split_opcode) * u32_ci_weight);
            }
        }

        std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
        XFieldElement log_deriv = LookupArg::default_initial();
        size_t inv_idx = 0;
        aux_table[0][aux_start + U32LookupClientLogDerivative] = log_deriv;
        for (size_t idx = 1; idx < num_rows; idx++) {
            for (size_t k = 0; k < inverse_counts[idx] && inv_idx < inverses.size(); ++k) {
                log_deriv += inverses[inv_idx++];
            }
            aux_table[idx][aux_start + U32LookupClientLogDerivative] = log_deriv;
        }
    });
    
    // ---- Thread 10: Column 10 (ClockJumpDifferenceLookupClientLogDerivative) ----
    threads.emplace_back([&]() {
        std::vector<XFieldElement> to_invert;
        std::vector<std::pair<size_t, BFieldElement>> idx_mult;
        to_invert.reserve(num_rows);
        idx_mult.reserve(num_rows);
        
        for (size_t idx = 0; idx < num_rows; idx++) {
            BFieldElement mult = main_table[idx][main_start + ClockJumpDifferenceLookupMultiplicity];
            if (!mult.is_zero()) {
                BFieldElement clk = main_table[idx][main_start + CLK];
                to_invert.push_back(cjd_indeterminate - XFieldElement(clk));
                idx_mult.emplace_back(idx, mult);
            }
        }
        
        std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
        XFieldElement log_deriv = LookupArg::default_initial();
        size_t inv_idx = 0;
        size_t next_inv = 0;
        for (size_t idx = 0; idx < num_rows; idx++) {
            if (next_inv < idx_mult.size() && idx_mult[next_inv].first == idx) {
                log_deriv += inverses[inv_idx++] * XFieldElement(idx_mult[next_inv].second);
                next_inv++;
            }
            aux_table[idx][aux_start + ClockJumpDifferenceLookupClientLogDerivative] = log_deriv;
        }
    });
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
}

void extend_processor_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    // Exact copy of the existing optimized implementation, but reading from `MainTableFlatView`.
    using namespace ProcessorMainColumn;
    using namespace ProcessorAuxColumn;
    
    const size_t main_start = PROCESSOR_TABLE_START;
    const size_t aux_start = AUX_PROCESSOR_TABLE_START;
    
    // ==========================================================================
    // OPTIMIZATION 1: Pre-decode ALL instructions ONCE (parallelized)
    // ==========================================================================
    std::vector<std::optional<TritonInstruction>> decoded_instructions(num_rows);
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < num_rows; ++idx) {
        uint32_t opcode = static_cast<uint32_t>(main_table[idx][main_start + CI].value());
        BFieldElement nia = main_table[idx][main_start + NIA];
        decoded_instructions[idx] = TritonInstruction::from_opcode(opcode, nia);
    }
    
    // ==========================================================================
    // OPTIMIZATION 2: Cache frequently-used challenge values
    // ==========================================================================
    const XFieldElement op_stack_indeterminate = challenges[ChallengeId::OpStackIndeterminate];
    const XFieldElement op_stack_clk_weight = challenges[ChallengeId::OpStackClkWeight];
    const XFieldElement op_stack_ib1_weight = challenges[ChallengeId::OpStackIb1Weight];
    const XFieldElement op_stack_pointer_weight = challenges[ChallengeId::OpStackPointerWeight];
    const XFieldElement op_stack_underflow_weight = challenges[ChallengeId::OpStackFirstUnderflowElementWeight];
    
    const XFieldElement ram_indeterminate = challenges[ChallengeId::RamIndeterminate];
    const XFieldElement ram_clk_weight = challenges[ChallengeId::RamClkWeight];
    const XFieldElement ram_type_weight = challenges[ChallengeId::RamInstructionTypeWeight];
    const XFieldElement ram_pointer_weight = challenges[ChallengeId::RamPointerWeight];
    const XFieldElement ram_value_weight = challenges[ChallengeId::RamValueWeight];
    
    const XFieldElement jump_stack_indeterminate = challenges[ChallengeId::JumpStackIndeterminate];
    const XFieldElement jump_stack_clk_weight = challenges[ChallengeId::JumpStackClkWeight];
    const XFieldElement jump_stack_ci_weight = challenges[ChallengeId::JumpStackCiWeight];
    const XFieldElement jump_stack_jsp_weight = challenges[ChallengeId::JumpStackJspWeight];
    const XFieldElement jump_stack_jso_weight = challenges[ChallengeId::JumpStackJsoWeight];
    const XFieldElement jump_stack_jsd_weight = challenges[ChallengeId::JumpStackJsdWeight];
    
    const XFieldElement hash_input_indeterminate = challenges[ChallengeId::HashInputIndeterminate];
    const XFieldElement hash_digest_indeterminate = challenges[ChallengeId::HashDigestIndeterminate];
    const XFieldElement sponge_indeterminate = challenges[ChallengeId::SpongeIndeterminate];
    const XFieldElement hash_ci_weight = challenges[ChallengeId::HashCIWeight];
    
    const XFieldElement standard_input_indeterminate = challenges[ChallengeId::StandardInputIndeterminate];
    const XFieldElement standard_output_indeterminate = challenges[ChallengeId::StandardOutputIndeterminate];
    
    const XFieldElement u32_indeterminate = challenges[ChallengeId::U32Indeterminate];
    const XFieldElement u32_lhs_weight = challenges[ChallengeId::U32LhsWeight];
    const XFieldElement u32_rhs_weight = challenges[ChallengeId::U32RhsWeight];
    const XFieldElement u32_ci_weight = challenges[ChallengeId::U32CiWeight];
    const XFieldElement u32_result_weight = challenges[ChallengeId::U32ResultWeight];
    
    const XFieldElement cjd_indeterminate = challenges[ChallengeId::ClockJumpDifferenceLookupIndeterminate];
    
    const XFieldElement instr_lookup_indeterminate = challenges[ChallengeId::InstructionLookupIndeterminate];
    const XFieldElement program_addr_weight = challenges[ChallengeId::ProgramAddressWeight];
    const XFieldElement program_instr_weight = challenges[ChallengeId::ProgramInstructionWeight];
    const XFieldElement program_nia_weight = challenges[ChallengeId::ProgramNextInstructionWeight];
    
    std::array<XFieldElement, 10> hash_state_weights;
    for (size_t i = 0; i < 10; ++i) {
        hash_state_weights[i] = challenges[ChallengeId::StackWeight0 + i];
    }
    
    // Constants
    const BFieldElement ram_type_write = BFieldElement::zero();
    const BFieldElement ram_type_read = BFieldElement::one();
    constexpr size_t NUM_OP_STACK_REGISTERS = 16;
    constexpr std::array<size_t, NUM_OP_STACK_REGISTERS> OP_STACK_COLUMNS = {
        ProcessorMainColumn::ST0, ProcessorMainColumn::ST1, ProcessorMainColumn::ST2, ProcessorMainColumn::ST3,
        ProcessorMainColumn::ST4, ProcessorMainColumn::ST5, ProcessorMainColumn::ST6, ProcessorMainColumn::ST7,
        ProcessorMainColumn::ST8, ProcessorMainColumn::ST9, ProcessorMainColumn::ST10, ProcessorMainColumn::ST11,
        ProcessorMainColumn::ST12, ProcessorMainColumn::ST13, ProcessorMainColumn::ST14, ProcessorMainColumn::ST15
    };
    constexpr std::array<size_t, 10> ST_INDICES = {ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9};
    constexpr std::array<size_t, 10> MERKLE_LEFT = {ST0, ST1, ST2, ST3, ST4, HV0, HV1, HV2, HV3, HV4};
    constexpr std::array<size_t, 10> MERKLE_RIGHT = {HV0, HV1, HV2, HV3, HV4, ST0, ST1, ST2, ST3, ST4};
    constexpr std::array<size_t, 4> ABSORB_MEM_STACK = {ST1, ST2, ST3, ST4};
    constexpr std::array<size_t, 6> HV_INDICES = {HV0, HV1, HV2, HV3, HV4, HV5};
    const BFieldElement sponge_absorb_opcode = opcode_to_b_field(AnInstruction::SpongeAbsorb);
    
    std::vector<std::thread> threads;
    
    // ---- Thread 1: Columns 0-1 (Input/Output Table Eval Args) ----
    threads.emplace_back([&]() {
        XFieldElement input_eval = EvalArg::default_initial();
        XFieldElement output_eval = EvalArg::default_initial();
        aux_table[0][aux_start + InputTableEvalArg] = input_eval;
        aux_table[0][aux_start + OutputTableEvalArg] = output_eval;

        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                size_t num_words = 0;
                switch (instr.num_words_arg) {
                    case NumberOfWords::N1: num_words = 1; break;
                    case NumberOfWords::N2: num_words = 2; break;
                    case NumberOfWords::N3: num_words = 3; break;
                    case NumberOfWords::N4: num_words = 4; break;
                    case NumberOfWords::N5: num_words = 5; break;
                    default: num_words = 0; break;
                }
                // OPTIMIZED: Unroll loops for better compiler optimization (max 5 iterations)
                if (instr.type == AnInstruction::ReadIo && num_words > 0) {
                    // Unroll for common cases (1-5 words)
                    if (num_words == 1) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else if (num_words == 2) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else if (num_words == 3) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST2]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else if (num_words == 4) {
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST3]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST2]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    } else { // num_words == 5
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST4]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST3]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST2]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST1]);
                        input_eval = input_eval * standard_input_indeterminate + XFieldElement(main_table[idx][main_start + ST0]);
                    }
                } else if (instr.type == AnInstruction::WriteIo && num_words > 0) {
                    // Unroll for common cases (1-5 words)
                    if (num_words == 1) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                    } else if (num_words == 2) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                    } else if (num_words == 3) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST2]);
                    } else if (num_words == 4) {
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST2]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST3]);
                    } else { // num_words == 5
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST0]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST1]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST2]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST3]);
                        output_eval = output_eval * standard_output_indeterminate + XFieldElement(main_table[idx - 1][main_start + ST4]);
                    }
                }
            }
            aux_table[idx][aux_start + InputTableEvalArg] = input_eval;
            aux_table[idx][aux_start + OutputTableEvalArg] = output_eval;
        }
    });

    // ---- Thread 2: Column 2 (InstructionLookupClientLogDerivative) ----
    threads.emplace_back([&]() {
        std::vector<XFieldElement> to_invert;
        to_invert.reserve(num_rows);
        for (size_t idx = 0; idx < num_rows; idx++) {
            if (main_table[idx][main_start + IsPadding].is_one()) break;

            BFieldElement ip = main_table[idx][main_start + IP];
            BFieldElement ci = main_table[idx][main_start + CI];
            BFieldElement nia = main_table[idx][main_start + NIA];

            XFieldElement compressed = XFieldElement(ip) * program_addr_weight
                + XFieldElement(ci) * program_instr_weight
                + XFieldElement(nia) * program_nia_weight;
            to_invert.push_back(instr_lookup_indeterminate - compressed);
        }
        
        std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
        XFieldElement log_deriv = LookupArg::default_initial();
        size_t inv_idx = 0;
        for (size_t idx = 0; idx < num_rows; idx++) {
            if (!main_table[idx][main_start + IsPadding].is_one() && inv_idx < inverses.size()) {
                log_deriv = log_deriv + inverses[inv_idx++];
            }
            aux_table[idx][aux_start + InstructionLookupClientLogDerivative] = log_deriv;
        }
    });
    
    // ---- Thread 3: Column 5 (JumpStackTablePermArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_product = PermArg::default_initial();
        for (size_t idx = 0; idx < num_rows; idx++) {
            BFieldElement clk = main_table[idx][main_start + CLK];
            BFieldElement ci = main_table[idx][main_start + CI];
            BFieldElement jsp = main_table[idx][main_start + JSP];
            BFieldElement jso = main_table[idx][main_start + JSO];
            BFieldElement jsd = main_table[idx][main_start + JSD];

            XFieldElement compressed = XFieldElement(clk) * jump_stack_clk_weight
                + XFieldElement(ci) * jump_stack_ci_weight
                + XFieldElement(jsp) * jump_stack_jsp_weight
                + XFieldElement(jso) * jump_stack_jso_weight
                + XFieldElement(jsd) * jump_stack_jsd_weight;
            
            running_product = running_product * (jump_stack_indeterminate - compressed);
            aux_table[idx][aux_start + JumpStackTablePermArg] = running_product;
        }
    });
    
    // ---- Thread 4: Column 3 (OpStackTablePermArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_product = PermArg::default_initial();
        aux_table[0][aux_start + OpStackTablePermArg] = running_product;
        
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_curr = main_table[idx];
            const auto& row_prev = main_table[idx - 1];
            
            if (row_curr[main_start + IsPadding].is_one()) {
                aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (!instr_opt.has_value()) {
                aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr = *instr_opt;
            int32_t influence = instr.op_stack_size_influence();
            if (influence == 0) {
                aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
                continue;
            }
            
            const auto& shorter_row = (influence > 0) ? row_prev : row_curr;
            size_t op_stack_delta = static_cast<size_t>(std::abs(influence));
            XFieldElement factor = XFieldElement::one();
            
            BFieldElement clk = row_prev[main_start + CLK];
            BFieldElement ib1 = row_prev[main_start + IB1];
            BFieldElement osp = shorter_row[main_start + OpStackPointer];
            
            for (size_t offset = 0; offset < op_stack_delta; ++offset) {
                size_t stack_idx = NUM_OP_STACK_REGISTERS - 1 - offset;
                BFieldElement underflow = shorter_row[main_start + OP_STACK_COLUMNS[stack_idx]];
                BFieldElement ptr = osp + BFieldElement(static_cast<uint64_t>(offset));
                
                XFieldElement compressed = XFieldElement(clk) * op_stack_clk_weight
                    + XFieldElement(ib1) * op_stack_ib1_weight
                    + XFieldElement(ptr) * op_stack_pointer_weight
                    + XFieldElement(underflow) * op_stack_underflow_weight;
                factor = factor * (op_stack_indeterminate - compressed);
            }
            running_product = running_product * factor;
            aux_table[idx][aux_start + OpStackTablePermArg] = running_product;
        }
    });
    
    // ---- Thread 5: Column 4 (RamTablePermArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_product = PermArg::default_initial();
        aux_table[0][aux_start + RamTablePermArg] = running_product;
        
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_curr = main_table[idx];
            const auto& row_prev = main_table[idx - 1];
            
            if (row_curr[main_start + IsPadding].is_one()) {
                aux_table[idx][aux_start + RamTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (!instr_opt.has_value()) {
                aux_table[idx][aux_start + RamTablePermArg] = running_product;
                continue;
            }
            
            const auto& instr = *instr_opt;
            BFieldElement instruction_type;
            std::vector<std::pair<BFieldElement, BFieldElement>> accesses;
            
            switch (instr.type) {
                case AnInstruction::ReadMem:
                case AnInstruction::WriteMem: {
                    instruction_type = (instr.type == AnInstruction::WriteMem) ? ram_type_write : ram_type_read;
                    const auto& longer_row = (instr.type == AnInstruction::ReadMem) ? row_curr : row_prev;
                    size_t op_stack_delta = static_cast<size_t>(std::abs(instr.op_stack_size_influence()));
                    BFieldElement ram_ptr = longer_row[main_start + ST0];
                    for (size_t off = 0; off < op_stack_delta; ++off) {
                        size_t val_col = OP_STACK_COLUMNS[off + 1];
                        BFieldElement ptr = ram_ptr + BFieldElement(static_cast<uint64_t>(off));
                        if (instr.type == AnInstruction::ReadMem) ptr = ptr + BFieldElement::one();
                        accesses.emplace_back(ptr, longer_row[main_start + val_col]);
                    }
                    break;
                }
                case AnInstruction::SpongeAbsorbMem: {
                    instruction_type = ram_type_read;
                    BFieldElement mp = row_prev[main_start + ST0];
                    accesses = {
                        {mp, row_curr[main_start + ST1]}, {mp + BFieldElement(1), row_curr[main_start + ST2]},
                        {mp + BFieldElement(2), row_curr[main_start + ST3]}, {mp + BFieldElement(3), row_curr[main_start + ST4]},
                        {mp + BFieldElement(4), row_prev[main_start + HV0]}, {mp + BFieldElement(5), row_prev[main_start + HV1]},
                        {mp + BFieldElement(6), row_prev[main_start + HV2]}, {mp + BFieldElement(7), row_prev[main_start + HV3]},
                        {mp + BFieldElement(8), row_prev[main_start + HV4]}, {mp + BFieldElement(9), row_prev[main_start + HV5]}
                    };
                    break;
                }
                case AnInstruction::MerkleStepMem: {
                    instruction_type = ram_type_read;
                    BFieldElement mp = row_prev[main_start + ST7];
                    accesses = {
                        {mp, row_prev[main_start + HV0]}, {mp + BFieldElement(1), row_prev[main_start + HV1]},
                        {mp + BFieldElement(2), row_prev[main_start + HV2]}, {mp + BFieldElement(3), row_prev[main_start + HV3]},
                        {mp + BFieldElement(4), row_prev[main_start + HV4]}
                    };
                    break;
                }
                case AnInstruction::XxDotStep: {
                    instruction_type = ram_type_read;
                    BFieldElement rhs = row_prev[main_start + ST0], lhs = row_prev[main_start + ST1];
                    accesses = {
                        {rhs, row_prev[main_start + HV0]}, {rhs + BFieldElement(1), row_prev[main_start + HV1]},
                        {rhs + BFieldElement(2), row_prev[main_start + HV2]}, {lhs, row_prev[main_start + HV3]},
                        {lhs + BFieldElement(1), row_prev[main_start + HV4]}, {lhs + BFieldElement(2), row_prev[main_start + HV5]}
                    };
                    break;
                }
                case AnInstruction::XbDotStep: {
                    instruction_type = ram_type_read;
                    BFieldElement rhs = row_prev[main_start + ST0], lhs = row_prev[main_start + ST1];
                    accesses = {
                        {rhs, row_prev[main_start + HV0]}, {lhs, row_prev[main_start + HV1]},
                        {lhs + BFieldElement(1), row_prev[main_start + HV2]}, {lhs + BFieldElement(2), row_prev[main_start + HV3]}
                    };
                    break;
                }
                default:
                    aux_table[idx][aux_start + RamTablePermArg] = running_product;
                    continue;
            }
            
            if (!accesses.empty()) {
                BFieldElement clk = row_prev[main_start + CLK];
                XFieldElement factor = XFieldElement::one();
                for (const auto& [ptr, val] : accesses) {
                    XFieldElement compressed = XFieldElement(clk) * ram_clk_weight
                        + XFieldElement(instruction_type) * ram_type_weight
                        + XFieldElement(ptr) * ram_pointer_weight
                        + XFieldElement(val) * ram_value_weight;
                    factor = factor * (ram_indeterminate - compressed);
                }
                running_product = running_product * factor;
            }
            aux_table[idx][aux_start + RamTablePermArg] = running_product;
        }
    });
    
    // ---- Thread 6: Column 6 (HashInputEvalArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_eval = EvalArg::default_initial();
        for (size_t idx = 0; idx < num_rows; idx++) {
            const auto& row = main_table[idx];
            const auto& instr_opt = decoded_instructions[idx];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                const std::array<size_t, 10>* cols = nullptr;
                if (instr.type == AnInstruction::Hash) {
                    cols = &ST_INDICES;
                } else if (instr.type == AnInstruction::MerkleStep || instr.type == AnInstruction::MerkleStepMem) {
                    bool is_left = (row[main_start + ST5].value() % 2ULL) == 0ULL;
                    cols = is_left ? &MERKLE_LEFT : &MERKLE_RIGHT;
                }
                if (cols) {
                    XFieldElement compressed = XFieldElement::zero();
                    for (size_t j = 0; j < 10; ++j) {
                        compressed += hash_state_weights[j] * XFieldElement(row[main_start + (*cols)[j]]);
                    }
                    running_eval = running_eval * hash_input_indeterminate + compressed;
                }
            }
            aux_table[idx][aux_start + HashInputEvalArg] = running_eval;
        }
    });
    
    // ---- Thread 7: Column 7 (HashDigestEvalArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_eval = EvalArg::default_initial();
        aux_table[0][aux_start + HashDigestEvalArg] = running_eval;
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                if (instr.type == AnInstruction::Hash || instr.type == AnInstruction::MerkleStep || instr.type == AnInstruction::MerkleStepMem) {
                    XFieldElement compressed = XFieldElement::zero();
                    for (size_t j = 0; j < 5; ++j) {
                        compressed += hash_state_weights[j] * XFieldElement(main_table[idx][main_start + ST0 + j]);
                    }
                    running_eval = running_eval * hash_digest_indeterminate + compressed;
                }
            }
            aux_table[idx][aux_start + HashDigestEvalArg] = running_eval;
        }
    });
    
    // ---- Thread 8: Column 8 (SpongeEvalArg) ----
    threads.emplace_back([&]() {
        XFieldElement running_eval = EvalArg::default_initial();
        aux_table[0][aux_start + SpongeEvalArg] = running_eval;
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_prev = main_table[idx - 1];
            const auto& row_curr = main_table[idx];
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (instr_opt.has_value()) {
                const auto& instr = *instr_opt;
                if (instr.type == AnInstruction::SpongeInit) {
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(row_prev[main_start + CI]);
                } else if (instr.type == AnInstruction::SpongeAbsorb) {
                    // OPTIMIZED: Unroll hash state compression (10 operations)
                    XFieldElement compressed = hash_state_weights[0] * XFieldElement(row_prev[main_start + ST_INDICES[0]]) +
                                             hash_state_weights[1] * XFieldElement(row_prev[main_start + ST_INDICES[1]]) +
                                             hash_state_weights[2] * XFieldElement(row_prev[main_start + ST_INDICES[2]]) +
                                             hash_state_weights[3] * XFieldElement(row_prev[main_start + ST_INDICES[3]]) +
                                             hash_state_weights[4] * XFieldElement(row_prev[main_start + ST_INDICES[4]]) +
                                             hash_state_weights[5] * XFieldElement(row_prev[main_start + ST_INDICES[5]]) +
                                             hash_state_weights[6] * XFieldElement(row_prev[main_start + ST_INDICES[6]]) +
                                             hash_state_weights[7] * XFieldElement(row_prev[main_start + ST_INDICES[7]]) +
                                             hash_state_weights[8] * XFieldElement(row_prev[main_start + ST_INDICES[8]]) +
                                             hash_state_weights[9] * XFieldElement(row_prev[main_start + ST_INDICES[9]]);
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(row_prev[main_start + CI]) + compressed;
                } else if (instr.type == AnInstruction::SpongeAbsorbMem) {
                    // OPTIMIZED: Unroll absorb mem compression (4 + 6 operations)
                    XFieldElement compressed = hash_state_weights[0] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[0]]) +
                                             hash_state_weights[1] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[1]]) +
                                             hash_state_weights[2] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[2]]) +
                                             hash_state_weights[3] * XFieldElement(row_curr[main_start + ABSORB_MEM_STACK[3]]) +
                                             hash_state_weights[4] * XFieldElement(row_prev[main_start + HV_INDICES[0]]) +
                                             hash_state_weights[5] * XFieldElement(row_prev[main_start + HV_INDICES[1]]) +
                                             hash_state_weights[6] * XFieldElement(row_prev[main_start + HV_INDICES[2]]) +
                                             hash_state_weights[7] * XFieldElement(row_prev[main_start + HV_INDICES[3]]) +
                                             hash_state_weights[8] * XFieldElement(row_prev[main_start + HV_INDICES[4]]) +
                                             hash_state_weights[9] * XFieldElement(row_prev[main_start + HV_INDICES[5]]);
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(sponge_absorb_opcode) + compressed;
                } else if (instr.type == AnInstruction::SpongeSqueeze) {
                    // OPTIMIZED: Unroll squeeze compression (10 operations)
                    XFieldElement compressed = hash_state_weights[0] * XFieldElement(row_curr[main_start + ST_INDICES[0]]) +
                                             hash_state_weights[1] * XFieldElement(row_curr[main_start + ST_INDICES[1]]) +
                                             hash_state_weights[2] * XFieldElement(row_curr[main_start + ST_INDICES[2]]) +
                                             hash_state_weights[3] * XFieldElement(row_curr[main_start + ST_INDICES[3]]) +
                                             hash_state_weights[4] * XFieldElement(row_curr[main_start + ST_INDICES[4]]) +
                                             hash_state_weights[5] * XFieldElement(row_curr[main_start + ST_INDICES[5]]) +
                                             hash_state_weights[6] * XFieldElement(row_curr[main_start + ST_INDICES[6]]) +
                                             hash_state_weights[7] * XFieldElement(row_curr[main_start + ST_INDICES[7]]) +
                                             hash_state_weights[8] * XFieldElement(row_curr[main_start + ST_INDICES[8]]) +
                                             hash_state_weights[9] * XFieldElement(row_curr[main_start + ST_INDICES[9]]);
                    running_eval = running_eval * sponge_indeterminate
                        + hash_ci_weight * XFieldElement(row_prev[main_start + CI]) + compressed;
                }
            }
            aux_table[idx][aux_start + SpongeEvalArg] = running_eval;
        }
    });

    // ---- Thread 9: Column 9 (U32LookupClientLogDerivative) ----
    threads.emplace_back([&]() {
        std::vector<XFieldElement> to_invert;
        to_invert.reserve(num_rows);
        std::vector<size_t> inverse_counts(num_rows, 0);  // Track how many inverses per row
        
        const BFieldElement and_opcode = opcode_to_b_field(AnInstruction::And);
        const BFieldElement lt_opcode = opcode_to_b_field(AnInstruction::Lt);
        const BFieldElement split_opcode = opcode_to_b_field(AnInstruction::Split);
        
        for (size_t idx = 1; idx < num_rows; idx++) {
            const auto& row_prev = main_table[idx - 1];
            const auto& row_curr = main_table[idx];
            const auto& instr_opt = decoded_instructions[idx - 1];
            if (!instr_opt.has_value()) continue;
            const auto& instr = *instr_opt;
            
            BFieldElement prev_ci = row_prev[main_start + CI];
            auto push = [&](const XFieldElement& term) {
                to_invert.push_back(u32_indeterminate - term);
                inverse_counts[idx]++;
            };

            if (instr.type == AnInstruction::Split) {
                push(XFieldElement(row_curr[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_curr[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(prev_ci) * u32_ci_weight);
            } else if (instr.type == AnInstruction::Lt || instr.type == AnInstruction::And || instr.type == AnInstruction::Pow) {
                push(XFieldElement(row_prev[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_prev[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(prev_ci) * u32_ci_weight
                    + XFieldElement(row_curr[main_start + ST0]) * u32_result_weight);
            } else if (instr.type == AnInstruction::Xor) {
                BFieldElement st0p = row_prev[main_start + ST0], st1p = row_prev[main_start + ST1], st0c = row_curr[main_start + ST0];
                BFieldElement from_xor = (st0p + st1p - st0c) / BFieldElement(2);
                push(XFieldElement(st0p) * u32_lhs_weight + XFieldElement(st1p) * u32_rhs_weight
                    + XFieldElement(and_opcode) * u32_ci_weight + XFieldElement(from_xor) * u32_result_weight);
            } else if (instr.type == AnInstruction::Log2Floor || instr.type == AnInstruction::PopCount) {
                push(XFieldElement(row_prev[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(prev_ci) * u32_ci_weight
                    + XFieldElement(row_curr[main_start + ST0]) * u32_result_weight);
            } else if (instr.type == AnInstruction::DivMod) {
                push(XFieldElement(row_curr[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_prev[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(lt_opcode) * u32_ci_weight
                    + XFieldElement(BFieldElement(1)) * u32_result_weight);
                push(XFieldElement(row_prev[main_start + ST0]) * u32_lhs_weight
                    + XFieldElement(row_curr[main_start + ST1]) * u32_rhs_weight
                    + XFieldElement(split_opcode) * u32_ci_weight);
            } else if (instr.type == AnInstruction::MerkleStep || instr.type == AnInstruction::MerkleStepMem) {
                push(XFieldElement(row_prev[main_start + ST5]) * u32_lhs_weight
                    + XFieldElement(row_curr[main_start + ST5]) * u32_rhs_weight
                    + XFieldElement(split_opcode) * u32_ci_weight);
            }
        }

        std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
        XFieldElement log_deriv = LookupArg::default_initial();
        size_t inv_idx = 0;
        aux_table[0][aux_start + U32LookupClientLogDerivative] = log_deriv;
        for (size_t idx = 1; idx < num_rows; idx++) {
            for (size_t k = 0; k < inverse_counts[idx] && inv_idx < inverses.size(); ++k) {
                log_deriv += inverses[inv_idx++];
            }
            aux_table[idx][aux_start + U32LookupClientLogDerivative] = log_deriv;
        }
    });
    
    // ---- Thread 10: Column 10 (ClockJumpDifferenceLookupClientLogDerivative) ----
    threads.emplace_back([&]() {
        std::vector<XFieldElement> to_invert;
        std::vector<std::pair<size_t, BFieldElement>> idx_mult;
        to_invert.reserve(num_rows);
        idx_mult.reserve(num_rows);
        
        for (size_t idx = 0; idx < num_rows; idx++) {
            BFieldElement mult = main_table[idx][main_start + ClockJumpDifferenceLookupMultiplicity];
            if (!mult.is_zero()) {
                BFieldElement clk = main_table[idx][main_start + CLK];
                to_invert.push_back(cjd_indeterminate - XFieldElement(clk));
                idx_mult.emplace_back(idx, mult);
            }
        }
        
        std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
        XFieldElement log_deriv = LookupArg::default_initial();
        size_t inv_idx = 0;
        size_t next_inv = 0;
        for (size_t idx = 0; idx < num_rows; idx++) {
            if (next_inv < idx_mult.size() && idx_mult[next_inv].first == idx) {
                log_deriv += inverses[inv_idx++] * XFieldElement(idx_mult[next_inv].second);
                next_inv++;
            }
            aux_table[idx][aux_start + ClockJumpDifferenceLookupClientLogDerivative] = log_deriv;
        }
    });
    
    for (auto& t : threads) {
        t.join();
    }
}

// =============================================================================
// OPTIMIZED extend_ram_table
// =============================================================================
// Optimizations:
// 1. Direct array access throughout
// 2. Fixed array cache for CJD inverses (0-99)
// 3. Single pass for finding padding boundary
// =============================================================================

void extend_ram_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace RamMainColumn;
    using namespace RamAuxColumn;

    const size_t main_start = RAM_TABLE_START;
    const size_t aux_start = AUX_RAM_TABLE_START;
    constexpr BFieldElement PADDING_INDICATOR(2);

    // Cache challenges
    const XFieldElement bezout_indeterminate = challenges[RamTableBezoutRelationIndeterminate];
    const XFieldElement ram_indeterminate = challenges[RamIndeterminate];
    const XFieldElement cjd_indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];
    const XFieldElement clk_weight = challenges[RamClkWeight];
    const XFieldElement type_weight = challenges[RamInstructionTypeWeight];
    const XFieldElement ptr_weight = challenges[RamPointerWeight];
    const XFieldElement val_weight = challenges[RamValueWeight];

    // Pre-compute CJD inverses for common values 0-99
    std::array<XFieldElement, 100> cached_cjd_inv;
    for (uint64_t i = 0; i < 100; i++) {
        cached_cjd_inv[i] = (cjd_indeterminate - XFieldElement(BFieldElement(i))).inverse();
    }

    // Find padding boundary first
    size_t padding_start = num_rows;
    for (size_t idx = 0; idx < num_rows; idx++) {
        if (main_table[idx][main_start + InstructionType] == PADDING_INDICATOR) {
            padding_start = idx;
            break;
        }
    }

    // Initialize running values
    XFieldElement running_product_ram_pointer = bezout_indeterminate - XFieldElement(main_table[0][main_start + RamPointer]);
    XFieldElement formal_derivative = XFieldElement::one();
    XFieldElement bezout_coeff_0 = XFieldElement(main_table[0][main_start + BezoutCoefficientPolynomialCoefficient0]);
    XFieldElement bezout_coeff_1 = XFieldElement(main_table[0][main_start + BezoutCoefficientPolynomialCoefficient1]);
    XFieldElement running_product_perm_arg = PermArg::default_initial();
    XFieldElement cjd_log_derivative = LookupArg::default_initial();

    // Process first row
    {
        const auto& row = main_table[0];
        BFieldElement clk = row[main_start + CLK];
        BFieldElement instr_type = row[main_start + InstructionType];
        BFieldElement ram_ptr = row[main_start + RamPointer];
        BFieldElement ram_val = row[main_start + RamValue];
        
        XFieldElement compressed = XFieldElement(clk) * clk_weight
            + XFieldElement(instr_type) * type_weight
            + XFieldElement(ram_ptr) * ptr_weight
            + XFieldElement(ram_val) * val_weight;
        running_product_perm_arg = running_product_perm_arg * (ram_indeterminate - compressed);
    }
    
    aux_table[0][aux_start + RunningProductOfRAMP] = running_product_ram_pointer;
    aux_table[0][aux_start + FormalDerivative] = formal_derivative;
        aux_table[0][aux_start + BezoutCoefficient0] = bezout_coeff_0;
        aux_table[0][aux_start + BezoutCoefficient1] = bezout_coeff_1;
    aux_table[0][aux_start + RunningProductPermArg] = running_product_perm_arg;
    aux_table[0][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_derivative;

    // Main processing loop (up to padding)
    for (size_t idx = 1; idx < padding_start; idx++) {
        const auto& curr = main_table[idx];
        const auto& prev = main_table[idx - 1];
        
        BFieldElement curr_ptr = curr[main_start + RamPointer];
        BFieldElement prev_ptr = prev[main_start + RamPointer];
        bool ptr_changed = (prev_ptr != curr_ptr);
        
        // Bezout coefficients
        if (ptr_changed) {
            bezout_coeff_0 = bezout_coeff_0 * bezout_indeterminate + XFieldElement(curr[main_start + BezoutCoefficientPolynomialCoefficient0]);
            bezout_coeff_1 = bezout_coeff_1 * bezout_indeterminate + XFieldElement(curr[main_start + BezoutCoefficientPolynomialCoefficient1]);
        }
        
        // RunningProductOfRAMP and FormalDerivative
        if (ptr_changed) {
            XFieldElement diff = bezout_indeterminate - XFieldElement(curr_ptr);
            formal_derivative = diff * formal_derivative + running_product_ram_pointer;
            running_product_ram_pointer = running_product_ram_pointer * diff;
        }
        
        // RunningProductPermArg
        BFieldElement clk = curr[main_start + CLK];
        BFieldElement instr_type = curr[main_start + InstructionType];
        BFieldElement ram_val = curr[main_start + RamValue];
        XFieldElement compressed = XFieldElement(clk) * clk_weight
            + XFieldElement(instr_type) * type_weight
            + XFieldElement(curr_ptr) * ptr_weight
            + XFieldElement(ram_val) * val_weight;
        running_product_perm_arg = running_product_perm_arg * (ram_indeterminate - compressed);
        
        // CJD log derivative
        if (!ptr_changed) {
            uint64_t clock_diff = (clk - prev[main_start + CLK]).value();
            XFieldElement inv = (clock_diff < 100) ? cached_cjd_inv[clock_diff]
                : (cjd_indeterminate - XFieldElement(BFieldElement(clock_diff))).inverse();
            cjd_log_derivative += inv;
        }
        
        aux_table[idx][aux_start + RunningProductOfRAMP] = running_product_ram_pointer;
        aux_table[idx][aux_start + FormalDerivative] = formal_derivative;
            aux_table[idx][aux_start + BezoutCoefficient0] = bezout_coeff_0;
            aux_table[idx][aux_start + BezoutCoefficient1] = bezout_coeff_1;
        aux_table[idx][aux_start + RunningProductPermArg] = running_product_perm_arg;
            aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_derivative;
    }
    
    // Fill padding rows with last values
    for (size_t idx = padding_start; idx < num_rows; idx++) {
        aux_table[idx][aux_start + RunningProductOfRAMP] = running_product_ram_pointer;
        aux_table[idx][aux_start + FormalDerivative] = formal_derivative;
        aux_table[idx][aux_start + BezoutCoefficient0] = bezout_coeff_0;
        aux_table[idx][aux_start + BezoutCoefficient1] = bezout_coeff_1;
        aux_table[idx][aux_start + RunningProductPermArg] = running_product_perm_arg;
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_derivative;
}
}

void extend_ram_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace RamMainColumn;
    using namespace RamAuxColumn;

    const size_t main_start = RAM_TABLE_START;
    const size_t aux_start = AUX_RAM_TABLE_START;
    constexpr BFieldElement PADDING_INDICATOR(2);

    auto at = [&](size_t r, size_t rel) -> BFieldElement { return main_table.at(r, main_start + rel); };

    // Cache challenges
    const XFieldElement bezout_indeterminate = challenges[RamTableBezoutRelationIndeterminate];
    const XFieldElement ram_indeterminate = challenges[RamIndeterminate];
    const XFieldElement cjd_indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];
    const XFieldElement clk_weight = challenges[RamClkWeight];
    const XFieldElement type_weight = challenges[RamInstructionTypeWeight];
    const XFieldElement ptr_weight = challenges[RamPointerWeight];
    const XFieldElement val_weight = challenges[RamValueWeight];

    std::array<XFieldElement, 100> cached_cjd_inv;
    for (uint64_t i = 0; i < 100; i++) {
        cached_cjd_inv[i] = (cjd_indeterminate - XFieldElement(BFieldElement(i))).inverse();
    }

    size_t padding_start = num_rows;
    for (size_t idx = 0; idx < num_rows; idx++) {
        if (at(idx, InstructionType) == PADDING_INDICATOR) {
            padding_start = idx;
            break;
        }
    }

    XFieldElement running_product_ram_pointer = bezout_indeterminate - XFieldElement(at(0, RamPointer));
    XFieldElement formal_derivative = XFieldElement::one();
    XFieldElement bezout_coeff_0 = XFieldElement(at(0, BezoutCoefficientPolynomialCoefficient0));
    XFieldElement bezout_coeff_1 = XFieldElement(at(0, BezoutCoefficientPolynomialCoefficient1));
    XFieldElement running_product_perm_arg = PermArg::default_initial();
    XFieldElement cjd_log_derivative = LookupArg::default_initial();

    // First row
    {
        BFieldElement clk = at(0, CLK);
        BFieldElement instr_type = at(0, InstructionType);
        BFieldElement ram_ptr = at(0, RamPointer);
        BFieldElement ram_val = at(0, RamValue);

        XFieldElement compressed = XFieldElement(clk) * clk_weight
            + XFieldElement(instr_type) * type_weight
            + XFieldElement(ram_ptr) * ptr_weight
            + XFieldElement(ram_val) * val_weight;
        running_product_perm_arg = running_product_perm_arg * (ram_indeterminate - compressed);
    }

    aux_table[0][aux_start + RunningProductOfRAMP] = running_product_ram_pointer;
    aux_table[0][aux_start + FormalDerivative] = formal_derivative;
    aux_table[0][aux_start + BezoutCoefficient0] = bezout_coeff_0;
    aux_table[0][aux_start + BezoutCoefficient1] = bezout_coeff_1;
    aux_table[0][aux_start + RunningProductPermArg] = running_product_perm_arg;
    aux_table[0][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_derivative;

    for (size_t idx = 1; idx < padding_start; idx++) {
        BFieldElement curr_ptr = at(idx, RamPointer);
        BFieldElement prev_ptr = at(idx - 1, RamPointer);
        bool ptr_changed = (prev_ptr != curr_ptr);

        if (ptr_changed) {
            bezout_coeff_0 = bezout_coeff_0 * bezout_indeterminate + XFieldElement(at(idx, BezoutCoefficientPolynomialCoefficient0));
            bezout_coeff_1 = bezout_coeff_1 * bezout_indeterminate + XFieldElement(at(idx, BezoutCoefficientPolynomialCoefficient1));
        }

        if (ptr_changed) {
            XFieldElement diff = bezout_indeterminate - XFieldElement(curr_ptr);
            formal_derivative = diff * formal_derivative + running_product_ram_pointer;
            running_product_ram_pointer = running_product_ram_pointer * diff;
        }

        BFieldElement clk = at(idx, CLK);
        BFieldElement instr_type = at(idx, InstructionType);
        BFieldElement ram_val = at(idx, RamValue);
        XFieldElement compressed = XFieldElement(clk) * clk_weight
            + XFieldElement(instr_type) * type_weight
            + XFieldElement(curr_ptr) * ptr_weight
            + XFieldElement(ram_val) * val_weight;
        running_product_perm_arg = running_product_perm_arg * (ram_indeterminate - compressed);

        if (!ptr_changed) {
            uint64_t clock_diff = (clk - at(idx - 1, CLK)).value();
            XFieldElement inv = (clock_diff < 100) ? cached_cjd_inv[clock_diff]
                : (cjd_indeterminate - XFieldElement(BFieldElement(clock_diff))).inverse();
            cjd_log_derivative += inv;
        }

        aux_table[idx][aux_start + RunningProductOfRAMP] = running_product_ram_pointer;
        aux_table[idx][aux_start + FormalDerivative] = formal_derivative;
        aux_table[idx][aux_start + BezoutCoefficient0] = bezout_coeff_0;
        aux_table[idx][aux_start + BezoutCoefficient1] = bezout_coeff_1;
        aux_table[idx][aux_start + RunningProductPermArg] = running_product_perm_arg;
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_derivative;
    }

    for (size_t idx = padding_start; idx < num_rows; idx++) {
        aux_table[idx][aux_start + RunningProductOfRAMP] = running_product_ram_pointer;
        aux_table[idx][aux_start + FormalDerivative] = formal_derivative;
        aux_table[idx][aux_start + BezoutCoefficient0] = bezout_coeff_0;
        aux_table[idx][aux_start + BezoutCoefficient1] = bezout_coeff_1;
        aux_table[idx][aux_start + RunningProductPermArg] = running_product_perm_arg;
        aux_table[idx][aux_start + ClockJumpDifferenceLookupLogDerivative] = cjd_log_derivative;
    }
}


void extend_u32_table(
    const std::vector<std::vector<BFieldElement>>& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    
    size_t main_start = U32_TABLE_START;
    size_t aux_start = AUX_U32_TABLE_START;
    
    // Exact copy of Rust U32 implementation
    XFieldElement running_sum_log_derivative = LookupArg::default_initial();
    XFieldElement lookup_indeterminate = challenges[U32Indeterminate];

    XFieldElement ci_weight = challenges[U32CiWeight];
    XFieldElement lhs_weight = challenges[U32LhsWeight];
    XFieldElement rhs_weight = challenges[U32RhsWeight];
    XFieldElement result_weight = challenges[U32ResultWeight];

    for (size_t idx = 0; idx < num_rows; idx++) {
        auto row = get_main_table_row(main_table, idx, main_start, U32_TABLE_COLS);

        if (row[U32MainColumn::CopyFlag].is_one()) {
            XFieldElement compressed_row = ci_weight * row[U32MainColumn::CI]
                + lhs_weight * row[U32MainColumn::LHS]
                + rhs_weight * row[U32MainColumn::RHS]
                + result_weight * row[U32MainColumn::Result];
            running_sum_log_derivative +=
                XFieldElement(row[U32MainColumn::LookupMultiplicity]) * (lookup_indeterminate - compressed_row).inverse();
                }

        aux_table[idx][aux_start + U32AuxColumn::LookupServerLogDerivative] = running_sum_log_derivative;
}
}

void extend_u32_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace U32MainColumn;
    using namespace U32AuxColumn;

    const size_t main_start = U32_TABLE_START;
    const size_t aux_start = AUX_U32_TABLE_START;

    auto at = [&](size_t r, size_t rel) -> BFieldElement { return main_table.at(r, main_start + rel); };

    XFieldElement running_sum_log_derivative = LookupArg::default_initial();
    XFieldElement lookup_indeterminate = challenges[U32Indeterminate];

    XFieldElement ci_weight = challenges[U32CiWeight];
    XFieldElement lhs_weight = challenges[U32LhsWeight];
    XFieldElement rhs_weight = challenges[U32RhsWeight];
    XFieldElement result_weight = challenges[U32ResultWeight];

    for (size_t idx = 0; idx < num_rows; idx++) {
        if (at(idx, CopyFlag).is_one()) {
            XFieldElement compressed_row = ci_weight * XFieldElement(at(idx, CI))
                + lhs_weight * XFieldElement(at(idx, LHS))
                + rhs_weight * XFieldElement(at(idx, RHS))
                + result_weight * XFieldElement(at(idx, Result));
            running_sum_log_derivative +=
                XFieldElement(at(idx, LookupMultiplicity)) * (lookup_indeterminate - compressed_row).inverse();
        }

        aux_table[idx][aux_start + LookupServerLogDerivative] = running_sum_log_derivative;
    }
}

void extend_cascade_table(
    const MainTableFlatView& main_table,
    std::vector<std::vector<XFieldElement>>& aux_table,
    const Challenges& challenges,
    size_t num_rows
) {
    using namespace ChallengeId;
    using namespace CascadeMainColumn;
    using namespace CascadeAuxColumn;
    
    const size_t main_start = CASCADE_TABLE_START;
    const size_t aux_start = AUX_CASCADE_TABLE_START;
    
    // Cache challenges
    const XFieldElement hash_indeterminate = challenges[HashCascadeLookupIndeterminate];
    const XFieldElement hash_input_weight = challenges[HashCascadeLookInWeight];
    const XFieldElement hash_output_weight = challenges[HashCascadeLookOutWeight];
    const XFieldElement lookup_indeterminate = challenges[CascadeLookupIndeterminate];
    const XFieldElement lookup_input_weight = challenges[LookupTableInputWeight];
    const XFieldElement lookup_output_weight = challenges[LookupTableOutputWeight];
    const BFieldElement two_pow_8(1ULL << 8);
    
    // PASS 1: collect values to invert
    std::vector<XFieldElement> to_invert;
    std::vector<size_t> active_rows;
    std::vector<BFieldElement> multiplicities;
    to_invert.reserve(num_rows * 3);
    active_rows.reserve(num_rows);
    multiplicities.reserve(num_rows);
    
    for (size_t idx = 0; idx < num_rows; idx++) {
        const auto row = main_table[idx];
        if (!row[main_start + IsPadding].is_one()) {
            active_rows.push_back(idx);
            multiplicities.push_back(row[main_start + CascadeMainColumn::LookupMultiplicity]);
            
            BFieldElement look_in_hi = row[main_start + LookInHi];
            BFieldElement look_in_lo = row[main_start + LookInLo];
            BFieldElement look_out_hi = row[main_start + LookOutHi];
            BFieldElement look_out_lo = row[main_start + LookOutLo];
            
            XFieldElement look_in = XFieldElement(two_pow_8 * look_in_hi + look_in_lo);
            XFieldElement look_out = XFieldElement(two_pow_8 * look_out_hi + look_out_lo);
            to_invert.push_back(hash_indeterminate - hash_input_weight * look_in - hash_output_weight * look_out);
            
            to_invert.push_back(lookup_indeterminate - lookup_input_weight * XFieldElement(look_in_lo) - lookup_output_weight * XFieldElement(look_out_lo));
            to_invert.push_back(lookup_indeterminate - lookup_input_weight * XFieldElement(look_in_hi) - lookup_output_weight * XFieldElement(look_out_hi));
        }
    }
    
    std::vector<XFieldElement> inverses = XFieldElement::batch_inversion(to_invert);
    
    XFieldElement hash_log_deriv = LookupArg::default_initial();
    XFieldElement lookup_log_deriv = LookupArg::default_initial();
    size_t active_idx = 0;
    size_t inv_idx = 0;
    
    for (size_t idx = 0; idx < num_rows; idx++) {
        if (active_idx < active_rows.size() && active_rows[active_idx] == idx) {
            hash_log_deriv += inverses[inv_idx] * XFieldElement(multiplicities[active_idx]);
            lookup_log_deriv += inverses[inv_idx + 1] + inverses[inv_idx + 2];
            inv_idx += 3;
            active_idx++;
        }
        aux_table[idx][aux_start + HashTableServerLogDerivative] = hash_log_deriv;
        aux_table[idx][aux_start + LookupTableClientLogDerivative] = lookup_log_deriv;
    }
}

} // namespace triton_vm
