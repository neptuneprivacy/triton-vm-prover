#include "vm/aet.hpp"
#include "vm/vm_state.hpp"
#include "hash/tip5.hpp"
#include "table/extend_helpers.hpp"
#include "parallel/thread_coordination.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#ifdef TVM_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/enumerable_thread_specific.h>
#endif
#ifdef TVM_USE_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

namespace triton_vm {

namespace {

constexpr size_t HASH_TABLE_WIDTH = 67;

constexpr BFieldElement montgomery_modulus() {
    return BFieldElement(BFieldElement::MONTGOMERY_R);
}

inline BFieldElement inverse_or_zero(const BFieldElement& x) {
    return x.is_zero() ? BFieldElement::zero() : x.inverse();
}

inline std::array<uint16_t, 4> base_field_element_into_16_bit_limbs(const BFieldElement& x) {
    // Match Rust `table::hash::base_field_element_into_16_bit_limbs`:
    // limbs of (R * x).value() in little-endian 16-bit chunks.
    uint64_t r_times_x = (montgomery_modulus() * x).value();
    return {
        static_cast<uint16_t>((r_times_x >> 0) & 0xFFFFULL),
        static_cast<uint16_t>((r_times_x >> 16) & 0xFFFFULL),
        static_cast<uint16_t>((r_times_x >> 32) & 0xFFFFULL),
        static_cast<uint16_t>((r_times_x >> 48) & 0xFFFFULL),
    };
}

inline BFieldElement lookup_16_bit_limb(uint16_t to_look_up) {
    uint8_t lo = static_cast<uint8_t>(to_look_up & 0xFF);
    uint8_t hi = static_cast<uint8_t>((to_look_up >> 8) & 0xFF);
    BFieldElement looked_up_lo(static_cast<uint64_t>(Tip5::LOOKUP_TABLE[lo]));
    BFieldElement looked_up_hi(static_cast<uint64_t>(Tip5::LOOKUP_TABLE[hi]));
    return BFieldElement(256) * looked_up_hi + looked_up_lo;
}

inline BFieldElement inverse_or_zero_of_highest_2_limbs(const BFieldElement& state_element) {
    auto limbs = base_field_element_into_16_bit_limbs(state_element);
    uint64_t highest = limbs[3];
    uint64_t mid_high = limbs[2];
    BFieldElement high_limbs((highest << 16) + mid_high);
    BFieldElement two_pow_32_minus_1((1ULL << 32) - 1);
    BFieldElement to_invert = two_pow_32_minus_1 - high_limbs;
    return inverse_or_zero(to_invert);
}

inline std::vector<BFieldElement> trace_row_to_hash_table_row(
    const std::array<BFieldElement, Tip5::STATE_SIZE>& trace_row,
    size_t round_number
) {
    using namespace HashMainColumn;
    std::vector<BFieldElement> row(HASH_TABLE_WIDTH, BFieldElement::zero());

    row[RoundNumber] = BFieldElement(static_cast<uint64_t>(round_number));

    // Split-and-lookup columns for state elements 0..3
    {
        auto limbs = base_field_element_into_16_bit_limbs(trace_row[0]);
        row[State0LowestLkIn] = BFieldElement(limbs[0]);
        row[State0MidLowLkIn] = BFieldElement(limbs[1]);
        row[State0MidHighLkIn] = BFieldElement(limbs[2]);
        row[State0HighestLkIn] = BFieldElement(limbs[3]);
        row[State0LowestLkOut] = lookup_16_bit_limb(limbs[0]);
        row[State0MidLowLkOut] = lookup_16_bit_limb(limbs[1]);
        row[State0MidHighLkOut] = lookup_16_bit_limb(limbs[2]);
        row[State0HighestLkOut] = lookup_16_bit_limb(limbs[3]);
    }
    {
        auto limbs = base_field_element_into_16_bit_limbs(trace_row[1]);
        row[State1LowestLkIn] = BFieldElement(limbs[0]);
        row[State1MidLowLkIn] = BFieldElement(limbs[1]);
        row[State1MidHighLkIn] = BFieldElement(limbs[2]);
        row[State1HighestLkIn] = BFieldElement(limbs[3]);
        row[State1LowestLkOut] = lookup_16_bit_limb(limbs[0]);
        row[State1MidLowLkOut] = lookup_16_bit_limb(limbs[1]);
        row[State1MidHighLkOut] = lookup_16_bit_limb(limbs[2]);
        row[State1HighestLkOut] = lookup_16_bit_limb(limbs[3]);
    }
    {
        auto limbs = base_field_element_into_16_bit_limbs(trace_row[2]);
        row[State2LowestLkIn] = BFieldElement(limbs[0]);
        row[State2MidLowLkIn] = BFieldElement(limbs[1]);
        row[State2MidHighLkIn] = BFieldElement(limbs[2]);
        row[State2HighestLkIn] = BFieldElement(limbs[3]);
        row[State2LowestLkOut] = lookup_16_bit_limb(limbs[0]);
        row[State2MidLowLkOut] = lookup_16_bit_limb(limbs[1]);
        row[State2MidHighLkOut] = lookup_16_bit_limb(limbs[2]);
        row[State2HighestLkOut] = lookup_16_bit_limb(limbs[3]);
    }
    {
        auto limbs = base_field_element_into_16_bit_limbs(trace_row[3]);
        row[State3LowestLkIn] = BFieldElement(limbs[0]);
        row[State3MidLowLkIn] = BFieldElement(limbs[1]);
        row[State3MidHighLkIn] = BFieldElement(limbs[2]);
        row[State3HighestLkIn] = BFieldElement(limbs[3]);
        row[State3LowestLkOut] = lookup_16_bit_limb(limbs[0]);
        row[State3MidLowLkOut] = lookup_16_bit_limb(limbs[1]);
        row[State3MidHighLkOut] = lookup_16_bit_limb(limbs[2]);
        row[State3HighestLkOut] = lookup_16_bit_limb(limbs[3]);
    }

    // Unsplit state elements 4..15
    row[State4] = trace_row[4];
    row[State5] = trace_row[5];
    row[State6] = trace_row[6];
    row[State7] = trace_row[7];
    row[State8] = trace_row[8];
    row[State9] = trace_row[9];
    row[State10] = trace_row[10];
    row[State11] = trace_row[11];
    row[State12] = trace_row[12];
    row[State13] = trace_row[13];
    row[State14] = trace_row[14];
    row[State15] = trace_row[15];

    // State inverses
    row[State0Inv] = inverse_or_zero_of_highest_2_limbs(trace_row[0]);
    row[State1Inv] = inverse_or_zero_of_highest_2_limbs(trace_row[1]);
    row[State2Inv] = inverse_or_zero_of_highest_2_limbs(trace_row[2]);
    row[State3Inv] = inverse_or_zero_of_highest_2_limbs(trace_row[3]);

    // Round constants
    if (round_number < Tip5::NUM_ROUNDS) {
        for (size_t i = 0; i < Tip5::STATE_SIZE; ++i) {
            row[Constant0 + i] = Tip5::ROUND_CONSTANTS[round_number * Tip5::STATE_SIZE + i];
        }
    } else {
        for (size_t i = 0; i < Tip5::STATE_SIZE; ++i) {
            row[Constant0 + i] = BFieldElement::zero();
        }
    }

    return row;
}

inline std::vector<std::vector<BFieldElement>> trace_to_table_rows(
    const std::vector<std::array<BFieldElement, Tip5::STATE_SIZE>>& trace
) {
    std::vector<std::vector<BFieldElement>> rows;
    rows.resize(trace.size());
    
#ifdef TVM_USE_TBB
    // Parallelize row conversion - each row is independent
    static bool tbb_verified = false;
    if (!tbb_verified && std::getenv("TVM_VERIFY_TBB")) {
        std::cout << "[TBB] Using TBB parallel_for for trace_to_table_rows (" << trace.size() << " rows)" << std::endl;
        tbb_verified = true;
    }
    tbb::parallel_for(size_t(0), trace.size(), [&](size_t round) {
        rows[round] = trace_row_to_hash_table_row(trace[round], round);
    });
#else
    // Sequential fallback
    static bool tbb_warning_shown = false;
    if (!tbb_warning_shown && std::getenv("TVM_VERIFY_TBB")) {
        std::cout << "[TBB] WARNING: TBB not available, using sequential loop for trace_to_table_rows" << std::endl;
        tbb_warning_shown = true;
    }
    for (size_t round = 0; round < trace.size(); ++round) {
        rows[round] = trace_row_to_hash_table_row(trace[round], round);
    }
#endif
    
    return rows;
}

} // namespace

AlgebraicExecutionTrace::AlgebraicExecutionTrace(std::vector<BFieldElement> program_bwords)
    // Initialize ALL members in declaration order - CRITICAL for proper initialization
    : instruction_multiplicities_()
    , program_bwords_(std::move(program_bwords))
    , processor_trace_flat_()
    , processor_trace_rows_(0)
    , processor_trace_capacity_(0)
    , processor_trace_legacy_()
    , op_stack_underflow_trace_()
    , ram_trace_()
    , program_hash_trace_()
    , hash_trace_()
    , sponge_trace_()
    , u32_entries_()
    , u32_entries_index_()
    , cascade_table_lookup_multiplicities_()
    , cascade_table_lookup_index_()
    , lookup_table_lookup_multiplicities_{}
{
    instruction_multiplicities_.resize(program_bwords_.size(), 0);
    fill_program_hash_trace();
}

void AlgebraicExecutionTrace::reserve_processor_trace(size_t capacity) {
    // OPTIMIZATION: Pre-allocate flat buffer to exact size (avoids resize during trace)
    processor_trace_flat_.resize(capacity * PROCESSOR_WIDTH, BFieldElement::zero());
    processor_trace_capacity_ = capacity;
}

void AlgebraicExecutionTrace::reserve_tables(size_t estimated_rows) {
    // Reserve space for sub-tables based on typical ratios
    op_stack_underflow_trace_.reserve(estimated_rows);
    ram_trace_.reserve(estimated_rows / 2);
    hash_trace_.reserve(estimated_rows / 10);
    sponge_trace_.reserve(estimated_rows / 10);
    u32_entries_.reserve(estimated_rows / 4);
}

void AlgebraicExecutionTrace::load_from_rust_ffi(
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
    const size_t* table_lengths_9
) {
    // Copy instruction multiplicities
    if (instruction_multiplicities_len != program_bwords_.size()) {
        throw std::runtime_error("Instruction multiplicities length mismatch: expected " + 
                                std::to_string(program_bwords_.size()) + 
                                ", got " + std::to_string(instruction_multiplicities_len));
    }
    instruction_multiplicities_.resize(instruction_multiplicities_len);
    for (size_t i = 0; i < instruction_multiplicities_len; ++i) {
        instruction_multiplicities_[i] = instruction_multiplicities[i];
    }
    
    // Copy processor trace from flat data (row-major)
    processor_trace_flat_.resize(processor_trace_rows * PROCESSOR_WIDTH);
    processor_trace_rows_ = processor_trace_rows;
    processor_trace_capacity_ = processor_trace_rows;
    
    for (size_t r = 0; r < processor_trace_rows; ++r) {
        for (size_t c = 0; c < PROCESSOR_WIDTH; ++c) {
            size_t idx = r * PROCESSOR_WIDTH + c;
            processor_trace_flat_[r * PROCESSOR_WIDTH + c] = BFieldElement(processor_trace_data[idx]);
        }
    }
    
    // Load op_stack trace
    op_stack_underflow_trace_.clear();
    op_stack_underflow_trace_.reserve(op_stack_trace_rows);
    for (size_t r = 0; r < op_stack_trace_rows; ++r) {
        std::vector<BFieldElement> row;
        row.reserve(op_stack_trace_cols);
        for (size_t c = 0; c < op_stack_trace_cols; ++c) {
            row.push_back(BFieldElement(op_stack_trace_data[r * op_stack_trace_cols + c]));
        }
        op_stack_underflow_trace_.push_back(std::move(row));
    }
    
    // Load ram trace
    ram_trace_.clear();
    ram_trace_.reserve(ram_trace_rows);
    for (size_t r = 0; r < ram_trace_rows; ++r) {
        std::vector<BFieldElement> row;
        row.reserve(ram_trace_cols);
        for (size_t c = 0; c < ram_trace_cols; ++c) {
            row.push_back(BFieldElement(ram_trace_data[r * ram_trace_cols + c]));
        }
        ram_trace_.push_back(std::move(row));
    }
    
    // Load program_hash trace
    program_hash_trace_.clear();
    program_hash_trace_.reserve(program_hash_trace_rows);
    for (size_t r = 0; r < program_hash_trace_rows; ++r) {
        std::vector<BFieldElement> row;
        row.reserve(program_hash_trace_cols);
        for (size_t c = 0; c < program_hash_trace_cols; ++c) {
            row.push_back(BFieldElement(program_hash_trace_data[r * program_hash_trace_cols + c]));
        }
        program_hash_trace_.push_back(std::move(row));
    }
    
    // Load hash trace
    hash_trace_.clear();
    hash_trace_.reserve(hash_trace_rows);
    for (size_t r = 0; r < hash_trace_rows; ++r) {
        std::vector<BFieldElement> row;
        row.reserve(hash_trace_cols);
        for (size_t c = 0; c < hash_trace_cols; ++c) {
            row.push_back(BFieldElement(hash_trace_data[r * hash_trace_cols + c]));
        }
        hash_trace_.push_back(std::move(row));
    }
    
    // Load sponge trace
    sponge_trace_.clear();
    sponge_trace_.reserve(sponge_trace_rows);
    for (size_t r = 0; r < sponge_trace_rows; ++r) {
        std::vector<BFieldElement> row;
        row.reserve(sponge_trace_cols);
        for (size_t c = 0; c < sponge_trace_cols; ++c) {
            row.push_back(BFieldElement(sponge_trace_data[r * sponge_trace_cols + c]));
        }
        sponge_trace_.push_back(std::move(row));
    }
    
    // Load U32 entries [instruction, op1, op2, mult] per entry
    u32_entries_.clear();
    u32_entries_.reserve(u32_entries_len);
    u32_entries_index_.clear();
    for (size_t i = 0; i < u32_entries_len; ++i) {
        U32TableEntry entry;
        entry.instruction_opcode = static_cast<uint32_t>(u32_entries_data[i * 4 + 0]);
        entry.left_operand = BFieldElement(u32_entries_data[i * 4 + 1]);
        entry.right_operand = BFieldElement(u32_entries_data[i * 4 + 2]);
        uint64_t mult = u32_entries_data[i * 4 + 3];
        
        size_t idx = u32_entries_.size();
        u32_entries_.push_back({entry, mult});
        u32_entries_index_[entry] = idx;
    }
    
    // Load cascade multiplicities [limb, mult] pairs
    cascade_table_lookup_multiplicities_.clear();
    cascade_table_lookup_multiplicities_.reserve(cascade_multiplicities_len);
    cascade_table_lookup_index_.clear();
    for (size_t i = 0; i < cascade_multiplicities_len; ++i) {
        uint16_t limb = static_cast<uint16_t>(cascade_multiplicities_data[i * 2 + 0]);
        uint64_t mult = cascade_multiplicities_data[i * 2 + 1];
        
        size_t idx = cascade_table_lookup_multiplicities_.size();
        cascade_table_lookup_multiplicities_.push_back({limb, mult});
        cascade_table_lookup_index_[limb] = idx;
    }
    
    // Load lookup multiplicities (256 u64)
    for (size_t i = 0; i < 256; ++i) {
        lookup_table_lookup_multiplicities_[i] = lookup_multiplicities_256[i];
    }
    
    // Store Rust table lengths for height override
    if (table_lengths_9 != nullptr) {
        std::array<size_t, 9> lengths;
        for (size_t i = 0; i < 9; ++i) {
            lengths[i] = table_lengths_9[i];
        }
        rust_table_lengths_ = lengths;
    }
}

void AlgebraicExecutionTrace::set_rust_table_lengths(const size_t* table_lengths_9) {
    if (table_lengths_9 != nullptr) {
        std::array<size_t, 9> lengths;
        for (size_t i = 0; i < 9; ++i) {
            lengths[i] = table_lengths_9[i];
        }
        rust_table_lengths_ = lengths;
    }
}

void AlgebraicExecutionTrace::record_state(const VMState& state) {
    // Record instruction lookup (increment multiplicity)
    size_t ip = state.instruction_pointer();
    if (ip < instruction_multiplicities_.size()) {
        instruction_multiplicities_[ip]++;
    }
    
    // OPTIMIZATION: Write directly to pre-allocated flat buffer (no allocation!)
    size_t row_start = processor_trace_rows_ * PROCESSOR_WIDTH;
    
    // Check if we need to grow (should be rare if reserve_processor_trace was called)
    if (row_start + PROCESSOR_WIDTH > processor_trace_flat_.size()) [[unlikely]] {
        size_t new_capacity = std::max(processor_trace_flat_.size() * 2, (processor_trace_rows_ + 1024) * PROCESSOR_WIDTH);
        processor_trace_flat_.resize(new_capacity, BFieldElement::zero());
    }
    
    // Fill directly into flat buffer (no allocation, no copy)
    state.fill_processor_row_flat(processor_trace_flat_.data() + row_start);
    processor_trace_rows_++;
}

void AlgebraicExecutionTrace::record_state_cached(const VMState& state, const TritonInstruction& instr) {
    // Record instruction lookup (increment multiplicity)
    size_t ip = state.instruction_pointer();
    if (ip < instruction_multiplicities_.size()) {
        instruction_multiplicities_[ip]++;
    }
    
    // OPTIMIZATION: Write directly to pre-allocated flat buffer with cached instruction
    size_t row_start = processor_trace_rows_ * PROCESSOR_WIDTH;
    
    // Check if we need to grow (should be rare if reserve_processor_trace was called)
    if (row_start + PROCESSOR_WIDTH > processor_trace_flat_.size()) [[unlikely]] {
        size_t new_capacity = std::max(processor_trace_flat_.size() * 2, (processor_trace_rows_ + 1024) * PROCESSOR_WIDTH);
        processor_trace_flat_.resize(new_capacity, BFieldElement::zero());
    }
    
    // Fill with cached instruction (avoids redundant current_instruction() call)
    state.fill_processor_row_flat_cached(processor_trace_flat_.data() + row_start, instr);
    processor_trace_rows_++;
}

void AlgebraicExecutionTrace::record_co_processor_call(const CoProcessorCall& call) {
    using namespace HashMainColumn;

    auto increase_lookup_multiplicities = [&](const std::vector<std::array<BFieldElement, Tip5::STATE_SIZE>>& trace) {
        if (trace.size() < 2) return;
        
#ifdef TVM_USE_TBB
        // Parallel version: collect per-thread cascade counts, then merge
        // Use thread-local storage to avoid contention during parallel phase
        static bool tbb_verified = false;
        if (!tbb_verified && std::getenv("TVM_VERIFY_TBB")) {
            std::cout << "[TBB] Using TBB parallel_for for increase_lookup_multiplicities (" << trace.size() << " rows)" << std::endl;
            tbb_verified = true;
        }
        
        struct ThreadLocalData {
            std::unordered_map<uint16_t, uint64_t> cascade_counts;
        };
        
        tbb::enumerable_thread_specific<ThreadLocalData> tls;
        
        // Parallel processing of rows - collect cascade counts per thread
        tbb::parallel_for(size_t(0), trace.size() - 1, [&](size_t row_idx) {
            auto& local = tls.local();
            const auto& row = trace[row_idx];
            for (size_t i = 0; i < Tip5::NUM_SPLIT_AND_LOOKUP; ++i) {
                for (uint16_t limb : base_field_element_into_16_bit_limbs(row[i])) {
                    local.cascade_counts[limb] += 1;
                }
            }
        });
        
        // Merge thread-local results into shared state (sequential merge preserves order)
        // Lookup table multiplicities are only incremented when cascade entry is first inserted
        std::mutex merge_mutex;
        for (auto& local : tls) {
            for (const auto& [limb, count] : local.cascade_counts) {
                std::lock_guard<std::mutex> lock(merge_mutex);
                size_t idx = 0;
                auto it = cascade_table_lookup_index_.find(limb);
                if (it == cascade_table_lookup_index_.end()) {
                    // First insertion: add cascade entry and increment lookup table multiplicities
                    idx = cascade_table_lookup_multiplicities_.size();
                    cascade_table_lookup_index_[limb] = idx;
                    cascade_table_lookup_multiplicities_.push_back({limb, 0});
                    // Rust: lookup-table multiplicities only increase when the cascade entry is first inserted.
                    uint8_t limb_lo = static_cast<uint8_t>(limb & 0xFF);
                    uint8_t limb_hi = static_cast<uint8_t>((limb >> 8) & 0xFF);
                    lookup_table_lookup_multiplicities_[limb_lo] += 1;
                    lookup_table_lookup_multiplicities_[limb_hi] += 1;
                } else {
                    idx = it->second;
                }
                cascade_table_lookup_multiplicities_[idx].second += count;
            }
        }
#else
        // Sequential version (original)
        for (size_t row_idx = 0; row_idx + 1 < trace.size(); ++row_idx) {
            const auto& row = trace[row_idx];
            for (size_t i = 0; i < Tip5::NUM_SPLIT_AND_LOOKUP; ++i) {
                for (uint16_t limb : base_field_element_into_16_bit_limbs(row[i])) {
                    // Cascade multiplicities: preserve insertion order (Rust IndexMap semantics).
                    size_t idx = 0;
                    auto it = cascade_table_lookup_index_.find(limb);
                    if (it == cascade_table_lookup_index_.end()) {
                        idx = cascade_table_lookup_multiplicities_.size();
                        cascade_table_lookup_index_[limb] = idx;
                        cascade_table_lookup_multiplicities_.push_back({limb, 0});
                        // Rust: lookup-table multiplicities only increase when the cascade entry is first inserted.
                        uint8_t limb_lo = static_cast<uint8_t>(limb & 0xFF);
                        uint8_t limb_hi = static_cast<uint8_t>((limb >> 8) & 0xFF);
                        lookup_table_lookup_multiplicities_[limb_lo] += 1;
                        lookup_table_lookup_multiplicities_[limb_hi] += 1;
                    } else {
                        idx = it->second;
                    }
                    cascade_table_lookup_multiplicities_[idx].second += 1;
                }
            }
        }
#endif
    };

    auto append_trace_rows = [&](std::vector<std::vector<BFieldElement>>& dest,
                                 const std::vector<std::array<BFieldElement, Tip5::STATE_SIZE>>& trace,
                                 uint32_t instruction_opcode) {
        increase_lookup_multiplicities(trace);
        auto rows = trace_to_table_rows(trace);
        for (auto& r : rows) {
            r[CI] = BFieldElement(static_cast<uint64_t>(instruction_opcode));
            dest.push_back(std::move(r));
        }
    };

    switch (call.type) {
        case CoProcessorCall::Type::SpongeStateReset: {
            // Rust: append_initial_sponge_state()
            auto initial_state = Tip5::init().state;
            auto row = trace_row_to_hash_table_row(initial_state, 0);
            row[CI] = BFieldElement(static_cast<uint64_t>(TritonInstruction{AnInstruction::SpongeInit}.opcode()));
            sponge_trace_.push_back(std::move(row));
            break;
        }
        case CoProcessorCall::Type::Tip5Trace: {
            if (!call.tip5_trace) break;
            const auto& trace = *call.tip5_trace;
            uint32_t opcode = call.instruction_opcode;
            if (opcode == TritonInstruction{AnInstruction::Hash}.opcode()) {
                append_trace_rows(hash_trace_, trace, opcode);
            } else {
                append_trace_rows(sponge_trace_, trace, opcode);
            }
            break;
        }
        case CoProcessorCall::Type::U32:
        case CoProcessorCall::Type::OpStack:
        case CoProcessorCall::Type::Ram:
            if (call.type == CoProcessorCall::Type::U32 && call.u32_entry) {
                const U32TableEntry key = *call.u32_entry;
                auto it = u32_entries_index_.find(key);
                if (it == u32_entries_index_.end()) {
                    size_t idx = u32_entries_.size();
                    u32_entries_index_[key] = idx;
                    u32_entries_.push_back({key, 1});
                } else {
                    u32_entries_[it->second].second += 1;
                }
            }
            if (call.type == CoProcessorCall::Type::Ram && call.ram_call) {
                // OPTIMIZED: Write directly to trace, avoid temporary vector
                ram_trace_.emplace_back(RAM_WIDTH, BFieldElement::zero());
                auto& row = ram_trace_.back();
                row[RamMainColumn::CLK] = BFieldElement(static_cast<uint64_t>(call.ram_call->clk));
                row[RamMainColumn::InstructionType] = call.ram_call->is_write ? BFieldElement(0) : BFieldElement(1);
                row[RamMainColumn::RamPointer] = call.ram_call->ram_pointer;
                row[RamMainColumn::RamValue] = call.ram_call->ram_value;
            }
            if (call.type == CoProcessorCall::Type::OpStack && call.op_stack_entry) {
                // OPTIMIZED: Write directly to trace, avoid temporary vector
                op_stack_underflow_trace_.emplace_back(OP_STACK_WIDTH, BFieldElement::zero());
                auto& row = op_stack_underflow_trace_.back();
                row[OpStackMainColumn::CLK] = BFieldElement(static_cast<uint64_t>(call.op_stack_entry->clk));
                row[OpStackMainColumn::IB1ShrinkStack] =
                    call.op_stack_entry->underflow_io.shrinks_stack() ? BFieldElement::one() : BFieldElement::zero();
                row[OpStackMainColumn::StackPointer] = call.op_stack_entry->op_stack_pointer;
                row[OpStackMainColumn::FirstUnderflowElement] = call.op_stack_entry->underflow_io.payload;
            }
            break;
    }
}

size_t AlgebraicExecutionTrace::padded_height() const {
    // Rust's padded_height uses height().height.next_power_of_two()
    // where height() returns the maximum height across all tables
    // So we need to use height() (which includes Program table), not just processor_trace_height()
    size_t h = height();
    // Next power of 2
    if (h == 0) return 1;
    // Use next_power_of_two logic: find the smallest power of 2 >= h
    size_t result = 1;
    while (result < h) {
        result <<= 1;
    }
    return result;
}

size_t AlgebraicExecutionTrace::height_of_table(size_t table_id) const {
    // If we have Rust table lengths (from FFI), use them directly
    // This is needed because Rust FFI doesn't return all co-processor traces
    if (rust_table_lengths_.has_value() && table_id < 9) {
        return rust_table_lengths_.value()[table_id];
    }
    
    // Based on Rust implementation in aet.rs:
    // TableId::Program => Self::padded_program_length(&self.program),
    // TableId::Processor => self.processor_trace.nrows(),
    // TableId::OpStack => self.op_stack_underflow_trace.nrows(),
    // TableId::Ram => self.ram_trace.nrows(),
    // TableId::JumpStack => self.processor_trace.nrows(),
    // TableId::Hash => hash_table_height(),
    // TableId::Cascade => self.cascade_table_lookup_multiplicities.len(),
    // TableId::Lookup => Self::LOOKUP_TABLE_HEIGHT,
    // TableId::U32 => self.u32_table_height(),
    
    // Table IDs match TableId enum order:
    // 0 = Program, 1 = Processor, 2 = OpStack, 3 = Ram, 4 = JumpStack,
    // 5 = Hash, 6 = Cascade, 7 = Lookup, 8 = U32
    
    switch (table_id) {
        case 0: { // Program
            // Rust: (program.len_bwords() + 1).next_multiple_of(Tip5::RATE)
            constexpr size_t RATE = Tip5::RATE;
            size_t program_len = program_bwords_.size();
            size_t padded_len = ((program_len + 1) + (RATE - 1)) / RATE * RATE;
            return padded_len;
        }
        case 1: // Processor
            return processor_trace_rows_;
        case 2: // OpStack
            return op_stack_underflow_trace_.size();
        case 3: // RAM
            return ram_trace_.size();
        case 4: // JumpStack (same as Processor)
            return processor_trace_rows_;
        case 5: { // Hash
            return hash_trace_.size() + sponge_trace_.size() + program_hash_trace_.size();
        }
        case 6: // Cascade
            return cascade_table_lookup_multiplicities_.size();
        case 7: // Lookup
            return LOOKUP_TABLE_HEIGHT;
        case 8: { // U32
            // Rust: sum of per-entry section heights
            auto bitlen_u32 = [](uint64_t x) -> uint32_t {
                uint32_t bits = 0;
                while (x) { bits++; x >>= 1; }
                return bits;
            };
            auto entry_height = [&](const U32TableEntry& e) -> uint32_t {
                uint32_t op = e.instruction_opcode;
                uint64_t lhs = e.left_operand.value() & 0xFFFFFFFFULL;
                uint64_t rhs = e.right_operand.value() & 0xFFFFFFFFULL;
                uint32_t max_rows = 33;
                if (op == TritonInstruction{AnInstruction::Pow}.opcode()) {
                    uint32_t bl = bitlen_u32(rhs);
                    uint32_t h = (bl == 0) ? 1U : (bl + 1U);
                    return std::min<uint32_t>(h, max_rows);
                }
                uint32_t bl_l = bitlen_u32(lhs);
                uint32_t bl_r = bitlen_u32(rhs);
                uint32_t bl = std::max(bl_l, bl_r);
                uint32_t h = (bl == 0) ? 1U : (bl + 1U);
                return std::min<uint32_t>(h, max_rows);
            };
            uint32_t total = 0;
            for (const auto& [entry, _mult] : u32_entries_) {
                total += entry_height(entry);
            }
            return static_cast<size_t>(total);
        }
        default:
            return 0;
    }
}

size_t AlgebraicExecutionTrace::height() const {
    // Return the maximum height across all tables
    // This matches Rust's height() which returns TableHeight with max height
    size_t max_height = 0;
    for (size_t i = 0; i < 9; ++i) { // 9 table types
        size_t table_height = height_of_table(i);
        max_height = std::max(max_height, table_height);
        // Debug output to see which table is the maximum
        if (i == 0 || table_height == max_height) {
            // Only print when we find a new max or for Program table
            // (commented out to avoid spam, uncomment for debugging)
            // std::cout << "DEBUG: Table " << i << " height: " << table_height << std::endl;
        }
    }
    return max_height;
}

const std::vector<std::vector<BFieldElement>>& AlgebraicExecutionTrace::processor_trace() const {
    // LAZY CONVERSION: Only build legacy format when needed
    if (processor_trace_legacy_.size() != processor_trace_rows_) {
        processor_trace_legacy_.resize(processor_trace_rows_);
        for (size_t r = 0; r < processor_trace_rows_; ++r) {
            processor_trace_legacy_[r].resize(PROCESSOR_WIDTH);
            const BFieldElement* row_ptr = processor_trace_flat_.data() + r * PROCESSOR_WIDTH;
            for (size_t c = 0; c < PROCESSOR_WIDTH; ++c) {
                processor_trace_legacy_[r][c] = row_ptr[c];
            }
        }
    }
    return processor_trace_legacy_;
}

void AlgebraicExecutionTrace::fill_program_hash_trace() {
    program_hash_trace_.clear();

    // Rust: `hash_input_pad_program(program)` => program_bwords + [1] + zeros to a multiple of Tip5::RATE
    std::vector<BFieldElement> padded = program_bwords_;
    padded.push_back(BFieldElement::one());
    while (padded.size() % Tip5::RATE != 0) {
        padded.push_back(BFieldElement::zero());
    }

    // Rust uses `Tip5::init()` here.
    Tip5 program_sponge = Tip5::init();

    using namespace HashMainColumn;
    uint32_t hash_opcode = TritonInstruction{AnInstruction::Hash}.opcode();

    for (size_t i = 0; i < padded.size(); i += Tip5::RATE) {
        for (size_t j = 0; j < Tip5::RATE; ++j) {
            program_sponge.state[j] = padded[i + j];
        }

        // Rust: `trace()` mutates the sponge to the permuted state.
        auto trace = program_sponge.trace();

        // Increase lookup multiplicities (matches Rust)
        if (trace.size() >= 2) {
            for (size_t row_idx = 0; row_idx + 1 < trace.size(); ++row_idx) {
                const auto& row = trace[row_idx];
                for (size_t k = 0; k < Tip5::NUM_SPLIT_AND_LOOKUP; ++k) {
                    for (uint16_t limb : base_field_element_into_16_bit_limbs(row[k])) {
                        size_t idx = 0;
                        auto it = cascade_table_lookup_index_.find(limb);
                        if (it == cascade_table_lookup_index_.end()) {
                            idx = cascade_table_lookup_multiplicities_.size();
                            cascade_table_lookup_index_[limb] = idx;
                            cascade_table_lookup_multiplicities_.push_back({limb, 0});
                            uint8_t limb_lo = static_cast<uint8_t>(limb & 0xFF);
                            uint8_t limb_hi = static_cast<uint8_t>((limb >> 8) & 0xFF);
                            lookup_table_lookup_multiplicities_[limb_lo] += 1;
                            lookup_table_lookup_multiplicities_[limb_hi] += 1;
                        } else {
                            idx = it->second;
                        }
                        cascade_table_lookup_multiplicities_[idx].second += 1;
                    }
                }
            }
        }

        auto rows = trace_to_table_rows(trace);
        for (auto& r : rows) {
            r[CI] = BFieldElement(static_cast<uint64_t>(hash_opcode));
            program_hash_trace_.push_back(std::move(r));
        }
    }
}

} // namespace triton_vm

