#include "vm/vm_state.hpp"
#include "vm/program.hpp"
#include "vm/processor_columns.hpp"
#include "table/extend_helpers.hpp"
#include <stdexcept>
#include <algorithm>
#include <array>
#include <iostream>
#include <cstring>

namespace triton_vm {

VMState::VMState(
    const Program& program,
    const std::vector<BFieldElement>& public_input,
    const std::vector<BFieldElement>& secret_input
)
    : program_(program)
    , public_input_(public_input.begin(), public_input.end())
    , secret_individual_tokens_(secret_input.begin(), secret_input.end())
    , op_stack_(std::make_unique<OpStack>(program.hash()))
    , cycle_count_(0)
    , instruction_pointer_(0)
    , halting_(false)
{
    // Debug: Verify input was copied correctly
    if (public_input_.size() != public_input.size()) {
        throw std::runtime_error("VMState constructor: Public input size mismatch - expected " + 
                                std::to_string(public_input.size()) + 
                                ", deque has " + std::to_string(public_input_.size()));
    }
}

std::vector<CoProcessorCall> VMState::step() {
    std::vector<CoProcessorCall> co_processor_calls;
    step_into(co_processor_calls);
    return co_processor_calls;
}

void VMState::step_into(std::vector<CoProcessorCall>& out_calls) {
    if (halting_) {
        return;
    }
    
    // Get current instruction
    auto instr_opt = current_instruction();
    if (!instr_opt.has_value()) {
        halting_ = true;
        return;
    }
    
    step_into_cached(instr_opt.value(), out_calls);
}

void VMState::step_into_cached(const TritonInstruction& instr, std::vector<CoProcessorCall>& out_calls) {
    if (halting_) {
        return;
    }

    // Record op-stack underflow IO sequence for this instruction (matches Rust)
    op_stack_->start_recording_underflow_io_sequence();

    // Execute instruction - append directly to out_calls (no allocation)
    execute_instruction_into(instr, out_calls);

    // Convert op-stack underflow IO sequence into OpStack table entries (matches Rust).
    {
        auto seq = op_stack_->stop_recording_underflow_io_sequence();
        UnderflowIO::canonicalize_sequence(seq);
        if (!seq.empty() && !UnderflowIO::is_uniform_sequence(seq)) {
            throw std::runtime_error("OpStack underflow IO sequence is not uniform");
        }
        if (!seq.empty()) {
            // Rust: OpStackTableEntry::from_underflow_io_sequence
            BFieldElement sp_after = op_stack_->pointer();
            BFieldElement seq_len(static_cast<uint64_t>(seq.size()));
            BFieldElement sp = (seq.front().kind == UnderflowIO::Kind::Write)
                ? (sp_after - seq_len)
                : (sp_after + seq_len);

            for (const auto& io : seq) {
                if (io.shrinks_stack()) {
                    sp -= BFieldElement::one();
                }
                out_calls.push_back(CoProcessorCall::make_op_stack(
                    static_cast<uint32_t>(cycle_count_),
                    sp,
                    io
                ));
                if (io.grows_stack()) {
                    sp += BFieldElement::one();
                }
            }
        }
    }
    
    // Advance instruction pointer (unless instruction modified it)
    bool ip_modified = (instr.type == AnInstruction::Call || 
                       instr.type == AnInstruction::Return ||
                       instr.type == AnInstruction::Skiz ||
                       instr.type == AnInstruction::Recurse);
    
    if (!ip_modified && !halting_) {
        instruction_pointer_ += instr.size();
        
        if (instruction_pointer_ >= program_.len_bwords()) {
            halting_ = true;
        }
    } else if (ip_modified) {
        if (instruction_pointer_ >= program_.len_bwords()) {
            halting_ = true;
        }
    }

    cycle_count_++;
}

void VMState::run() {
    while (!halting_) {
        step();
    }
}

std::vector<CoProcessorCall> VMState::execute_instruction(const TritonInstruction& instr) {
    std::vector<CoProcessorCall> co_processor_calls;
    execute_instruction_into(instr, co_processor_calls);
    return co_processor_calls;
}

void VMState::execute_instruction_into(const TritonInstruction& instr, std::vector<CoProcessorCall>& co_processor_calls) {
    switch (instr.type) {
        case AnInstruction::Push:
            op_stack_->push(instr.bfield_arg);
            break;
            
        case AnInstruction::Pop: {
            // NumberOfWords is an enum, convert to size_t
            size_t n = 1;
            switch (instr.num_words_arg) {
                case NumberOfWords::N1: n = 1; break;
                case NumberOfWords::N2: n = 2; break;
                case NumberOfWords::N3: n = 3; break;
                case NumberOfWords::N4: n = 4; break;
                case NumberOfWords::N5: n = 5; break;
            }
            op_stack_->pop(n);
            break;
        }
        
        case AnInstruction::Add: {
            if (op_stack_->size() < 2) {
                throw std::runtime_error("Stack underflow in add");
            }
            BFieldElement b = op_stack_->pop();
            BFieldElement a = op_stack_->pop();
            op_stack_->push(a + b);
            break;
        }
        
        case AnInstruction::Mul: {
            if (op_stack_->size() < 2) {
                throw std::runtime_error("Stack underflow in mul");
            }
            BFieldElement b = op_stack_->pop();
            BFieldElement a = op_stack_->pop();
            op_stack_->push(a * b);
            break;
        }
        
        case AnInstruction::Dup: {
            size_t depth = static_cast<size_t>(instr.op_stack_arg);
            op_stack_->dup(depth);
            break;
        }
        
        case AnInstruction::Swap: {
            size_t depth = static_cast<size_t>(instr.op_stack_arg);
            op_stack_->swap(depth);
            break;
        }
        
        case AnInstruction::Halt:
            halting_ = true;
            break;
            
        case AnInstruction::Nop:
            // Do nothing
            break;
            
        case AnInstruction::ReadIo: {
            size_t n = 1;
            switch (instr.num_words_arg) {
                case NumberOfWords::N1: n = 1; break;
                case NumberOfWords::N2: n = 2; break;
                case NumberOfWords::N3: n = 3; break;
                case NumberOfWords::N4: n = 4; break;
                case NumberOfWords::N5: n = 5; break;
            }
            // Debug: Check input before reading
            if (public_input_.empty()) {
                throw std::runtime_error("Public input exhausted in read_io: trying to read " + 
                                         std::to_string(n) + " word(s) but input is empty (cycle " + 
                                         std::to_string(cycle_count_) + ", IP " + 
                                         std::to_string(instruction_pointer_) + 
                                         "). This suggests input was consumed in a previous iteration.");
            }
            // Consume input and push to stack
            // If push() throws, input is already consumed, so IP won't advance
            // This is the bug: we need to push BEFORE consuming, or handle exceptions properly
            for (size_t i = 0; i < n; ++i) {
                if (public_input_.empty()) {
                    throw std::runtime_error("Public input exhausted in read_io: trying to read " + 
                                           std::to_string(n) + " word(s), " + std::to_string(i) + 
                                           " already read (cycle " + std::to_string(cycle_count_) + 
                                           ", IP " + std::to_string(instruction_pointer_) + ")");
                }
                BFieldElement value = public_input_.front();
                public_input_.pop_front();  // Consume input first
                op_stack_->push(value);      // Then push (may throw if stack is full)
            }
            break;
        }
        
        case AnInstruction::WriteIo: {
            size_t n = 1;
            switch (instr.num_words_arg) {
                case NumberOfWords::N1: n = 1; break;
                case NumberOfWords::N2: n = 2; break;
                case NumberOfWords::N3: n = 3; break;
                case NumberOfWords::N4: n = 4; break;
                case NumberOfWords::N5: n = 5; break;
            }
            for (size_t i = 0; i < n; ++i) {
                if (op_stack_->empty()) {
                    throw std::runtime_error("Stack underflow in write_io");
                }
                public_output_.push_back(op_stack_->pop());
            }
            break;
        }
        
        case AnInstruction::Call: {
            // Call: push (call_origin, call_destination) to jump stack
            // Matching Rust: let size_of_instruction_call = 2;
            //                let call_origin = (self.instruction_pointer as u32 + size_of_instruction_call).into();
            //                let jump_stack_entry = (call_origin, call_destination);
            //                self.jump_stack.push(jump_stack_entry);
            //                self.instruction_pointer = call_destination.value().try_into().unwrap();
            constexpr size_t size_of_instruction_call = 2; // call instruction takes 2 words (opcode + arg)
            BFieldElement call_origin(instruction_pointer_ + size_of_instruction_call);
            BFieldElement call_destination(instr.bfield_arg.value());
            jump_stack_.push_back({call_origin, call_destination});
            
            // Jump to function address
            instruction_pointer_ = call_destination.value();
            break;
        }
        
        case AnInstruction::Return: {
            // Return: pop jump stack, restore stack height, jump to call_origin
            // Matching Rust: let (call_origin, _) = self.jump_stack_pop()?;
            //                self.instruction_pointer = call_origin.value().try_into().unwrap();
            // Note: Rust doesn't restore stack height in return, it just jumps
            if (jump_stack_.empty()) {
                halting_ = true;
                break;
            }
            
            auto [call_origin, call_destination] = jump_stack_.back();
            jump_stack_.pop_back();
            
            // Jump to return address (call_origin)
            instruction_pointer_ = call_origin.value();
            break;
        }
        
        case AnInstruction::Assert: {
            // Assert: pop stack[0], verify it's 1, throw if not
            // Matching Rust: assert checks stack[0] == 1, pops it, then advances IP
            if (op_stack_->empty()) {
                throw std::runtime_error("Stack underflow in assert");
            }
            BFieldElement value = op_stack_->pop();
            if (value != BFieldElement::one()) {
                // Get error_id from instruction argument if available (error_id is in bfield_arg)
                std::string error_msg = "Assertion failed: expected 1, got " + std::to_string(value.value());
                uint64_t error_id_val = instr.bfield_arg.value();
                if (error_id_val != 0) {
                    error_msg += " (error_id " + std::to_string(error_id_val) + ")";
                }
                throw std::runtime_error(error_msg);
            }
            break;
        }
        
        case AnInstruction::Lt: {
            // Less than: pop lhs (ST0) then rhs (ST1), push 1 if lhs < rhs, else 0
            // Matching Rust: let lhs = self.op_stack.pop_u32()?; let rhs = self.op_stack.pop_u32()?;
            if (op_stack_->size() < 2) {
                throw std::runtime_error("Stack underflow in lt");
            }
            BFieldElement lhs = op_stack_->pop(); // ST0
            BFieldElement rhs = op_stack_->pop(); // ST1
            op_stack_->push(lhs < rhs ? BFieldElement::one() : BFieldElement::zero());

            // Record U32 co-processor call (matches Rust)
            U32TableEntry entry{
                TritonInstruction{AnInstruction::Lt}.opcode(),
                lhs,
                rhs
            };
            co_processor_calls.push_back(CoProcessorCall::make_u32(entry));
            break;
        }
        
        case AnInstruction::AddI: {
            // Add immediate: modify stack[0] in place (matching Rust: self.op_stack[0] += i)
            // Matching Rust: fn addi(&mut self, i: BFieldElement) -> Vec<CoProcessorCall> {
            //                    self.op_stack[0] += i;
            //                    self.instruction_pointer += 2;
            //                    vec![]
            //                }
            if (op_stack_->empty()) {
                throw std::runtime_error("Stack underflow in addi");
            }
            BFieldElement immediate = instr.bfield_arg;
            // Modify top element in place
            BFieldElement top = op_stack_->pop();
            op_stack_->push(top + immediate);
            break;
        }
        
        case AnInstruction::Pow: {
            // Power: pop base and exponent, push base^exponent
            if (op_stack_->size() < 2) {
                throw std::runtime_error("Stack underflow in pow");
            }
            // Match Rust:
            //   let base = pop();
            //   let exponent = pop_u32();
            BFieldElement base = op_stack_->pop();      // ST0
            BFieldElement exp = op_stack_->pop();       // ST1
            op_stack_->push(base.pow(exp.value()));

            // Record U32 co-processor call (matches Rust)
            U32TableEntry entry{
                TritonInstruction{AnInstruction::Pow}.opcode(),
                base,
                exp
            };
            co_processor_calls.push_back(CoProcessorCall::make_u32(entry));
            break;
        }
        
        case AnInstruction::SpongeInit: {
            // Sponge init: initialize the sponge state (matching Rust)
            sponge_ = Tip5::init();
            co_processor_calls.push_back(CoProcessorCall::sponge_state_reset());
            break;
        }
        
        case AnInstruction::SpongeSqueeze: {
            // Sponge squeeze: squeeze 10 elements from sponge to stack (matching Rust)
            if (!sponge_.has_value()) {
                throw std::runtime_error("Sponge not initialized");
            }
            Tip5& sponge = sponge_.value();

            // Rust pushes the current RATE state elements first, then performs the permutation trace.
            // Push elements in reverse order (matching Rust: for i in (0..tip5::RATE).rev())
            for (int i = static_cast<int>(Tip5::RATE) - 1; i >= 0; --i) {
                op_stack_->push(sponge.state[i]);
            }

            // Rust: let tip5_trace = sponge.trace(); (this mutates sponge state)
            auto trace = sponge.trace();
            
            // Create CoProcessorCall with SpongeSqueeze opcode
            uint32_t opcode = instr.opcode(); // Should be 56 for SpongeSqueeze
            co_processor_calls.push_back(CoProcessorCall::make_tip5_trace(opcode, std::move(trace)));
            break;
        }
        
        case AnInstruction::WriteMem: {
            // Write memory: pop address and values, write to RAM
            size_t n = 1;
            switch (instr.num_words_arg) {
                case NumberOfWords::N1: n = 1; break;
                case NumberOfWords::N2: n = 2; break;
                case NumberOfWords::N3: n = 3; break;
                case NumberOfWords::N4: n = 4; break;
                case NumberOfWords::N5: n = 5; break;
            }
            if (op_stack_->size() < n + 1) {
                throw std::runtime_error("Stack underflow in write_mem");
            }
            BFieldElement address = op_stack_->pop();
            for (size_t i = 0; i < n; ++i) {
                BFieldElement value = op_stack_->pop();
                ram_[address] = value;

                // Record RAM table call (matches Rust `ram_write`)
                RamTableCall call_payload{
                    static_cast<uint32_t>(cycle_count_),
                    address,
                    value,
                    true
                };
                co_processor_calls.push_back(CoProcessorCall::make_ram(call_payload));

                address = address + BFieldElement::one();
            }
            op_stack_->push(address);
            break;
        }
        
        case AnInstruction::Split: {
            // Split: pop value, push low 32 bits and high 32 bits
            // Matching Rust: splits value into low 32 bits and high 32 bits
            if (op_stack_->empty()) {
                throw std::runtime_error("Stack underflow in split");
            }
            BFieldElement value = op_stack_->pop();
            uint64_t val = value.value();
            // Low 32 bits: val & 0xFFFFFFFF
            uint32_t low = static_cast<uint32_t>(val & 0xFFFFFFFFULL);
            // High 32 bits: (val >> 32) & 0xFFFFFFFF
            uint32_t high = static_cast<uint32_t>((val >> 32) & 0xFFFFFFFFULL);
            BFieldElement lo_bfe(static_cast<uint64_t>(low));
            BFieldElement hi_bfe(static_cast<uint64_t>(high));

            // Match Rust: push hi then lo (so lo ends up on top)
            op_stack_->push(hi_bfe);
            op_stack_->push(lo_bfe);

            // Record U32 co-processor call (matches Rust)
            U32TableEntry entry{
                TritonInstruction{AnInstruction::Split}.opcode(),
                lo_bfe,
                hi_bfe
            };
            co_processor_calls.push_back(CoProcessorCall::make_u32(entry));
            break;
        }
        
        case AnInstruction::Skiz: {
            // Skiz: skip if zero - pop value, if zero skip next instruction
            // Matching Rust: let top_of_stack = self.op_stack.pop()?;
            //                self.instruction_pointer += match top_of_stack.is_zero() {
            //                    true => 1 + self.next_instruction()?.size(),
            //                    false => 1,
            //                };
            if (op_stack_->empty()) {
                throw std::runtime_error("Stack underflow in skiz");
            }
            BFieldElement value = op_stack_->pop();
            if (value == BFieldElement::zero()) {
                // Skip next instruction: advance IP by 1 (past skiz) + size of next instruction
                instruction_pointer_++; // Skip the skiz instruction itself
                // Get next instruction to determine its size
                // Instructions with arguments take 2 words (opcode + arg), others take 1 word
                if (instruction_pointer_ < program_.len_bwords()) {
                    std::optional<TritonInstruction> next_instr = program_.instruction_at(instruction_pointer_);
                    if (next_instr.has_value()) {
                        // Skip the next instruction (it may have an argument)
                        // Check if instruction has an argument using arg() method
                        std::optional<BFieldElement> arg_opt = next_instr.value().arg();
                        size_t next_instr_size = arg_opt.has_value() ? 2 : 1;
                        instruction_pointer_ += next_instr_size;
                    } else {
                        instruction_pointer_++; // Fallback: just skip one word
                    }
                }
            } else {
                // Don't skip: just advance IP by 1 (past skiz instruction)
                instruction_pointer_++;
            }
            // IP is modified, so step() won't auto-increment it
            break;
        }
        
        case AnInstruction::Recurse: {
            // Recurse: jump to call_destination (the address where function was called)
            // Matching Rust: let (_, call_destination) = self.jump_stack_peek()?;
            //                self.instruction_pointer = call_destination.value().try_into().unwrap();
            if (jump_stack_.empty()) {
                throw std::runtime_error("Jump stack is empty in recurse");
            }
            // jump_stack_ entry is (call_origin, call_destination)
            // call_destination is where the function starts (where we jump to)
            // call_origin is where we return to (after the call instruction)
            // For recurse, we jump to call_destination (function start)
            BFieldElement call_destination = jump_stack_.back().second; // destination = function start address
            instruction_pointer_ = call_destination.value();
            // Note: IP is modified, so step() won't auto-increment it
            break;
        }
        
        case AnInstruction::Eq: {
            // Equal: pop lhs then rhs, push 1 if equal, else 0
            // Matching Rust: let lhs = self.op_stack.pop()?; let rhs = self.op_stack.pop()?;
            if (op_stack_->size() < 2) {
                throw std::runtime_error("Stack underflow in eq");
            }
            BFieldElement lhs = op_stack_->pop();
            BFieldElement rhs = op_stack_->pop();
            op_stack_->push(lhs == rhs ? BFieldElement::one() : BFieldElement::zero());
            break;
        }
        
        default:
            // TODO: Implement remaining instructions
            throw std::runtime_error("Instruction not yet implemented: " + instr.name());
    }
}

std::optional<TritonInstruction> VMState::current_instruction() const {
    return program_.instruction_at(instruction_pointer_);
}

void VMState::fill_processor_row(std::vector<BFieldElement>& row) const {
    // OPTIMIZED: Cache instruction lookup - called only once instead of 3x
    auto instr_opt = current_instruction();
    const bool has_instr = instr_opt.has_value();
    
    size_t stack_size = op_stack_->size();
    
    // CLK - Cycle count
    row[processor_column_index(ProcessorMainColumn::CLK)] = BFieldElement(cycle_count_);
    
    // IP - Instruction pointer
    row[processor_column_index(ProcessorMainColumn::IP)] = BFieldElement(static_cast<uint32_t>(instruction_pointer_));
    
    // CI and IB0-IB6 - Current instruction opcode and bits
    if (has_instr) {
        const TritonInstruction& instr = instr_opt.value();
        row[processor_column_index(ProcessorMainColumn::CI)] = BFieldElement(instr.opcode());
        row[processor_column_index(ProcessorMainColumn::IB0)] = instr.ib(InstructionBit::IB0);
        row[processor_column_index(ProcessorMainColumn::IB1)] = instr.ib(InstructionBit::IB1);
        row[processor_column_index(ProcessorMainColumn::IB2)] = instr.ib(InstructionBit::IB2);
        row[processor_column_index(ProcessorMainColumn::IB3)] = instr.ib(InstructionBit::IB3);
        row[processor_column_index(ProcessorMainColumn::IB4)] = instr.ib(InstructionBit::IB4);
        row[processor_column_index(ProcessorMainColumn::IB5)] = instr.ib(InstructionBit::IB5);
        row[processor_column_index(ProcessorMainColumn::IB6)] = instr.ib(InstructionBit::IB6);
        
        // NIA - Next instruction or argument (inline to avoid repeated current_instruction lookup)
        auto arg_opt = instr.arg();
        if (arg_opt.has_value()) {
            row[processor_column_index(ProcessorMainColumn::NIA)] = arg_opt.value();
        } else {
            size_t next_ip = instruction_pointer_ + instr.size();
            auto next_instr = program_.instruction_at(next_ip);
            row[processor_column_index(ProcessorMainColumn::NIA)] = next_instr.has_value() 
                ? BFieldElement(next_instr.value().opcode()) 
                : BFieldElement(1);
        }
        
        // HV0-HV5 - Helper variables (pass cached instruction)
        auto helper_vars = derive_helper_variables_fast(instr);
        row[processor_column_index(ProcessorMainColumn::HV0)] = helper_vars[0];
        row[processor_column_index(ProcessorMainColumn::HV1)] = helper_vars[1];
        row[processor_column_index(ProcessorMainColumn::HV2)] = helper_vars[2];
        row[processor_column_index(ProcessorMainColumn::HV3)] = helper_vars[3];
        row[processor_column_index(ProcessorMainColumn::HV4)] = helper_vars[4];
        row[processor_column_index(ProcessorMainColumn::HV5)] = helper_vars[5];
    } else {
        row[processor_column_index(ProcessorMainColumn::CI)] = BFieldElement(8); // Nop opcode
        row[processor_column_index(ProcessorMainColumn::NIA)] = BFieldElement::zero();
    }
    
    // JSP, JSO, JSD - Jump stack
    row[processor_column_index(ProcessorMainColumn::JSP)] = jump_stack_pointer();
    row[processor_column_index(ProcessorMainColumn::JSO)] = jump_stack_origin();
    row[processor_column_index(ProcessorMainColumn::JSD)] = jump_stack_destination();
    
    // ST0-ST15 - Op stack elements
    const size_t st0_idx = processor_column_index(ProcessorMainColumn::ST0);
    const size_t copy_count = std::min(stack_size, size_t(16));
    for (size_t i = 0; i < copy_count; ++i) {
        row[st0_idx + i] = op_stack_->peek_at(i);
    }
    // Zero unused stack slots (important when stack shrinks!)
    for (size_t i = copy_count; i < 16; ++i) {
        row[st0_idx + i] = BFieldElement::zero();
    }
    
    // OpStackPointer
    row[processor_column_index(ProcessorMainColumn::OpStackPointer)] = BFieldElement(stack_size);
}

// OPTIMIZED: Fill directly to flat buffer (zero-copy, cache-friendly)
void VMState::fill_processor_row_flat(BFieldElement* row) const {
    // Get instruction (calls current_instruction())
    auto instr_opt = current_instruction();
    if (instr_opt.has_value()) {
        fill_processor_row_flat_cached(row, instr_opt.value());
    } else {
        // No instruction - fill with defaults
        const size_t stack_size = op_stack_->size();
        row[0] = BFieldElement(cycle_count_);
        row[1] = BFieldElement::zero();  // IsPadding
        row[2] = BFieldElement(static_cast<uint32_t>(instruction_pointer_));
        row[3] = BFieldElement(8); // Nop opcode
        row[4] = BFieldElement::zero(); // NIA
        for (size_t i = 5; i < 12; ++i) row[i] = BFieldElement::zero(); // IB0-6
        row[12] = jump_stack_pointer();
        row[13] = jump_stack_origin();
        row[14] = jump_stack_destination();
        for (size_t i = 0; i < 16; ++i) {
            row[15 + i] = (i < stack_size) ? op_stack_->peek_at(i) : BFieldElement::zero();
        }
        row[31] = BFieldElement(stack_size);
        for (size_t i = 32; i < 38; ++i) row[i] = BFieldElement::zero(); // HV0-5
        row[38] = BFieldElement::zero(); // ClockJumpDiff
    }
}

// FASTEST: Fill directly to flat buffer with pre-cached instruction
void VMState::fill_processor_row_flat_cached(BFieldElement* row, const TritonInstruction& instr) const {
    const size_t stack_size = op_stack_->size();
    
    // Column layout: CLK(0), IsPadding(1), IP(2), CI(3), NIA(4), IB0-6(5-11), 
    //                JSP(12), JSO(13), JSD(14), ST0-15(15-30), OpStackPointer(31), 
    //                HV0-5(32-37), ClockJumpDiff(38)
    
    // Only zero specific columns that need it (faster than full memset)
    row[1] = BFieldElement::zero();  // IsPadding
    row[38] = BFieldElement::zero(); // ClockJumpDifferenceLookupMultiplicity
    
    // CLK
    row[0] = BFieldElement(cycle_count_);
    // IP
    row[2] = BFieldElement(static_cast<uint32_t>(instruction_pointer_));
    
    // CI - instruction opcode
    row[3] = BFieldElement(instr.opcode());
    
    // NIA - Next instruction or argument
    auto arg_opt = instr.arg();
    if (arg_opt.has_value()) {
        row[4] = arg_opt.value();
    } else {
        size_t next_ip = instruction_pointer_ + instr.size();
        auto next_instr = program_.instruction_at(next_ip);
        row[4] = next_instr.has_value() 
            ? BFieldElement(next_instr.value().opcode()) 
            : BFieldElement(1);
    }
    
    // IB0-IB6 (columns 5-11)
    row[5] = instr.ib(InstructionBit::IB0);
    row[6] = instr.ib(InstructionBit::IB1);
    row[7] = instr.ib(InstructionBit::IB2);
    row[8] = instr.ib(InstructionBit::IB3);
    row[9] = instr.ib(InstructionBit::IB4);
    row[10] = instr.ib(InstructionBit::IB5);
    row[11] = instr.ib(InstructionBit::IB6);
    
    // JSP, JSO, JSD (columns 12-14)
    row[12] = jump_stack_pointer();
    row[13] = jump_stack_origin();
    row[14] = jump_stack_destination();
    
    // ST0-ST15 (columns 15-30) - zero unused slots
    const size_t copy_count = std::min(stack_size, size_t(16));
    for (size_t i = 0; i < copy_count; ++i) {
        row[15 + i] = op_stack_->peek_at(i);
    }
    for (size_t i = copy_count; i < 16; ++i) {
        row[15 + i] = BFieldElement::zero();
    }
    
    // OpStackPointer (column 31)
    row[31] = BFieldElement(stack_size);
    
    // Helper variables HV0-HV5 (columns 32-37)
    auto hvs = derive_helper_variables_fast(instr);
    row[32] = hvs[0];
    row[33] = hvs[1];
    row[34] = hvs[2];
    row[35] = hvs[3];
    row[36] = hvs[4];
    row[37] = hvs[5];
}

std::vector<BFieldElement> VMState::to_processor_row() const {
    std::vector<BFieldElement> row(PROCESSOR_COLUMN_COUNT, BFieldElement::zero());
    fill_processor_row(row);
    return row;
}

BFieldElement VMState::next_instruction_or_argument() const {
    auto instr_opt = current_instruction();
    if (!instr_opt.has_value()) {
        return BFieldElement::zero();
    }
    
    // If instruction has an argument, return it
    auto arg_opt = instr_opt.value().arg();
    if (arg_opt.has_value()) {
        return arg_opt.value();
    }
    
    // Otherwise, return opcode of next instruction
    auto next_instr_opt = next_instruction();
    if (next_instr_opt.has_value()) {
        return BFieldElement(next_instr_opt.value().opcode());
    }
    
    // If no next instruction, return 1 (for hash-input padding separator)
    return BFieldElement(1);
}

std::optional<TritonInstruction> VMState::next_instruction() const {
    auto curr_instr_opt = current_instruction();
    if (!curr_instr_opt.has_value()) {
        return std::nullopt;
    }
    
    // Skip argument if current instruction has one
    size_t next_ip = instruction_pointer_ + curr_instr_opt.value().size();
    return program_.instruction_at(next_ip);
}

std::array<BFieldElement, 6> VMState::derive_helper_variables() const {
    auto instr_opt = current_instruction();
    if (!instr_opt.has_value()) {
        return {BFieldElement::zero(), BFieldElement::zero(), 
                BFieldElement::zero(), BFieldElement::zero(),
                BFieldElement::zero(), BFieldElement::zero()};
    }
    return derive_helper_variables_fast(instr_opt.value());
}

std::array<BFieldElement, 6> VMState::derive_helper_variables_fast(const TritonInstruction& instr) const {
    std::array<BFieldElement, 6> hvs = {BFieldElement::zero(), BFieldElement::zero(), 
                                        BFieldElement::zero(), BFieldElement::zero(),
                                        BFieldElement::zero(), BFieldElement::zero()};
    
    // Helper function to decompose argument into 4 bits
    auto decompose_arg = [](uint64_t a) -> std::array<BFieldElement, 4> {
        return {
            BFieldElement(a % 2),
            BFieldElement((a >> 1) % 2),
            BFieldElement((a >> 2) % 2),
            BFieldElement((a >> 3) % 2)
        };
    };
    
    // Helper function to read from RAM
    auto ram_read = [this](BFieldElement address) -> BFieldElement {
        auto it = ram_.find(address);
        if (it != ram_.end()) {
            return it->second;
        }
        return BFieldElement::zero();
    };

    // Match Rust `inverse_or_zero()`
    auto inverse_or_zero_local = [](const BFieldElement& x) -> BFieldElement {
        return x.is_zero() ? BFieldElement::zero() : x.inverse();
    };
    
    switch (instr.type) {
        case AnInstruction::Pop:
        case AnInstruction::Divine:
        case AnInstruction::Pick:
        case AnInstruction::Place:
        case AnInstruction::Dup:
        case AnInstruction::Swap:
        case AnInstruction::ReadMem:
        case AnInstruction::WriteMem:
        case AnInstruction::ReadIo:
        case AnInstruction::WriteIo: {
            auto arg_opt = instr.arg();
            if (arg_opt.has_value()) {
                uint64_t arg_val = arg_opt.value().value();
                auto decomposed = decompose_arg(arg_val);
                hvs[0] = decomposed[0];
                hvs[1] = decomposed[1];
                hvs[2] = decomposed[2];
                hvs[3] = decomposed[3];
            }
            break;
        }
        
        case AnInstruction::Skiz: {
            // HV0 = inverse of ST0 (or 0 if ST0 is 0)
            if (op_stack_->size() > 0) {
                BFieldElement st0 = op_stack_->peek();
                if (!st0.is_zero()) {
                    hvs[0] = st0.inverse();
                }
            }
            // HV1-HV5 = decomposition of next instruction-or-argument opcode
            // Match Rust `decompose_opcode_for_instruction_skiz`:
            // [ opcode % 2,
            //   (opcode >> 1) % 4,
            //   (opcode >> 3) % 4,
            //   (opcode >> 5) % 4,
            //   opcode >> 7 ]
            uint64_t next_opcode = next_instruction_or_argument().value();
            hvs[1] = BFieldElement(next_opcode % 2);
            hvs[2] = BFieldElement((next_opcode >> 1) % 4);
            hvs[3] = BFieldElement((next_opcode >> 3) % 4);
            hvs[4] = BFieldElement((next_opcode >> 5) % 4);
            hvs[5] = BFieldElement(next_opcode >> 7);
            break;
        }
        
        case AnInstruction::RecurseOrReturn: {
            // HV0 = inverse of (ST6 - ST5)
            if (op_stack_->size() > 6) {
                BFieldElement st6 = op_stack_->peek_at(6);
                BFieldElement st5 = op_stack_->peek_at(5);
                BFieldElement diff = st6 - st5;
                if (!diff.is_zero()) {
                    hvs[0] = diff.inverse();
                }
            }
            break;
        }
        
        case AnInstruction::SpongeAbsorbMem: {
            // HV0-HV5 = RAM reads at ST0+4 through ST0+9
            if (op_stack_->size() > 0) {
                BFieldElement base = op_stack_->peek();
                for (size_t i = 0; i < 6; ++i) {
                    BFieldElement addr = base + BFieldElement(4 + i);
                    hvs[i] = ram_read(addr);
                }
            }
            break;
        }
        
        case AnInstruction::MerkleStep: {
            // HV0-HV4 = divined digest values
            // HV5 = node_index % 2
            // TODO: Implement when secret_digests is available
            if (op_stack_->size() > 5) {
                uint64_t node_index = op_stack_->peek_at(5).value();
                hvs[5] = BFieldElement(node_index % 2);
            }
            break;
        }
        
        case AnInstruction::MerkleStepMem: {
            // Similar to MerkleStep but reads from RAM
            // TODO: Implement when secret_digests is available
            if (op_stack_->size() > 7) {
                uint64_t node_index = op_stack_->peek_at(5).value();
                BFieldElement ram_pointer = op_stack_->peek_at(7);
                // Read digest from RAM
                for (size_t i = 0; i < 5; ++i) {
                    hvs[i] = ram_read(ram_pointer + BFieldElement(i));
                }
                hvs[5] = BFieldElement(node_index % 2);
            }
            break;
        }

        case AnInstruction::Split: {
            // Match Rust `derive_helper_variables` for Split:
            // Let top = ST0, lo = top & 0xffff_ffff, hi = top >> 32.
            // If lo != 0: HV0 = (hi - (2^32 - 1)).inverse_or_zero()
            if (op_stack_->size() > 0) {
                uint64_t top = op_stack_->peek().value();
                uint64_t lo_u64 = top & 0xffff'ffffULL;
                uint64_t hi_u64 = top >> 32;
                BFieldElement lo(lo_u64);
                if (!lo.is_zero()) {
                    BFieldElement hi(hi_u64);
                    BFieldElement max_val_of_hi((1ULL << 32) - 1);
                    BFieldElement diff = hi - max_val_of_hi;
                    hvs[0] = inverse_or_zero_local(diff);
                }
            }
            break;
        }

        case AnInstruction::Eq: {
            // Match Rust: HV0 = (ST1 - ST0).inverse_or_zero()
            if (op_stack_->size() > 1) {
                BFieldElement st0 = op_stack_->peek();
                BFieldElement st1 = op_stack_->peek_at(1);
                hvs[0] = inverse_or_zero_local(st1 - st0);
            }
            break;
        }
        
        default:
            // Most instructions have zero helper variables
            break;
    }
    
    return hvs;
}

BFieldElement VMState::jump_stack_pointer() const {
    return BFieldElement(jump_stack_.size());
}

BFieldElement VMState::jump_stack_origin() const {
    if (jump_stack_.empty()) {
        return BFieldElement::zero();
    }
    return jump_stack_.back().first; // Return address
}

BFieldElement VMState::jump_stack_destination() const {
    if (jump_stack_.empty()) {
        return BFieldElement::zero();
    }
    return jump_stack_.back().second; // Stack height
}

} // namespace triton_vm

