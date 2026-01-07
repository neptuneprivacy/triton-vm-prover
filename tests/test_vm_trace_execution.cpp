#include <gtest/gtest.h>
#include "vm/vm.hpp"
#include "vm/aet.hpp"
#include "vm/vm_state.hpp"
#include "vm/program.hpp"
#include "vm/processor_columns.hpp"
#include "table/extend_helpers.hpp"
#include "table/master_table.hpp"
#include "types/b_field_element.hpp"
#include "proof_stream/proof_stream.hpp"
#include "types/digest.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <random>

// Forward declarations to avoid including stark.hpp (which has conflicting AlgebraicExecutionTrace)
namespace triton_vm {
    struct Claim {
        Digest program_digest;
        uint32_t version;
        std::vector<BFieldElement> input;
        std::vector<BFieldElement> output;
    };
}

using namespace triton_vm;

/**
 * Test fixture for VM trace execution tests
 */
class VMTraceExecutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup common test data
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
};

/**
 * Test: Create AET from program
 * 
 * Verify that we can create an AlgebraicExecutionTrace for a program.
 */
TEST_F(VMTraceExecutionTest, CreateAET) {
    // Test AET creation with a small program in bword form.
    // Create on heap to test if stack allocation is the issue.
    std::vector<BFieldElement> program_bwords = {
        BFieldElement(static_cast<uint64_t>(TritonInstruction{AnInstruction::Halt}.opcode()))
    };
    
    std::unique_ptr<AlgebraicExecutionTrace> aet_ptr;
    try {
        aet_ptr = std::make_unique<AlgebraicExecutionTrace>(program_bwords);
    } catch (const std::exception& e) {
        FAIL() << "Failed to create AET: " << e.what();
    }
    
    EXPECT_NE(aet_ptr, nullptr);
    EXPECT_EQ(aet_ptr->program_length(), program_bwords.size());
    EXPECT_EQ(aet_ptr->instruction_multiplicities().size(), program_bwords.size());
    EXPECT_EQ(aet_ptr->processor_trace_height(), 0); // No execution yet
    
    // AET will be destroyed when aet_ptr goes out of scope
    // This tests if the destructor works properly
}

/**
 * Test: Trace execution of simple program
 * 
 * Execute a simple program and verify the trace is generated correctly.
 */
TEST_F(VMTraceExecutionTest, SimpleProgramTrace) {
    Program program = Program::from_code("push 1\npush 2\nadd\nhalt");
    std::vector<BFieldElement> input = {};
    
    auto result = VM::trace_execution(program, input);
    
    EXPECT_FALSE(result.aet.processor_trace().empty());
    EXPECT_EQ(result.output.size(), 0); // No output for this program
    EXPECT_GT(result.aet.processor_trace_height(), 0);
    EXPECT_EQ(result.aet.processor_trace_height(), 4); // 4 instructions executed
}

/**
 * Test: Instruction multiplicities
 * 
 * Verify that instruction execution counts are tracked correctly.
 */
TEST_F(VMTraceExecutionTest, InstructionMultiplicities) {
    Program program = Program::from_code("push 1\npush 1\nadd\nhalt");
    
    auto result = VM::trace_execution(program, {});
    
    const auto& multiplicities = result.aet.instruction_multiplicities();
    // Program is in bword space: [push, 1, push, 1, add, halt]
    EXPECT_EQ(multiplicities[0], 1); // push opcode
    EXPECT_EQ(multiplicities[1], 0); // push argument word (never an IP)
    EXPECT_EQ(multiplicities[2], 1); // push opcode
    EXPECT_EQ(multiplicities[3], 0); // push argument word (never an IP)
    EXPECT_EQ(multiplicities[4], 1); // add opcode
    EXPECT_EQ(multiplicities[5], 1); // halt opcode
}

/**
 * Test: Processor state extraction
 * 
 * Verify that to_processor_row() extracts all 39 processor columns correctly.
 */
TEST_F(VMTraceExecutionTest, ProcessorStateExtraction) {
    // Create a minimal program
    TritonProgram triton_program;
    TritonInstruction push_instr{AnInstruction::Push, BFieldElement(42)};
    triton_program.add_instruction(push_instr);
    
    // Convert to Program
    Program program = Program::from_code("push 42\nhalt");
    
    // Create VMState with the program
    std::vector<BFieldElement> public_input = {};
    std::vector<BFieldElement> secret_input = {};
    VMState state(program, public_input, secret_input);
    
    // Extract processor row
    std::vector<BFieldElement> row = state.to_processor_row();
    
    // Verify row has correct width (39 columns)
    EXPECT_EQ(row.size(), 39);
    
    // Verify CLK is 0 initially
    EXPECT_EQ(row[processor_column_index(ProcessorMainColumn::CLK)], BFieldElement(0));
    
    // Verify IP is 0 initially
    EXPECT_EQ(row[processor_column_index(ProcessorMainColumn::IP)], BFieldElement(0));
    
    // Verify IsPadding is 0
    EXPECT_EQ(row[processor_column_index(ProcessorMainColumn::IsPadding)], BFieldElement(0));
    
    // Rust-compatible OpStack: always at least 16 registers, preloaded with program digest
    EXPECT_EQ(row[processor_column_index(ProcessorMainColumn::OpStackPointer)], BFieldElement(16));
    
    // Verify JSP, JSO, JSD are 0 (empty jump stack)
    EXPECT_EQ(row[processor_column_index(ProcessorMainColumn::JSP)], BFieldElement(0));
    EXPECT_EQ(row[processor_column_index(ProcessorMainColumn::JSO)], BFieldElement(0));
    EXPECT_EQ(row[processor_column_index(ProcessorMainColumn::JSD)], BFieldElement(0));
    
    // Verify ST0-ST10 are 0, and ST11-ST15 contain the program digest (matches Rust OpStack::new).
    Digest d = program.hash();
    for (size_t i = 0; i < 16; ++i) {
        ProcessorMainColumn col = static_cast<ProcessorMainColumn>(
            static_cast<size_t>(ProcessorMainColumn::ST0) + i);
        if (i <= 10) {
            EXPECT_EQ(row[processor_column_index(col)], BFieldElement(0));
        } else {
            // i=11..15 maps to digest[0..4] (deepest registers)
            EXPECT_EQ(row[processor_column_index(col)], d[i - 11]);
        }
    }
    
    // Verify HV0-HV5 are 0 initially
    for (size_t i = 0; i < 6; ++i) {
        ProcessorMainColumn col = static_cast<ProcessorMainColumn>(
            static_cast<size_t>(ProcessorMainColumn::HV0) + i);
        EXPECT_EQ(row[processor_column_index(col)], BFieldElement(0));
    }
}

/**
 * Test: Processor trace recording
 * 
 * Verify that processor state is recorded after each instruction.
 */
TEST_F(VMTraceExecutionTest, ProcessorTraceRecording) {
    // TODO: Implement once VM is functional
    // Verify that processor_trace has correct number of rows
    // Verify that each row has correct width (PROCESSOR_WIDTH = 39)
}

/**
 * Test: Match Rust output
 * 
 * Compare C++ trace output with Rust-generated test data.
 */
TEST_F(VMTraceExecutionTest, MatchRustOutput) {
    // TODO: Load Rust test data and compare
    // This will be the ultimate verification test
}

/**
 * Test: Fiat-Shamir claim hashing after trace execution
 * 
 * Verify that we can hash a claim into the proof stream after trace execution.
 * This tests the integration of trace execution → Fiat-Shamir claim step.
 */
TEST_F(VMTraceExecutionTest, FiatShamirClaimAfterTrace) {
    // Create a simple program
    Program program = Program::from_code("push 1\npush 2\nadd\nhalt");
    std::vector<BFieldElement> input = {};
    
    // Step 1: Trace execution
    auto result = VM::trace_execution(program, input);
    
    EXPECT_GT(result.aet.processor_trace_height(), 0);
    EXPECT_EQ(result.aet.processor_trace_height(), 4); // 4 instructions
    
    // Step 2: Create claim from trace result
    Claim claim;
    claim.program_digest = program.hash();
    claim.version = 1;
    claim.input = input;
    claim.output = result.output;
    
    // Step 3: Initialize proof stream and hash claim (Fiat-Shamir: claim)
    ProofStream proof_stream;
    
    // Encode claim according to Rust BFieldCodec format
    std::vector<BFieldElement> claim_encoding;
    claim_encoding.reserve(16);
    
    auto encode_vec = [&](const std::vector<BFieldElement>& v) {
        const size_t vec_encoding_len = 1 + v.size();
        claim_encoding.push_back(BFieldElement(static_cast<uint64_t>(vec_encoding_len)));
        claim_encoding.push_back(BFieldElement(static_cast<uint64_t>(v.size())));
        for (const auto& e : v) {
            claim_encoding.push_back(e);
        }
    };
    
    // Encode in reverse field order: output, input, version, program_digest
    encode_vec(claim.output);
    encode_vec(claim.input);
    claim_encoding.push_back(BFieldElement(static_cast<uint64_t>(claim.version)));
    for (size_t i = 0; i < Digest::LEN; ++i) {
        claim_encoding.push_back(claim.program_digest[i]);
    }
    
    // Hash claim into proof stream
    proof_stream.alter_fiat_shamir_state_with(claim_encoding);
    
    // Verify proof stream was updated (sponge state should be non-zero)
    auto sponge = proof_stream.sponge();
    bool any_nonzero = false;
    for (const auto& state : sponge.state) {
        if (!state.is_zero()) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "Sponge state should be non-zero after hashing claim";
    
    // Verify we can sample challenges (proves Fiat-Shamir is working)
    auto challenges = proof_stream.sample_scalars(5);
    EXPECT_EQ(challenges.size(), 5);
    
    // Verify challenges are non-zero (with high probability)
    bool any_nonzero_challenge = false;
    for (const auto& ch : challenges) {
        if (!ch.is_zero()) {
            any_nonzero_challenge = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero_challenge) << "Challenges should be non-zero";
}

/**
 * Test: Derive additional parameters after trace execution
 * 
 * Verify that we can derive domain parameters (padded_height, FRI domain, ProverDomains)
 * from the AET after trace execution. This tests the "derive additional parameters" step.
 */
TEST_F(VMTraceExecutionTest, DeriveAdditionalParameters) {
    // Step 1: Trace execution
    Program program = Program::from_code("push 1\npush 2\nadd\nhalt");
    std::vector<BFieldElement> input = {};
    
    auto result = VM::trace_execution(program, input);
    
    EXPECT_GT(result.aet.processor_trace_height(), 0);
    size_t trace_height = result.aet.processor_trace_height();
    
    // Step 2: Compute padded_height (next power of 2)
    size_t padded_height = result.aet.padded_height();
    EXPECT_GE(padded_height, trace_height);
    EXPECT_TRUE((padded_height & (padded_height - 1)) == 0) << "padded_height should be a power of 2";
    
    // Step 3: Derive FRI domain
    // Use default parameters: num_trace_randomizers=30, security_level=160
    constexpr size_t num_trace_randomizers = 30;
    size_t randomized_trace_len = padded_height + num_trace_randomizers;
    // Round up to next power of 2
    size_t rand_trace_pow2 = 1;
    while (rand_trace_pow2 < randomized_trace_len) {
        rand_trace_pow2 <<= 1;
    }
    EXPECT_GT(rand_trace_pow2, padded_height);
    
    // Create FRI domain
    size_t fri_domain_length = 4096; // Default for testing
    ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length);
    BFieldElement fri_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(fri_domain_length)) + 1
    );
    fri_domain = fri_domain.with_offset(fri_offset);
    
    EXPECT_EQ(fri_domain.length, fri_domain_length);
    EXPECT_FALSE(fri_domain.offset.is_zero());
    
    // Step 4: Derive ProverDomains
    // Approximate max_degree (simplified formula)
    int64_t interpolant_degree = static_cast<int64_t>(rand_trace_pow2 - 1);
    int64_t max_degree = interpolant_degree * 4; // Approximate
    EXPECT_GT(max_degree, 0);
    
    ProverDomains domains = ProverDomains::derive(
        padded_height,
        num_trace_randomizers,
        fri_domain,
        max_degree
    );
    
    // Verify domains are properly derived
    EXPECT_GT(domains.trace.length, 0);
    EXPECT_GT(domains.randomized_trace.length, 0);
    EXPECT_GT(domains.quotient.length, 0);
    EXPECT_EQ(domains.fri.length, fri_domain_length);
    
    // Trace domain should be half of randomized trace domain
    EXPECT_EQ(domains.trace.length * 2, domains.randomized_trace.length);
    
    // Quotient domain should be at least as large as max_degree
    EXPECT_GE(domains.quotient.length, static_cast<size_t>(max_degree));
    
    // Step 5: Enqueue log2 padded height to proof stream
    ProofStream proof_stream;
    uint32_t log2_padded_height = static_cast<uint32_t>(std::log2(padded_height));
    proof_stream.enqueue(ProofItem::make_log2_padded_height(log2_padded_height));
    
    // Verify proof stream was updated
    auto sponge = proof_stream.sponge();
    bool any_nonzero = false;
    for (const auto& state : sponge.state) {
        if (!state.is_zero()) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "Sponge state should be non-zero after enqueueing log2_padded_height";
    
    std::cout << "  ✓ Padded height: " << padded_height << " (log2: " << log2_padded_height << ")" << std::endl;
    std::cout << "  ✓ Trace domain length: " << domains.trace.length << std::endl;
    std::cout << "  ✓ Randomized trace domain length: " << domains.randomized_trace.length << std::endl;
    std::cout << "  ✓ Quotient domain length: " << domains.quotient.length << std::endl;
    std::cout << "  ✓ FRI domain length: " << domains.fri.length << std::endl;
}

/**
 * Test: Main table creation from AET
 * 
 * Verify that we can create a MasterMainTable from AET.
 * This tests the "main tables - create" step.
 */
TEST_F(VMTraceExecutionTest, MainTableCreationFromAET) {
    // Step 1: Trace execution
    Program program = Program::from_code("push 1\npush 2\nadd\nhalt");
    std::vector<BFieldElement> input = {};
    
    auto result = VM::trace_execution(program, input);
    
    EXPECT_GT(result.aet.processor_trace_height(), 0);
    size_t trace_height = result.aet.processor_trace_height();
    
    // Step 2: Derive parameters
    size_t padded_height = result.aet.padded_height();
    EXPECT_GE(padded_height, trace_height);
    
    // Create FRI domain
    size_t fri_domain_length = 4096;
    ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length);
    BFieldElement fri_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(fri_domain_length)) + 1
    );
    fri_domain = fri_domain.with_offset(fri_offset);
    
    // Derive ProverDomains
    constexpr size_t num_trace_randomizers = 30;
    size_t randomized_trace_len = padded_height + num_trace_randomizers;
    size_t rand_trace_pow2 = 1;
    while (rand_trace_pow2 < randomized_trace_len) {
        rand_trace_pow2 <<= 1;
    }
    int64_t interpolant_degree = static_cast<int64_t>(rand_trace_pow2 - 1);
    int64_t max_degree = interpolant_degree * 4;
    
    ProverDomains domains = ProverDomains::derive(
        padded_height,
        num_trace_randomizers,
        fri_domain,
        max_degree
    );
    
    // Step 3: Create main table from AET
    std::array<uint8_t, 32> trace_randomizer_seed{};
    std::random_device rd;
    for (size_t i = 0; i < 32; ++i) {
        trace_randomizer_seed[i] = static_cast<uint8_t>(rd() & 0xFF);
    }
    
    MasterMainTable main_table = MasterMainTable::from_aet(
        result.aet,
        domains,
        num_trace_randomizers,
        trace_randomizer_seed
    );
    
    // Verify table dimensions
    EXPECT_EQ(main_table.num_columns(), 379) << "Main table should have 379 columns";
    EXPECT_EQ(main_table.num_rows(), domains.trace.length) << "Table rows should match trace domain length";
    
    // Verify processor table is filled (columns 7-45)
    using namespace TableColumnOffsets;
    bool has_nonzero_processor = false;
    for (size_t row = 0; row < trace_height && row < main_table.num_rows(); ++row) {
        for (size_t col = PROCESSOR_TABLE_START; col < PROCESSOR_TABLE_START + PROCESSOR_TABLE_COLS; ++col) {
            if (!main_table.get(row, col).is_zero()) {
                has_nonzero_processor = true;
                break;
            }
        }
        if (has_nonzero_processor) break;
    }
    EXPECT_TRUE(has_nonzero_processor) << "Processor table should have non-zero values";
    
    // Verify Program table is filled (columns 0-6)
    // Check that addresses are set correctly
    size_t program_length = result.aet.program_length();
    bool has_program_data = false;
    for (size_t addr = 0; addr < program_length && addr < main_table.num_rows(); ++addr) {
        BFieldElement address_val = main_table.get(addr, PROGRAM_TABLE_START + ProgramMainColumn::Address);
        if (address_val.value() == static_cast<uint64_t>(addr)) {
            has_program_data = true;
            break;
        }
    }
    EXPECT_TRUE(has_program_data) << "Program table should have address values";
    
    // Verify Program table has instruction multiplicities
    bool has_multiplicities = false;
    for (size_t addr = 0; addr < program_length && addr < main_table.num_rows(); ++addr) {
        BFieldElement mult_val = main_table.get(addr, PROGRAM_TABLE_START + ProgramMainColumn::LookupMultiplicity);
        if (!mult_val.is_zero()) {
            has_multiplicities = true;
            break;
        }
    }
    EXPECT_TRUE(has_multiplicities) << "Program table should have instruction multiplicities";
    
    // Verify trace randomizers are set
    EXPECT_EQ(main_table.num_trace_randomizers(), num_trace_randomizers);
    
    // Verify table structure - all columns should be accessible
    for (size_t col = 0; col < main_table.num_columns(); ++col) {
        BFieldElement val = main_table.get(0, col);
        // Just verify we can access all columns without errors
        (void)val;
    }
    
    std::cout << "  ✓ Main table created: " << main_table.num_rows() << " x " << main_table.num_columns() << std::endl;
    std::cout << "  ✓ Processor table filled (columns " << PROCESSOR_TABLE_START 
              << "-" << (PROCESSOR_TABLE_START + PROCESSOR_TABLE_COLS - 1) << ")" << std::endl;
    std::cout << "  ✓ Program table filled (columns " << PROGRAM_TABLE_START 
              << "-" << (PROGRAM_TABLE_START + PROGRAM_TABLE_COLS - 1) << ")" << std::endl;
    std::cout << "  ✓ All 379 columns accessible" << std::endl;
}

/**
 * Test: Main table padding
 * 
 * Verify that we can pad the main table to a power-of-2 height.
 * This tests the "main tables - pad" step.
 */
TEST_F(VMTraceExecutionTest, MainTablePadding) {
    // Step 1: Trace execution
    Program program = Program::from_code("push 1\npush 2\nadd\nhalt");
    std::vector<BFieldElement> input = {};
    
    auto result = VM::trace_execution(program, input);
    
    EXPECT_GT(result.aet.processor_trace_height(), 0);
    size_t trace_height = result.aet.processor_trace_height();
    
    // Step 2: Derive parameters
    size_t padded_height = result.aet.padded_height();
    EXPECT_GE(padded_height, trace_height);
    EXPECT_TRUE((padded_height & (padded_height - 1)) == 0) << "padded_height should be a power of 2";
    
    // Create FRI domain
    size_t fri_domain_length = 4096;
    ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length);
    BFieldElement fri_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(fri_domain_length)) + 1
    );
    fri_domain = fri_domain.with_offset(fri_offset);
    
    // Derive ProverDomains
    constexpr size_t num_trace_randomizers = 30;
    size_t randomized_trace_len = padded_height + num_trace_randomizers;
    size_t rand_trace_pow2 = 1;
    while (rand_trace_pow2 < randomized_trace_len) {
        rand_trace_pow2 <<= 1;
    }
    int64_t interpolant_degree = static_cast<int64_t>(rand_trace_pow2 - 1);
    int64_t max_degree = interpolant_degree * 4;
    
    ProverDomains domains = ProverDomains::derive(
        padded_height,
        num_trace_randomizers,
        fri_domain,
        max_degree
    );
    
    // Step 3: Create main table from AET
    std::array<uint8_t, 32> trace_randomizer_seed{};
    std::random_device rd;
    for (size_t i = 0; i < 32; ++i) {
        trace_randomizer_seed[i] = static_cast<uint8_t>(rd() & 0xFF);
    }
    
    MasterMainTable main_table = MasterMainTable::from_aet(
        result.aet,
        domains,
        num_trace_randomizers,
        trace_randomizer_seed
    );
    
    size_t original_height = main_table.num_rows();
    
    // The table might already be at padded_height if domains.trace.length == padded_height
    // In that case, we still want to test padding logic, so we'll pad to a larger power of 2
    size_t target_padded_height = padded_height;
    if (original_height >= padded_height) {
        // If already at or above padded_height, pad to next power of 2
        target_padded_height = original_height;
        if ((target_padded_height & (target_padded_height - 1)) != 0) {
            // Round up to next power of 2
            target_padded_height = 1;
            while (target_padded_height < original_height) {
                target_padded_height <<= 1;
            }
        } else {
            // Already a power of 2, double it
            target_padded_height <<= 1;
        }
    }
    
    EXPECT_LT(original_height, target_padded_height) << "Table should need padding";
    
    // Step 4: Pad table
    main_table.pad(target_padded_height);
    
    // Verify padding
    EXPECT_EQ(main_table.num_rows(), target_padded_height) << "Table should be padded to target_padded_height";
    
    // Verify processor table padding rules
    using namespace TableColumnOffsets;
    
    // Check that padding rows have IsPadding = 1
    bool has_padding_marker = false;
    for (size_t row = original_height; row < target_padded_height; ++row) {
        size_t is_padding_col = PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::IsPadding);
        if (main_table.get(row, is_padding_col).is_one()) {
            has_padding_marker = true;
            break;
        }
    }
    EXPECT_TRUE(has_padding_marker) << "Padding rows should have IsPadding = 1";
    
    // Check that CLK values are sequential in padding rows
    bool clk_values_correct = true;
    for (size_t row = original_height; row < target_padded_height; ++row) {
        size_t clk_col = PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::CLK);
        BFieldElement clk_val = main_table.get(row, clk_col);
        if (clk_val.value() != static_cast<uint64_t>(row)) {
            clk_values_correct = false;
            break;
        }
    }
    EXPECT_TRUE(clk_values_correct) << "CLK values should be sequential in padding rows";
    
    std::cout << "  ✓ Table padded from " << original_height << " to " << target_padded_height << " rows" << std::endl;
    std::cout << "  ✓ Padding rows have IsPadding = 1" << std::endl;
    std::cout << "  ✓ CLK values are sequential" << std::endl;
}

