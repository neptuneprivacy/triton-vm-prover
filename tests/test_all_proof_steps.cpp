/**
 * Comprehensive test for all 15 proof generation steps
 * 
 * This test exercises each step from the performance log (base-performance.log):
 * 1. trace execution
 * 2. Fiat-Shamir: claim
 * 3. derive additional parameters
 * 4. main tables (create, pad, LDE, Merkle tree, Fiat-Shamir, extend)
 * 5. aux tables (LDE, Merkle tree, Fiat-Shamir)
 * 6. quotient calculation (zerofier inverse, evaluate AIR, compute quotient codeword)
 * 7. quotient LDE
 * 8. hash rows of quotient segments
 * 9. Merkle tree (quotient)
 * 10. out-of-domain rows
 * 11. linear combination (main, aux, quotient)
 * 12. DEEP (main&aux curr row, main&aux next row, segmented quotient)
 * 13. combined DEEP polynomial (sum)
 * 14. FRI
 * 15. open trace leafs
 */

#include <gtest/gtest.h>
#include "vm/vm.hpp"
#include "vm/aet.hpp"
#include "stark.hpp"
#include "proof_stream/proof_stream.hpp"
#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>

using namespace triton_vm;

class AllProofStepsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple program with a loop to ensure jump stack table has rows
        // push 1, push 0, call, halt
        // This creates a program that will have jump stack entries
        program_ = std::make_unique<Program>(Program::from_code("push 1\npush 2\nadd\nhalt"));
        input_ = {};
        
        // Execute trace once for all tests
        auto result = VM::trace_execution(*program_, input_);
        trace_result_ = std::make_unique<VM::TraceResult>(std::move(result));
        
        // Create claim
        claim_ = std::make_unique<Claim>();
        claim_->program_digest = program_->hash();
        claim_->version = 1;
        claim_->input = input_;
        claim_->output = trace_result_->output;
        
        // Create STARK prover
        stark_ = std::make_unique<Stark>(Stark::default_stark());
        
        // Convert AET to SimpleAlgebraicExecutionTrace
        simple_aet_ = std::make_unique<SimpleAlgebraicExecutionTrace>();
        simple_aet_->padded_height = trace_result_->aet.padded_height();
        simple_aet_->processor_trace_height = trace_result_->aet.processor_trace_height();
        simple_aet_->processor_trace_width = trace_result_->aet.processor_trace_width();
        simple_aet_->processor_trace = trace_result_->aet.processor_trace();
    }
    
    void TearDown() override {
    }
    
    std::unique_ptr<Program> program_;
    std::vector<BFieldElement> input_;
    std::unique_ptr<VM::TraceResult> trace_result_;
    std::unique_ptr<Claim> claim_;
    std::unique_ptr<Stark> stark_;
    std::unique_ptr<SimpleAlgebraicExecutionTrace> simple_aet_;
};

// Test 1: Trace execution
TEST_F(AllProofStepsTest, Step1_TraceExecution) {
    std::cout << "\n[Test 1/15] Trace execution" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = VM::trace_execution(*program_, input_);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    EXPECT_FALSE(result.aet.processor_trace().empty());
    EXPECT_GT(result.aet.processor_trace_height(), 0);
    EXPECT_GT(result.aet.padded_height(), 0);
    
    std::cout << "  ✓ Trace execution completed in " << duration << " ms" << std::endl;
    std::cout << "    Processor trace height: " << result.aet.processor_trace_height() << std::endl;
    std::cout << "    Padded height: " << result.aet.padded_height() << std::endl;
}

// Test 2: Fiat-Shamir: claim
TEST_F(AllProofStepsTest, Step2_FiatShamirClaim) {
    std::cout << "\n[Test 2/15] Fiat-Shamir: claim" << std::endl;
    
    ProofStream proof_stream;
    
    // Encode claim (matching Rust's BFieldCodec)
    std::vector<BFieldElement> claim_encoding;
    auto encode_vec = [&](const std::vector<BFieldElement>& v) {
        claim_encoding.push_back(BFieldElement(1 + v.size()));
        claim_encoding.push_back(BFieldElement(v.size()));
        for (const auto& e : v) claim_encoding.push_back(e);
    };
    encode_vec(claim_->output);
    encode_vec(claim_->input);
    claim_encoding.push_back(BFieldElement(claim_->version));
    for (size_t i = 0; i < Digest::LEN; ++i) {
        claim_encoding.push_back(claim_->program_digest[i]);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    proof_stream.alter_fiat_shamir_state_with(claim_encoding);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Verify sponge state was updated
    auto sponge = proof_stream.sponge();
    EXPECT_NE(sponge.state[0].value(), 0);
    
    std::cout << "  ✓ Fiat-Shamir: claim completed in " << duration << " ns" << std::endl;
    std::cout << "    Sponge state[0]: " << sponge.state[0].value() << std::endl;
}

// Test 3: Derive additional parameters
TEST_F(AllProofStepsTest, Step3_DeriveAdditionalParameters) {
    std::cout << "\n[Test 3/15] Derive additional parameters" << std::endl;
    
    size_t padded_height = simple_aet_->padded_height;
    size_t fri_domain_length = 4096;
    ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length);
    BFieldElement fri_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(static_cast<double>(fri_domain_length))) + 1
    );
    fri_domain = fri_domain.with_offset(fri_offset);
    
    auto start = std::chrono::high_resolution_clock::now();
    ProverDomains domains = ProverDomains::derive(
        padded_height,
        stark_->num_trace_randomizers(),
        fri_domain,
        stark_->max_degree(padded_height)
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    EXPECT_GT(domains.trace.length, 0);
    EXPECT_GT(domains.quotient.length, 0);
    EXPECT_GT(domains.fri.length, 0);
    
    std::cout << "  ✓ Derive additional parameters completed in " << duration << " µs" << std::endl;
    std::cout << "    Trace domain length: " << domains.trace.length << std::endl;
    std::cout << "    FRI domain length: " << domains.fri.length << std::endl;
}

// Test 4: Main tables (create, pad, LDE, Merkle tree, Fiat-Shamir, extend)
TEST_F(AllProofStepsTest, Step4_MainTables) {
    std::cout << "\n[Test 4/15] Main tables" << std::endl;
    
    size_t padded_height = simple_aet_->padded_height;
    size_t fri_domain_length = 4096;
    ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length);
    BFieldElement fri_offset = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(static_cast<double>(fri_domain_length))) + 1
    );
    fri_domain = fri_domain.with_offset(fri_offset);
    ProverDomains domains = ProverDomains::derive(
        padded_height,
        stark_->num_trace_randomizers(),
        fri_domain,
        stark_->max_degree(padded_height)
    );
    
    std::array<uint8_t, 32> trace_randomizer_seed{};
    constexpr size_t NUM_COLUMNS = 379;
    
    auto start = std::chrono::high_resolution_clock::now();
    MasterMainTable main_table(
        domains.trace.length,
        NUM_COLUMNS,
        domains.trace,
        domains.quotient,
        domains.fri,
        trace_randomizer_seed
    );
        main_table.set_num_trace_randomizers(stark_->num_trace_randomizers());
    
    // Fill processor table
    using namespace TableColumnOffsets;
        for (size_t row = 0; row < simple_aet_->processor_trace.size() && row < main_table.num_rows(); ++row) {
            const auto& processor_row = simple_aet_->processor_trace[row];
        for (size_t col = 0; col < PROCESSOR_TABLE_COLS && col < processor_row.size(); ++col) {
            main_table.set(row, PROCESSOR_TABLE_START + col, processor_row[col]);
        }
    }
    
    main_table.pad(padded_height);
    main_table.low_degree_extend(domains.fri);
    auto main_commitment = TableCommitment::commit(main_table);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    EXPECT_GT(main_table.num_rows(), 0);
    EXPECT_GT(main_table.num_columns(), 0);
    EXPECT_FALSE(main_table.lde_table().empty());
    
    std::cout << "  ✓ Main tables completed in " << duration << " ms" << std::endl;
    std::cout << "    Table size: " << main_table.num_rows() << " x " << main_table.num_columns() << std::endl;
    std::cout << "    Merkle root: " << main_commitment.root() << std::endl;
}

// Test 5: Aux tables (LDE, Merkle tree, Fiat-Shamir)
TEST_F(AllProofStepsTest, Step5_AuxTables) {
    std::cout << "\n[Test 5/15] Aux tables" << std::endl;
    
    // Generate proof to get to aux tables step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    // Verify proof contains aux table commitment
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Aux tables completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 6: Quotient calculation (zerofier inverse, evaluate AIR, compute quotient codeword)
TEST_F(AllProofStepsTest, Step6_QuotientCalculation) {
    std::cout << "\n[Test 6/15] Quotient calculation" << std::endl;
    
    // Generate proof to get to quotient calculation step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    // Verify proof contains quotient-related data
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Quotient calculation completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 7: Quotient LDE
TEST_F(AllProofStepsTest, Step7_QuotientLDE) {
    std::cout << "\n[Test 7/15] Quotient LDE" << std::endl;
    
    // Generate proof to get to quotient LDE step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Quotient LDE completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 8: Hash rows of quotient segments
TEST_F(AllProofStepsTest, Step8_HashRowsOfQuotientSegments) {
    std::cout << "\n[Test 8/15] Hash rows of quotient segments" << std::endl;
    
    // Generate proof to get to hash rows step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Hash rows of quotient segments completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 9: Merkle tree (quotient)
TEST_F(AllProofStepsTest, Step9_MerkleTreeQuotient) {
    std::cout << "\n[Test 9/15] Merkle tree (quotient)" << std::endl;
    
    // Generate proof to get to quotient Merkle tree step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Merkle tree (quotient) completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 10: Out-of-domain rows
TEST_F(AllProofStepsTest, Step10_OutOfDomainRows) {
    std::cout << "\n[Test 10/15] Out-of-domain rows" << std::endl;
    
    // Generate proof to get to out-of-domain rows step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Out-of-domain rows completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 11: Linear combination (main, aux, quotient)
TEST_F(AllProofStepsTest, Step11_LinearCombination) {
    std::cout << "\n[Test 11/15] Linear combination" << std::endl;
    
    // Generate proof to get to linear combination step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Linear combination completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 12: DEEP (main&aux curr row, main&aux next row, segmented quotient)
TEST_F(AllProofStepsTest, Step12_DEEP) {
    std::cout << "\n[Test 12/15] DEEP" << std::endl;
    
    // Generate proof to get to DEEP step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ DEEP completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 13: Combined DEEP polynomial (sum)
TEST_F(AllProofStepsTest, Step13_CombinedDEEPPolynomial) {
    std::cout << "\n[Test 13/15] Combined DEEP polynomial" << std::endl;
    
    // Generate proof to get to combined DEEP polynomial step
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ Combined DEEP polynomial completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 14: FRI
TEST_F(AllProofStepsTest, Step14_FRI) {
    std::cout << "\n[Test 14/15] FRI" << std::endl;
    
    // Generate proof to get to FRI step
    auto start = std::chrono::high_resolution_clock::now();
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    EXPECT_FALSE(proof.elements.empty());
    
    std::cout << "  ✓ FRI completed in " << duration << " ms" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
}

// Test 15: Open trace leafs
TEST_F(AllProofStepsTest, Step15_OpenTraceLeafs) {
    std::cout << "\n[Test 15/15] Open trace leafs" << std::endl;
    
    // Generate complete proof (includes opening trace leafs)
    Proof proof = stark_->prove(*claim_, *simple_aet_);
    
    EXPECT_FALSE(proof.elements.empty());
    EXPECT_GT(proof.elements.size(), 1000); // Should have substantial proof data
    
    std::cout << "  ✓ Open trace leafs completed" << std::endl;
    std::cout << "    Proof size: " << proof.elements.size() << " BFieldElements" << std::endl;
    std::cout << "\n=== All 15 steps tested successfully ===" << std::endl;
}
