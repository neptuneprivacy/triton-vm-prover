#include <gtest/gtest.h>

#include "stark.hpp"
#include "vm/vm.hpp"
#include "vm/program.hpp"
#include "vm/aet.hpp"

using namespace triton_vm;

TEST(PureCppProofVerificationTest, ProveAndVerifySmallProgram) {
    // Program: simple arithmetic, terminates quickly.
    Program program = Program::from_code("push 1\npush 2\nadd\nhalt\n");
    std::vector<BFieldElement> public_input = {};

    // Trace execution (pure C++)
    auto trace_result = VM::trace_execution(program, public_input);
    const AlgebraicExecutionTrace& aet = trace_result.aet;

    // Claim (match Rust layout assumptions used by Fiat-Shamir)
    Claim claim;
    claim.program_digest = program.hash();
    claim.version = 0;
    claim.input = public_input;
    claim.output = trace_result.output;

    // Domains and main table creation (pure C++ main tables + padding; degree lowering may still be filled during pad)
    Stark stark = Stark::default_stark();
    const size_t padded_height = aet.padded_height();

    const size_t rand_trace_len = stark.randomized_trace_len(padded_height);
    const size_t fri_domain_length = stark.fri_expansion_factor() * rand_trace_len;
    ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length).with_offset(BFieldElement::generator());

    ProverDomains domains = ProverDomains::derive(
        padded_height,
        stark.num_trace_randomizers(),
        fri_domain,
        stark.max_degree(padded_height)
    );

    std::array<uint8_t, 32> seed{};
    MasterMainTable main_table = MasterMainTable::from_aet(aet, domains, stark.num_trace_randomizers(), seed);

    std::array<size_t, 9> table_lengths = {
        aet.height_of_table(0),
        aet.height_of_table(1),
        aet.height_of_table(2),
        aet.height_of_table(3),
        aet.height_of_table(4),
        aet.height_of_table(5),
        aet.height_of_table(6),
        aet.height_of_table(7),
        aet.height_of_table(8)
    };
    main_table.pad(padded_height, table_lengths);

    ProofStream ps;
    Proof proof = stark.prove_with_table(claim, main_table, ps, /*proof_path=*/"");
    ASSERT_FALSE(proof.elements.empty());

    // Pure C++ verifier (no Rust CLI / no verifier FFI)
    EXPECT_TRUE(stark.verify(claim, proof));
}


