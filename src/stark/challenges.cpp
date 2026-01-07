#include "stark/challenges.hpp"
#include "stark/cross_table_arg.hpp"
#include "types/b_field_element.hpp"
#include "hash/tip5.hpp"
#include <stdexcept>
#include <iostream>

namespace triton_vm {

Challenges Challenges::from_sampled(const std::vector<XFieldElement>& sampled) {
    if (sampled.size() != SAMPLE_COUNT) {
        throw std::invalid_argument("Expected " + std::to_string(SAMPLE_COUNT) + " sampled challenges, got " + std::to_string(sampled.size()));
    }
    
    Challenges challenges;
    
    // Copy sampled challenges (indices 0-58)
    for (size_t i = 0; i < SAMPLE_COUNT; i++) {
        challenges.challenges_[i] = sampled[i];
    }
    
    // Leave derived challenges as zeros (for cases where they're not needed)
    return challenges;
}

Challenges Challenges::from_sampled_and_claim(
    const std::vector<XFieldElement>& sampled,
    const std::vector<BFieldElement>& program_digest,
    const std::vector<BFieldElement>& input,
    const std::vector<BFieldElement>& output,
    const std::vector<BFieldElement>& lookup_table
) {
    Challenges challenges = Challenges::from_sampled(sampled);
    challenges.compute_derived_challenges(program_digest, input, output, lookup_table);
    return challenges;
}

void Challenges::compute_derived_challenges(
    const std::vector<BFieldElement>& program_digest,
    const std::vector<BFieldElement>& input,
    const std::vector<BFieldElement>& output,
    const std::vector<BFieldElement>& lookup_table
) {
    using namespace ChallengeId;
    
    // Debug: Check if program_digest is empty
    if (program_digest.empty()) {
        std::cerr << "⚠ Warning: program_digest is empty in compute_derived_challenges" << std::endl;
        return;  // Can't compute derived challenges without program_digest
    }
    
    // Debug output
    std::cerr << "DEBUG compute_derived_challenges:" << std::endl;
    std::cerr << "  program_digest size: " << program_digest.size() << std::endl;
    std::cerr << "  challenges[0] (CompressProgramDigestIndeterminate): " << challenges_[CompressProgramDigestIndeterminate].to_string() << std::endl;
    
    // Compute CompressedProgramDigest using EvalArg
    XFieldElement compressed_digest = EvalArg::compute_terminal(
        program_digest,
        EvalArg::default_initial(),
        challenges_[CompressProgramDigestIndeterminate]
    );
    
    std::cerr << "  compressed_digest result: " << compressed_digest.to_string() << std::endl;
    
    // Compute StandardInputTerminal
    XFieldElement input_terminal = EvalArg::compute_terminal(
        input,
        EvalArg::default_initial(),
        challenges_[StandardInputIndeterminate]
    );
    
    // Compute StandardOutputTerminal
    XFieldElement output_terminal = EvalArg::compute_terminal(
        output,
        EvalArg::default_initial(),
        challenges_[StandardOutputIndeterminate]
    );
    
    // Compute LookupTablePublicTerminal
    XFieldElement lookup_terminal = EvalArg::compute_terminal(
        lookup_table,
        EvalArg::default_initial(),
        challenges_[LookupTablePublicIndeterminate]
    );
    
    // Set derived challenges
    challenges_[StandardInputTerminal] = input_terminal;
    challenges_[StandardOutputTerminal] = output_terminal;
    challenges_[LookupTablePublicTerminal] = lookup_terminal;
    challenges_[CompressedProgramDigest] = compressed_digest;
    
    // Debug: Verify challenges[62] is set
    if (challenges_[CompressedProgramDigest].is_zero()) {
        std::cerr << "⚠ Warning: CompressedProgramDigest (challenges[62]) is zero after computation" << std::endl;
        std::cerr << "  program_digest size: " << program_digest.size() << std::endl;
        std::cerr << "  challenges[0] (CompressProgramDigestIndeterminate): " << challenges_[CompressProgramDigestIndeterminate].to_string() << std::endl;
    }
}

} // namespace triton_vm

