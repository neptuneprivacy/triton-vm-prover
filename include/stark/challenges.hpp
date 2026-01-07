#pragma once

#include "types/x_field_element.hpp"
#include <vector>
#include <cassert>

namespace triton_vm {

/**
 * Challenges - Fiat-Shamir challenges used for cross-table arguments
 * 
 * Challenges are XFieldElements used for:
 * - Permutation arguments
 * - Evaluation arguments  
 * - Lookup arguments
 * - RAM table contiguity argument
 * 
 * Total count is 63 (59 sampled + 4 derived)
 */
class Challenges {
public:
    static constexpr size_t COUNT = 63;  // Total challenges
    static constexpr size_t NUM_DERIVED = 4;  // Derived challenges
    static constexpr size_t SAMPLE_COUNT = COUNT - NUM_DERIVED;  // 59
    
    Challenges() : challenges_(COUNT) {}
    
    // Construct from sampled challenges (59) - derived ones will be computed
    static Challenges from_sampled(const std::vector<XFieldElement>& sampled);
    
    // Construct from sampled challenges and claim data (like Rust does)
    static Challenges from_sampled_and_claim(
        const std::vector<XFieldElement>& sampled,
        const std::vector<BFieldElement>& program_digest,
        const std::vector<BFieldElement>& input,
        const std::vector<BFieldElement>& output,
        const std::vector<BFieldElement>& lookup_table
    );
    
    // Compute derived challenges (public for testing)
    void compute_derived_challenges(
        const std::vector<BFieldElement>& program_digest,
        const std::vector<BFieldElement>& input,
        const std::vector<BFieldElement>& output,
        const std::vector<BFieldElement>& lookup_table
    );
    
    // Access challenges by index
    const XFieldElement& operator[](size_t idx) const {
        assert(idx < COUNT);
        return challenges_[idx];
    }
    
    XFieldElement& operator[](size_t idx) {
        assert(idx < COUNT);
        return challenges_[idx];
    }
    
    // Get all challenges
    const std::vector<XFieldElement>& all() const { return challenges_; }

private:
    std::vector<XFieldElement> challenges_;
};

// Challenge ID indices (matching Rust ChallengeId enum)
namespace ChallengeId {
    // Indeterminates
    constexpr size_t CompressProgramDigestIndeterminate = 0;
    constexpr size_t StandardInputIndeterminate = 1;
    constexpr size_t StandardOutputIndeterminate = 2;
    constexpr size_t InstructionLookupIndeterminate = 3;
    constexpr size_t HashInputIndeterminate = 4;
    constexpr size_t HashDigestIndeterminate = 5;
    constexpr size_t SpongeIndeterminate = 6;
    constexpr size_t OpStackIndeterminate = 7;
    constexpr size_t RamIndeterminate = 8;
    constexpr size_t JumpStackIndeterminate = 9;
    constexpr size_t U32Indeterminate = 10;
    constexpr size_t ClockJumpDifferenceLookupIndeterminate = 11;
    constexpr size_t RamTableBezoutRelationIndeterminate = 12;
    constexpr size_t ProgramAddressWeight = 13;
    constexpr size_t ProgramInstructionWeight = 14;
    constexpr size_t ProgramNextInstructionWeight = 15;
    constexpr size_t OpStackClkWeight = 16;
    constexpr size_t OpStackIb1Weight = 17;
    constexpr size_t OpStackPointerWeight = 18;
    constexpr size_t OpStackFirstUnderflowElementWeight = 19;
    constexpr size_t RamClkWeight = 20;
    constexpr size_t RamPointerWeight = 21;
    constexpr size_t RamValueWeight = 22;
    constexpr size_t RamInstructionTypeWeight = 23;
    constexpr size_t JumpStackClkWeight = 24;
    constexpr size_t JumpStackCiWeight = 25;
    constexpr size_t JumpStackJspWeight = 26;
    constexpr size_t JumpStackJsoWeight = 27;
    constexpr size_t JumpStackJsdWeight = 28;
    constexpr size_t ProgramAttestationPrepareChunkIndeterminate = 29;
    constexpr size_t ProgramAttestationSendChunkIndeterminate = 30;
    constexpr size_t HashCIWeight = 31;
    constexpr size_t StackWeight0 = 32;
    constexpr size_t StackWeight1 = 33;
    constexpr size_t StackWeight2 = 34;
    constexpr size_t StackWeight3 = 35;
    constexpr size_t StackWeight4 = 36;
    constexpr size_t StackWeight5 = 37;
    constexpr size_t StackWeight6 = 38;
    constexpr size_t StackWeight7 = 39;
    constexpr size_t StackWeight8 = 40;
    constexpr size_t StackWeight9 = 41;
    constexpr size_t StackWeight10 = 42;
    constexpr size_t StackWeight11 = 43;
    constexpr size_t StackWeight12 = 44;
    constexpr size_t StackWeight13 = 45;
    constexpr size_t StackWeight14 = 46;
    constexpr size_t StackWeight15 = 47;
    constexpr size_t HashCascadeLookupIndeterminate = 48;
    constexpr size_t HashCascadeLookInWeight = 49;
    constexpr size_t HashCascadeLookOutWeight = 50;
    constexpr size_t CascadeLookupIndeterminate = 51;
    constexpr size_t LookupTableInputWeight = 52;
    constexpr size_t LookupTableOutputWeight = 53;
    constexpr size_t LookupTablePublicIndeterminate = 54;
    constexpr size_t U32LhsWeight = 55;
    constexpr size_t U32RhsWeight = 56;
    constexpr size_t U32CiWeight = 57;
    constexpr size_t U32ResultWeight = 58;
    
    // Derived challenges (last 4)
    constexpr size_t StandardInputTerminal = Challenges::COUNT - 4;
    constexpr size_t StandardOutputTerminal = Challenges::COUNT - 3;
    constexpr size_t LookupTablePublicTerminal = Challenges::COUNT - 2;
    constexpr size_t CompressedProgramDigest = Challenges::COUNT - 1;
}

} // namespace triton_vm

