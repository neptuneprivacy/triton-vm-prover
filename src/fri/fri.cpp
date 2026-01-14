#include "fri/fri.hpp"
#include "proof_stream/proof_stream.hpp"
#include "ntt/ntt.hpp"
#include "merkle/merkle_tree.hpp"
#include "common/debug_control.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <chrono>
#include <optional>

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/fri_kernel.cuh"
#include "gpu/kernels/merkle_kernel.cuh"
#include <cuda_runtime.h>
#endif

namespace triton_vm {

// FriRound implementation
FriRound::FriRound(const ArithmeticDomain& domain, const std::vector<XFieldElement>& codeword)
    : domain(domain)
    , codeword(codeword)
{
    // Convert each codeword element to Digest (no hashing - matches Rust's From<XFieldElement> for Digest)
    // Rust: Digest::new([c0, c1, c2, BFieldElement::ZERO, BFieldElement::ZERO])
    std::vector<Digest> leaves;
    leaves.reserve(codeword.size());
    
    for (const auto& elem : codeword) {
        // Direct conversion: XFieldElement -> Digest by padding with zeros
        // This matches Rust's impl From<XFieldElement> for Digest
        Digest digest(
            elem.coeff(0),  // c0
            elem.coeff(1),  // c1
            elem.coeff(2),  // c2
            BFieldElement::zero(),  // pad with zero
            BFieldElement::zero()   // pad with zero
        );
        leaves.push_back(digest);
    }
    
    merkle_tree = std::make_unique<MerkleTree>(leaves);
}

std::vector<XFieldElement> FriRound::split_and_fold(const XFieldElement& folding_challenge) const {
    const size_t n = codeword.size();
    const size_t half_n = n / 2;
    
    // Get domain points (matches Rust: domain.domain_values())
    std::vector<BFieldElement> domain_points;
    domain_points.reserve(n);
    BFieldElement current = domain.offset;
    for (size_t i = 0; i < n; ++i) {
        domain_points.push_back(current);
        current = current * domain.generator;
    }
    
    // Compute inverses of all domain points (matches Rust exactly)
    // Rust: BFieldElement::batch_inversion(domain_points)
    // Then only uses first half: domain_point_inverses[i] for i in 0..n/2
    auto all_domain_point_inverses = BFieldElement::batch_inversion(domain_points);
    // Extract first half (which is what we actually use)
    std::vector<BFieldElement> domain_point_inverses(
        all_domain_point_inverses.begin(), 
        all_domain_point_inverses.begin() + half_n
    );
    
    XFieldElement one = XFieldElement::one();
    XFieldElement two_inv = XFieldElement(BFieldElement(2)).inverse();
    
    std::vector<XFieldElement> folded;
    folded.reserve(half_n);
    
    for (size_t i = 0; i < half_n; ++i) {
        // scaled_offset_inv = folding_challenge * domain_point_inverses[i]
        XFieldElement scaled_offset_inv = folding_challenge * domain_point_inverses[i];
        
        // left_summand = (1 + scaled_offset_inv) * codeword[i]
        XFieldElement left_summand = (one + scaled_offset_inv) * codeword[i];
        
        // right_summand = (1 - scaled_offset_inv) * codeword[n/2 + i]
        XFieldElement right_summand = (one - scaled_offset_inv) * codeword[half_n + i];
        
        // result = (left + right) / 2
        folded.push_back((left_summand + right_summand) * two_inv);
    }
    
    return folded;
}

Digest FriRound::merkle_root() const {
    return merkle_tree->root();
}

// Fri implementation
Fri::Fri(const ArithmeticDomain& domain, 
         size_t expansion_factor,
         size_t num_collinearity_checks)
    : domain_(domain)
    , expansion_factor_(expansion_factor)
    , num_collinearity_checks_(num_collinearity_checks)
{
    if (expansion_factor <= 1) {
        throw std::invalid_argument("Expansion factor must be > 1");
    }
    if ((expansion_factor & (expansion_factor - 1)) != 0) {
        throw std::invalid_argument("Expansion factor must be power of 2");
    }
    if (expansion_factor > domain.length) {
        throw std::invalid_argument("Expansion factor must not exceed domain length");
    }
}

size_t Fri::first_round_max_degree() const {
    return (domain_.length / expansion_factor_) - 1;
}

size_t Fri::last_round_max_degree() const {
    return first_round_max_degree() >> num_rounds();
}

size_t Fri::num_rounds() const {
    // FRI rounds calculation matches Rust exactly:
    // See triton-vm/src/fri.rs::num_rounds()
    
    size_t first_round_code_dimension = first_round_max_degree() + 1;
    
    // next_power_of_two and ilog2
    size_t code_dim_pow2 = 1;
    while (code_dim_pow2 < first_round_code_dimension) {
        code_dim_pow2 <<= 1;
    }
    
    size_t max_num_rounds = 0;
    size_t temp = code_dim_pow2;
    while (temp > 1) {
        temp >>= 1;
        max_num_rounds++;
    }
    
    // Skip rounds for which Merkle tree verification cost exceeds
    // arithmetic cost, because more than half the codeword's locations are
    // queried.
    size_t num_rounds_checking_all = 0;
    size_t temp_checks = num_collinearity_checks_;
    while (temp_checks > 1) {
        temp_checks >>= 1;
        num_rounds_checking_all++;
    }
    size_t num_rounds_checking_most = num_rounds_checking_all + 1;
    
    // saturating_sub
    if (max_num_rounds >= num_rounds_checking_most) {
        return max_num_rounds - num_rounds_checking_most;
    }
    return 0;
}

ArithmeticDomain Fri::round_domain(size_t round) const {
    ArithmeticDomain result = domain_;
    for (size_t i = 0; i < round; ++i) {
        result = result.halve();
    }
    return result;
}

size_t Fri::fold_index(size_t index, size_t domain_length) {
    // In folding, index i and (i + n/2) both map to index i in the folded domain
    return index % (domain_length / 2);
}

std::vector<size_t> Fri::collinearity_check_b_indices(
    const std::vector<size_t>& a_indices,
    size_t domain_length
) const {
    std::vector<size_t> b_indices;
    b_indices.reserve(a_indices.size());
    for (size_t a_index : a_indices) {
        b_indices.push_back((a_index + domain_length / 2) % domain_length);
    }
    return b_indices;
}

std::vector<size_t> Fri::prove(
    const std::vector<XFieldElement>& codeword,
    ProofStream& proof_stream
) const {
    if (codeword.size() != domain_.length) {
        throw std::invalid_argument("Codeword length must match domain length");
    }
    
    size_t rounds = num_rounds();
    std::cout << "DEBUG: Full FRI.prove() - num_rounds() = " << rounds << std::endl;
    
    // FIXME: Temporary workaround - use actual number of rounds from proof stream
    // The issue is that num_rounds() returns 2, but Rust executes 3 rounds.
    // This suggests num_rounds() calculation might not match Rust exactly.
    // For now, we'll continue with the computed value, but this needs investigation.
    
    std::vector<FriRound> round_data;
    
    // Commit phase: fold codeword through all rounds
    // Rust pattern: commit to each round, then sample challenge for next
    std::vector<XFieldElement> current_codeword = codeword;
    
    // First round: commit
    {
        FriRound round(domain_, current_codeword);
        Digest first_merkle_root = round.merkle_root();
        
        // DEBUG: Capture sponge state before and after enqueuing first Merkle root
        Tip5 sponge_before = proof_stream.sponge();
        std::cout << "DEBUG: Full FRI.prove() - Sponge before first Merkle root: " 
                  << sponge_before.state[0].value() << ","
                  << sponge_before.state[1].value() << ","
                  << sponge_before.state[2].value() << std::endl;
        std::cout << "DEBUG: Full FRI.prove() - First Merkle root: " 
                  << first_merkle_root[0].value() << ","
                  << first_merkle_root[1].value() << ","
                  << first_merkle_root[2].value() << ","
                  << first_merkle_root[3].value() << ","
                  << first_merkle_root[4].value() << std::endl;
        
        proof_stream.enqueue(ProofItem::merkle_root(first_merkle_root));
        
        Tip5 sponge_after = proof_stream.sponge();
        std::cout << "DEBUG: Full FRI.prove() - Sponge after first Merkle root: " 
                  << sponge_after.state[0].value() << ","
                  << sponge_after.state[1].value() << ","
                  << sponge_after.state[2].value() << std::endl;
        
        round_data.push_back(std::move(round));
    }

    // Subsequent rounds: sample challenge, fold, commit
    // Matches Rust: previous_round.split_and_fold(challenge), then previous_round.domain.halve()
    for (size_t r = 0; r < rounds; ++r) {
        // DEBUG: Track sponge state before each round
        Tip5 sponge_before_round = proof_stream.sponge();
        std::cout << "DEBUG: Full FRI.prove() - Round " << (r+1) << " - Sponge before: " 
                  << sponge_before_round.state[0].value() << ","
                  << sponge_before_round.state[1].value() << ","
                  << sponge_before_round.state[2].value() << std::endl;
        
        // Sample folding challenge
        XFieldElement folding_challenge = proof_stream.sample_scalars(1)[0];
        
        Tip5 sponge_after_challenge = proof_stream.sponge();
        std::cout << "DEBUG: Full FRI.prove() - Round " << (r+1) << " - Sponge after challenge: " 
                  << sponge_after_challenge.state[0].value() << ","
                  << sponge_after_challenge.state[1].value() << ","
                  << sponge_after_challenge.state[2].value() << std::endl;

        // Fold (matches Rust: split_and_fold first)
        current_codeword = round_data.back().split_and_fold(folding_challenge);
        
        // Halve domain (matches Rust: previous_round.domain.halve())
        ArithmeticDomain next_domain = round_data.back().domain.halve();
        std::cout << "DEBUG: Previous domain length: " << round_data.back().domain.length << std::endl;
        std::cout << "DEBUG: Next domain length: " << next_domain.length << std::endl;

        // Commit to new round
        FriRound round(next_domain, current_codeword);
        Digest round_merkle_root = round.merkle_root();
        proof_stream.enqueue(ProofItem::merkle_root(round_merkle_root));
        round_data.push_back(std::move(round));
        
        // DEBUG: Track sponge state after each round
        Tip5 sponge_after_round = proof_stream.sponge();
        std::cout << "DEBUG: Full FRI.prove() - Round " << (r+1) << " - Sponge after Merkle root: " 
                  << sponge_after_round.state[0].value() << ","
                  << sponge_after_round.state[1].value() << ","
                  << sponge_after_round.state[2].value() << std::endl;
    }
    
    // Send last codeword (matches Rust: self.rounds.last().unwrap().codeword)
    const std::vector<XFieldElement>& last_codeword = round_data.back().codeword;
    std::cout << "Enqueuing FriCodeword with " << last_codeword.size() << " elements" << std::endl;
    proof_stream.enqueue(ProofItem::fri_codeword(last_codeword));
    
    // Send last polynomial (interpolate last codeword)
    // Rust: let last_polynomial = ArithmeticDomain::of_length(last_codeword.len()).interpolate(&last_codeword);
    std::vector<XFieldElement> last_polynomial;
    {
        // Pure C++ interpolation on the multiplicative subgroup (offset = 1).
        // ArithmeticDomain::of_length(n).interpolate(codeword) is equivalent to component-wise INTT.
        std::vector<BFieldElement> comp0(last_codeword.size());
        std::vector<BFieldElement> comp1(last_codeword.size());
        std::vector<BFieldElement> comp2(last_codeword.size());
        for (size_t i = 0; i < last_codeword.size(); ++i) {
            comp0[i] = last_codeword[i].coeff(0);
            comp1[i] = last_codeword[i].coeff(1);
            comp2[i] = last_codeword[i].coeff(2);
        }
        auto coeffs0 = NTT::interpolate(comp0);
        auto coeffs1 = NTT::interpolate(comp1);
        auto coeffs2 = NTT::interpolate(comp2);
        last_polynomial.resize(coeffs0.size());
        for (size_t i = 0; i < coeffs0.size(); ++i) {
            last_polynomial[i] = XFieldElement(coeffs0[i], coeffs1[i], coeffs2[i]);
        }
        // Trim trailing zeros to match Rust `Polynomial` encoding and verifier expectations.
        while (!last_polynomial.empty() && last_polynomial.back().is_zero()) {
            last_polynomial.pop_back();
        }
    }
    Tip5 sponge_before_polynomial = proof_stream.sponge();
    proof_stream.enqueue(ProofItem::fri_polynomial(last_polynomial));
    Tip5 sponge_after_polynomial = proof_stream.sponge();
    std::cout << "DEBUG: Full FRI.prove() - Sponge before last polynomial: " 
              << sponge_before_polynomial.state[0].value() << ","
              << sponge_before_polynomial.state[1].value() << ","
              << sponge_before_polynomial.state[2].value() << std::endl;
    std::cout << "DEBUG: Full FRI.prove() - Sponge after last polynomial: "
              << sponge_after_polynomial.state[0].value() << ","
              << sponge_after_polynomial.state[1].value() << ","
              << sponge_after_polynomial.state[2].value() << std::endl;

    // DEBUG: Capture sponge state right before sampling indices (for comparison with Rust)
    // Note: In Rust, FriCodeword is sent BEFORE FriPolynomial, so the sponge state
    // after FriPolynomial is the state before query sampling
    // This should match Rust's sponge_state_before_query
    Tip5 sponge_before_query = proof_stream.sponge();
    std::cout << "DEBUG: Sponge state before query (first 3 elements): ";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << sponge_before_query.state[i].value();
        if (i < 2) std::cout << ",";
    }
    std::cout << std::endl;
    
    // Query phase: sample indices, then reveal
    // Store original indices for return (they get modified in the loop)
    std::vector<size_t> first_round_indices = proof_stream.sample_indices(
        domain_.length,
        num_collinearity_checks_);
    std::vector<size_t> original_first_round_indices = first_round_indices;
    
    // Reveal at a-indices for first round
    {
        const FriRound& round = round_data[0];
        std::vector<XFieldElement> revealed_leaves;
        for (size_t idx : first_round_indices) {
            revealed_leaves.push_back(round.codeword[idx]);
        }
        // Debug: Print first few indices
        static bool debug_fri = std::getenv("TVM_DEBUG_FRI_INDICES") != nullptr;
        if (debug_fri) {
            std::cout << "DEBUG: FRI Round 0 (a-indices) - first 10 indices: ";
            for (size_t i = 0; i < 10 && i < first_round_indices.size(); ++i) {
                std::cout << first_round_indices[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "DEBUG: FRI Round 0 codeword length: " << round.codeword.size() << std::endl;
            std::cout << "DEBUG: FRI Round 0 Merkle tree num_leaves: " << round.merkle_tree->num_leaves() << std::endl;
        }
        
        std::vector<Digest> auth_structure = round.merkle_tree->authentication_structure(first_round_indices);
        
        // Debug: Print auth structure size
        std::cout << "DEBUG: FRI Round 0 (a-indices) - " << first_round_indices.size() 
                  << " indices, " << auth_structure.size() << " auth digests" << std::endl;
        
        FriResponse response;
        response.auth_structure = auth_structure;
        response.revealed_leaves = revealed_leaves;
        
        // Debug: Print FriResponse sizes
        bool debug_fri_response_sizes = std::getenv("TVM_DEBUG_FRI_RESPONSE_SIZES") != nullptr;
        if (debug_fri_response_sizes) {
            std::cout << "DEBUG: FriResponse Round 0:" << std::endl;
            std::cout << "  auth_structure: " << auth_structure.size() << " digests (" << (auth_structure.size() * 5) << " BFieldElements)" << std::endl;
            std::cout << "  revealed_leaves: " << revealed_leaves.size() << " XFieldElements (" << (revealed_leaves.size() * 3) << " BFieldElements)" << std::endl;
            std::cout << "  Total field encoding: " << (1 + auth_structure.size() * 5) << " + " << (1 + revealed_leaves.size() * 3) << " = " 
                      << (1 + auth_structure.size() * 5 + 1 + revealed_leaves.size() * 3) << " BFieldElements" << std::endl;
        }
        
        proof_stream.enqueue(ProofItem::fri_response(response));
    }
    
    // For each round (except the last), reveal b-indices
    // Rust uses original_first_round_indices for ALL rounds' b_indices calculation
    for (size_t r = 0; r < rounds; ++r) {
        const FriRound& round = round_data[r];
        
        // Compute b-indices using ORIGINAL first-round indices (matching Rust exactly)
        // Rust: self.first_round_collinearity_check_indices for all rounds
        std::vector<size_t> b_indices = collinearity_check_b_indices(
            original_first_round_indices, round.domain.length);
        
        // Validate b_indices are within bounds
        for (size_t idx : b_indices) {
            if (idx >= round.codeword.size()) {
                throw std::out_of_range("FRI Round " + std::to_string(r+1) + 
                    ": b_index " + std::to_string(idx) + 
                    " >= codeword size " + std::to_string(round.codeword.size()));
            }
            if (idx >= round.merkle_tree->num_leaves()) {
                throw std::out_of_range("FRI Round " + std::to_string(r+1) + 
                    ": b_index " + std::to_string(idx) + 
                    " >= num_leaves " + std::to_string(round.merkle_tree->num_leaves()));
            }
        }
        
        std::vector<XFieldElement> revealed_leaves;
        for (size_t idx : b_indices) {
            revealed_leaves.push_back(round.codeword[idx]);
        }
        std::vector<Digest> auth_structure = round.merkle_tree->authentication_structure(b_indices);
        
        // Debug: Print auth structure size
        std::cout << "DEBUG: FRI Round " << (r+1) << " (b-indices) - " << b_indices.size() 
                  << " indices, " << auth_structure.size() << " auth digests" << std::endl;
        std::cout << "DEBUG: FRI Round " << (r+1) << " - domain.length=" << round.domain.length 
                  << ", codeword.size()=" << round.codeword.size() 
                  << ", num_leaves=" << round.merkle_tree->num_leaves() << std::endl;
        
        FriResponse response;
        response.auth_structure = auth_structure;
        response.revealed_leaves = revealed_leaves;
        
        // Debug: Print FriResponse sizes
        bool debug_fri_response_sizes = std::getenv("TVM_DEBUG_FRI_RESPONSE_SIZES") != nullptr;
        if (debug_fri_response_sizes) {
            std::cout << "DEBUG: FriResponse Round " << (r+1) << ":" << std::endl;
            std::cout << "  auth_structure: " << auth_structure.size() << " digests (" << (auth_structure.size() * 5) << " BFieldElements)" << std::endl;
            std::cout << "  revealed_leaves: " << revealed_leaves.size() << " XFieldElements (" << (revealed_leaves.size() * 3) << " BFieldElements)" << std::endl;
            std::cout << "  Total field encoding: " << (1 + auth_structure.size() * 5) << " + " << (1 + revealed_leaves.size() * 3) << " = " 
                      << (1 + auth_structure.size() * 5 + 1 + revealed_leaves.size() * 3) << " BFieldElements" << std::endl;
        }
        
        proof_stream.enqueue(ProofItem::fri_response(response));
    }
    
    // Sample one XFieldElement from Fiat-Shamir and then throw it away.
    // This scalar is the indeterminate for the low degree test using the
    // barycentric evaluation formula. This indeterminate is used only by
    // the verifier, but it is important to modify the sponge state the same
    // way. (Matches Rust line 576)
    proof_stream.sample_scalars(1);
    
    // Return the original first-round indices for trace openings
    return original_first_round_indices;
}

bool Fri::verify(ProofStream& proof_stream) const {
    return verify_and_get_first_round(proof_stream).has_value();
}

std::optional<Fri::VerifyResult> Fri::verify_and_get_first_round(ProofStream& proof_stream) const {
    auto fail = [&](const char* why) -> std::optional<VerifyResult> {
        std::cerr << "FRI verify failed: " << why << std::endl;
        return std::nullopt;
    };
    const size_t rounds = num_rounds();
    if (domain_.length == 0 || (domain_.length & (domain_.length - 1)) != 0) {
        return fail("invalid domain length");
    }
    if (num_collinearity_checks_ == 0) {
        return fail("num_collinearity_checks == 0");
    }

    auto dequeue_root = [&](Digest& out) -> bool {
        ProofItem item = proof_stream.dequeue();
        return item.try_into_merkle_root(out);
    };
    auto dequeue_codeword = [&](std::vector<XFieldElement>& out) -> bool {
        ProofItem item = proof_stream.dequeue();
        return item.try_into_fri_codeword(out);
    };
    auto dequeue_polynomial = [&](std::vector<XFieldElement>& out) -> bool {
        ProofItem item = proof_stream.dequeue();
        return item.try_into_fri_polynomial(out);
    };
    auto dequeue_response = [&](FriResponse& out) -> bool {
        ProofItem item = proof_stream.dequeue();
        return item.try_into_fri_response(out);
    };

    // 1) Read Merkle roots and sample folding challenges between them (must match prover order)
    std::vector<Digest> roots;
    roots.resize(rounds + 1);
    if (!dequeue_root(roots[0])) return fail("missing first FRI Merkle root");

    std::vector<XFieldElement> folding_challenges;
    folding_challenges.reserve(rounds);
    for (size_t r = 0; r < rounds; ++r) {
        folding_challenges.push_back(proof_stream.sample_scalars(1)[0]);
        if (!dequeue_root(roots[r + 1])) return fail("missing FRI Merkle root");
    }

    // 2) Read last codeword + last polynomial, and check they are consistent.
    std::vector<XFieldElement> last_codeword;
    if (!dequeue_codeword(last_codeword)) return fail("missing last FRI codeword");
    std::vector<XFieldElement> last_polynomial;
    if (!dequeue_polynomial(last_polynomial)) return fail("missing last FRI polynomial");

    if (last_codeword.empty() || (last_codeword.size() & (last_codeword.size() - 1)) != 0) {
        return fail("invalid last codeword length");
    }

    // Recompute interpolation (matches prover: component-wise inverse NTT on subgroup offset=1).
    std::vector<BFieldElement> comp0(last_codeword.size());
    std::vector<BFieldElement> comp1(last_codeword.size());
    std::vector<BFieldElement> comp2(last_codeword.size());
    for (size_t i = 0; i < last_codeword.size(); ++i) {
        comp0[i] = last_codeword[i].coeff(0);
        comp1[i] = last_codeword[i].coeff(1);
        comp2[i] = last_codeword[i].coeff(2);
    }
    auto coeffs0 = NTT::interpolate(comp0);
    auto coeffs1 = NTT::interpolate(comp1);
    auto coeffs2 = NTT::interpolate(comp2);
    std::vector<XFieldElement> recomputed(coeffs0.size());
    for (size_t i = 0; i < coeffs0.size(); ++i) {
        recomputed[i] = XFieldElement(coeffs0[i], coeffs1[i], coeffs2[i]);
    }
    // Trim trailing zeros (matches Rust Polynomial encoding)
    while (!recomputed.empty() && recomputed.back().is_zero()) recomputed.pop_back();
    if (recomputed.size() != last_polynomial.size()) return fail("last polynomial length mismatch");
    for (size_t i = 0; i < recomputed.size(); ++i) {
        if (recomputed[i] != last_polynomial[i]) return fail("last polynomial content mismatch");
    }

    // 3) Sample first-round indices and verify openings through rounds.
    std::vector<size_t> original_indices = proof_stream.sample_indices(domain_.length, num_collinearity_checks_);
    if (original_indices.size() != num_collinearity_checks_) return fail("failed to sample query indices");

    // a-indices response (round 0)
    FriResponse a_resp;
    if (!dequeue_response(a_resp)) return fail("missing round-0 a-response");
    if (a_resp.revealed_leaves.size() != original_indices.size()) return fail("round-0 a-response leaf count mismatch");

    // Verify Merkle auth structure for round 0 a-indices
    {
        std::vector<Digest> leaves;
        leaves.reserve(a_resp.revealed_leaves.size());
        for (const auto& x : a_resp.revealed_leaves) {
            leaves.emplace_back(x.coeff(0), x.coeff(1), x.coeff(2), BFieldElement::zero(), BFieldElement::zero());
        }
        if (!MerkleTree::verify_authentication_structure(
                roots[0], domain_.length, original_indices, leaves, a_resp.auth_structure)) {
            return fail("round-0 a-response authentication failed");
        }
    }

    // Current a-values start as the opened round-0 a-values.
    std::vector<XFieldElement> a_values = a_resp.revealed_leaves;

    // For each round r, consume b-indices response, verify Merkle auth, and fold to next a-values.
    size_t dom_len = domain_.length;
    for (size_t r = 0; r < rounds; ++r) {
        if (dom_len < 2) return fail("domain length underflow during folding");
        const size_t half = dom_len / 2;

        // Compute current-round a-indices (original reduced mod dom_len)
        std::vector<size_t> a_idx(dom_len ? original_indices.size() : 0);
        for (size_t i = 0; i < original_indices.size(); ++i) {
            a_idx[i] = original_indices[i] % dom_len;
        }
        // Compute b-indices (pair indices)
        std::vector<size_t> b_idx;
        b_idx.reserve(original_indices.size());
        for (size_t i = 0; i < original_indices.size(); ++i) {
            b_idx.push_back((a_idx[i] + half) % dom_len);
        }

        FriResponse b_resp;
        if (!dequeue_response(b_resp)) return fail("missing b-response");
        if (b_resp.revealed_leaves.size() != b_idx.size()) return fail("b-response leaf count mismatch");

        // Verify Merkle auth structure for b-indices against root of round r.
        {
            std::vector<Digest> leaves;
            leaves.reserve(b_resp.revealed_leaves.size());
            for (const auto& x : b_resp.revealed_leaves) {
                leaves.emplace_back(x.coeff(0), x.coeff(1), x.coeff(2), BFieldElement::zero(), BFieldElement::zero());
            }
            if (!MerkleTree::verify_authentication_structure(
                    roots[r], dom_len, b_idx, leaves, b_resp.auth_structure)) {
                return fail("b-response authentication failed");
            }
        }

        // Fold to obtain next round a-values at indices i = a_idx % half.
        ArithmeticDomain rd = this->round_domain(r);
        XFieldElement alpha = folding_challenges[r];
        XFieldElement one = XFieldElement::one();
        XFieldElement two_inv = XFieldElement(BFieldElement(2)).inverse();

        std::vector<XFieldElement> next_a_values;
        next_a_values.resize(a_values.size());
        for (size_t i = 0; i < a_values.size(); ++i) {
            size_t ai = a_idx[i];
            size_t ii = ai % half;
            // Determine left/right order per split_and_fold definition.
            XFieldElement left, right;
            if (ai < half) {
                left = a_values[i];
                right = b_resp.revealed_leaves[i];
            } else {
                left = b_resp.revealed_leaves[i];
                right = a_values[i];
            }

            BFieldElement dom_point = rd.element(ii);
            XFieldElement scaled_offset_inv = alpha * dom_point.inverse();
            XFieldElement left_summand = (one + scaled_offset_inv) * left;
            XFieldElement right_summand = (one - scaled_offset_inv) * right;
            next_a_values[i] = (left_summand + right_summand) * two_inv;
        }

        a_values = std::move(next_a_values);
        dom_len = half;
    }

    // Final consistency check: the folded values must match the provided last codeword at the folded indices.
    if (dom_len != last_codeword.size()) {
        return fail("folded domain length != last codeword length");
    }
    for (size_t i = 0; i < original_indices.size(); ++i) {
        size_t idx = original_indices[i] % dom_len;
        if (a_values[i] != last_codeword[idx]) {
            return fail("folded values do not match last codeword");
        }
    }

    // Rust samples one extra scalar for barycentric indeterminate; keep sponge aligned.
    proof_stream.sample_scalars(1);

    VerifyResult res;
    res.first_round_indices = std::move(original_indices);
    res.first_round_values = std::move(a_resp.revealed_leaves);
    return res;
}

// Batch inversion using Montgomery's trick
std::vector<BFieldElement> batch_inverse(const std::vector<BFieldElement>& elements) {
    if (elements.empty()) {
        return {};
    }
    
    size_t n = elements.size();
    std::vector<BFieldElement> result(n);
    
    // Compute running products
    std::vector<BFieldElement> products(n);
    products[0] = elements[0];
    for (size_t i = 1; i < n; ++i) {
        products[i] = products[i - 1] * elements[i];
    }
    
    // Invert the product of all elements
    BFieldElement all_inv = products[n - 1].inverse();
    
    // Compute individual inverses
    for (size_t i = n - 1; i > 0; --i) {
        result[i] = all_inv * products[i - 1];
        all_inv = all_inv * elements[i];
    }
    result[0] = all_inv;
    
    return result;
}

#ifdef TRITON_CUDA_ENABLED

std::vector<size_t> Fri::prove_gpu(
    const std::vector<XFieldElement>& codeword,
    ProofStream& proof_stream
) const {
    if (codeword.size() != domain_.length) {
        throw std::invalid_argument("Codeword length must match domain length");
    }
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    size_t rounds = num_rounds();
    std::cout << "DEBUG: GPU FRI.prove() - num_rounds() = " << rounds << std::endl;
    
    // Store round data for query phase
    std::vector<std::vector<XFieldElement>> round_codewords;
    std::vector<ArithmeticDomain> round_domains;
    std::vector<std::unique_ptr<MerkleTree>> round_trees;
    
    // Convert XFE codeword to flat GPU format
    size_t current_len = codeword.size();
    std::vector<uint64_t> h_codeword(current_len * 3);
    for (size_t i = 0; i < current_len; ++i) {
        h_codeword[i * 3 + 0] = codeword[i].coeff(0).value();
        h_codeword[i * 3 + 1] = codeword[i].coeff(1).value();
        h_codeword[i * 3 + 2] = codeword[i].coeff(2).value();
    }
    
    // Allocate GPU buffers for codeword
    uint64_t* d_codeword;
    uint64_t* d_folded;
    cudaMalloc(&d_codeword, current_len * 3 * sizeof(uint64_t));
    cudaMalloc(&d_folded, (current_len / 2) * 3 * sizeof(uint64_t));
    cudaMemcpy(d_codeword, h_codeword.data(), current_len * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Allocate domain inverses buffer
    uint64_t* d_domain_inv;
    cudaMalloc(&d_domain_inv, (current_len / 2) * sizeof(uint64_t));
    
    // Challenge buffer
    uint64_t* d_challenge;
    cudaMalloc(&d_challenge, 3 * sizeof(uint64_t));
    
    // Two inverse (constant)
    uint64_t two_inv = BFieldElement(2).inverse().value();
    
    // Current domain
    ArithmeticDomain current_domain = domain_;
    
    // Store first round codeword and commit
    round_codewords.push_back(codeword);
    round_domains.push_back(current_domain);
    
    // Build Merkle tree for first round (XFE -> Digest)
    std::vector<Digest> first_leaves;
    first_leaves.reserve(current_len);
    for (const auto& elem : codeword) {
        first_leaves.emplace_back(
            elem.coeff(0), elem.coeff(1), elem.coeff(2),
            BFieldElement::zero(), BFieldElement::zero()
        );
    }
    round_trees.push_back(std::make_unique<MerkleTree>(first_leaves));
    Digest first_root = round_trees.back()->root();
    proof_stream.enqueue(ProofItem::merkle_root(first_root));
    
    // Folding rounds
    for (size_t r = 0; r < rounds; ++r) {
        // Sample folding challenge
        XFieldElement folding_challenge = proof_stream.sample_scalars(1)[0];
        
        // Upload challenge to GPU
        uint64_t h_challenge[3] = {
            folding_challenge.coeff(0).value(),
            folding_challenge.coeff(1).value(),
            folding_challenge.coeff(2).value()
        };
        cudaMemcpy(d_challenge, h_challenge, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        // Compute domain inverses on GPU
        size_t half_len = current_len / 2;
        gpu::kernels::compute_domain_inverses_gpu(
            current_domain.offset.value(),
            current_domain.generator.value(),
            d_domain_inv,
            half_len,
            0
        );
        
        // Fold on GPU
        gpu::kernels::fri_fold_gpu(
            d_codeword,
            current_len,
            d_challenge,
            d_domain_inv,
            two_inv,
            d_folded,
            0
        );
        cudaDeviceSynchronize();
        
        // Download folded codeword
        std::vector<uint64_t> h_folded(half_len * 3);
        cudaMemcpy(h_folded.data(), d_folded, half_len * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        // Convert to XFE
        std::vector<XFieldElement> folded_codeword(half_len);
        for (size_t i = 0; i < half_len; ++i) {
            folded_codeword[i] = XFieldElement(
                BFieldElement(h_folded[i * 3 + 0]),
                BFieldElement(h_folded[i * 3 + 1]),
                BFieldElement(h_folded[i * 3 + 2])
            );
        }
        
        // Halve domain
        current_domain = current_domain.halve();
        current_len = half_len;
        
        // Store round data
        round_codewords.push_back(folded_codeword);
        round_domains.push_back(current_domain);
        
        // Build Merkle tree and commit
        std::vector<Digest> leaves;
        leaves.reserve(half_len);
        for (const auto& elem : folded_codeword) {
            leaves.emplace_back(
                elem.coeff(0), elem.coeff(1), elem.coeff(2),
                BFieldElement::zero(), BFieldElement::zero()
            );
        }
        round_trees.push_back(std::make_unique<MerkleTree>(leaves));
        Digest root = round_trees.back()->root();
        proof_stream.enqueue(ProofItem::merkle_root(root));
        
        // Swap buffers for next round
        std::swap(d_codeword, d_folded);
        
        // Upload new codeword to GPU
        cudaMemcpy(d_codeword, h_folded.data(), half_len * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    
    // Free GPU buffers
    cudaFree(d_codeword);
    cudaFree(d_folded);
    cudaFree(d_domain_inv);
    cudaFree(d_challenge);
    
    // Send last codeword
    const std::vector<XFieldElement>& last_codeword = round_codewords.back();
    proof_stream.enqueue(ProofItem::fri_codeword(last_codeword));
    
    // Interpolate and send last polynomial
    std::vector<XFieldElement> last_polynomial;
    {
        std::vector<BFieldElement> comp0(last_codeword.size());
        std::vector<BFieldElement> comp1(last_codeword.size());
        std::vector<BFieldElement> comp2(last_codeword.size());
        for (size_t i = 0; i < last_codeword.size(); ++i) {
            comp0[i] = last_codeword[i].coeff(0);
            comp1[i] = last_codeword[i].coeff(1);
            comp2[i] = last_codeword[i].coeff(2);
        }
        auto coeffs0 = NTT::interpolate(comp0);
        auto coeffs1 = NTT::interpolate(comp1);
        auto coeffs2 = NTT::interpolate(comp2);
        last_polynomial.resize(coeffs0.size());
        for (size_t i = 0; i < coeffs0.size(); ++i) {
            last_polynomial[i] = XFieldElement(coeffs0[i], coeffs1[i], coeffs2[i]);
        }
        // Trim trailing zeros to match Rust `Polynomial` encoding and verifier expectations.
        while (!last_polynomial.empty() && last_polynomial.back().is_zero()) {
            last_polynomial.pop_back();
        }
    }
    proof_stream.enqueue(ProofItem::fri_polynomial(last_polynomial));
    
    // Query phase: sample indices
    std::vector<size_t> first_round_indices = proof_stream.sample_indices(
        domain_.length, num_collinearity_checks_);
    std::vector<size_t> original_first_round_indices = first_round_indices;
    
    // Reveal at a-indices for first round
    {
        const auto& tree = round_trees[0];
        const auto& cw = round_codewords[0];
        std::vector<XFieldElement> revealed_leaves;
        for (size_t idx : first_round_indices) {
            revealed_leaves.push_back(cw[idx]);
        }
        std::vector<Digest> auth_structure = tree->authentication_structure(first_round_indices);
        
        FriResponse response;
        response.auth_structure = auth_structure;
        response.revealed_leaves = revealed_leaves;
        proof_stream.enqueue(ProofItem::fri_response(response));
    }
    
    // For each round, reveal b-indices
    for (size_t r = 0; r < rounds; ++r) {
        const auto& tree = round_trees[r];
        const auto& cw = round_codewords[r];
        size_t dom_len = round_domains[r].length;
        
        std::vector<size_t> b_indices = collinearity_check_b_indices(
            original_first_round_indices, dom_len);
        
        std::vector<XFieldElement> revealed_leaves;
        for (size_t idx : b_indices) {
            if (idx < cw.size()) {
                revealed_leaves.push_back(cw[idx]);
            }
        }
        std::vector<Digest> auth_structure = tree->authentication_structure(b_indices);
        
        FriResponse response;
        response.auth_structure = auth_structure;
        response.revealed_leaves = revealed_leaves;
        proof_stream.enqueue(ProofItem::fri_response(response));
    }
    
    // Sample one XFieldElement and throw it away (matches CPU behavior)
    // This is the indeterminate for the low degree test using barycentric evaluation
    proof_stream.sample_scalars(1);
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;
    std::cout << "GPU: FRI prove: " << gpu_ms << " ms" << std::endl;
    
    return original_first_round_indices;
}

#endif // TRITON_CUDA_ENABLED

} // namespace triton_vm
