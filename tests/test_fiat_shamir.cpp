#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <regex>
#include <sstream>
#include <nlohmann/json.hpp>
#include "proof_stream/proof_stream.hpp"
#include "test_data_loader.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"

using namespace triton_vm;
using json = nlohmann::json;

/**
 * Parse XFieldElement from string like "(06684776751427307721·x² + 02215282505576409730·x + 11814865297276416494)"
 */
static XFieldElement parse_xfield_from_string(const std::string& str) {
    // Pattern: (coeff2·x² + coeff1·x + coeff0)
    std::regex pattern(R"(\((\d+)·x² \+ (\d+)·x \+ (\d+)\))");
    std::smatch match;
    
    if (!std::regex_search(str, match, pattern)) {
        throw std::runtime_error("Failed to parse XFieldElement: " + str);
    }
    
    uint64_t coeff2 = std::stoull(match[1].str());
    uint64_t coeff1 = std::stoull(match[2].str());
    uint64_t coeff0 = std::stoull(match[3].str());
    
    return XFieldElement(
        BFieldElement(coeff0),
        BFieldElement(coeff1),
        BFieldElement(coeff2)
    );
}

/**
 * FiatShamirTest - Test Fiat-Shamir challenge generation
 */
class FiatShamirTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data_dir_ = std::string(TEST_DATA_DIR) + "/../test_data_lde";
        
        if (!std::filesystem::exists(test_data_dir_)) {
            GTEST_SKIP() << "Test data directory not found. Run 'gen_test_data spin.tasm 8 test_data_lde' first.";
        }
    }
    
    std::string test_data_dir_;
    
    json load_json(const std::string& filename) {
        std::string path = test_data_dir_ + "/" + filename;
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + path);
        }
        return json::parse(f);
    }
    
    Digest load_merkle_root(const std::string& filename, const std::string& key) {
        auto json = load_json(filename);
        std::string hex_str = json[key].get<std::string>();
        
        // Convert hex string to Digest
        Digest root;
        for (size_t i = 0; i < Digest::LEN; i++) {
            // Each BFieldElement is 16 hex chars (8 bytes = 64 bits)
            std::string hex_chunk = hex_str.substr(i * 16, 16);
            uint64_t value = std::stoull(hex_chunk, nullptr, 16);
            root[i] = BFieldElement(value);
        }
        
        return root;
    }
    
    std::vector<BFieldElement> load_claim_encoded() {
        auto json = load_json("06_claim.json");
        std::vector<BFieldElement> encoded;
        
        auto& encoded_json = json["encoded_for_fiat_shamir"];
        for (const auto& val : encoded_json) {
            encoded.push_back(BFieldElement(val.get<uint64_t>()));
        }
        
        return encoded;
    }
    
    std::vector<uint64_t> load_sponge_state(const std::string& filename) {
        auto json = load_json(filename);
        std::vector<uint64_t> state;
        
        auto& state_json = json["state"];
        for (const auto& val : state_json) {
            state.push_back(val.get<uint64_t>());
        }
        
        return state;
    }
    
    void compare_sponge_state(const Tip5& sponge, const std::vector<uint64_t>& rust_state, const std::string& step_name) {
        std::cout << "\n=== Sponge State Comparison: " << step_name << " ===" << std::endl;
        
        EXPECT_EQ(sponge.state.size(), rust_state.size()) << "State size should match";
        
        size_t matches = 0;
        for (size_t i = 0; i < std::min(sponge.state.size(), rust_state.size()); i++) {
            uint64_t cpp_val = sponge.state[i].value();
            uint64_t rust_val = rust_state[i];
            
            if (cpp_val == rust_val) {
                matches++;
            } else {
                std::cout << "  Mismatch at index " << i << ": C++=" << cpp_val << ", Rust=" << rust_val << std::endl;
            }
        }
        
        std::cout << "  Matches: " << matches << "/" << rust_state.size() << std::endl;
        
        if (matches == rust_state.size()) {
            std::cout << "  ✓ Sponge state matches Rust exactly!" << std::endl;
        } else {
            std::cout << "  ✗ Sponge state mismatch - need to debug absorption logic" << std::endl;
        }
    }
};

// Test: Load Fiat-Shamir challenges from Rust output
TEST_F(FiatShamirTest, LoadChallengesFromRust) {
    auto json = load_json("07_fiat_shamir_challenges.json");
    
    auto& challenge_strings = json["challenge_values"];
    size_t count = json["challenges_sample_count"].get<size_t>();
    
    EXPECT_EQ(challenge_strings.size(), count);
    
    std::vector<XFieldElement> challenges;
    for (const auto& str : challenge_strings) {
        XFieldElement xfe = parse_xfield_from_string(str.get<std::string>());
        challenges.push_back(xfe);
    }
    
    EXPECT_EQ(challenges.size(), count);
    std::cout << "\n=== Loaded " << challenges.size() << " challenges from Rust ===" << std::endl;
}

// Test: Compute Fiat-Shamir challenges after enqueueing Merkle root
TEST_F(FiatShamirTest, ComputeChallengesAfterMerkleRoot) {
    // Load Merkle root from step 6
    // NOTE: For now, we'll use the Merkle root from Rust's encoding to debug the absorption
    // The actual root from JSON might have encoding differences
    auto rust_merkle_encoded = load_json("merkle_root_encoding.json");
    auto& rust_encoded_json = rust_merkle_encoded["encoded"];
    
    Digest main_merkle_root;
    // Skip index 0 (discriminant), use indices 1-5 for the digest
    for (size_t i = 0; i < Digest::LEN; i++) {
        main_merkle_root[i] = BFieldElement(rust_encoded_json[i + 1].get<uint64_t>());
    }
    
    // Also load from JSON for comparison
    Digest main_merkle_root_from_json = load_merkle_root("06_main_tables_merkle.json", "merkle_root");
    
    // Load expected challenges
    auto json = load_json("07_fiat_shamir_challenges.json");
    size_t expected_count = json["challenges_sample_count"].get<size_t>();
    
    std::vector<XFieldElement> expected_challenges;
    for (const auto& str : json["challenge_values"]) {
        expected_challenges.push_back(parse_xfield_from_string(str.get<std::string>()));
    }
    
    std::cout << "\n=== Computing Fiat-Shamir Challenges ===" << std::endl;
    std::cout << "  Expected count: " << expected_count << std::endl;
    
    // Load claim encoded data
    std::vector<BFieldElement> claim_encoded = load_claim_encoded();
    std::cout << "  Loaded claim encoded data: " << claim_encoded.size() << " BFieldElements" << std::endl;
    
    // Verify claim encoding matches expected length
    auto claim_json = load_json("06_claim.json");
    size_t expected_encoded_len = claim_json["encoded_length"].get<size_t>();
    EXPECT_EQ(claim_encoded.size(), expected_encoded_len) 
        << "Claim encoded length should match";
    
    // Create proof stream and absorb claim (as Rust does)
    ProofStream proof_stream;
    proof_stream.alter_fiat_shamir_state_with(claim_encoded);
    
    // Compare sponge state after claim absorption
    auto rust_state_after_claim = load_sponge_state("sponge_state_after_claim.json");
    compare_sponge_state(proof_stream.sponge(), rust_state_after_claim, "After Claim");
    
    // Debug: Print loaded Merkle root
    std::cout << "\n=== Merkle Root Debug ===" << std::endl;
    std::cout << "  Loaded Merkle root values:" << std::endl;
    for (size_t i = 0; i < Digest::LEN; i++) {
        std::cout << "    [" << i << "] = " << main_merkle_root[i].value() << std::endl;
    }
    
    // Verify Merkle root encoding matches Rust (reuse already loaded data)
    auto merkle_encoded = ProofItem::merkle_root(main_merkle_root).encode();
    
    EXPECT_EQ(merkle_encoded.size(), rust_encoded_json.size()) 
        << "Merkle root encoding length should match";
    
    std::cout << "  Rust encoded values:" << std::endl;
    for (size_t i = 0; i < rust_encoded_json.size(); i++) {
        std::cout << "    [" << i << "] = " << rust_encoded_json[i].get<uint64_t>() << std::endl;
    }
    
    size_t encoding_matches = 0;
    for (size_t i = 0; i < std::min(merkle_encoded.size(), rust_encoded_json.size()); i++) {
        uint64_t cpp_val = merkle_encoded[i].value();
        uint64_t rust_val = rust_encoded_json[i].get<uint64_t>();
        if (cpp_val == rust_val) {
            encoding_matches++;
        } else {
            std::cout << "  Encoding mismatch at index " << i << ": C++=" << cpp_val << ", Rust=" << rust_val << std::endl;
        }
    }
    
    std::cout << "  Merkle root encoding matches: " << encoding_matches << "/" << rust_encoded_json.size() << std::endl;
    EXPECT_EQ(encoding_matches, rust_encoded_json.size()) << "Merkle root encoding should match Rust";
    
    // Then enqueue Merkle root
    proof_stream.enqueue(ProofItem::merkle_root(main_merkle_root));
    
    // Compare sponge state after Merkle root enqueue
    auto rust_state_after_merkle = load_sponge_state("sponge_state_after_merkle_root.json");
    compare_sponge_state(proof_stream.sponge(), rust_state_after_merkle, "After Merkle Root");
    
    // Sample challenges
    std::vector<XFieldElement> computed_challenges = proof_stream.sample_scalars(expected_count);
    
    EXPECT_EQ(computed_challenges.size(), expected_count);
    
    // Compare each challenge
    size_t matches = 0;
    for (size_t i = 0; i < std::min(computed_challenges.size(), expected_challenges.size()); i++) {
        bool match = true;
        for (int j = 0; j < 3; j++) {
            if (computed_challenges[i].coeff(j).value() != expected_challenges[i].coeff(j).value()) {
                match = false;
                break;
            }
        }
        
        if (match) {
            matches++;
        } else {
            std::cout << "  Mismatch at index " << i << ":" << std::endl;
            std::cout << "    Expected: (" 
                      << expected_challenges[i].coeff(2).value() << "·x² + "
                      << expected_challenges[i].coeff(1).value() << "·x + "
                      << expected_challenges[i].coeff(0).value() << ")" << std::endl;
            std::cout << "    Got:      (" 
                      << computed_challenges[i].coeff(2).value() << "·x² + "
                      << computed_challenges[i].coeff(1).value() << "·x + "
                      << computed_challenges[i].coeff(0).value() << ")" << std::endl;
        }
    }
    
    std::cout << "  Matches: " << matches << "/" << expected_count << std::endl;
    
    if (matches == expected_count) {
        std::cout << "  ✓ All challenges match Rust exactly!" << std::endl;
    } else {
        std::cout << "  ✗ Challenges don't match. Need to debug pad_and_absorb_all or sample_scalars." << std::endl;
    }
    
    EXPECT_EQ(matches, expected_count) << "All challenges should match Rust";
}

// Test: Verify Merkle root encoding matches
TEST_F(FiatShamirTest, MerkleRootEncoding) {
    Digest root = load_merkle_root("06_main_tables_merkle.json", "merkle_root");

    ProofItem item = ProofItem::merkle_root(root);
    auto encoded = item.encode();

    // Rust's BFieldCodec includes discriminant: 1 discriminant + 5 digest elements
    EXPECT_EQ(encoded.size(), Digest::LEN + 1) << "Merkle root should encode to 6 BFieldElements (1 discriminant + 5 digest)";

    // Verify discriminant
    EXPECT_EQ(encoded[0].value(), 0) << "Discriminant should be 0 for MerkleRoot";

    // Verify encoding matches original (after discriminant)
    for (size_t i = 0; i < Digest::LEN; i++) {
        EXPECT_EQ(encoded[i + 1].value(), root[i].value())
            << "Encoded element " << (i + 1) << " should match original digest element " << i;
    }

    std::cout << "\n=== Merkle Root Encoding ===" << std::endl;
    std::cout << "  ✓ Merkle root encodes correctly with discriminant" << std::endl;
}

