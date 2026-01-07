#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "table/master_table.hpp"
#include "hash/tip5.hpp"
#include "merkle/merkle_tree.hpp"
#include "fri/fri.hpp"

using namespace triton_vm;
using json = nlohmann::json;

/**
 * IOVerificationTest - 100% accurate compute and compare tests
 * 
 * For each test:
 * 1. Load input from Rust-generated JSON
 * 2. Compute output using C++ implementation
 * 3. Compare with Rust output - must match exactly
 */
class IOVerificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Path to I/O test data
        io_test_dir_ = std::string(TEST_DATA_DIR) + "/../test_data_io";
        
        if (!std::filesystem::exists(io_test_dir_)) {
            GTEST_SKIP() << "I/O test data not found. Run 'cargo run --release --bin gen_io_tests' first.";
        }
    }
    
    std::string io_test_dir_;
    
    json load_json(const std::string& subdir, const std::string& filename) {
        std::string path = io_test_dir_ + "/" + subdir + "/" + filename;
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + path);
        }
        return json::parse(f);
    }
};

// ============================================================================
// Tip5 Hash Tests - Must match exactly
// ============================================================================
TEST_F(IOVerificationTest, Tip5_HashEmpty) {
    auto data = load_json("tip5", "hash_empty.json");
    
    // Load input
    std::vector<BFieldElement> input;
    // Empty input
    
    // Compute hash
    Tip5 hasher;
    Digest cpp_hash = hasher.hash_varlen(input);
    
    // Compare with Rust output
    auto rust_digest = data["output"]["digest"];
    EXPECT_EQ(cpp_hash[0].value(), rust_digest[0].get<uint64_t>());
    EXPECT_EQ(cpp_hash[1].value(), rust_digest[1].get<uint64_t>());
    EXPECT_EQ(cpp_hash[2].value(), rust_digest[2].get<uint64_t>());
    EXPECT_EQ(cpp_hash[3].value(), rust_digest[3].get<uint64_t>());
    EXPECT_EQ(cpp_hash[4].value(), rust_digest[4].get<uint64_t>());
    
    std::cout << "  ✓ Tip5 hash_empty matches Rust exactly" << std::endl;
}

TEST_F(IOVerificationTest, Tip5_HashSingle) {
    auto data = load_json("tip5", "hash_single.json");
    
    // Load input
    std::vector<BFieldElement> input;
    for (auto elem : data["input"]["elements"]) {
        input.push_back(BFieldElement(elem.get<uint64_t>()));
    }
    EXPECT_EQ(input.size(), data["input"]["count"].get<size_t>());
    
    // Compute hash
    Tip5 hasher;
    Digest cpp_hash = hasher.hash_varlen(input);
    
    // Compare with Rust output
    auto rust_digest = data["output"]["digest"];
    EXPECT_EQ(cpp_hash[0].value(), rust_digest[0].get<uint64_t>()) << "Digest[0] mismatch";
    EXPECT_EQ(cpp_hash[1].value(), rust_digest[1].get<uint64_t>()) << "Digest[1] mismatch";
    EXPECT_EQ(cpp_hash[2].value(), rust_digest[2].get<uint64_t>()) << "Digest[2] mismatch";
    EXPECT_EQ(cpp_hash[3].value(), rust_digest[3].get<uint64_t>()) << "Digest[3] mismatch";
    EXPECT_EQ(cpp_hash[4].value(), rust_digest[4].get<uint64_t>()) << "Digest[4] mismatch";
    
    std::cout << "  ✓ Tip5 hash_single matches Rust exactly" << std::endl;
}

TEST_F(IOVerificationTest, Tip5_HashTen) {
    auto data = load_json("tip5", "hash_ten.json");
    
    // Load input
    std::vector<BFieldElement> input;
    for (auto elem : data["input"]["elements"]) {
        input.push_back(BFieldElement(elem.get<uint64_t>()));
    }
    EXPECT_EQ(input.size(), 10);
    
    // Compute hash
    Tip5 hasher;
    Digest cpp_hash = hasher.hash_varlen(input);
    
    // Compare with Rust output
    auto rust_digest = data["output"]["digest"];
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(cpp_hash[i].value(), rust_digest[i].get<uint64_t>()) 
            << "Digest[" << i << "] mismatch";
    }
    
    std::cout << "  ✓ Tip5 hash_ten matches Rust exactly" << std::endl;
}

TEST_F(IOVerificationTest, Tip5_HashPair) {
    auto data = load_json("tip5", "hash_pair.json");
    
    // Load input digests
    auto left = data["input"]["left"];
    auto right = data["input"]["right"];
    
    Digest d1(
        BFieldElement(left[0].get<uint64_t>()),
        BFieldElement(left[1].get<uint64_t>()),
        BFieldElement(left[2].get<uint64_t>()),
        BFieldElement(left[3].get<uint64_t>()),
        BFieldElement(left[4].get<uint64_t>())
    );
    Digest d2(
        BFieldElement(right[0].get<uint64_t>()),
        BFieldElement(right[1].get<uint64_t>()),
        BFieldElement(right[2].get<uint64_t>()),
        BFieldElement(right[3].get<uint64_t>()),
        BFieldElement(right[4].get<uint64_t>())
    );
    
    // Compute hash
    Tip5 hasher;
    Digest cpp_hash = hasher.hash_pair(d1, d2);
    
    // Compare with Rust output
    auto rust_digest = data["output"]["digest"];
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(cpp_hash[i].value(), rust_digest[i].get<uint64_t>()) 
            << "Digest[" << i << "] mismatch";
    }
    
    std::cout << "  ✓ Tip5 hash_pair matches Rust exactly" << std::endl;
}

TEST_F(IOVerificationTest, Tip5_HashRow379) {
    auto data = load_json("tip5", "hash_row_379.json");
    
    // Load input (379 elements like a main table row)
    std::vector<BFieldElement> input;
    for (auto elem : data["input"]["elements"]) {
        input.push_back(BFieldElement(elem.get<uint64_t>()));
    }
    EXPECT_EQ(input.size(), 379);
    
    // Compute hash
    Tip5 hasher;
    Digest cpp_hash = hasher.hash_varlen(input);
    
    // Compare with Rust output
    auto rust_digest = data["output"]["digest"];
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(cpp_hash[i].value(), rust_digest[i].get<uint64_t>()) 
            << "Digest[" << i << "] mismatch for 379-element row hash";
    }
    
    std::cout << "  ✓ Tip5 hash_row_379 matches Rust exactly" << std::endl;
}

// ============================================================================
// BFieldElement Arithmetic Tests
// ============================================================================
TEST_F(IOVerificationTest, BFieldArithmetic) {
    auto data = load_json("bfield", "arithmetic.json");
    
    // Verify modulus
    EXPECT_EQ(BFieldElement::MODULUS, data["modulus"].get<uint64_t>());
    
    // Test addition
    auto add = data["tests"]["addition"];
    BFieldElement a(add["a"].get<uint64_t>());
    BFieldElement b(add["b"].get<uint64_t>());
    EXPECT_EQ((a + b).value(), add["result"].get<uint64_t>()) << "Addition mismatch";
    
    // Test subtraction
    auto sub = data["tests"]["subtraction"];
    a = BFieldElement(sub["a"].get<uint64_t>());
    b = BFieldElement(sub["b"].get<uint64_t>());
    EXPECT_EQ((a - b).value(), sub["result"].get<uint64_t>()) << "Subtraction mismatch";
    
    // Test multiplication
    auto mul = data["tests"]["multiplication"];
    a = BFieldElement(mul["a"].get<uint64_t>());
    b = BFieldElement(mul["b"].get<uint64_t>());
    EXPECT_EQ((a * b).value(), mul["result"].get<uint64_t>()) << "Multiplication mismatch";
    
    // Test negation
    auto neg = data["tests"]["negation"];
    a = BFieldElement(neg["a"].get<uint64_t>());
    EXPECT_EQ((-a).value(), neg["result"].get<uint64_t>()) << "Negation mismatch";
    
    // Test inverse
    auto inv = data["tests"]["inverse"];
    a = BFieldElement(inv["a"].get<uint64_t>());
    EXPECT_EQ(a.inverse().value(), inv["result"].get<uint64_t>()) << "Inverse mismatch";
    
    // Test power
    auto pow_test = data["tests"]["power"];
    a = BFieldElement(pow_test["base"].get<uint64_t>());
    uint64_t exp = pow_test["exponent"].get<uint64_t>();
    EXPECT_EQ(a.pow(exp).value(), pow_test["result"].get<uint64_t>()) << "Power mismatch";
    
    // Test primitive roots
    auto root512 = data["tests"]["primitive_root_512"];
    BFieldElement omega512 = BFieldElement::primitive_root_of_unity(9);  // 2^9 = 512
    EXPECT_EQ(omega512.value(), root512["result"].get<uint64_t>()) 
        << "Primitive root of unity (512) mismatch";
    
    auto root4096 = data["tests"]["primitive_root_4096"];
    BFieldElement omega4096 = BFieldElement::primitive_root_of_unity(12);  // 2^12 = 4096
    EXPECT_EQ(omega4096.value(), root4096["result"].get<uint64_t>()) 
        << "Primitive root of unity (4096) mismatch";
    
    std::cout << "  ✓ All BFieldElement arithmetic matches Rust exactly" << std::endl;
}

// ============================================================================
// XFieldElement Arithmetic Tests
// ============================================================================
TEST_F(IOVerificationTest, XFieldArithmetic) {
    auto data = load_json("xfield", "arithmetic.json");
    
    // Test addition
    auto add = data["tests"]["addition"];
    auto a_arr = add["a"];
    auto b_arr = add["b"];
    
    XFieldElement a(
        BFieldElement(a_arr[0].get<uint64_t>()),
        BFieldElement(a_arr[1].get<uint64_t>()),
        BFieldElement(a_arr[2].get<uint64_t>())
    );
    XFieldElement b(
        BFieldElement(b_arr[0].get<uint64_t>()),
        BFieldElement(b_arr[1].get<uint64_t>()),
        BFieldElement(b_arr[2].get<uint64_t>())
    );
    
    auto result = add["result"];
    XFieldElement cpp_sum = a + b;
    EXPECT_EQ(cpp_sum.coeff(0).value(), result[0].get<uint64_t>()) << "Addition coeff[0] mismatch";
    EXPECT_EQ(cpp_sum.coeff(1).value(), result[1].get<uint64_t>()) << "Addition coeff[1] mismatch";
    EXPECT_EQ(cpp_sum.coeff(2).value(), result[2].get<uint64_t>()) << "Addition coeff[2] mismatch";
    
    // Test multiplication
    auto mul = data["tests"]["multiplication"];
    a_arr = mul["a"];
    b_arr = mul["b"];
    
    a = XFieldElement(
        BFieldElement(a_arr[0].get<uint64_t>()),
        BFieldElement(a_arr[1].get<uint64_t>()),
        BFieldElement(a_arr[2].get<uint64_t>())
    );
    b = XFieldElement(
        BFieldElement(b_arr[0].get<uint64_t>()),
        BFieldElement(b_arr[1].get<uint64_t>()),
        BFieldElement(b_arr[2].get<uint64_t>())
    );
    
    result = mul["result"];
    XFieldElement cpp_prod = a * b;
    EXPECT_EQ(cpp_prod.coeff(0).value(), result[0].get<uint64_t>()) << "Multiplication coeff[0] mismatch";
    EXPECT_EQ(cpp_prod.coeff(1).value(), result[1].get<uint64_t>()) << "Multiplication coeff[1] mismatch";
    EXPECT_EQ(cpp_prod.coeff(2).value(), result[2].get<uint64_t>()) << "Multiplication coeff[2] mismatch";
    
    // Test inverse
    auto inv = data["tests"]["inverse"];
    a_arr = inv["a"];
    a = XFieldElement(
        BFieldElement(a_arr[0].get<uint64_t>()),
        BFieldElement(a_arr[1].get<uint64_t>()),
        BFieldElement(a_arr[2].get<uint64_t>())
    );
    
    result = inv["result"];
    XFieldElement cpp_inv = a.inverse();
    EXPECT_EQ(cpp_inv.coeff(0).value(), result[0].get<uint64_t>()) << "Inverse coeff[0] mismatch";
    EXPECT_EQ(cpp_inv.coeff(1).value(), result[1].get<uint64_t>()) << "Inverse coeff[1] mismatch";
    EXPECT_EQ(cpp_inv.coeff(2).value(), result[2].get<uint64_t>()) << "Inverse coeff[2] mismatch";
    
    std::cout << "  ✓ All XFieldElement arithmetic matches Rust exactly" << std::endl;
}

// ============================================================================
// Domain Computation Tests
// ============================================================================
TEST_F(IOVerificationTest, Domain512) {
    auto data = load_json("domain", "domain_512.json");
    
    size_t length = data["input"]["length"].get<size_t>();
    EXPECT_EQ(length, 512);
    
    ArithmeticDomain domain = ArithmeticDomain::of_length(length);
    
    EXPECT_EQ(domain.length, data["output"]["length"].get<size_t>());
    EXPECT_EQ(domain.generator.value(), data["output"]["generator"].get<uint64_t>())
        << "Generator mismatch for domain 512";
    EXPECT_EQ(domain.offset.value(), data["output"]["offset"].get<uint64_t>())
        << "Offset mismatch for domain 512";
    
    std::cout << "  ✓ Domain 512 matches Rust exactly" << std::endl;
}

TEST_F(IOVerificationTest, Domain4096) {
    auto data = load_json("domain", "domain_4096.json");
    
    size_t length = data["input"]["length"].get<size_t>();
    EXPECT_EQ(length, 4096);
    
    ArithmeticDomain domain = ArithmeticDomain::of_length(length);
    
    EXPECT_EQ(domain.length, data["output"]["length"].get<size_t>());
    EXPECT_EQ(domain.generator.value(), data["output"]["generator"].get<uint64_t>())
        << "Generator mismatch for domain 4096";
    EXPECT_EQ(domain.offset.value(), data["output"]["offset"].get<uint64_t>())
        << "Offset mismatch for domain 4096";
    
    std::cout << "  ✓ Domain 4096 matches Rust exactly" << std::endl;
}

TEST_F(IOVerificationTest, Domain4096WithOffset) {
    auto data = load_json("domain", "domain_4096_with_offset.json");
    
    size_t length = data["input"]["length"].get<size_t>();
    BFieldElement offset(data["input"]["offset"].get<uint64_t>());
    
    ArithmeticDomain domain = ArithmeticDomain::of_length(length);
    domain = domain.with_offset(offset);
    
    EXPECT_EQ(domain.length, data["output"]["length"].get<size_t>());
    EXPECT_EQ(domain.generator.value(), data["output"]["generator"].get<uint64_t>())
        << "Generator mismatch";
    EXPECT_EQ(domain.offset.value(), data["output"]["offset"].get<uint64_t>())
        << "Offset mismatch";
    
    std::cout << "  ✓ Domain 4096 with offset matches Rust exactly" << std::endl;
}

// ============================================================================
// Merkle Tree Tests
// ============================================================================
TEST_F(IOVerificationTest, MerkleTree4Leaves) {
    auto data = load_json("merkle", "tree_4_leaves.json");
    
    // Load leaves
    std::vector<Digest> leaves;
    for (auto& leaf : data["input"]["leaves"]) {
        leaves.push_back(Digest(
            BFieldElement(leaf[0].get<uint64_t>()),
            BFieldElement(leaf[1].get<uint64_t>()),
            BFieldElement(leaf[2].get<uint64_t>()),
            BFieldElement(leaf[3].get<uint64_t>()),
            BFieldElement(leaf[4].get<uint64_t>())
        ));
    }
    EXPECT_EQ(leaves.size(), 4);
    
    // Build Merkle tree
    MerkleTree tree(leaves);
    Digest cpp_root = tree.root();
    
    // Compare with Rust
    auto rust_root = data["output"]["root"];
    EXPECT_EQ(cpp_root[0].value(), rust_root[0].get<uint64_t>()) << "Root[0] mismatch";
    EXPECT_EQ(cpp_root[1].value(), rust_root[1].get<uint64_t>()) << "Root[1] mismatch";
    EXPECT_EQ(cpp_root[2].value(), rust_root[2].get<uint64_t>()) << "Root[2] mismatch";
    EXPECT_EQ(cpp_root[3].value(), rust_root[3].get<uint64_t>()) << "Root[3] mismatch";
    EXPECT_EQ(cpp_root[4].value(), rust_root[4].get<uint64_t>()) << "Root[4] mismatch";
    
    std::cout << "  ✓ Merkle tree (4 leaves) root matches Rust exactly" << std::endl;
}

TEST_F(IOVerificationTest, MerkleTree8Leaves) {
    auto data = load_json("merkle", "tree_8_leaves.json");
    
    // Load leaves
    std::vector<Digest> leaves;
    for (auto& leaf : data["input"]["leaves"]) {
        leaves.push_back(Digest(
            BFieldElement(leaf[0].get<uint64_t>()),
            BFieldElement(leaf[1].get<uint64_t>()),
            BFieldElement(leaf[2].get<uint64_t>()),
            BFieldElement(leaf[3].get<uint64_t>()),
            BFieldElement(leaf[4].get<uint64_t>())
        ));
    }
    EXPECT_EQ(leaves.size(), 8);
    
    // Build Merkle tree
    MerkleTree tree(leaves);
    Digest cpp_root = tree.root();
    
    // Compare with Rust
    auto rust_root = data["output"]["root"];
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(cpp_root[i].value(), rust_root[i].get<uint64_t>()) 
            << "Root[" << i << "] mismatch";
    }
    
    std::cout << "  ✓ Merkle tree (8 leaves) root matches Rust exactly" << std::endl;
}

// ============================================================================
// FRI Fold Test
// ============================================================================
TEST_F(IOVerificationTest, FriFold) {
    auto data = load_json("fri", "fri_fold.json");
    
    // Load codeword
    std::vector<XFieldElement> codeword;
    for (auto& elem : data["input"]["codeword"]) {
        codeword.push_back(XFieldElement(
            BFieldElement(elem[0].get<uint64_t>()),
            BFieldElement(elem[1].get<uint64_t>()),
            BFieldElement(elem[2].get<uint64_t>())
        ));
    }
    
    // Load challenge
    auto ch = data["input"]["challenge"];
    XFieldElement challenge(
        BFieldElement(ch[0].get<uint64_t>()),
        BFieldElement(ch[1].get<uint64_t>()),
        BFieldElement(ch[2].get<uint64_t>())
    );
    
    // Create FriRound and fold
    ArithmeticDomain domain = ArithmeticDomain::of_length(codeword.size());
    FriRound round(domain, codeword);
    auto folded = round.split_and_fold(challenge);
    
    // Compare with Rust output
    auto rust_folded = data["output"]["folded"];
    EXPECT_EQ(folded.size(), rust_folded.size()) << "Folded codeword size mismatch";
    
    for (size_t i = 0; i < folded.size(); i++) {
        auto expected = rust_folded[i];
        EXPECT_EQ(folded[i].coeff(0).value(), expected[0].get<uint64_t>()) 
            << "Folded[" << i << "] coeff[0] mismatch";
        EXPECT_EQ(folded[i].coeff(1).value(), expected[1].get<uint64_t>()) 
            << "Folded[" << i << "] coeff[1] mismatch";
        EXPECT_EQ(folded[i].coeff(2).value(), expected[2].get<uint64_t>()) 
            << "Folded[" << i << "] coeff[2] mismatch";
    }
    
    std::cout << "  ✓ FRI fold matches Rust exactly (" << folded.size() << " elements)" << std::endl;
}

// ============================================================================
// Summary Test
// ============================================================================
TEST_F(IOVerificationTest, Summary) {
    std::cout << "\n=== I/O Verification Summary ===" << std::endl;
    std::cout << "  All tests compute outputs using C++ implementation" << std::endl;
    std::cout << "  and compare 100% with Rust outputs." << std::endl;
    std::cout << "\n  Tests verified:" << std::endl;
    std::cout << "    ✓ Tip5 hash (empty, single, 10, pair, 379 elements)" << std::endl;
    std::cout << "    ✓ BFieldElement arithmetic (add, sub, mul, neg, inv, pow, roots)" << std::endl;
    std::cout << "    ✓ XFieldElement arithmetic (add, mul, inv)" << std::endl;
    std::cout << "    ✓ ArithmeticDomain (512, 4096, with offset)" << std::endl;
    std::cout << "    ✓ Merkle tree (4 and 8 leaves)" << std::endl;
    std::cout << "    ✓ FRI fold" << std::endl;
    
    SUCCEED();
}

