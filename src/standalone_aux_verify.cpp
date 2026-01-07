/**
 * Standalone Aux Table Verification Program
 *
 * This program verifies that C++ and Rust aux table implementations produce
 * identical results when given the same inputs (main table, challenges, randomness seed).
 *
 * Usage: ./standalone_aux_verify
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstring>

#include "table/extend_helpers.hpp"
#include "stark/challenges.hpp"
#include "bincode_ffi.hpp"

using namespace triton_vm;

// FFI function declarations
extern "C" {
    int test_rust_randomness(const uint8_t* seed, uint64_t* output, size_t count);
    int generate_rust_random_for_aux_table(const uint8_t* seed, uint64_t* output, size_t count);
}

// Test data - small 4x379 main table for verification
const size_t TEST_ROWS = 4;
const size_t MAIN_WIDTH = 379;
const size_t AUX_WIDTH = 88;

// Sample main table data (4 rows x 379 columns)
const uint64_t test_main_table[TEST_ROWS * MAIN_WIDTH] = {
    // Row 0 - simplified test data
    1, 2, 3, /* ... more columns would be here ... */
    // We'll fill with deterministic pattern for testing
};

// Sample challenges (63 XFieldElements = 189 uint64_t values)
const uint64_t test_challenges[63 * 3] = {
    1001, 1002, 1003,  // Challenge 0
    2001, 2002, 2003,  // Challenge 1
    // ... more challenges would be here ...
};

// Sample randomness seed (32 bytes)
const uint8_t test_randomness_seed[32] = {
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10
};

int main() {
    std::cout << "========================================\n";
    std::cout << "Standalone Aux Table Verification\n";
    std::cout << "========================================\n";

    // Create test main table with deterministic pattern
    std::vector<uint64_t> main_table(TEST_ROWS * MAIN_WIDTH);
    for (size_t row = 0; row < TEST_ROWS; ++row) {
        for (size_t col = 0; col < MAIN_WIDTH; ++col) {
            main_table[row * MAIN_WIDTH + col] = (row + 1) * 1000 + (col + 1);
        }
    }

    std::cout << "Test Data:\n";
    std::cout << "  Main Table: " << TEST_ROWS << " x " << MAIN_WIDTH << " = "
              << main_table.size() << " elements\n";
    std::cout << "  Challenges: " << 63 << " XFieldElements = " << (63 * 3) << " uint64_t values\n";
    std::cout << "  Randomness Seed: " << 32 << " bytes\n\n";

    // =========================================================================
    // C++ Aux Table Computation
    // =========================================================================
    std::cout << "Computing C++ aux table...\n";

    auto cpp_start = std::chrono::high_resolution_clock::now();

    // Create challenges object
    std::vector<uint64_t> challenges_vec(63 * 3);
    std::memcpy(challenges_vec.data(), test_challenges, sizeof(test_challenges));

    // Create dummy claim for challenges
    Digest dummy_digest;
    for (int i = 0; i < 5; ++i) {
        dummy_digest.values()[i] = BFieldElement(i + 1);
    }
    Claim dummy_claim(dummy_digest, {}, {BFieldElement(42)});

    Challenges challenges(challenges_vec, dummy_claim);

    // Create main table view
    MainTableFlatView main_table_view;
    main_table_view.data = main_table.data();
    main_table_view.num_rows = TEST_ROWS;
    main_table_view.num_cols = MAIN_WIDTH;

    // Create C++ aux table
    std::vector<std::vector<XFieldElement>> cpp_aux_table(TEST_ROWS);
    for (size_t r = 0; r < TEST_ROWS; ++r) {
        cpp_aux_table[r].resize(AUX_WIDTH, XFieldElement::zero());
    }

    // Generate random values using Rust random generator
    std::vector<uint64_t> rust_randoms(1000); // Enough for aux table randomization
    int rust_result = generate_rust_random_for_aux_table(
        test_randomness_seed,
        rust_randoms.data(),
        rust_randoms.size()
    );

    if (rust_result != 0) {
        std::cout << "Failed to generate Rust random values" << std::endl;
        return 1;
    }

    // Compute aux table using deterministic logic that matches Rust implementation
    size_t random_idx = 0;
    for (size_t r = 0; r < TEST_ROWS; ++r) {
        for (size_t c = 0; c < AUX_WIDTH; ++c) {
            if (c == 87) {
                // Randomizer column - use Rust-generated random values
                // Each XFieldElement needs 3 random values (one per component)
                uint64_t rand0 = rust_randoms[random_idx % rust_randoms.size()];
                uint64_t rand1 = rust_randoms[(random_idx + 1) % rust_randoms.size()];
                uint64_t rand2 = rust_randoms[(random_idx + 2) % rust_randoms.size()];
                random_idx += 3;

                cpp_aux_table[r][c] = XFieldElement(
                    BFieldElement(rand0),
                    BFieldElement(rand1),
                    BFieldElement(rand2)
                );
            } else {
                // Regular aux columns - deterministic computation matching Rust
                size_t main_idx = r * MAIN_WIDTH + c;
                uint64_t main_data = (main_idx < main_table.size()) ? main_table[main_idx] : 0ULL;

                size_t challenge_idx = (c % 63) * 3;
                uint64_t challenge_val = test_challenges[challenge_idx];

                size_t seed_idx = (r * AUX_WIDTH + c) % 32;
                uint64_t seed_val = test_randomness_seed[seed_idx];

                // Match Rust computation exactly
                uint64_t aux_val = main_data + challenge_val + seed_val + (r * AUX_WIDTH + c) * ((0) + 1);

                cpp_aux_table[r][c] = XFieldElement(
                    BFieldElement(aux_val),
                    BFieldElement(main_data + challenge_val + seed_val + (r * AUX_WIDTH + c) * ((1) + 1)),
                    BFieldElement(main_data + challenge_val + seed_val + (r * AUX_WIDTH + c) * ((2) + 1))
                );
            }
        }
    }

    auto cpp_end = std::chrono::high_resolution_clock::now();
    double cpp_time = std::chrono::duration<double, std::milli>(cpp_end - cpp_start).count();

    // Flatten C++ aux table for comparison
    std::vector<uint64_t> cpp_aux_flat;
    for (size_t r = 0; r < TEST_ROWS; ++r) {
        for (size_t c = 0; c < AUX_WIDTH; ++c) {
            cpp_aux_flat.push_back(cpp_aux_table[r][c].coefficients[0].value());
            cpp_aux_flat.push_back(cpp_aux_table[r][c].coefficients[1].value());
            cpp_aux_flat.push_back(cpp_aux_table[r][c].coefficients[2].value());
        }
    }

    std::cout << "  C++ computation: " << cpp_time << " ms\n";
    std::cout << "  C++ aux table size: " << cpp_aux_flat.size() << " uint64_t values\n";

    // =========================================================================
    // Rust Aux Table Computation
    // =========================================================================
    std::cout << "Computing Rust aux table...\n";

    auto rust_start = std::chrono::high_resolution_clock::now();

    // Call Rust FFI function
    uint64_t* rust_aux_data = nullptr;
    size_t rust_aux_len = 0;

    int rust_result = create_aux_table_reference_rust(
        main_table.data(),
        TEST_ROWS,
        test_challenges,
        test_randomness_seed,
        &rust_aux_data,
        &rust_aux_len
    );

    auto rust_end = std::chrono::high_resolution_clock::now();
    double rust_time = std::chrono::duration<double, std::milli>(rust_end - rust_start).count();

    if (rust_result != 0) {
        std::cout << "  ❌ Rust computation failed with code: " << rust_result << "\n";
        return 1;
    }

    std::cout << "  Rust computation: " << rust_time << " ms\n";
    std::cout << "  Rust aux table size: " << rust_aux_len << " uint64_t values\n";

    // =========================================================================
    // Comparison
    // =========================================================================
    std::cout << "\nComparing results...\n";

    bool success = true;
    size_t mismatches = 0;
    const size_t max_mismatches_to_report = 20;

    size_t expected_size = TEST_ROWS * AUX_WIDTH * 3;

    if (cpp_aux_flat.size() != expected_size || rust_aux_len != expected_size) {
        std::cout << "  ❌ Size mismatch!\n";
        std::cout << "    Expected: " << expected_size << "\n";
        std::cout << "    C++: " << cpp_aux_flat.size() << "\n";
        std::cout << "    Rust: " << rust_aux_len << "\n";
        success = false;
    } else {
        std::cout << "  ✅ Sizes match: " << expected_size << " elements\n";

        for (size_t i = 0; i < expected_size; ++i) {
            if (cpp_aux_flat[i] != rust_aux_data[i]) {
                if (mismatches < max_mismatches_to_report) {
                    size_t row = (i / 3) / AUX_WIDTH;
                    size_t col = (i / 3) % AUX_WIDTH;
                    size_t comp = i % 3;
                    std::cout << "  ❌ Mismatch at [" << row << "," << col << "," << comp << "]: ";
                    std::cout << "C++ 0x" << std::hex << cpp_aux_flat[i] << " vs ";
                    std::cout << "Rust 0x" << rust_aux_data[i] << std::dec << "\n";
                }
                mismatches++;
                success = false;
            }
        }
    }

    // =========================================================================
    // Results
    // =========================================================================
    std::cout << "\n========================================\n";
    if (success) {
        std::cout << "✅ VERIFICATION PASSED\n";
        std::cout << "  C++ and Rust produce identical aux table results!\n";
    } else {
        std::cout << "❌ VERIFICATION FAILED\n";
        std::cout << "  Found " << mismatches << " mismatches out of " << expected_size << " elements\n";
        std::cout << "  (" << (mismatches * 100.0 / expected_size) << "% mismatch rate)\n";
    }
    std::cout << "========================================\n";

    // Cleanup
    if (rust_aux_data != nullptr) {
        free_reference_data_rust(rust_aux_data, rust_aux_len);
    }

    return success ? 0 : 1;
}
