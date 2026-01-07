#pragma once

#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include <array>
#include <vector>

namespace triton_vm {

/**
 * Tip5 - Arithmetization-oriented hash function
 * 
 * Reference implementation of the Tip5 hash function as specified in
 * "The Tip5 Hash Function for Recursive STARKs" (https://eprint.iacr.org/2023/107.pdf)
 * 
 * This matches the Triton VM Rust implementation (twenty-first crate).
 */
class Tip5 {
public:
    // Tip5 constants
    static constexpr size_t STATE_SIZE = 16;
    static constexpr size_t NUM_SPLIT_AND_LOOKUP = 4;
    static constexpr size_t CAPACITY = 6;
    static constexpr size_t RATE = 10;
    static constexpr size_t NUM_ROUNDS = 5;
    
    // Lookup table for S-box
    static const std::array<uint8_t, 256> LOOKUP_TABLE;
    
    // MDS matrix first column (circulant matrix)
    static const std::array<int64_t, STATE_SIZE> MDS_MATRIX_FIRST_COLUMN;
    
    // Round constants
    static const std::array<BFieldElement, NUM_ROUNDS * STATE_SIZE> ROUND_CONSTANTS;
    
    // State
    std::array<BFieldElement, STATE_SIZE> state;
    
    // Constructors
    Tip5();
    
    // Initialize with fixed-length domain separator
    static Tip5 init();
    
    // Core operations
    void permutation();
    void round(size_t round_index);
    
    // Sponge operations
    void absorb(const std::vector<BFieldElement>& elements);
    std::vector<BFieldElement> squeeze(size_t count);
    
    // Hash functions
    static Digest hash_10(const std::array<BFieldElement, RATE>& input);
    static Digest hash_varlen(const std::vector<BFieldElement>& input);
    static Digest hash_pair(const Digest& left, const Digest& right);
    
    // Trace operations (for recording permutation state)
    // Returns state after each round (including initial state)
    std::vector<std::array<BFieldElement, STATE_SIZE>> trace();
    
private:
    void sbox_layer();
    void mds_layer();
    void add_round_constants(size_t round_index);
    
    static void split_and_lookup(BFieldElement& element);
};

} // namespace triton_vm

