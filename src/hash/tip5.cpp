#include "hash/tip5.hpp"
#include <cstring>

namespace triton_vm {

// Lookup table for the Tip5 S-box (from CUDA reference implementation)
const std::array<uint8_t, 256> Tip5::LOOKUP_TABLE = {
      0,   7,  26,  63, 124, 215,  85, 254, 214, 228,  45, 185, 140, 173,  33, 240,
     29, 177, 176,  32,   8, 110,  87, 202, 204,  99, 150, 106, 230,  14, 235, 128,
    213, 239, 212, 138,  23, 130, 208,   6,  44,  71,  93, 116, 146, 189, 251,  81,
    199,  97,  38,  28,  73, 179,  95,  84, 152,  48,  35, 119,  49,  88, 242,   3,
    148, 169,  72, 120,  62, 161, 166,  83, 175, 191, 137,  19, 100, 129, 112,  55,
    221, 102, 218,  61, 151, 237,  68, 164,  17, 147,  46, 234, 203, 216,  22, 141,
     65,  57, 123,  12, 244,  54, 219, 231,  96,  77, 180, 154,   5, 253, 133, 165,
     98, 195, 205, 134, 245,  30,   9, 188,  59, 142, 186, 197, 181, 144,  92,  31,
    224, 163, 111,  74,  58,  69, 113, 196,  67, 246, 225,  10, 121,  50,  60, 157,
     90, 122,   2, 250, 101,  75, 178, 159,  24,  36, 201,  11, 243, 132, 198, 190,
    114, 233,  39,  52,  21, 209, 108, 238,  91, 187,  18, 104, 194,  37, 153,  34,
    200, 143, 126, 155, 236, 118,  64,  80, 172,  89,  94, 193, 135, 183,  86, 107,
    252,  13, 167, 206, 136, 220, 207, 103, 171, 160,  76, 182, 227, 217, 158,  56,
    174,   4,  66, 109, 139, 162, 184, 211, 249,  47, 125, 232, 117,  43,  16,  42,
    127,  20, 241,  25, 149, 105, 156,  51,  53, 168, 145, 247, 223,  79,  78, 226,
     15, 222,  82, 115,  70, 210,  27,  41,   1, 170,  40, 131, 192, 229, 248, 255
};

// MDS matrix first column (circulant matrix)
const std::array<int64_t, Tip5::STATE_SIZE> Tip5::MDS_MATRIX_FIRST_COLUMN = {
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034,
    56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845
};

// Round constants (from CUDA reference - hex values)
const std::array<BFieldElement, Tip5::NUM_ROUNDS * Tip5::STATE_SIZE> Tip5::ROUND_CONSTANTS = {
    // Round 0
    BFieldElement(0xBD2A3DEB61AB60DEULL), BFieldElement(0xEA7DF21AD9547ED2ULL),
    BFieldElement(0x900B3677A1DE063FULL), BFieldElement(0x1B46887E876C8677ULL),
    BFieldElement(0xD364D977889CFB97ULL), BFieldElement(0xDC8DFAC843699F02ULL),
    BFieldElement(0x375C405D7190DB58ULL), BFieldElement(0x27924006D2B0D4B1ULL),
    BFieldElement(0x78DD1172D483CD38ULL), BFieldElement(0x3346C66244882A56ULL),
    BFieldElement(0xB0249B279F498AA5ULL), BFieldElement(0x94CD51BE79338D4DULL),
    BFieldElement(0xB0E0DC7052C5B218ULL), BFieldElement(0xF8DCC4D248ADAD95ULL),
    BFieldElement(0x68E3C635FEC868B7ULL), BFieldElement(0xD7D06B3FFB6B0D8CULL),
    // Round 1
    BFieldElement(0xF3500DEA20EF032AULL), BFieldElement(0x4865BF175BBA5803ULL),
    BFieldElement(0xD5F7FE3027287A27ULL), BFieldElement(0xA57333F44E193412ULL),
    BFieldElement(0x8726E153A977EAE2ULL), BFieldElement(0x3014A98463FC191BULL),
    BFieldElement(0xBA145461AF39B212ULL), BFieldElement(0x03AB70105933202FULL),
    BFieldElement(0x3D90B7EEBFCF71E5ULL), BFieldElement(0x386322B1CC520BFDULL),
    BFieldElement(0x27C2C8DAF774F675ULL), BFieldElement(0x4FCB83F50309BC6AULL),
    BFieldElement(0x5E6D5CE8275F3CB3ULL), BFieldElement(0xECC2F6592C8F905CULL),
    BFieldElement(0x837F532461E609B4ULL), BFieldElement(0xB2B1F6B95C92C93CULL),
    // Round 2
    BFieldElement(0xC0027AF556411DC1ULL), BFieldElement(0x16E18C885FC2A26CULL),
    BFieldElement(0x8880EF183D9F2BF3ULL), BFieldElement(0xB2930BDB5CA88C45ULL),
    BFieldElement(0x9C2EC8322E1C1553ULL), BFieldElement(0xE5B05EAF3220A674ULL),
    BFieldElement(0xA49CC6AE4B861C4EULL), BFieldElement(0x11708E0AEB86EBD7ULL),
    BFieldElement(0xC09DE92BBC3902E0ULL), BFieldElement(0x929B3C79516BCBC1ULL),
    BFieldElement(0xE006E5BF738F27D1ULL), BFieldElement(0x2D9E1EC0EAC8EA38ULL),
    BFieldElement(0x0984D8D94BF937C5ULL), BFieldElement(0x4959273C220E6747ULL),
    BFieldElement(0xFE1D934207E796FAULL), BFieldElement(0x2B9B9298F2F6DD73ULL),
    // Round 3
    BFieldElement(0x07A1F5A67D6E3A41ULL), BFieldElement(0x4407593EE73743D9ULL),
    BFieldElement(0x9F054720EF802E59ULL), BFieldElement(0x78D4B711336E6AA6ULL),
    BFieldElement(0xADC638AEF3C8B228ULL), BFieldElement(0xA4D6D3E86AFB2114ULL),
    BFieldElement(0x9D4808E725531968ULL), BFieldElement(0x369804DF3866D0EFULL),
    BFieldElement(0xE6DBD9A9D2215024ULL), BFieldElement(0x8ED22CA212EE85B2ULL),
    BFieldElement(0x397BB882FCD23EB6ULL), BFieldElement(0xEB8F8786D7277531ULL),
    BFieldElement(0x9999D4CDAFF543B5ULL), BFieldElement(0xF382A61217F192D6ULL),
    BFieldElement(0x49C37260B026ADC1ULL), BFieldElement(0x3FF8918CE35C1019ULL),
    // Round 4
    BFieldElement(0x2E7DF8B76080BD07ULL), BFieldElement(0xF5DBAC250B8A28B9ULL),
    BFieldElement(0x853C3727AE9DA4CCULL), BFieldElement(0xB2F1F5F3D9E5A26DULL),
    BFieldElement(0x3FCE22012D337847ULL), BFieldElement(0x6B5A3E6DB7EEE347ULL),
    BFieldElement(0x171582CD59DDE50DULL), BFieldElement(0xC0C0B3095EE62A8AULL),
    BFieldElement(0x665B25C6F6A203D2ULL), BFieldElement(0x3099AED93B6AE69FULL),
    BFieldElement(0x801DF6092BE69C38ULL), BFieldElement(0x8066AD0CDFFF43CDULL),
    BFieldElement(0x8AF9D44A5F4FDC6BULL), BFieldElement(0xD80219CD97C0D762ULL),
    BFieldElement(0x10C9CEBA14148EBBULL), BFieldElement(0x539BD4C3F2F24474ULL)
};

Tip5::Tip5() {
    state.fill(BFieldElement::zero());
}

Tip5 Tip5::init() {
    Tip5 tip5;
    // Rust `Tip5::init()` is used for both Fiat-Shamir and sponge init/program hashing.
    // Its initial state has *zero* capacity elements (the domain handling happens in the sponge API).
    return tip5;
}

void Tip5::permutation() {
    for (size_t i = 0; i < NUM_ROUNDS; ++i) {
        round(i);
    }
}

std::vector<std::array<BFieldElement, Tip5::STATE_SIZE>> Tip5::trace() {
    // Record state after each round (including initial state) and advance the sponge.
    // This matches Rust's `Tip5::trace()` which mutates `self` and returns `PermutationTrace`.
    std::vector<std::array<BFieldElement, STATE_SIZE>> trace_result;
    trace_result.reserve(NUM_ROUNDS + 1); // Initial state + one per round

    // Record initial state (before any rounds)
    trace_result.push_back(state);

    // Apply rounds, recording state after each one.
    for (size_t i = 0; i < NUM_ROUNDS; ++i) {
        round(i);
        trace_result.push_back(state);
    }
    return trace_result;
}

void Tip5::round(size_t round_index) {
    sbox_layer();
    mds_layer();
    add_round_constants(round_index);
}

void Tip5::sbox_layer() {
    // Split-and-lookup for first 4 elements
    for (size_t i = 0; i < NUM_SPLIT_AND_LOOKUP; ++i) {
        split_and_lookup(state[i]);
    }
    
    // x^7 for remaining elements
    for (size_t i = NUM_SPLIT_AND_LOOKUP; i < STATE_SIZE; ++i) {
        BFieldElement sq = state[i] * state[i];
        BFieldElement qu = sq * sq;
        state[i] = state[i] * sq * qu; // x * x^2 * x^4 = x^7
    }
}

// Montgomery reduction: (hi, lo) * R^(-1) mod P
static uint64_t monty_reduce(uint64_t hi, uint64_t lo) {
    // Matches CUDA montyred_from_parts
    uint64_t shifted = lo << 32;
    uint64_t a = lo + shifted;
    bool overflow_a = (a < lo) || (a < shifted);
    
    uint64_t b = a - (a >> 32);
    if (overflow_a) b -= 1;
    
    uint64_t r = hi - b;
    bool underflow = (hi < b);
    
    if (underflow) {
        r -= (1ULL + ~BFieldElement::MODULUS);
    }
    
    return r;
}

void Tip5::split_and_lookup(BFieldElement& element) {
    // R^2 mod P for converting to Montgomery form
    constexpr uint64_t R2 = 0xFFFFFFFE00000001ULL;
    
    uint64_t val = element.value();
    
    // Convert to Montgomery form: val * R^2 * R^(-1) = val * R
    __uint128_t product = static_cast<__uint128_t>(val) * R2;
    uint64_t lo = static_cast<uint64_t>(product);
    uint64_t hi = static_cast<uint64_t>(product >> 64);
    uint64_t montgomery_val = monty_reduce(hi, lo);
    
    // Do lookup on Montgomery form bytes
    uint64_t sbox_out = 0;
    for (size_t i = 0; i < 8; ++i) {
        uint8_t byte = static_cast<uint8_t>((montgomery_val >> (i * 8)) & 0xFF);
        uint8_t looked_up = LOOKUP_TABLE[byte];
        sbox_out |= static_cast<uint64_t>(looked_up) << (i * 8);
    }
    
    // Convert back from Montgomery form: sbox_out * R^(-1)
    uint64_t result = monty_reduce(0, sbox_out);
    
    element = BFieldElement(result);
}

void Tip5::mds_layer() {
    std::array<BFieldElement, STATE_SIZE> new_state;
    new_state.fill(BFieldElement::zero());
    
    // Circulant matrix multiplication using naive approach (matches Rust)
    // Uses CUDA-style indexing: idx = (i - j) & (STATE_SIZE - 1)
    for (size_t row = 0; row < STATE_SIZE; ++row) {
        for (size_t col = 0; col < STATE_SIZE; ++col) {
            size_t idx = (row - col) & (STATE_SIZE - 1);
            BFieldElement coeff(static_cast<uint64_t>(MDS_MATRIX_FIRST_COLUMN[idx]));
            new_state[row] = new_state[row] + (coeff * state[col]);
        }
    }
    
    state = new_state;
}

void Tip5::add_round_constants(size_t round_index) {
    size_t offset = round_index * STATE_SIZE;
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        state[i] += ROUND_CONSTANTS[offset + i];
    }
}

void Tip5::absorb(const std::vector<BFieldElement>& elements) {
    // Overwrite mode (matching CUDA)
    for (size_t i = 0; i < elements.size() && i < RATE; ++i) {
        state[i] = elements[i];
    }
    permutation();
}

std::vector<BFieldElement> Tip5::squeeze(size_t count) {
    std::vector<BFieldElement> result;
    result.reserve(count);
    
    for (size_t i = 0; i < count && i < RATE; ++i) {
        result.push_back(state[i]);
    }
    
    return result;
}

Digest Tip5::hash_10(const std::array<BFieldElement, RATE>& input) {
    // Rust uses `Tip5::new(sponge::Domain::FixedLength)` for fixed-length hashing.
    // We model that domain separation by setting capacity to ones.
    Tip5 tip5;
    for (size_t i = RATE; i < STATE_SIZE; ++i) {
        tip5.state[i] = BFieldElement::one();
    }
    
    // Overwrite rate portion with input
    for (size_t i = 0; i < RATE; ++i) {
        tip5.state[i] = input[i];
    }
    
    tip5.permutation();
    
    // Extract digest from first 5 elements
    return Digest(
        tip5.state[0], tip5.state[1], tip5.state[2],
        tip5.state[3], tip5.state[4]
    );
}

Digest Tip5::hash_varlen(const std::vector<BFieldElement>& input) {
    Tip5 tip5; // Variable-length domain: all zeros initially
    
    size_t consumed = 0;
    
    // Process full chunks using OVERWRITE mode (matching CUDA)
    while (consumed + RATE <= input.size()) {
        for (size_t i = 0; i < RATE; ++i) {
            tip5.state[i] = input[consumed + i];
        }
        tip5.permutation();
        consumed += RATE;
    }
    
    // Handle last chunk with padding (matching CUDA)
    // Remaining elements, then 1, then zeros to fill RATE
    std::array<BFieldElement, RATE> last_chunk;
    last_chunk.fill(BFieldElement::zero());
    
    for (size_t i = 0; i < input.size() - consumed; ++i) {
        last_chunk[i] = input[consumed + i];
    }
    // Add padding indicator right after last element
    last_chunk[input.size() - consumed] = BFieldElement::one();
    
    // Overwrite state with last chunk
    for (size_t i = 0; i < RATE; ++i) {
        tip5.state[i] = last_chunk[i];
    }
    tip5.permutation();
    
    // Extract digest from first 5 elements
    return Digest(
        tip5.state[0], tip5.state[1], tip5.state[2],
        tip5.state[3], tip5.state[4]
    );
}

Digest Tip5::hash_pair(const Digest& left, const Digest& right) {
    std::array<BFieldElement, RATE> input;
    
    // Left digest in positions 0-4
    for (size_t i = 0; i < Digest::LEN; ++i) {
        input[i] = left[i];
    }
    // Right digest in positions 5-9
    for (size_t i = 0; i < Digest::LEN; ++i) {
        input[Digest::LEN + i] = right[i];
    }
    
    return hash_10(input);
}

} // namespace triton_vm

