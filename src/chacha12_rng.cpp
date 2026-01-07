#include "chacha12_rng.hpp"
#include <cstring>

namespace triton_vm {

// ChaCha quarter round function
void ChaCha12Rng::quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8) | (d >> 24);
    c += d; b ^= c; b = (b << 7) | (b >> 25);
}

// Generate next block of keystream
void ChaCha12Rng::generate_block() {
    std::array<uint32_t, 16> working_state = state_;

    // 12 rounds (6 double rounds)
    for (int i = 0; i < 6; ++i) {
        // Column rounds
        quarter_round(working_state[0], working_state[4], working_state[8], working_state[12]);
        quarter_round(working_state[1], working_state[5], working_state[9], working_state[13]);
        quarter_round(working_state[2], working_state[6], working_state[10], working_state[14]);
        quarter_round(working_state[3], working_state[7], working_state[11], working_state[15]);

        // Diagonal rounds
        quarter_round(working_state[0], working_state[5], working_state[10], working_state[15]);
        quarter_round(working_state[1], working_state[6], working_state[11], working_state[12]);
        quarter_round(working_state[2], working_state[7], working_state[8], working_state[13]);
        quarter_round(working_state[3], working_state[4], working_state[9], working_state[14]);
    }

    // Add original state
    for (int i = 0; i < 16; ++i) {
        working_state[i] += state_[i];
    }

    // Convert to bytes
    for (int i = 0; i < 16; ++i) {
        uint32_t word = working_state[i];
        buffer_[i * 4] = word & 0xFF;
        buffer_[i * 4 + 1] = (word >> 8) & 0xFF;
        buffer_[i * 4 + 2] = (word >> 16) & 0xFF;
        buffer_[i * 4 + 3] = (word >> 24) & 0xFF;
    }

    // Increment counter
    state_[12]++;
    if (state_[12] == 0) {
        state_[13]++; // Handle overflow
    }

    index_ = 0;
}

// Initialize with 32-byte seed
ChaCha12Rng::ChaCha12Rng(const Seed& seed) : index_(64) {
    // ChaCha constants
    state_[0] = 0x61707865; // "expa"
    state_[1] = 0x3320646e; // "nd 3"
    state_[2] = 0x79622d32; // "2-by"
    state_[3] = 0x6b206574; // "te k"

    // Key (seed)
    for (int i = 0; i < 8; ++i) {
        state_[4 + i] = (seed[i * 4]) |
                       (seed[i * 4 + 1] << 8) |
                       (seed[i * 4 + 2] << 16) |
                       (seed[i * 4 + 3] << 24);
    }

    // Counter (starts at 0)
    state_[12] = 0;
    state_[13] = 0;

    // Nonce (all zeros for rand compatibility)
    state_[14] = 0;
    state_[15] = 0;
}

// Generate random u64
uint64_t ChaCha12Rng::next_u64() {
    if (index_ >= 64) {
        generate_block();
    }

    uint64_t result = 0;
    for (int i = 0; i < 8; ++i) {
        result |= static_cast<uint64_t>(buffer_[index_ + i]) << (i * 8);
    }

    index_ += 8;
    return result;
}

// Fill a buffer with random bytes
void ChaCha12Rng::fill_bytes(uint8_t* buffer, size_t length) {
    size_t remaining = length;
    uint8_t* ptr = buffer;

    while (remaining > 0) {
        if (index_ >= 64) {
            generate_block();
        }

        size_t to_copy = std::min(remaining, static_cast<size_t>(64 - index_));
        std::memcpy(ptr, &buffer_[index_], to_copy);

        ptr += to_copy;
        index_ += to_copy;
        remaining -= to_copy;
    }
}

} // namespace triton_vm
