#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace triton_vm {

/**
 * ChaCha12Rng - ChaCha12 random number generator matching Rust's rand ChaCha12Rng
 *
 * This implementation generates deterministic random numbers for STARK proof generation,
 * ensuring reproducible proofs when using the same seed.
 */
class ChaCha12Rng {
private:
    std::array<uint32_t, 16> state_;
    std::array<uint8_t, 64> buffer_;
    size_t index_;

    // ChaCha quarter round function
    static void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d);

    // Generate next block of keystream
    void generate_block();

public:
    using Seed = std::array<uint8_t, 32>;

    // Initialize with 32-byte seed
    explicit ChaCha12Rng(const Seed& seed);

    // Generate random u64
    uint64_t next_u64();

    // Generate random u32
    uint32_t next_u32() {
        return static_cast<uint32_t>(next_u64() & 0xFFFFFFFFULL);
    }

    // Fill a buffer with random bytes
    void fill_bytes(uint8_t* buffer, size_t length);
};

} // namespace triton_vm
