#include "types/digest.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace triton_vm {

bool Digest::operator==(const Digest& rhs) const {
    for (size_t i = 0; i < LEN; ++i) {
        if (elements_[i] != rhs.elements_[i]) {
            return false;
        }
    }
    return true;
}

bool Digest::operator!=(const Digest& rhs) const {
    return !(*this == rhs);
}

std::string Digest::to_hex() const {
    std::ostringstream oss;
    // Match Rust's `Digest::to_hex()`: each BFieldElement is written as 8 bytes in little-endian order.
    for (size_t i = 0; i < LEN; ++i) {
        uint64_t val = elements_[i].value();
        for (size_t byte_idx = 0; byte_idx < 8; ++byte_idx) {
            uint8_t byte_val = static_cast<uint8_t>((val >> (byte_idx * 8)) & 0xFF);
            oss << std::setfill('0') << std::setw(2) << std::hex << static_cast<int>(byte_val);
        }
    }
    return oss.str();
}

Digest Digest::from_hex(const std::string& hex) {
    if (hex.length() != LEN * 16) {
        throw std::invalid_argument("Invalid hex string length for Digest");
    }
    
    std::array<BFieldElement, LEN> elements;
    for (size_t i = 0; i < LEN; ++i) {
        // Rust's to_hex() uses little-endian byte order
        // Each BFieldElement is 8 bytes (16 hex chars)
        std::string chunk = hex.substr(i * 16, 16);
        
        // Parse hex string as little-endian bytes
        // Rust's to_hex() outputs bytes in little-endian order
        // Each pair of hex chars = 1 byte, bytes are in little-endian order
        uint64_t value = 0;
        for (size_t byte_idx = 0; byte_idx < 8; ++byte_idx) {
            size_t hex_pos = byte_idx * 2;
            std::string byte_str = chunk.substr(hex_pos, 2);
            uint8_t byte_val = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
            // Little-endian: byte 0 is least significant (bits 0-7), byte 7 is most significant (bits 56-63)
            value |= static_cast<uint64_t>(byte_val) << (byte_idx * 8);
        }
        
        elements[i] = BFieldElement(value);
    }
    
    return Digest(elements);
}

std::ostream& operator<<(std::ostream& os, const Digest& digest) {
    return os << digest.to_hex();
}

} // namespace triton_vm

