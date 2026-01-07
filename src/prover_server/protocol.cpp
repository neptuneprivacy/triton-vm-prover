#include "protocol.hpp"

#include <unistd.h>
#include <cstring>
#include <iostream>

namespace triton_vm {
namespace prover_server {

// SocketReader implementation

bool SocketReader::read_exact(void* buf, size_t n) {
    uint8_t* ptr = static_cast<uint8_t*>(buf);
    size_t remaining = n;
    
    while (remaining > 0) {
        ssize_t bytes_read = ::read(fd_, ptr, remaining);
        if (bytes_read <= 0) {
            if (bytes_read == 0) {
                // Connection closed
                return false;
            }
            if (errno == EINTR) {
                continue;  // Interrupted, retry
            }
            return false;  // Error
        }
        ptr += bytes_read;
        remaining -= bytes_read;
    }
    return true;
}

std::optional<uint32_t> SocketReader::read_u32_le() {
    uint8_t buf[4];
    if (!read_exact(buf, 4)) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(buf[0]) |
           (static_cast<uint32_t>(buf[1]) << 8) |
           (static_cast<uint32_t>(buf[2]) << 16) |
           (static_cast<uint32_t>(buf[3]) << 24);
}

std::optional<uint64_t> SocketReader::read_u64_le() {
    uint8_t buf[8];
    if (!read_exact(buf, 8)) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(buf[0]) |
           (static_cast<uint64_t>(buf[1]) << 8) |
           (static_cast<uint64_t>(buf[2]) << 16) |
           (static_cast<uint64_t>(buf[3]) << 24) |
           (static_cast<uint64_t>(buf[4]) << 32) |
           (static_cast<uint64_t>(buf[5]) << 40) |
           (static_cast<uint64_t>(buf[6]) << 48) |
           (static_cast<uint64_t>(buf[7]) << 56);
}

std::optional<std::string> SocketReader::read_length_prefixed_string() {
    auto len_opt = read_u32_le();
    if (!len_opt) {
        return std::nullopt;
    }
    
    uint32_t len = *len_opt;
    
    // Sanity check: JSON blobs shouldn't be > 100MB
    if (len > 100'000'000) {
        std::cerr << "[protocol] String too large: " << len << " bytes" << std::endl;
        return std::nullopt;
    }
    
    std::string result(len, '\0');
    if (!read_exact(result.data(), len)) {
        return std::nullopt;
    }
    
    return result;
}

std::optional<ProverRequest> SocketReader::read_request() {
    // Read and verify magic
    auto magic_opt = read_u32_le();
    if (!magic_opt) {
        return std::nullopt;
    }
    if (*magic_opt != MAGIC_REQUEST) {
        std::cerr << "[protocol] Invalid magic: expected 0x" << std::hex << MAGIC_REQUEST
                  << ", got 0x" << *magic_opt << std::dec << std::endl;
        return std::nullopt;
    }
    
    // Read and verify version
    auto version_opt = read_u32_le();
    if (!version_opt) {
        return std::nullopt;
    }
    if (*version_opt != PROTOCOL_VERSION) {
        std::cerr << "[protocol] Unsupported version: " << *version_opt << std::endl;
        return std::nullopt;
    }
    
    // Read job_id
    auto job_id_opt = read_u32_le();
    if (!job_id_opt) {
        return std::nullopt;
    }
    
    // Read 5 length-prefixed JSON strings
    auto claim_json = read_length_prefixed_string();
    if (!claim_json) return std::nullopt;
    
    auto program_json = read_length_prefixed_string();
    if (!program_json) return std::nullopt;
    
    auto nondet_json = read_length_prefixed_string();
    if (!nondet_json) return std::nullopt;
    
    auto max_log2_json = read_length_prefixed_string();
    if (!max_log2_json) return std::nullopt;
    
    auto env_vars_json = read_length_prefixed_string();
    if (!env_vars_json) return std::nullopt;
    
    ProverRequest request;
    request.job_id = *job_id_opt;
    request.claim_json = std::move(*claim_json);
    request.program_json = std::move(*program_json);
    request.nondet_json = std::move(*nondet_json);
    request.max_log2_json = std::move(*max_log2_json);
    request.env_vars_json = std::move(*env_vars_json);
    
    return request;
}

// SocketWriter implementation

bool SocketWriter::write_all(const void* buf, size_t n) {
    const uint8_t* ptr = static_cast<const uint8_t*>(buf);
    size_t remaining = n;
    
    while (remaining > 0) {
        ssize_t bytes_written = ::write(fd_, ptr, remaining);
        if (bytes_written <= 0) {
            if (errno == EINTR) {
                continue;  // Interrupted, retry
            }
            return false;  // Error
        }
        ptr += bytes_written;
        remaining -= bytes_written;
    }
    return true;
}

bool SocketWriter::write_u32_le(uint32_t v) {
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>(v);
    buf[1] = static_cast<uint8_t>(v >> 8);
    buf[2] = static_cast<uint8_t>(v >> 16);
    buf[3] = static_cast<uint8_t>(v >> 24);
    return write_all(buf, 4);
}

bool SocketWriter::write_u64_le(uint64_t v) {
    uint8_t buf[8];
    buf[0] = static_cast<uint8_t>(v);
    buf[1] = static_cast<uint8_t>(v >> 8);
    buf[2] = static_cast<uint8_t>(v >> 16);
    buf[3] = static_cast<uint8_t>(v >> 24);
    buf[4] = static_cast<uint8_t>(v >> 32);
    buf[5] = static_cast<uint8_t>(v >> 40);
    buf[6] = static_cast<uint8_t>(v >> 48);
    buf[7] = static_cast<uint8_t>(v >> 56);
    return write_all(buf, 8);
}

bool SocketWriter::write_response(const ProverResponse& response) {
    // Write magic
    if (!write_u32_le(MAGIC_RESPONSE)) return false;
    
    // Write status
    if (!write_u32_le(static_cast<uint32_t>(response.status))) return false;
    
    // Write job_id
    if (!write_u32_le(response.job_id)) return false;
    
    switch (response.status) {
        case ResponseStatus::Ok:
            // Write proof length and data
            if (!write_u64_le(response.proof_bincode.size())) return false;
            if (!write_all(response.proof_bincode.data(), response.proof_bincode.size())) return false;
            break;
            
        case ResponseStatus::PaddedHeightTooBig:
            // Write observed_log2
            if (!write_u32_le(response.observed_log2)) return false;
            break;
            
        case ResponseStatus::Error:
            // Write error message length and data
            if (!write_u32_le(static_cast<uint32_t>(response.error_message.size()))) return false;
            if (!write_all(response.error_message.data(), response.error_message.size())) return false;
            break;
    }
    
    return true;
}

} // namespace prover_server
} // namespace triton_vm

