#pragma once

/**
 * Socket protocol for GPU prover server
 * 
 * Wire format matches the Rust implementation:
 * 
 * Request:
 *   [4 bytes: magic "TVMP" = 0x54564D50]
 *   [4 bytes: version = 1]
 *   [4 bytes: job_id]
 *   [4 bytes: claim_json_len]     [claim_json bytes]
 *   [4 bytes: program_json_len]   [program_json bytes]
 *   [4 bytes: nondet_json_len]    [nondet_json bytes]
 *   [4 bytes: max_log2_json_len]  [max_log2_json bytes]
 *   [4 bytes: env_vars_json_len]  [env_vars_json bytes]
 * 
 * Response:
 *   [4 bytes: magic "TVMR" = 0x54564D52]
 *   [4 bytes: status: 0=OK, 1=PADDED_HEIGHT_TOO_BIG, 2=ERROR]
 *   [4 bytes: job_id]
 *   if status == OK:
 *     [8 bytes: proof_len]
 *     [proof_len bytes: proof bincode]
 *   if status == PADDED_HEIGHT_TOO_BIG:
 *     [4 bytes: observed_log2]
 *   if status == ERROR:
 *     [4 bytes: error_msg_len]
 *     [error_msg_len bytes: error message UTF-8]
 */

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace triton_vm {
namespace prover_server {

// Protocol constants
constexpr uint32_t MAGIC_REQUEST = 0x54564D50;  // "TVMP"
constexpr uint32_t MAGIC_RESPONSE = 0x54564D52; // "TVMR"
constexpr uint32_t PROTOCOL_VERSION = 1;

// Response status codes
enum class ResponseStatus : uint32_t {
    Ok = 0,
    PaddedHeightTooBig = 1,
    Error = 2,
};

// Request from Neptune's triton-vm-prover proxy
struct ProverRequest {
    uint32_t job_id;
    std::string claim_json;
    std::string program_json;
    std::string nondet_json;
    std::string max_log2_json;
    std::string env_vars_json;
};

// Response to Neptune's triton-vm-prover proxy
struct ProverResponse {
    ResponseStatus status;
    uint32_t job_id;
    
    // For Ok response
    std::vector<uint8_t> proof_bincode;
    
    // For PaddedHeightTooBig response
    uint32_t observed_log2;
    
    // For Error response
    std::string error_message;
    
    // Factory methods for creating responses
    static ProverResponse ok(uint32_t job_id, std::vector<uint8_t> proof) {
        ProverResponse r;
        r.status = ResponseStatus::Ok;
        r.job_id = job_id;
        r.proof_bincode = std::move(proof);
        return r;
    }
    
    static ProverResponse padded_height_too_big(uint32_t job_id, uint32_t observed) {
        ProverResponse r;
        r.status = ResponseStatus::PaddedHeightTooBig;
        r.job_id = job_id;
        r.observed_log2 = observed;
        return r;
    }
    
    static ProverResponse error(uint32_t job_id, const std::string& msg) {
        ProverResponse r;
        r.status = ResponseStatus::Error;
        r.job_id = job_id;
        r.error_message = msg;
        return r;
    }
};

// Read/write helpers for socket I/O
class SocketReader {
public:
    explicit SocketReader(int fd) : fd_(fd) {}
    
    // Read exactly n bytes
    bool read_exact(void* buf, size_t n);
    
    // Read a u32 in little-endian
    std::optional<uint32_t> read_u32_le();
    
    // Read a u64 in little-endian
    std::optional<uint64_t> read_u64_le();
    
    // Read a length-prefixed string
    std::optional<std::string> read_length_prefixed_string();
    
    // Read a complete request
    std::optional<ProverRequest> read_request();
    
private:
    int fd_;
};

class SocketWriter {
public:
    explicit SocketWriter(int fd) : fd_(fd) {}
    
    // Write exactly n bytes
    bool write_all(const void* buf, size_t n);
    
    // Write a u32 in little-endian
    bool write_u32_le(uint32_t v);
    
    // Write a u64 in little-endian
    bool write_u64_le(uint64_t v);
    
    // Write a response
    bool write_response(const ProverResponse& response);
    
private:
    int fd_;
};

} // namespace prover_server
} // namespace triton_vm

