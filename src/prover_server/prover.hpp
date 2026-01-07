#pragma once

/**
 * Prover Logic
 * 
 * Parses Neptune's JSON inputs, runs the GPU prover, and returns bincode proof.
 */

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

#include "protocol.hpp"
#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include "stark.hpp"

namespace triton_vm {
namespace prover_server {

/**
 * Parse Claim from Neptune's JSON format
 */
std::optional<Claim> parse_claim_json(const std::string& json);

/**
 * Parse max_log2_padded_height from JSON (may be null)
 */
std::optional<uint8_t> parse_max_log2_json(const std::string& json);

/**
 * Main prover function
 * 
 * Takes a proving request from the socket, parses inputs, runs GPU prover,
 * and returns bincode-serialized proof.
 */
ProverResponse prove_request(const ProverRequest& request);

} // namespace prover_server
} // namespace triton_vm

