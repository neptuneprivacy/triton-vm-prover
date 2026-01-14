#include "prover.hpp"
#include "common/debug_control.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <array>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <set>
#include <map>

#include <unistd.h>
#include <sys/wait.h>
#include <cmath>
#include <regex>

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "stark.hpp"
#include "bincode_ffi.hpp"

namespace triton_vm {
namespace prover_server {

namespace {

double elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Get path to GPU prover binary
std::string get_gpu_prover_path() {
    // Check environment variable first
    const char* env_path = std::getenv("TRITON_GPU_PROVER_PATH");
    if (env_path && std::filesystem::exists(env_path)) {
        return env_path;
    }
    
    // Try relative to current executable
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        
        // Try same directory
        auto prover_path = exe_dir / "triton_vm_prove_gpu_full";
        if (std::filesystem::exists(prover_path)) {
            return prover_path.string();
        }
        
        // Try build directory
        prover_path = exe_dir / "../build/triton_vm_prove_gpu_full";
        if (std::filesystem::exists(prover_path)) {
            return prover_path.string();
        }
    }
    
    // Default path
    return "./triton_vm_prove_gpu_full";
}

// Convert instruction name to lowercase TASM format
std::string instruction_name_to_tasm(const std::string& name) {
    std::string result = name;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

// Convert string argument (like "ST0", "N1") to numeric value
// Returns -1 if conversion fails
int64_t parse_string_argument(const std::string& arg) {
    // Handle OpStackElement format: "ST0", "ST1", etc.
    if (arg.size() >= 3 && arg.substr(0, 2) == "ST") {
        try {
            return std::stoll(arg.substr(2));
        } catch (...) {
            return -1;
        }
    }
    // Handle NumberOfWords format: "N1", "N2", etc.
    if (arg.size() >= 2 && arg[0] == 'N') {
        try {
            return std::stoll(arg.substr(1));
        } catch (...) {
            return -1;
        }
    }
    return -1;
}

// Check if instruction has an argument (affects bword count)
// In Triton VM, instructions with args take 2 bwords (opcode + arg)
int instruction_size(const std::string& opcode) {
    // Instructions that take an argument (size = 2 bwords)
    static const std::set<std::string> with_arg = {
        "push", "pop", "divine", "pick", "place", "dup", "swap",
        "call", "readmem", "writemem", "addi", "readio", "writeio"
    };
    std::string lower = opcode;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return with_arg.count(lower) > 0 ? 2 : 1;
}

// Write Neptune JSON program to TASM file format
// IMPORTANT: In the Neptune JSON, instructions array has each instruction repeated by its size!
// A 2-bword instruction (like Push) appears TWICE consecutively in the array.
// The index in the array IS the bword address.
// We need to:
// 1. Skip duplicate entries (only output each unique instruction once)
// 2. Put labels at the correct positions
bool write_program_to_tasm(const std::string& program_json, const std::string& tasm_path) {
    try {
        auto json = nlohmann::json::parse(program_json);
        
        std::ofstream out(tasm_path);
        if (!out.is_open()) {
            std::cerr << "[prover] Failed to open TASM file for writing: " << tasm_path << std::endl;
            return false;
        }
        
        // Build bword_address -> label_name map from address_to_label
        std::map<uint64_t, std::string> bword_to_label;
        std::map<uint64_t, std::string> call_addr_to_label;  // For Call instruction targets
        if (json.contains("address_to_label") && json["address_to_label"].is_object()) {
            for (auto& [addr_str, label] : json["address_to_label"].items()) {
                uint64_t addr = std::stoull(addr_str);
                std::string label_name = label.get<std::string>();
                bword_to_label[addr] = label_name;
                call_addr_to_label[addr] = label_name;
            }
        }
        
        if (!json.contains("instructions") || !json["instructions"].is_array()) {
            std::cerr << "[prover] No instructions array in program JSON" << std::endl;
            return false;
        }
        
        const auto& instructions = json["instructions"];
        size_t total_bwords = instructions.size();  // In Neptune's format, array length = total bwords
        
        // Process instructions - the bword index IS the array index
        size_t bword_idx = 0;
        while (bword_idx < total_bwords) {
            const auto& instr = instructions[bword_idx];
            
            // Check if there's a label at this bword address
            auto label_it = bword_to_label.find(bword_idx);
            if (label_it != bword_to_label.end()) {
                out << label_it->second << ":\n";
            }
            
            std::string opcode;
            int size = 1;
            
            if (instr.is_string()) {
                // Simple instruction like "Halt", "Return", etc. (size = 1)
                opcode = instr.get<std::string>();
                size = 1;
                out << instruction_name_to_tasm(opcode) << "\n";
            } else if (instr.is_object()) {
                // Complex instruction like {"Push": value} or {"Dup": "ST0"}
                for (auto& [key, val] : instr.items()) {
                    opcode = key;
                    size = instruction_size(key);
                    
                    std::string opcode_lower = instruction_name_to_tasm(key);
                    out << opcode_lower;
                    
                    // Special handling for Call - use label if available
                    if (opcode_lower == "call" && val.is_number()) {
                        uint64_t call_target = val.get<uint64_t>();
                        auto target_label = call_addr_to_label.find(call_target);
                        if (target_label != call_addr_to_label.end()) {
                            out << " " << target_label->second;
                        } else {
                            out << " " << call_target;
                        }
                    } else if (val.is_number()) {
                        // Numeric argument - write as-is
                        uint64_t num_val = val.get<uint64_t>();
                        out << " " << num_val;
                    } else if (val.is_string()) {
                        // String argument like "ST0", "N1" - convert to number
                        std::string str_val = val.get<std::string>();
                        int64_t num_arg = parse_string_argument(str_val);
                        if (num_arg >= 0) {
                            out << " " << num_arg;
                        } else {
                            out << " " << str_val;
                        }
                    } else if (val.is_array()) {
                        for (const auto& v : val) {
                            if (v.is_number()) {
                                out << " " << v.get<uint64_t>();
                            } else if (v.is_string()) {
                                std::string str_val = v.get<std::string>();
                                int64_t num_arg = parse_string_argument(str_val);
                                if (num_arg >= 0) {
                                    out << " " << num_arg;
                                } else {
                                    out << " " << str_val;
                                }
                            }
                        }
                    }
                    out << "\n";
                    break;  // Only process first key in object
                }
            }
            
            // Skip to next instruction (skip the duplicated entries for size-2 instructions)
            bword_idx += size;
        }
        
        out.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[prover] Failed to write TASM: " << e.what() << std::endl;
        return false;
    }
}

// Format public input as comma-separated string
std::string format_public_input(const std::string& claim_json) {
    try {
        auto json = nlohmann::json::parse(claim_json);
        
        if (!json.contains("input") || !json["input"].is_array()) {
            return "";
        }
        
        std::stringstream ss;
        bool first = true;
        for (const auto& elem : json["input"]) {
            if (!first) ss << ",";
            first = false;
            
            if (elem.is_number()) {
                ss << elem.get<uint64_t>();
            } else if (elem.is_string()) {
                ss << elem.get<std::string>();
            }
        }
        
        return ss.str();
    } catch (...) {
        return "";
    }
}

// Read binary file into vector
std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return {};
    }
    
    return buffer;
}

} // namespace

std::optional<Claim> parse_claim_json(const std::string& json_str) {
    try {
        auto json = nlohmann::json::parse(json_str);
        
        Claim claim;
        
        // Parse program_digest (hex string)
        std::string digest_hex = json["program_digest"].get<std::string>();
        
        if (digest_hex.length() != 80) {
            std::cerr << "[prover] Invalid digest hex length: " << digest_hex.length() << std::endl;
            return std::nullopt;
        }
        
        std::array<BFieldElement, 5> elements;
        for (int i = 0; i < 5; ++i) {
            std::string chunk = digest_hex.substr(i * 16, 16);
            uint64_t value = std::stoull(chunk, nullptr, 16);
            elements[i] = BFieldElement(value);
        }
        claim.program_digest = Digest(elements);
        
        claim.version = json["version"].get<uint32_t>();
        
        if (json.contains("input") && json["input"].is_array()) {
            for (const auto& elem : json["input"]) {
                if (elem.is_number()) {
                    claim.input.push_back(BFieldElement(elem.get<uint64_t>()));
                } else if (elem.is_string()) {
                    claim.input.push_back(BFieldElement(std::stoull(elem.get<std::string>())));
                }
            }
        }
        
        if (json.contains("output") && json["output"].is_array()) {
            for (const auto& elem : json["output"]) {
                if (elem.is_number()) {
                    claim.output.push_back(BFieldElement(elem.get<uint64_t>()));
                } else if (elem.is_string()) {
                    claim.output.push_back(BFieldElement(std::stoull(elem.get<std::string>())));
                }
            }
        }
        
        return claim;
    } catch (const std::exception& e) {
        std::cerr << "[prover] Failed to parse claim JSON: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::optional<uint8_t> parse_max_log2_json(const std::string& json_str) {
    try {
        auto json = nlohmann::json::parse(json_str);
        if (json.is_null()) {
            return std::nullopt;
        }
        return json.get<uint8_t>();
    } catch (...) {
        return std::nullopt;
    }
}

ProverResponse prove_request(const ProverRequest& request) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "[prover] =========================================" << std::endl;
    std::cout << "[prover] Starting prove job_id=" << request.job_id << std::endl;
    std::cout << "[prover] Claim JSON length: " << request.claim_json.size() << " bytes" << std::endl;
    std::cout << "[prover] Program JSON length: " << request.program_json.size() << " bytes" << std::endl;
    std::cout << "[prover] NonDet JSON length: " << request.nondet_json.size() << " bytes" << std::endl;
    
    // Create temp directory
    //
    // By default we delete this directory at the end of the request. When debugging nondeterminism
    // mismatches between GPU/C++ and Rust, it is extremely helpful to preserve the exact inputs
    // and outputs (program.tasm, program.json, nondet.json, output.claim, output.proof).
    //
    // Set TRITON_GPU_PROVER_PRESERVE_TMP=1 to keep the directory (both on success and failure).
    const bool preserve_tmp_dir = (std::getenv("TRITON_GPU_PROVER_PRESERVE_TMP") != nullptr);

    std::string temp_dir = "/tmp/gpu_prover_" + std::to_string(getpid()) + "_" + std::to_string(request.job_id);
    std::filesystem::create_directories(temp_dir);
    
    std::string tasm_path = temp_dir + "/program.tasm";
    std::string claim_json_path = temp_dir + "/claim.json";
    std::string claim_path = temp_dir + "/output.claim";
    std::string proof_path = temp_dir + "/output.proof";
    std::string nondet_path = temp_dir + "/nondet.json";
    std::string program_json_path = temp_dir + "/program.json";
    
    // Write program to TASM file
    std::cout << "[prover] Writing program to TASM..." << std::endl;
    if (!write_program_to_tasm(request.program_json, tasm_path)) {
        if (!preserve_tmp_dir) std::filesystem::remove_all(temp_dir);
        return ProverResponse::error(request.job_id, "Failed to write program to TASM format");
    }

    // Write Claim JSON to file (for reproducibility/debugging)
    {
        std::ofstream claim_file(claim_json_path);
        if (!claim_file.is_open()) {
            if (!preserve_tmp_dir) std::filesystem::remove_all(temp_dir);
            return ProverResponse::error(request.job_id, "Failed to write Claim JSON");
        }
        claim_file << request.claim_json;
        claim_file.close();
    }
    
    // Write NonDeterminism JSON to file (for Rust FFI trace execution)
    {
        std::ofstream nondet_file(nondet_path);
        if (!nondet_file.is_open()) {
            if (!preserve_tmp_dir) std::filesystem::remove_all(temp_dir);
            return ProverResponse::error(request.job_id, "Failed to write NonDeterminism JSON");
        }
        nondet_file << request.nondet_json;
        nondet_file.close();
    }
    
    // Write Program JSON to file (for Rust FFI trace execution)
    {
        std::ofstream program_file(program_json_path);
        if (!program_file.is_open()) {
            std::filesystem::remove_all(temp_dir);
            return ProverResponse::error(request.job_id, "Failed to write Program JSON");
        }
        program_file << request.program_json;
        program_file.close();
    }
    
    // Get public input
    std::string public_input = format_public_input(request.claim_json);
    std::cout << "[prover] Public input: " << (public_input.empty() ? "(none)" : public_input) << std::endl;
    
    // Get GPU prover path
    std::string gpu_prover = get_gpu_prover_path();
    std::cout << "[prover] GPU prover: " << gpu_prover << std::endl;
    
    if (!std::filesystem::exists(gpu_prover)) {
        std::cerr << "[prover] GPU prover not found at: " << gpu_prover << std::endl;
        std::cerr << "[prover] Set TRITON_GPU_PROVER_PATH environment variable" << std::endl;
        std::filesystem::remove_all(temp_dir);
        return ProverResponse::error(request.job_id, "GPU prover binary not found");
    }
    
    // Check if NonDeterminism has RAM or secret input (requires Rust FFI)
    bool has_nondet = request.nondet_json.size() > 10;  // More than just empty object "{}"
    
    // Build command
    std::cout << "[prover] =========================================" << std::endl;
    std::cout << "[prover] Running GPU prover..." << std::endl;
    if (has_nondet) {
        std::cout << "[prover] NonDeterminism detected (" << request.nondet_json.size() 
                  << " bytes), using Rust FFI for trace" << std::endl;
    }
    std::cout << "[prover] Command: " << gpu_prover << " " << tasm_path << " " 
              << public_input << " " << claim_path << " " << proof_path;
    if (has_nondet) {
        std::cout << " " << nondet_path << " " << program_json_path;
    }
    std::cout << std::endl;
    std::cout << "[prover] =========================================" << std::endl;
    
    auto prove_start = std::chrono::high_resolution_clock::now();
    
    // Create pipes to capture stdout and stderr from GPU prover
    int stdout_pipe[2], stderr_pipe[2];
    if (pipe(stdout_pipe) < 0 || pipe(stderr_pipe) < 0) {
        if (pipe(stdout_pipe) >= 0) {
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
        }
        if (pipe(stderr_pipe) >= 0) {
            close(stderr_pipe[0]);
            close(stderr_pipe[1]);
        }
        std::filesystem::remove_all(temp_dir);
        return ProverResponse::error(request.job_id, "Failed to create pipes for GPU prover output");
    }
    
    // Fork and exec the GPU prover
    pid_t pid = fork();
    if (pid < 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        std::filesystem::remove_all(temp_dir);
        return ProverResponse::error(request.job_id, "Failed to fork GPU prover process");
    }
    
    if (pid == 0) {
        // Child process - exec GPU prover
        // Redirect stdout and stderr to pipes
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        
        // Set environment variables for GPU prover
        setenv("TRITON_FIXED_SEED", "1", 1);  // Use fixed seed for deterministic proofs
        
        // Build argument list - conditionally include NonDet paths
        if (has_nondet) {
            execl(gpu_prover.c_str(), "triton_vm_prove_gpu_full",
                  tasm_path.c_str(),
                  public_input.c_str(),
                  claim_path.c_str(),
                  proof_path.c_str(),
                  nondet_path.c_str(),
                  program_json_path.c_str(),
                  nullptr);
        } else {
            execl(gpu_prover.c_str(), "triton_vm_prove_gpu_full",
                  tasm_path.c_str(),
                  public_input.c_str(),
                  claim_path.c_str(),
                  proof_path.c_str(),
                  nullptr);
        }
        
        // If exec fails
        std::cerr << "[prover] Failed to exec GPU prover: " << strerror(errno) << std::endl;
        _exit(1);
    }
    
    // Parent process - close write ends of pipes
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);
    
    // Read stdout and stderr output to capture padded height
    std::string output;
    char buffer[4096];
    ssize_t n;
    
    // Read from stdout
    while ((n = read(stdout_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
        buffer[n] = '\0';
        output += buffer;
    }
    close(stdout_pipe[0]);
    
    // Read from stderr
    while ((n = read(stderr_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
        buffer[n] = '\0';
        output += buffer;
    }
    close(stderr_pipe[0]);
    
    // Wait for child process
    int status;
    waitpid(pid, &status, 0);
    
    double prove_time = elapsed_ms(prove_start);
    
    // Parse padded height from output and check for log2=21
    std::regex padded_height_regex(R"(Padded height:\s*(\d+))");
    std::smatch match;
    if (std::regex_search(output, match, padded_height_regex)) {
        try {
            uint64_t padded_height = std::stoull(match[1].str());
            uint8_t log2_padded_height = static_cast<uint8_t>(std::log2(padded_height));
            
            if (log2_padded_height == 21) {
                // Red warning for log2=21 (very large proof: 2^21 = 2,097,152 rows)
                std::cerr << "\033[1;31m"  // Red color
                          << "[prover] ⚠️  WARNING: Padded height log2=21 detected! "
                          << "This is a VERY LARGE proof (2^21 = 2,097,152 rows, actual: " 
                          << padded_height << " rows). "
                          << "Proof generation may take significant time and memory."
                          << "\033[0m"  // Reset color
                          << std::endl;
            }
        } catch (...) {
            // Ignore parsing errors
        }
    }
    
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        std::cerr << "[prover] GPU prover failed with exit code: " << exit_code << std::endl;
        
        // Log captured output for debugging
        if (!output.empty()) {
            std::cerr << "[prover] GPU prover output:" << std::endl;
            std::cerr << output << std::endl;
        }
        
        // For debugging: save the failed TASM and JSON to /tmp for inspection
        std::string debug_dir = "/tmp/gpu_prover_failed_" + std::to_string(request.job_id);
        try {
            std::filesystem::create_directories(debug_dir);
            std::filesystem::copy(tasm_path, debug_dir + "/program.tasm", std::filesystem::copy_options::overwrite_existing);
            
            // Also save the original JSON inputs
            std::ofstream claim_file(debug_dir + "/claim.json");
            claim_file << request.claim_json;
            claim_file.close();
            
            std::ofstream program_file(debug_dir + "/program.json");
            program_file << request.program_json;
            program_file.close();

            std::ofstream nondet_file(debug_dir + "/nondet.json");
            nondet_file << request.nondet_json;
            nondet_file.close();

            // If the prover managed to produce any outputs, preserve those too.
            if (std::filesystem::exists(claim_path)) {
                std::filesystem::copy(claim_path, debug_dir + "/output.claim", std::filesystem::copy_options::overwrite_existing);
            }
            if (std::filesystem::exists(proof_path)) {
                std::filesystem::copy(proof_path, debug_dir + "/output.proof", std::filesystem::copy_options::overwrite_existing);
            }
            
            std::cerr << "[prover] Debug files saved to: " << debug_dir << std::endl;
        } catch (...) {
            // Ignore debug file save errors
        }
        
        // Extract the most relevant error message from output
        std::string error_msg = "GPU prover failed with exit code " + std::to_string(exit_code);
        
        // Look for specific error patterns and include them in the error message
        if (output.find("OpStack underflow") != std::string::npos) {
            error_msg = "OpStack underflow: Program execution failed";
        } else if (output.find("Error:") != std::string::npos) {
            // Extract the error line
            size_t error_pos = output.find("Error:");
            size_t newline_pos = output.find('\n', error_pos);
            if (newline_pos != std::string::npos) {
                error_msg = output.substr(error_pos, newline_pos - error_pos);
            } else {
                error_msg = output.substr(error_pos);
            }
        }
        
        if (!preserve_tmp_dir) {
            std::filesystem::remove_all(temp_dir);
        } else {
            std::cerr << "[prover] TRITON_GPU_PROVER_PRESERVE_TMP=1: preserving temp dir: " << temp_dir << std::endl;
        }
        return ProverResponse::error(request.job_id, error_msg);
    }
    
    TRITON_PROFILE_COUT("[prover] GPU prover completed in " << prove_time << " ms" << std::endl);
    
    // Check if proof file exists
    if (!std::filesystem::exists(proof_path)) {
        std::cerr << "[prover] Proof file not found: " << proof_path << std::endl;
        std::filesystem::remove_all(temp_dir);
        return ProverResponse::error(request.job_id, "GPU prover did not generate proof file");
    }
    
    // Read proof file
    std::cout << "[prover] Reading proof file..." << std::endl;
    std::vector<uint8_t> proof_bytes = read_binary_file(proof_path);
    
    if (proof_bytes.empty()) {
        std::cerr << "[prover] Failed to read proof file or proof is empty" << std::endl;
        std::filesystem::remove_all(temp_dir);
        return ProverResponse::error(request.job_id, "Failed to read proof file");
    }
    
    // Get proof size for logging
    auto proof_size_kb = proof_bytes.size() / 1024.0;
    
    // Cleanup temp files
    if (!preserve_tmp_dir) {
        std::filesystem::remove_all(temp_dir);
    } else {
        std::cerr << "[prover] TRITON_GPU_PROVER_PRESERVE_TMP=1: preserving temp dir: " << temp_dir << std::endl;
    }
    
    double total_time = elapsed_ms(total_start);
    
    std::cout << "[prover] =========================================" << std::endl;
    std::cout << "[prover] SUCCESS: proof_size=" << proof_bytes.size() 
              << " bytes (" << proof_size_kb << " KB)" << std::endl;
    std::cout << "[prover] GPU prove time: " << prove_time << " ms" << std::endl;
    std::cout << "[prover] Total time: " << total_time << " ms" << std::endl;
    std::cout << "[prover] =========================================" << std::endl;
    
    return ProverResponse::ok(request.job_id, std::move(proof_bytes));
}

} // namespace prover_server
} // namespace triton_vm
