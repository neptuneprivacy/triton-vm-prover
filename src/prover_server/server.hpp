#pragma once

/**
 * GPU Prover Server
 * 
 * TCP socket server that receives proving requests from Neptune's triton-vm-prover
 * proxy and returns GPU-generated proofs.
 * 
 * Usage:
 *   ./gpu_prover_server --tcp 127.0.0.1:5555
 *   ./gpu_prover_server --unix /tmp/gpu-prover.sock
 */

#include <string>
#include <functional>
#include <atomic>
#include <memory>

#include "protocol.hpp"

namespace triton_vm {
namespace prover_server {

// Callback type for handling proving requests
using ProveHandler = std::function<ProverResponse(const ProverRequest&)>;

class Server {
public:
    Server();
    ~Server();
    
    // Set the proving handler
    void set_prove_handler(ProveHandler handler);
    
    // Start listening on TCP
    bool listen_tcp(const std::string& host, uint16_t port);
    
    // Start listening on Unix socket
    bool listen_unix(const std::string& path);
    
    // Run the server (blocks until stop() is called)
    void run();
    
    // Stop the server
    void stop();
    
    // Check if server is running
    bool is_running() const { return running_.load(); }
    
private:
    void handle_client(int client_fd);
    
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    ProveHandler prove_handler_;
    std::string unix_socket_path_;  // For cleanup
};

} // namespace prover_server
} // namespace triton_vm

