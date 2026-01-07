#include "server.hpp"

#include <iostream>
#include <cstring>
#include <chrono>

#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>

namespace triton_vm {
namespace prover_server {

Server::Server() = default;

Server::~Server() {
    stop();
    
    // Clean up Unix socket file if we created one
    if (!unix_socket_path_.empty()) {
        ::unlink(unix_socket_path_.c_str());
    }
}

void Server::set_prove_handler(ProveHandler handler) {
    prove_handler_ = std::move(handler);
}

bool Server::listen_tcp(const std::string& host, uint16_t port) {
    // Create socket
    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "[server] Failed to create socket: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Set SO_REUSEADDR
    int opt = 1;
    if (::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "[server] Failed to set SO_REUSEADDR: " << strerror(errno) << std::endl;
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }
    
    // Bind
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        std::cerr << "[server] Invalid address: " << host << std::endl;
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }
    
    if (::bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[server] Failed to bind: " << strerror(errno) << std::endl;
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }
    
    // Listen
    if (::listen(server_fd_, 16) < 0) {
        std::cerr << "[server] Failed to listen: " << strerror(errno) << std::endl;
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }
    
    std::cout << "[server] Listening on " << host << ":" << port << std::endl;
    return true;
}

bool Server::listen_unix(const std::string& path) {
    // Remove existing socket file
    ::unlink(path.c_str());
    
    // Create socket
    server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "[server] Failed to create socket: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Bind
    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    
    if (::bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[server] Failed to bind: " << strerror(errno) << std::endl;
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }
    
    // Listen
    if (::listen(server_fd_, 16) < 0) {
        std::cerr << "[server] Failed to listen: " << strerror(errno) << std::endl;
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }
    
    unix_socket_path_ = path;
    std::cout << "[server] Listening on " << path << std::endl;
    return true;
}

void Server::run() {
    if (server_fd_ < 0) {
        std::cerr << "[server] Not listening" << std::endl;
        return;
    }
    
    running_.store(true);
    
    // Ignore SIGPIPE (we handle write errors explicitly)
    signal(SIGPIPE, SIG_IGN);
    
    std::cout << "[server] Ready to accept connections" << std::endl;
    
    while (running_.load()) {
        struct sockaddr_storage client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = ::accept(server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
        
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted, check running_ flag
            }
            if (!running_.load()) {
                break;  // Server stopped
            }
            std::cerr << "[server] Accept failed: " << strerror(errno) << std::endl;
            continue;
        }
        
        // Log connection
        if (client_addr.ss_family == AF_INET) {
            auto* addr = reinterpret_cast<struct sockaddr_in*>(&client_addr);
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &addr->sin_addr, ip, sizeof(ip));
            std::cout << "[server] Accepted connection from " << ip << ":" << ntohs(addr->sin_port) << std::endl;
        } else {
            std::cout << "[server] Accepted connection" << std::endl;
        }
        
        // Handle client (single-threaded for now, matching Rust implementation)
        handle_client(client_fd);
        
        ::close(client_fd);
    }
}

void Server::stop() {
    running_.store(false);
    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }
}

void Server::handle_client(int client_fd) {
    SocketReader reader(client_fd);
    SocketWriter writer(client_fd);
    
    // Read request
    std::cout << "[server] Reading request..." << std::endl;
    auto request_opt = reader.read_request();
    
    if (!request_opt) {
        std::cerr << "[server] Failed to read request" << std::endl;
        return;
    }
    
    const ProverRequest& request = *request_opt;
    
    std::cout << "[server] =========================================" << std::endl;
    std::cout << "[server] Request received: job_id=" << request.job_id << std::endl;
    std::cout << "[server] Claim JSON length: " << request.claim_json.size() << " bytes" << std::endl;
    std::cout << "[server] Program JSON length: " << request.program_json.size() << " bytes" << std::endl;
    std::cout << "[server] =========================================" << std::endl;
    
    // Process request
    ProverResponse response;
    
    if (prove_handler_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::cout << "[server] Starting proof generation pipeline..." << std::endl;
        response = prove_handler_(request);
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "[server] =========================================" << std::endl;
        if (response.status == ResponseStatus::Ok) {
            std::cout << "[server] Proof generation SUCCESS job_id=" << response.job_id
                      << " proof_size=" << response.proof_bincode.size()
                      << " duration_ms=" << duration_ms << std::endl;
        } else if (response.status == ResponseStatus::PaddedHeightTooBig) {
            std::cout << "[server] Proof generation PADDED_HEIGHT_TOO_BIG job_id=" << response.job_id
                      << " observed_log2=" << response.observed_log2 << std::endl;
        } else {
            std::cout << "[server] Proof generation ERROR job_id=" << response.job_id
                      << " error=" << response.error_message << std::endl;
        }
        std::cout << "[server] =========================================" << std::endl;
    } else {
        std::cerr << "[server] No prove handler set!" << std::endl;
        response = ProverResponse::error(request.job_id, "No prove handler configured");
    }
    
    // Send response
    std::cout << "[server] Sending response..." << std::endl;
    if (!writer.write_response(response)) {
        std::cerr << "[server] Failed to send response" << std::endl;
    }
    std::cout << "[server] Response sent, connection complete" << std::endl;
}

} // namespace prover_server
} // namespace triton_vm

