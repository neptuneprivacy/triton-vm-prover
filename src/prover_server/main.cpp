/**
 * GPU Prover Server - Main Entry Point
 * 
 * TCP socket server that receives proving requests from Neptune's triton-vm-prover
 * proxy and returns GPU-generated proofs.
 * 
 * Usage:
 *   ./gpu_prover_server --tcp 127.0.0.1:5555
 *   ./gpu_prover_server --unix /tmp/gpu-prover.sock
 *   
 * Environment Variables:
 *   TRITON_GPU_PROVER_THREADS - Number of OpenMP threads (default: auto)
 *   TVM_DEBUG - Enable debug output
 */

#include <iostream>
#include <string>
#include <cstring>
#include <csignal>

#include "server.hpp"
#include "prover.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace triton_vm::prover_server;

// Global server pointer for signal handling
static Server* g_server = nullptr;

void signal_handler(int sig) {
    std::cout << "\n[main] Received signal " << sig << ", stopping server..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [OPTIONS]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --tcp HOST:PORT    Listen on TCP socket (e.g., 127.0.0.1:5555)" << std::endl;
    std::cerr << "  --unix PATH        Listen on Unix socket (e.g., /tmp/gpu-prover.sock)" << std::endl;
    std::cerr << "  --help             Show this help message" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Environment Variables:" << std::endl;
    std::cerr << "  TRITON_GPU_PROVER_THREADS  Number of OpenMP threads" << std::endl;
    std::cerr << "  TVM_DEBUG                  Enable debug output" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Prover Server for Neptune" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configure OpenMP
#ifdef _OPENMP
    const char* threads_env = std::getenv("TRITON_GPU_PROVER_THREADS");
    if (threads_env) {
        int threads = std::atoi(threads_env);
        if (threads > 0) {
            omp_set_num_threads(threads);
        }
    }
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif

    // Show GPU prover path
    const char* gpu_prover_path = std::getenv("TRITON_GPU_PROVER_PATH");
    if (gpu_prover_path) {
        std::cout << "GPU Prover: " << gpu_prover_path << std::endl;
    } else {
        std::cout << "GPU Prover: (auto-detect from build dir)" << std::endl;
    }
    std::cout << "Mode: Calls triton_vm_prove_gpu_full (GPU STARK prover)" << std::endl;
    
    std::cout << std::endl;
    
    // Parse command line arguments
    std::string mode;
    std::string address;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--tcp") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --tcp requires HOST:PORT argument" << std::endl;
                return 1;
            }
            mode = "tcp";
            address = argv[++i];
        } else if (arg == "--unix") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --unix requires PATH argument" << std::endl;
                return 1;
            }
            mode = "unix";
            address = argv[++i];
        } else {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Default to TCP if not specified
    if (mode.empty()) {
        mode = "tcp";
        address = "127.0.0.1:5555";
        std::cout << "No mode specified, using default: --tcp " << address << std::endl;
    }
    
    // Create server
    Server server;
    g_server = &server;
    
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Set prove handler
    server.set_prove_handler([](const ProverRequest& request) {
        return prove_request(request);
    });
    
    // Start listening
    bool ok = false;
    if (mode == "tcp") {
        // Parse HOST:PORT
        size_t colon = address.find(':');
        if (colon == std::string::npos) {
            std::cerr << "Error: Invalid TCP address, expected HOST:PORT" << std::endl;
            return 1;
        }
        std::string host = address.substr(0, colon);
        uint16_t port = static_cast<uint16_t>(std::stoi(address.substr(colon + 1)));
        ok = server.listen_tcp(host, port);
    } else if (mode == "unix") {
        ok = server.listen_unix(address);
    }
    
    if (!ok) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "Server ready. Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;
    
    // Run server (blocks until stopped)
    server.run();
    
    std::cout << "Server stopped." << std::endl;
    return 0;
}

