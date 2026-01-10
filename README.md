# Triton VM Prover

High-performance C++/CUDA GPU-accelerated STARK prover for Triton VM.

## Overview

This project provides:
1. High-performance C++/CUDA implementation of the Triton VM STARK prover
2. GPU-accelerated proof generation with zero-copy memory management
3. Functional verification against test data from the Rust reference implementation
4. Hybrid CPU/GPU and full GPU execution modes for optimal performance

## Project Structure

```
triton-vm-prover/
├── CMakeLists.txt          # Build configuration
├── include/                 # Header files
│   ├── types/              # Core data types
│   │   ├── b_field_element.hpp
│   │   ├── x_field_element.hpp
│   │   └── digest.hpp
│   ├── table/              # AIR tables
│   │   └── master_table.hpp
│   ├── fri/                # FRI protocol
│   │   └── fri.hpp
│   ├── proof_stream/       # Fiat-Shamir
│   │   └── proof_stream.hpp
│   ├── stark.hpp           # Main STARK prover
│   └── test_data_loader.hpp
├── src/                    # Implementation files
│   ├── types/
│   ├── table/
│   ├── fri/
│   ├── proof_stream/
│   └── stark.cpp
└── tests/                  # Google Test unit tests
    ├── test_b_field_element.cpp
    ├── test_x_field_element.cpp
    ├── test_data_loader.cpp
    └── test_stark.cpp
```

## Features

### Core Types
- BFieldElement - Goldilocks prime field (2^64 - 2^32 + 1)
- XFieldElement - Degree-3 extension field
- Digest - 5-element hash digest

### STARK Prover Pipeline
- Trace execution and VM integration
- Main table creation, padding, and degree lowering
- Low-degree extension (LDE) with GPU acceleration
- Fiat-Shamir challenge sampling
- Auxiliary table creation and extension
- Quotient computation and evaluation
- Out-of-domain evaluation
- FRI protocol implementation
- Merkle tree construction and authentication paths
- Proof encoding and serialization

### GPU Acceleration
- CUDA kernels for field arithmetic, NTT, and hash functions
- GPU-accelerated LDE and Merkle tree construction
- Zero-copy memory management for minimal host-device transfers
- Hybrid CPU/GPU and full GPU execution modes
- Multi-GPU support for large proofs

## Building and Usage

The project includes a build script that automatically configures and builds the prover. To build and run:

```bash
./run_gpu_prover.sh spin_input21.tasm 19 --multi-gpu --cpu-aux --gpu-count=2
```

To clean and rebuild from scratch:

```bash
./run_gpu_prover.sh spin_input21.tasm 19 --multi-gpu --cpu-aux --gpu-count=2 --clean
```

The script will automatically:
- Configure CMake with CUDA support
- Build all necessary components (C++ library, Rust FFI libraries, GPU prover)
- Run the prover with the specified program and input
- Verify the proof against the Rust reference implementation

## Running the Prover Server

To run the Triton VM prover server with GPU support:

```bash
./run_rust_prover_server_gpu.sh \
    --tcp 127.0.0.1:5555 \
    --num-gpus 2 \
    --omp-threads 60 \
    --omp-init 0
```

Server options:
- `--tcp <ADDR>` - TCP address to listen on (default: 127.0.0.1:5555)
- `--num-gpus <N>` - Number of GPU devices to use
- `--omp-threads <N>` - OpenMP thread count
- `--omp-init <0|1>` - Enable/disable OpenMP init parallelization (0 = disabled, 1 = enabled)
- `--unix <PATH>` - Use Unix socket instead of TCP
- `--max-jobs <N>` - Maximum concurrent jobs (default: matches num-gpus)

### Forwarding Proof Generation to GPU Prover

To forward proof generation requests from external tools (such as xnt-core) to the GPU prover server, set the `TRITON_VM_PROVER_SOCKET` environment variable to match the server's socket address:

```bash
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
```

This tells the external tool to connect to the GPU prover server running on the specified address instead of using CPU-based proof generation. The socket address must match the `--tcp` argument used when starting the prover server.

## Performance Optimization

To optimize performance, you can set the following environment variables:

```bash
export TRITON_OMP_UPLOAD=1
export TRITON_OMP_INIT=0
export TRITON_OMP_QUOTIENT=1
export TRITON_OMP_PROCESSOR=1
export OMP_NUM_THREADS=96
export TRITON_AUX_CPU=1
export TVM_USE_TASKFLOW=1
export TVM_USE_TBB=1
export TRITON_GPU_DEGREE_LOWERING=1
export TRITON_GPU_U32=1
export TVM_USE_RUST_TRACE=1
export TRITON_GPU_USE_RAM_OVERFLOW=1
export TRITON_MULTI_GPU=0
export TRITON_GPU_COUNT=1
```

These variables control:
- OpenMP parallelization settings (`TRITON_OMP_*`, `OMP_NUM_THREADS`)
- CPU/GPU execution modes (`TRITON_AUX_CPU`, `TRITON_GPU_*`)
- Task scheduling libraries (`TVM_USE_TASKFLOW`, `TVM_USE_TBB`)
- Multi-GPU support (`TRITON_MULTI_GPU`, `TRITON_GPU_COUNT`)
- GPU memory management (`TRITON_GPU_USE_RAM_OVERFLOW` - uses system RAM as VRAM buffer)
- Rust trace integration (`TVM_USE_RUST_TRACE`)

Adjust `OMP_NUM_THREADS` based on your CPU core count for optimal performance.

## License

Apache 2.0 (matching Triton VM)

