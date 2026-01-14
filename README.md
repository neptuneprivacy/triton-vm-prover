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

## Running with xnt-core

To run xnt-core with GPU-accelerated proof generation, simply run:

```bash
./run_xnt_core_with_gpu.sh
```

This script automatically:
- Configures all GPU and performance optimization settings
- Sets up the direct GPU prover integration (no separate server needed)
- Launches xnt-core with optimal settings for GPU acceleration

The script uses sensible defaults for most settings, but you can customize them by setting environment variables before running the script. See `run_xnt_core_with_gpu.sh` for all available configuration options.

## License

Apache 2.0 (matching Triton VM)

