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

## License

Apache 2.0 (matching Triton VM)

