# Triton VM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![crates.io](https://img.shields.io/crates/v/triton-vm.svg)](https://crates.io/crates/triton-vm)
[![GitHub CI](https://github.com/TritonVM/triton-vm/actions/workflows/main.yml/badge.svg)](https://github.com/TritonVM/triton-vm/actions)
[![Check Links](https://github.com/TritonVM/triton-vm/actions/workflows/link_checker.yml/badge.svg)](https://github.com/TritonVM/triton-vm/actions/workflows/link_checker.yml)
[![Spec: online](https://img.shields.io/badge/Spec-online-success)](https://triton-vm.org/spec/)
[![Coverage Status](https://coveralls.io/repos/github/TritonVM/triton-vm/badge.svg?branch=master)](https://coveralls.io/github/TritonVM/triton-vm?branch=master)

Triton is a virtual machine that comes with Algebraic Execution Tables (AET) and Arithmetic
Intermediate Representations (AIR) for use in combination with
a [STARK proof system](https://aszepieniec.github.io/stark-anatomy/). It defines a Turing
complete [Instruction Set Architecture](https://triton-vm.org/spec/isa.html), as well as the
corresponding [arithmetization](https://triton-vm.org/spec/arithmetization.html) of the VM. The
really cool thing about Triton VM is its efficient _recursive_ verification of the STARKs produced
when running Triton VM.

## GPU Acceleration

This fork of Triton VM includes CUDA GPU acceleration for significantly faster proof generation.

### Pre-built Kernels

Pre-compiled CUDA kernels (`.cubin` files) are included in the `built_kernels/` directory. These are ready to use and support NVIDIA GPUs with compute capability sm_100 (B200) or sm_89 (RTX 4090).

### Rebuilding CUDA Kernels

If you need to recompile the CUDA kernels for a different GPU architecture:

1. Ensure you have the CUDA toolkit installed (nvcc)
2. First run a release build to generate the required header file:
   ```bash
   cargo build --release
   ```
3. Edit `build_cuda_kernels.sh` and set `ARCH` to your target architecture (e.g., `sm_89` for RTX 4090, `sm_100` for B200)
4. Run the build script:
   ```bash
   ./build_cuda_kernels.sh
   ```

The CUDA source files are located in `twenty-first-local/twenty-first/cuda/`.

### Using This Fork in the Core/Composer

To use this GPU-accelerated version of Triton VM, update the dependency in your `Cargo.toml` to point to this repository instead of the upstream crates.io version:

```toml
[dependencies]
triton-vm = { git = "https://github.com/neptuneprivacy/triton-vm", branch = "master" }
```

Or if using a local path:

```toml
[dependencies]
triton-vm = { path = "../path/to/triton-vm-public/triton-vm" }
```

## Building xnt-core with this Triton VM

Pre-built binaries are available in the `xnt-core-build/` directory.

To build xnt-core (neptune-privacy) using this GPU-accelerated Triton VM from source:

### Prerequisites

- Rust toolchain (edition 2024, rustc 1.85+)
- CUDA toolkit (for GPU acceleration)
- CMake and C++ compiler

### Build Instructions

1. Clone xnt-core alongside this repository:
   ```
   /your-workspace/
   ├── triton-vm/   (this repository)
   └── xnt-core/
   ```

2. Add the following `[patch.crates-io]` section to xnt-core's `Cargo.toml`:
   ```toml
   [patch.crates-io]
   triton-vm = { path = "../triton-vm/triton-vm" }
   triton-air = { path = "../triton-vm/triton-air" }
   triton-isa = { path = "../triton-vm/triton-isa" }
   triton-constraint-builder = { path = "../triton-vm/triton-constraint-builder" }
   triton-constraint-circuit = { path = "../triton-vm/triton-constraint-circuit" }
   twenty-first = { path = "../triton-vm/twenty-first-local/twenty-first" }
   bfieldcodec_derive = { path = "../triton-vm/twenty-first-local/bfieldcodec_derive" }
   ```

3. Build in release mode:
   ```bash
   cd xnt-core
   CXXFLAGS="-Wno-unknown-pragmas" cargo build --release -p neptune-privacy
   ```

4. The binary will be at `target/release/xnt-core`

## Getting Started

If you want to start writing programs for Triton VM, check out the
[Triton TUI](https://github.com/TritonVM/triton-tui). If you want to run a Triton assembly program,
generate or verify proofs of correct execution, or
[profile the performance](https://github.com/TritonVM/triton-cli#profiling) of either of these,
check out the [Triton CLI](https://github.com/TritonVM/triton-cli). If you want to use Triton VM as
a library, check out the [examples](triton-vm/examples).

## Project Status

We consider Triton VM to be usable in production.
That said, there are plans to change some of the internals significantly.
Some of these changes might have effects on the interface as well as the ISA.

As is common for rust projects, Triton VM follows
[Cargo-style SemVer](https://doc.rust-lang.org/cargo/reference/semver.html).

## Specification

Triton VM’s specification can be found [online](https://triton-vm.org/spec/).
Alternatively, you can [self-host](specification/README.md) it.

## Recursive STARKs of Computational Integrity

Normally, when executing a machine – virtual or not – the flow of information can be regarded as
follows. The tuple of (`input`, `program`) is given to the machine, which takes the `program`,
evaluates it on the `input`, and produces some `output`.

![](./specification/src/img/recursive-1.svg)

If the – now almost definitely virtual – machine also has an associated STARK engine, one additional
output is a `proof` of computational integrity.

![](./specification/src/img/recursive-2.svg)

Only if `input`, `program`, and `output` correspond to one another, i.e., if `output` is indeed the
result of evaluating the `program` on the `input` according to the rules defined by the virtual
machine, then producing such a `proof` is easy. Otherwise, producing a `proof` is next to
impossible.

The routine that checks whether a `proof` is, in fact, a valid one, is called the Verifier. It takes
as input a 4-tuple (`input`, `program`, `output`, `proof`) and evaluates to `true` if and only if
that 4-tuple is consistent with the rules of the virtual machine.

![](./specification/src/img/recursive-3.svg)

Since the Verifier is a program taking some input and producing some output, the original virtual
machine can be used to perform the computation.

![](./specification/src/img/recursive-4.svg)

The associated STARK engine will then produce a proof of computational integrity of _verifying_ some
other proof of computational integrity – recursion!
Of course, the Verifier can be a subroutine in a larger program.

Triton VM is specifically designed to allow fast recursive verification.
