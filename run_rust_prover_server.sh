#!/bin/bash
# Rust Prover Server for Neptune Integration
#
# This script runs the Rust prover server that accepts Neptune's JSON via socket
# and calls the GPU prover binary (triton_vm_prove_gpu_full) as a subprocess.
#
# Usage:
#   ./run_rust_prover_server.sh [--tcp <addr>] [--unix <path>]
#
# Example:
#   ./run_rust_prover_server.sh --tcp 127.0.0.1:5555

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_SERVER="$SCRIPT_DIR/rust/prover_server/target/release/prover-server"
GPU_PROVER="$SCRIPT_DIR/build/triton_vm_prove_gpu_full"

# Check if Rust server binary exists
if [ ! -f "$RUST_SERVER" ]; then
    echo "Building Rust prover server..."
    cd "$SCRIPT_DIR/rust/prover_server"
    cargo build --release
    cd "$SCRIPT_DIR"
fi

# Check if GPU prover binary exists
if [ ! -f "$GPU_PROVER" ]; then
    echo "Building GPU prover..."
    cd "$SCRIPT_DIR/build"
    if [ ! -f "Makefile" ]; then
        cmake ..
    fi
    make triton_vm_prove_gpu_full -j$(nproc)
    cd "$SCRIPT_DIR"
fi

echo "=============================================="
echo "Rust Prover Server for Neptune Integration"
echo "=============================================="
echo "  Rust server: $RUST_SERVER"
echo "  GPU prover:  $GPU_PROVER"
echo ""
echo "Mode: Rust server → GPU prover subprocess"
echo ""
echo "Environment variables:"
echo "  TRITON_GPU_PROVER_PATH=$GPU_PROVER"
echo "  RUST_LOG=info,prover_server=debug"
echo "=============================================="

# Set environment variables
export TRITON_GPU_PROVER_PATH="$GPU_PROVER"
export RUST_LOG="${RUST_LOG:-info,prover_server=info}"

# Default arguments
ARGS="--tcp 127.0.0.1:5555"

# Override with command-line arguments if provided
if [ $# -gt 0 ]; then
    ARGS="$@"
fi

echo ""
echo "Starting server with: $RUST_SERVER $ARGS"
echo ""

exec "$RUST_SERVER" $ARGS

