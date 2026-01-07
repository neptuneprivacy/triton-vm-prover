#!/bin/bash
# Start GPU Prover Server with Multi-GPU Support
# 
# This script starts the Rust prover server configured for:
# - 2 GPUs (H200 setup)
# - Hybrid CPU/GPU routing (proof collections on CPU, single proofs on GPU)
# - TCP socket on 127.0.0.1:5555

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_SERVER="$SCRIPT_DIR/rust/prover_server/target/release/prover-server"
GPU_PROVER="${TRITON_GPU_PROVER_PATH:-$SCRIPT_DIR/build/triton_vm_prove_gpu_full}"

# Check if Rust server exists
if [ ! -f "$RUST_SERVER" ]; then
    echo "Building Rust prover server..."
    cd "$SCRIPT_DIR/rust/prover_server"
    cargo build --release
    cd "$SCRIPT_DIR"
fi

# Set GPU prover path if binary exists
if [ -f "$GPU_PROVER" ]; then
    export TRITON_GPU_PROVER_PATH="$GPU_PROVER"
    echo "✓ GPU prover found: $GPU_PROVER"
else
    echo "⚠ GPU prover not found at $GPU_PROVER"
    echo "  Server will use Rust proving (CPU-only mode)"
    echo "  To enable GPU: Build triton_vm_prove_gpu_full and set TRITON_GPU_PROVER_PATH"
fi

# Set logging
export RUST_LOG="${RUST_LOG:-info,prover_server=info}"

# Default: 2 GPUs for H200 setup
NUM_GPUS="${TRITON_GPU_COUNT:-2}"

echo "=============================================="
echo "Starting GPU Prover Server"
echo "=============================================="
echo "  Server: $RUST_SERVER"
echo "  TCP:    127.0.0.1:5555"
echo "  GPUs:   $NUM_GPUS"
echo "  Mode:   Hybrid CPU/GPU (if GPU available)"
echo "=============================================="
echo ""

# Start server with multi-GPU support
exec "$RUST_SERVER" --tcp 127.0.0.1:5555 --num-gpus "$NUM_GPUS"
