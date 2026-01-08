#!/bin/bash
# GPU Prover Server Runner for XNT-Core
#
# This script runs the GPU prover server that xnt-core connects to for parallel proof generation.
# The server should be started BEFORE starting xnt-core.
#
# Usage:
#   ./run-gpu-prover-server.sh [server-args]
#
# Example:
#   ./run-gpu-prover-server.sh --tcp 127.0.0.1:5555 --num-gpus 2
#
# Environment Variables:
#   TRITON_GPU_PROVER_PATH - Path to GPU prover binary (required for GPU acceleration)
#   OMP_NUM_THREADS - OpenMP thread count (optional)
#   TRITON_OMP_INIT - Enable/disable OpenMP init parallelization (optional)
#   TRITON_GPU_COUNT - Number of GPUs (default: 2)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XNT_CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$XNT_CORE_DIR/.." && pwd)"

# Paths
PROVER_SERVER_DIR="$WORKSPACE_DIR/triton-vm-cpp/rust/prover_server"
RUST_SERVER="$PROVER_SERVER_DIR/target/release/prover-server"
GPU_PROVER="$WORKSPACE_DIR/triton-vm-cpp/build/triton_vm_prove_gpu_full"

# Check if prover server directory exists
if [ ! -d "$PROVER_SERVER_DIR" ]; then
    echo "Error: Prover server directory not found at $PROVER_SERVER_DIR"
    echo "Please ensure triton-vm-cpp is cloned in the workspace directory."
    exit 1
fi

# Build Rust server if needed
if [ ! -f "$RUST_SERVER" ]; then
    echo "Building Rust prover server..."
    cd "$PROVER_SERVER_DIR"
    cargo build --release
    cd "$SCRIPT_DIR"
fi

# Check if GPU prover binary exists (optional)
if [ ! -f "$GPU_PROVER" ]; then
    echo "Warning: GPU prover binary not found at $GPU_PROVER"
    echo "The server will use CPU-only Rust proving (slower but still works)."
    echo "To enable GPU acceleration, build the GPU prover first."
    echo ""
    GPU_PROVER_PATH=""
else
    GPU_PROVER_PATH="$GPU_PROVER"
    echo "Found GPU prover at: $GPU_PROVER"
fi

echo "=============================================="
echo "GPU Prover Server for XNT-Core"
echo "=============================================="
echo "  Rust server: $RUST_SERVER"
if [ -n "$GPU_PROVER_PATH" ]; then
    echo "  GPU prover:  $GPU_PROVER_PATH"
else
    echo "  GPU prover:  (not found, using CPU-only)"
fi
echo ""
echo "Environment:"
if [ -n "$GPU_PROVER_PATH" ]; then
    echo "  TRITON_GPU_PROVER_PATH: $GPU_PROVER_PATH"
else
    echo "  TRITON_GPU_PROVER_PATH: (not set)"
fi
if [ -n "$OMP_NUM_THREADS" ]; then
    echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
fi
if [ -n "$TRITON_OMP_INIT" ]; then
    echo "  TRITON_OMP_INIT: $TRITON_OMP_INIT"
fi
if [ -n "$TRITON_GPU_COUNT" ]; then
    echo "  TRITON_GPU_COUNT: $TRITON_GPU_COUNT"
fi
echo "=============================================="
echo ""
echo "Server Options:"
echo "  --tcp <ADDR>      TCP address to listen on (default: 127.0.0.1:5555)"
echo "  --unix <PATH>     Unix socket path to listen on"
echo "  --max-jobs <N>    Maximum concurrent jobs (default: matches num-gpus)"
echo "  --num-gpus <N>    Number of GPU devices available (default: 2, or TRITON_GPU_COUNT env var)"
echo "  --omp-threads <N> OpenMP thread count (OMP_NUM_THREADS)"
echo "  --omp-init <0|1>  Enable/disable OpenMP init parallelization (TRITON_OMP_INIT)"
echo ""
echo "To use with xnt-core:"
echo "  1. Start this server first"
echo "  2. Set TRITON_VM_PROVER_SOCKET=127.0.0.1:5555 (or your socket address)"
echo "  3. Start xnt-core"
echo "=============================================="

# Set required environment variables
if [ -n "$GPU_PROVER_PATH" ]; then
    export TRITON_GPU_PROVER_PATH="$GPU_PROVER_PATH"
fi
export RUST_LOG="${RUST_LOG:-info,prover_server=info}"

# Default arguments
ARGS="--tcp 127.0.0.1:5555"

# Add OpenMP thread count if specified via environment variable
if [ -n "$OMP_NUM_THREADS" ]; then
    ARGS="$ARGS --omp-threads $OMP_NUM_THREADS"
fi

# Add OpenMP init setting if specified via environment variable
if [ -n "$TRITON_OMP_INIT" ]; then
    # Convert to 0/1 format
    if [ "$TRITON_OMP_INIT" = "1" ] || [ "$TRITON_OMP_INIT" = "true" ] || [ "$TRITON_OMP_INIT" = "yes" ] || [ "$TRITON_OMP_INIT" = "enabled" ]; then
        ARGS="$ARGS --omp-init 1"
    elif [ "$TRITON_OMP_INIT" = "0" ] || [ "$TRITON_OMP_INIT" = "false" ] || [ "$TRITON_OMP_INIT" = "no" ] || [ "$TRITON_OMP_INIT" = "disabled" ]; then
        ARGS="$ARGS --omp-init 0"
    fi
fi

# Override with command-line arguments if provided
if [ $# -gt 0 ]; then
    ARGS="$@"
fi

echo ""
echo "Starting server with: $RUST_SERVER $ARGS"
echo "  Environment:"
if [ -n "$GPU_PROVER_PATH" ]; then
    echo "    TRITON_GPU_PROVER_PATH=$TRITON_GPU_PROVER_PATH"
fi
if [ -n "$OMP_NUM_THREADS" ]; then
    echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS (passed to GPU prover)"
fi
if [ -n "$TRITON_OMP_INIT" ]; then
    echo "    TRITON_OMP_INIT=$TRITON_OMP_INIT (passed to GPU prover)"
fi
echo ""

# Start the server
exec "$RUST_SERVER" $ARGS

