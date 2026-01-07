#!/bin/bash
# Rust Prover Server with GPU Mode
#
# This script ensures TRITON_FORCE_RUST_PROVER is unset and GPU prover is enabled

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
    echo "Error: GPU prover binary not found at $GPU_PROVER"
    echo "Build it with: cd build && make triton_vm_prove_gpu_full"
    exit 1
fi

echo "=============================================="
echo "Rust Prover Server - GPU MODE"
echo "=============================================="
echo "  Rust server: $RUST_SERVER"
echo "  GPU prover:  $GPU_PROVER"
echo ""
echo "Environment:"
echo "  TRITON_FORCE_RUST_PROVER: <explicitly unset>"
echo "  TRITON_GPU_PROVER_PATH: $GPU_PROVER"
if [ -n "$OMP_NUM_THREADS" ]; then
    echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
fi
if [ -n "$TRITON_OMP_INIT" ]; then
    echo "  TRITON_OMP_INIT: $TRITON_OMP_INIT"
fi
echo "=============================================="
echo ""
echo "Usage:"
echo "  $0 [server-args"
echo "  OR"
echo "  OMP_NUM_THREADS=60 TRITON_OMP_INIT=0 $0 [server-args]"
echo ""
echo "Server options:"
echo "  --omp-threads <N>  Set OpenMP thread count"
echo "  --omp-init <0|1>   Enable/disable OpenMP init parallelization"
echo "=============================================="

# Set required environment variables
export TRITON_GPU_PROVER_PATH="$GPU_PROVER"
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
echo "    TRITON_GPU_PROVER_PATH=$TRITON_GPU_PROVER_PATH"
echo "    TRITON_FORCE_RUST_PROVER=<unset>"
if [ -n "$OMP_NUM_THREADS" ]; then
    echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS (passed to GPU prover)"
fi
if [ -n "$TRITON_OMP_INIT" ]; then
    echo "    TRITON_OMP_INIT=$TRITON_OMP_INIT (passed to GPU prover)"
fi
echo ""

# Use env -u to explicitly remove TRITON_FORCE_RUST_PROVER before starting
# This ensures it's not in the environment passed to the server process
exec env -u TRITON_FORCE_RUST_PROVER "$RUST_SERVER" $ARGS

