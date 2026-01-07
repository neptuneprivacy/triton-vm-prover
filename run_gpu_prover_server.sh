#!/bin/bash
#
# GPU Prover Server for Neptune Integration
#
# Usage:
#   ./run_gpu_prover_server.sh [OPTIONS]
#
# Options:
#   --tcp HOST:PORT    Listen on TCP socket (default: 127.0.0.1:5555)
#   --unix PATH        Listen on Unix socket
#   --help             Show help message
#
# Environment Variables:
#   TRITON_GPU_PROVER_PATH      Path to triton_vm_prove_gpu_full binary (auto-detected)
#   TRITON_GPU_PROVER_THREADS   OpenMP threads for GPU prover
#   TRITON_FIXED_SEED           Set to 1 for deterministic proofs
#
# Example:
#   ./run_gpu_prover_server.sh --tcp 127.0.0.1:5555
#   TRITON_VM_PROVER_SOCKET=127.0.0.1:5555 neptune-core --compose
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Check if server binary exists, build if missing
if [ ! -f "$BUILD_DIR/gpu_prover_server" ]; then
    echo "gpu_prover_server not found, building..."
    cd "$BUILD_DIR"
    make gpu_prover_server -j$(nproc) || {
        echo "Error: Failed to build gpu_prover_server"
        exit 1
    }
    cd "$SCRIPT_DIR"
fi

# Check if GPU prover binary exists, build if missing
if [ ! -f "$BUILD_DIR/triton_vm_prove_gpu_full" ]; then
    echo "triton_vm_prove_gpu_full not found, building..."
    cd "$BUILD_DIR"
    make triton_vm_prove_gpu_full -j$(nproc) || {
        echo "Error: Failed to build triton_vm_prove_gpu_full"
        exit 1
    }
    cd "$SCRIPT_DIR"
fi

# Set up environment
# Note: Rust FFI libraries are statically linked - no LD_LIBRARY_PATH needed
export TRITON_GPU_PROVER_PATH="${TRITON_GPU_PROVER_PATH:-$BUILD_DIR/triton_vm_prove_gpu_full}"

# GPU prover environment (match run_gpu_prover.sh settings)
export TRITON_FIXED_SEED="${TRITON_FIXED_SEED:-1}"
export TVM_USE_TASKFLOW="${TVM_USE_TASKFLOW:-1}"
export TVM_USE_RUST_TRACE="${TVM_USE_RUST_TRACE:-1}"
export TRITON_GPU_DEGREE_LOWERING="${TRITON_GPU_DEGREE_LOWERING:-1}"
export TVM_USE_TBB="${TVM_USE_TBB:-1}"
export TRITON_AUX_CPU="${TRITON_AUX_CPU:-1}"
export TRITON_MULTI_GPU="${TRITON_MULTI_GPU:-1}"
export TRITON_GPU_COUNT="${TRITON_GPU_COUNT:-2}"

echo "=========================================="
echo "GPU Prover Server for Neptune"
echo "=========================================="
echo "GPU Prover: $TRITON_GPU_PROVER_PATH"
echo "Server: $BUILD_DIR/gpu_prover_server"
echo ""
echo "Environment:"
echo "  TRITON_FIXED_SEED=$TRITON_FIXED_SEED"
echo "  TVM_USE_TASKFLOW=$TVM_USE_TASKFLOW"
echo "  TVM_USE_RUST_TRACE=$TVM_USE_RUST_TRACE"
echo "  TRITON_GPU_DEGREE_LOWERING=$TRITON_GPU_DEGREE_LOWERING"
echo "  TVM_USE_TBB=$TVM_USE_TBB"
echo "  TRITON_AUX_CPU=$TRITON_AUX_CPU"
echo "  TRITON_MULTI_GPU=$TRITON_MULTI_GPU"
echo "  TRITON_GPU_COUNT=$TRITON_GPU_COUNT"
echo "=========================================="
echo ""

# Run server
exec "$BUILD_DIR/gpu_prover_server" "$@"

