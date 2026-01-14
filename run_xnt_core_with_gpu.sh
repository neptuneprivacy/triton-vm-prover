#!/usr/bin/env bash
# Run xnt-core with direct GPU prover integration (no server needed)
#
# This script replaces the need to run both:
#   1. ./run_rust_prover_server_gpu.sh (NO LONGER NEEDED)
#   2. ./target/release/xnt-core
#
# Now you only need to run this single script!

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration - use relative paths from script location
GPU_PROVER_PATH="${TRITON_GPU_PROVER_PATH:-$SCRIPT_DIR/build/triton_vm_prove_gpu_full}"
XNT_CORE_PATH="${XNT_CORE_PATH:-$SCRIPT_DIR/xnt-core/target/release/xnt-core}"

# GPU/OpenMP settings (adjust to match your hardware)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"  # Use GPU 0 and 1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export TRITON_OMP_INIT="${TRITON_OMP_INIT:-0}"

# GPU optimization settings - Performance tuning for GPU prover
export TRITON_AUX_CPU="${TRITON_AUX_CPU:-1}"                     # Use CPU for auxiliary tables
export TVM_USE_TASKFLOW="${TVM_USE_TASKFLOW:-1}"                # Enable Intel TBB task scheduling
export TVM_USE_TBB="${TVM_USE_TBB:-1}"                          # Enable Intel TBB parallelism
export TRITON_GPU_DEGREE_LOWERING="${TRITON_GPU_DEGREE_LOWERING:-1}"  # GPU degree lowering optimization
export TRITON_GPU_U32="${TRITON_GPU_U32:-1}"                    # Use 32-bit operations on GPU
export TVM_USE_RUST_TRACE="${TVM_USE_RUST_TRACE:-1}"            # Use Rust trace execution
export TRITON_GPU_USE_RAM_OVERFLOW="${TRITON_GPU_USE_RAM_OVERFLOW:-1}"  # Use system RAM as VRAM buffer
export TRITON_MULTI_GPU="${TRITON_MULTI_GPU:-0}"                # Disable multi-GPU (use single GPU)
export TRITON_GPU_COUNT="${TRITON_GPU_COUNT:-1}"                # Number of GPUs to use
export TRITON_NTT_REG6STAGE="${TRITON_NTT_REG6STAGE:-1}"        # NTT register optimization (6-stage)
export TRITON_NTT_FUSED12="${TRITON_NTT_FUSED12:-1}"            # NTT fused kernel optimization
export TRITON_NTT_COALESCED="${TRITON_NTT_COALESCED:-1}"        # NTT coalesced memory access
export TRITON_PAD_SCALE_MODE="${TRITON_PAD_SCALE_MODE:-4}"      # Table padding/scaling mode

# Enable GPU prover (this is the key change - no socket needed!)
export TRITON_GPU_PROVER_PATH="$GPU_PROVER_PATH"

echo "========================================"
echo "xnt-core with Direct GPU Prover"
echo "========================================"
echo "GPU Prover:       $TRITON_GPU_PROVER_PATH"
echo "XNT-Core:         $XNT_CORE_PATH"
echo ""
echo "GPU Settings:"
echo "  GPUs:           $CUDA_VISIBLE_DEVICES"
echo "  Multi-GPU:      $TRITON_MULTI_GPU (0=single, 1=multi)"
echo "  GPU Count:      $TRITON_GPU_COUNT"
echo ""
echo "OpenMP/Threading:"
echo "  OMP Threads:    $OMP_NUM_THREADS"
echo "  OMP Init:       $TRITON_OMP_INIT"
echo "  Taskflow:       $TVM_USE_TASKFLOW"
echo "  TBB:            $TVM_USE_TBB"
echo ""
echo "GPU Optimizations:"
echo "  Degree Lower:   $TRITON_GPU_DEGREE_LOWERING"
echo "  U32 Mode:       $TRITON_GPU_U32"
echo "  RAM Overflow:   $TRITON_GPU_USE_RAM_OVERFLOW"
echo "  AUX CPU:        $TRITON_AUX_CPU"
echo ""
echo "NTT Optimizations:"
echo "  REG6STAGE:      $TRITON_NTT_REG6STAGE"
echo "  FUSED12:        $TRITON_NTT_FUSED12"
echo "  COALESCED:      $TRITON_NTT_COALESCED"
echo ""
echo "Other:"
echo "  Rust Trace:     $TVM_USE_RUST_TRACE"
echo "  Pad Scale Mode: $TRITON_PAD_SCALE_MODE"
echo "========================================"
echo ""

# Check if GPU prover exists
if [[ ! -x "$TRITON_GPU_PROVER_PATH" ]]; then
    echo "ERROR: GPU prover not found or not executable: $TRITON_GPU_PROVER_PATH"
    echo ""
    echo "Please set TRITON_GPU_PROVER_PATH to the path of triton_vm_prove_gpu_full"
    echo "Example:"
    echo "  export TRITON_GPU_PROVER_PATH=/path/to/triton_vm_prove_gpu_full"
    exit 1
fi

# Check if xnt-core exists
if [[ ! -x "$XNT_CORE_PATH" ]]; then
    echo "ERROR: xnt-core not found or not executable: $XNT_CORE_PATH"
    echo ""
    echo "Building xnt-core..."
    cd "$SCRIPT_DIR/xnt-core"
    cargo build --release
    cd "$SCRIPT_DIR"
fi

# Run xnt-core with GPU prover enabled
# No separate prover server needed!
exec "$XNT_CORE_PATH" \
  --network main \
  --peer 161.97.150.88:9898 \
  --peer 154.38.160.61:9898 \
  --peer 103.78.0.72:9898 \
  --peer 5.21.91.33:9898 \
  --compose \
  --guesser-fraction 0.871 \
  --tx-proof-upgrading \
  --tx-proving-capability=singleproof \
  --gobbling-fraction=0.6 \
  --min-gobbling-fee=0.0001 \
  --tx-upgrade-filter=1:0 \
  --max-num-peers 20 \
  "$@"
