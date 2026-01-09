#!/bin/bash
#
# GPU Triton VM Prover - Build, Prove, and Verify
#
# Usage:
#   ./run_gpu_prover.sh [program.tasm] [input] [OPTIONS]
#
# Options:
#   --multi-gpu      Enable multi-GPU unified memory (for larger inputs)
#   --gpu-count=N    Limit to N GPUs (e.g., --gpu-count=2 uses only 2 of 8 GPUs)
#   --cpu-aux        Use hybrid CPU/GPU for aux table (faster, now working!)
#   --arch=SM        Override CUDA architecture (e.g., 90 for Hopper, 120 for Blackwell)
#   --clean          Force clean rebuild (removes build directory)
#   --profile        Enable detailed profiling output
#   --profile-lde    Enable detailed LDE kernel profiling
#   --pad-mode=N     Select pad_and_scale kernel variant (0=original, 4=branchless)
#
# Examples:
#   ./run_gpu_prover.sh spin_input21.tasm 18                     # Single GPU
#   ./run_gpu_prover.sh spin_input21.tasm 19 --multi-gpu         # Multi-GPU
#   ./run_gpu_prover.sh spin_input21.tasm 19 --multi-gpu --cpu-aux  # Hybrid CPU/GPU (fastest, now working!)
#   ./run_gpu_prover.sh spin_input21.tasm 19 --gpu-count=2       # Use only 2 GPUs
#   ./run_gpu_prover.sh spin_input21.tasm 18 --arch=90           # Force Hopper build
#   ./run_gpu_prover.sh spin_input21.tasm 18 --clean             # Clean rebuild
#   ./run_gpu_prover.sh spin_input21.tasm 19 --profile           # With profiling
#   ./run_gpu_prover.sh spin_input21.tasm 19 --profile-lde       # With LDE kernel profiling
#
# Auto-detected architectures:
#   H100/H200     -> sm_90  (Hopper)
#   RTX 40xx/L40  -> sm_89  (Ada Lovelace)
#   RTX 50xx      -> sm_90  (Blackwell, compatibility mode)
#   A100/RTX 30xx -> sm_80  (Ampere)
#
# Kernel optimizations applied:
#   - Fused zerofier_add + pad_scale (eliminates intermediate buffer)
#   - ILP optimization (4 elements per thread)
#   - Precomputed powers for coset interpolation
#   - Optimized NTT butterfly kernels
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROGRAM="${1:-spin_input21.tasm}"
INPUT="${2:-18}"
MULTI_GPU=0
GPU_COUNT=""
CPU_AUX=0
ARCH_OVERRIDE=""
CLEAN_BUILD=0
PROFILE_MODE=0
PROFILE_LDE=0
PAD_SCALE_MODE=""

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --multi-gpu)
            MULTI_GPU=1
            ;;
        --gpu-count=*)
            GPU_COUNT="${arg#*=}"
            MULTI_GPU=1  # GPU count implies multi-GPU mode
            ;;
        --cpu-aux)
            CPU_AUX=1
            ;;
        --arch=*)
            ARCH_OVERRIDE="${arg#*=}"
            ;;
        --clean)
            CLEAN_BUILD=1
            ;;
        --profile)
            PROFILE_MODE=1
            ;;
        --profile-lde)
            PROFILE_LDE=1
            ;;
        --pad-mode=*)
            PAD_SCALE_MODE="${arg#*=}"
            ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
PROVER="$BUILD_DIR/triton_vm_prove_gpu_full"
VERIFIER_DIR="$SCRIPT_DIR/triton-cli-1.0.0"
VERIFIER="$VERIFIER_DIR/target/release/triton-cli"

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CLAIM_FILE="/tmp/claim_${TIMESTAMP}.bin"
PROOF_FILE="/tmp/proof_${TIMESTAMP}.bin"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  GPU Triton VM Prover - Build, Prove, Verify                 ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Program: ${YELLOW}$PROGRAM${NC}"
echo -e "${BLUE}║  Input:   ${YELLOW}$INPUT${NC}"
if [[ $MULTI_GPU -eq 1 ]]; then
    if [[ -n "$GPU_COUNT" ]]; then
echo -e "${BLUE}║  Mode:    ${GREEN}Multi-GPU (limit: $GPU_COUNT GPUs)${NC}"
    else
echo -e "${BLUE}║  Mode:    ${GREEN}Multi-GPU Unified Memory${NC}"
    fi
else
echo -e "${BLUE}║  Mode:    Single GPU${NC}"
fi
if [[ $CPU_AUX -eq 1 ]]; then
echo -e "${BLUE}║  Aux:     ${GREEN}Hybrid CPU/GPU (optimized)${NC}"
fi
if [[ $PROFILE_LDE -eq 1 ]]; then
echo -e "${BLUE}║  LDE:     ${YELLOW}Detailed profiling enabled${NC}"
fi
if [[ -n "$PAD_SCALE_MODE" ]]; then
echo -e "${BLUE}║  Kernel:  ${YELLOW}pad_scale mode=$PAD_SCALE_MODE${NC}"
fi
if [[ $CLEAN_BUILD -eq 1 ]]; then
echo -e "${BLUE}║  Build:   ${YELLOW}Clean rebuild${NC}"
fi
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Validate input for spin_input21.tasm (requires log₂(padded height) between 16-32)
if [[ "$PROGRAM" == "spin_input21.tasm" ]] && [[ $INPUT -lt 16 || $INPUT -gt 32 ]]; then
    echo -e "${RED}Error: spin_input21.tasm requires input between 16-32 (log₂ padded height), got: $INPUT${NC}"
    echo -e "${YELLOW}Try: ./run_gpu_prover.sh spin_input21.tasm 18${NC}"
    exit 1
fi

# ============================================================================
# Step 1: Build
# ============================================================================
echo -e "${YELLOW}━━━ Step 1: Building Project ━━━${NC}"

cd "$SCRIPT_DIR"

# Set CUDA environment variables FIRST (before checking version)
# Prefer system CUDA 12.0 (compiles faster), then fall back to CUDA 13.x
# Note: nvcc 13.0/13.1 with sm_90 can hang on complex kernels during ptxas
if [[ -x "/usr/bin/nvcc" ]]; then
    # Use system nvcc (usually CUDA 12.0)
    export PATH=/usr/bin:$PATH
elif [[ -d "/usr/local/cuda-13.0" ]]; then
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CUDAToolkit_ROOT=/usr/local/cuda-13.0
elif [[ -d "/usr/local/cuda-13.1" ]]; then
    export CUDA_HOME=/usr/local/cuda-13.1
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CUDAToolkit_ROOT=/usr/local/cuda-13.1
fi

# Get CUDA version to determine supported architectures
CUDA_VERSION=""
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -n1)
    echo "CUDA Toolkit version: ${CUDA_VERSION}"
fi

# Auto-detect GPU architecture for optimal compile
DETECTED_ARCH=""
if [[ -z "$ARCH_OVERRIDE" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr -d '\r' | xargs)
        echo "Detected GPU: ${GPU_NAME}"
        
        # Hopper architecture (H100, H200) - sm_90 (requires CUDA 11.8+)
        if [[ "$GPU_NAME" == *"H200"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
            DETECTED_ARCH="90"
            echo "  -> Hopper architecture detected, using sm_90"
        # Ada Lovelace (RTX 40xx, L40) - sm_89 (requires CUDA 11.8+)
        elif [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"L40"* ]]; then
            DETECTED_ARCH="89"
            echo "  -> Ada Lovelace architecture detected, using sm_89"
        # Blackwell (RTX 50xx, RTX PRO 6000) - sm_90 (Hopper compatibility mode)
        # Using sm_90 with CUDA 12.0 (nvcc 13.x hangs on sm_90 for complex kernels)
        elif [[ "$GPU_NAME" == *"RTX 50"* ]] || [[ "$GPU_NAME" == *"RTX PRO 6000"* ]] || [[ "$GPU_NAME" == *"Blackwell"* ]]; then
            DETECTED_ARCH="90"
            echo "  -> Blackwell architecture detected, using sm_90 (Hopper compatibility mode)"
        # Ampere (A100, RTX 30xx) - sm_80
        elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"RTX 30"* ]]; then
            DETECTED_ARCH="80"
            echo "  -> Ampere architecture detected, using sm_80"
        fi
    else
        echo "Warning: nvidia-smi not found, using default CUDA architectures"
    fi
fi

if [[ -n "$ARCH_OVERRIDE" ]]; then
    TARGET_ARCH="$ARCH_OVERRIDE"
    echo "Using user-specified CUDA architectures: ${TARGET_ARCH}"
elif [[ -n "$DETECTED_ARCH" ]]; then
    TARGET_ARCH="$DETECTED_ARCH"
else
    TARGET_ARCH=""
    echo "Using default CUDA architectures (may be slow to compile)"
fi

# Handle --clean flag
if [[ $CLEAN_BUILD -eq 1 ]]; then
    echo "Clean build requested, removing build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory and configure if needed
NEED_CMAKE=0
if [[ ! -d "$BUILD_DIR" ]]; then
    NEED_CMAKE=1
elif [[ ! -f "$BUILD_DIR/Makefile" ]]; then
    NEED_CMAKE=1
elif ! grep -q "triton_vm_prove_gpu_full" "$BUILD_DIR/Makefile" 2>/dev/null; then
    echo "GPU target not found in Makefile, reconfiguring..."
    NEED_CMAKE=1
fi

# Only check critical configuration changes that actually require rebuild
if [[ $NEED_CMAKE -eq 0 ]] && [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
    # Only reconfigure if CUDA architecture changed (affects compiled binaries)
    if [[ -n "$TARGET_ARCH" ]] && ! grep -q "CMAKE_CUDA_ARCHITECTURES:STRING=${TARGET_ARCH}" "$CACHE_FILE" 2>/dev/null; then
        echo "CUDA architecture changed to ${TARGET_ARCH}, reconfiguring..."
        NEED_CMAKE=1
    fi
fi

if [[ $NEED_CMAKE -eq 1 ]]; then
    echo "Configuring build directory with CUDA support..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Environment variables are already set at the top of the script
    CMAKE_ARGS=(
        ..
        -DENABLE_CUDA=ON
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_TESTS=OFF
        -DBUILD_GPU_TESTS=OFF
        -DBUILD_BENCHMARKS=OFF
    )

    # Prefer system nvcc 12.0 (faster compilation), then CUDA 13.x
    if [[ -x "/usr/bin/nvcc" ]]; then
        CMAKE_ARGS+=("-DCMAKE_CUDA_COMPILER=/usr/bin/nvcc")
    elif [[ -d "/usr/local/cuda-13.0" ]]; then
        CMAKE_ARGS+=("-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc")
        CMAKE_ARGS+=("-DCUDAToolkit_ROOT=/usr/local/cuda-13.0")
    elif [[ -d "/usr/local/cuda-13.1" ]]; then
        CMAKE_ARGS+=("-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc")
        CMAKE_ARGS+=("-DCUDAToolkit_ROOT=/usr/local/cuda-13.1")
    fi

    if [[ -n "$TARGET_ARCH" ]]; then
        CMAKE_ARGS+=("-DTRITON_CUDA_ARCHITECTURES=${TARGET_ARCH}")
        # Set CMAKE_CUDA_ARCHITECTURES as environment variable for compiler detection
        export CMAKE_CUDA_ARCHITECTURES="${TARGET_ARCH}"
    else
        # Default to 90 if not specified (for RTX 5090 / Blackwell compatibility)
        export CMAKE_CUDA_ARCHITECTURES="90"
    fi

    cmake "${CMAKE_ARGS[@]}"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Error: CMake configuration failed${NC}"
        echo "Make sure CUDA toolkit is installed and nvcc is in PATH"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# Build
cd "$BUILD_DIR"
echo "Building triton_vm_prove_gpu_full..."
make -j$(nproc) triton_vm_prove_gpu_full 2>&1 | tail -15

if [[ ! -f "$PROVER" ]]; then
    echo -e "${RED}Error: Build failed - prover not found${NC}"
    echo "Try running manually:"
    echo "  cd $BUILD_DIR"
    echo "  cmake .. -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  make -j\$(nproc) triton_vm_prove_gpu_full"
    exit 1
fi

echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# ============================================================================
# Step 2: Generate Proof
# ============================================================================
echo -e "${YELLOW}━━━ Step 2: Generating Proof ━━━${NC}"

cd "$SCRIPT_DIR"

# Check if program file exists
if [[ ! -f "$PROGRAM" ]]; then
    echo -e "${RED}Error: Program file not found: $PROGRAM${NC}"
    exit 1
fi

# Set environment for multi-GPU mode
if [[ $MULTI_GPU -eq 1 ]]; then
    export TRITON_MULTI_GPU=1
fi

# Set GPU count limit (for systems with many GPUs where only a few are needed)
if [[ -n "$GPU_COUNT" ]]; then
    export TRITON_GPU_COUNT=$GPU_COUNT
fi

# Set environment for hybrid CPU/GPU aux computation (35% faster)
if [[ $CPU_AUX -eq 1 ]]; then
    export TRITON_AUX_CPU=1
fi

# Set OpenMP thread count for maximum CPU utilization
# Threadripper 9995WX: 96 cores, 192 threads (Zen 5)
# Set to 96 threads to match physical cores
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-96}
# Enable thread binding for better performance
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
# Use dynamic scheduling for better load balancing
export OMP_SCHEDULE=dynamic

# Set optimal OpenMP parallelization controls (based on benchmark results)
# Upload + Quotient configuration was found to be fastest
export TRITON_OMP_UPLOAD=${TRITON_OMP_UPLOAD:-1}
export TRITON_OMP_INIT=${TRITON_OMP_INIT:-0}
export TRITON_OMP_QUOTIENT=${TRITON_OMP_QUOTIENT:-1}
export TRITON_OMP_PROCESSOR=${TRITON_OMP_PROCESSOR:-0}

echo "OpenMP configuration:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "  OMP_PLACES=$OMP_PLACES"
echo "  TRITON_OMP_UPLOAD=$TRITON_OMP_UPLOAD"
echo "  TRITON_OMP_INIT=$TRITON_OMP_INIT"
echo "  TRITON_OMP_QUOTIENT=$TRITON_OMP_QUOTIENT"
echo "  TRITON_OMP_PROCESSOR=$TRITON_OMP_PROCESSOR"

# Set environment for profiling
if [[ $PROFILE_MODE -eq 1 ]]; then
    export TRITON_PROFILE_QUOT=1
    export TRITON_PROFILE_AUX=1
    export TRITON_PROFILE_MAIN=1
    export TVM_PROFILE_AUX=1
fi

# Set environment for detailed LDE profiling
if [[ $PROFILE_LDE -eq 1 ]]; then
    export TRITON_PROFILE_LDE_DETAIL=1
fi

# Set pad_and_scale kernel mode (0=original, 4=branchless)
if [[ -n "$PAD_SCALE_MODE" ]]; then
    export TRITON_PAD_SCALE_MODE=$PAD_SCALE_MODE
fi

# Run prover
START_TIME=$(date +%s.%N)

"$PROVER" "$PROGRAM" "$INPUT" "$CLAIM_FILE" "$PROOF_FILE"
PROVE_STATUS=$?

END_TIME=$(date +%s.%N)
PROVE_TIME=$(awk "BEGIN {printf \"%.3f\", $END_TIME - $START_TIME}")

if [[ $PROVE_STATUS -ne 0 ]]; then
    echo -e "${RED}Error: Proof generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Proof generated in ${PROVE_TIME}s${NC}"
echo "  Claim: $CLAIM_FILE"
echo "  Proof: $PROOF_FILE"
echo ""

# ============================================================================
# Step 3: Verify Proof
# ============================================================================
echo -e "${YELLOW}━━━ Step 3: Verifying Proof ━━━${NC}"

# Check if verifier exists
if [[ ! -f "$VERIFIER" ]]; then
    echo -e "${YELLOW}Building verifier...${NC}"
    cd "$VERIFIER_DIR"
    cargo build --release 2>&1 | tail -3
    cd "$SCRIPT_DIR"
fi

# Run verifier
START_TIME=$(date +%s.%N)

VERIFY_OUTPUT=$("$VERIFIER" verify --claim "$CLAIM_FILE" --proof "$PROOF_FILE" 2>&1)
VERIFY_STATUS=$?

END_TIME=$(date +%s.%N)
VERIFY_TIME=$(awk "BEGIN {printf \"%.3f\", $END_TIME - $START_TIME}")

echo "$VERIFY_OUTPUT"

if [[ $VERIFY_STATUS -eq 0 ]] && [[ "$VERIFY_OUTPUT" == *"verified"* ]]; then
    echo -e "${GREEN}✓ Verification complete in ${VERIFY_TIME}s${NC}"
else
    echo -e "${RED}✗ Verification failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
PROOF_SIZE=$(ls -lh "$PROOF_FILE" | awk '{print $5}')

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ${GREEN}SUCCESS${BLUE}                                                     ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Program:     $PROGRAM${NC}"
echo -e "${BLUE}║  Input:       $INPUT${NC}"
echo -e "${BLUE}║  Prove time:  ${PROVE_TIME}s${NC}"
echo -e "${BLUE}║  Verify time: ${VERIFY_TIME}s${NC}"
echo -e "${BLUE}║  Proof size:  $PROOF_SIZE${NC}"
echo -e "${BLUE}║${NC}"
echo -e "${BLUE}║  Claim: $CLAIM_FILE${NC}"
echo -e "${BLUE}║  Proof: $PROOF_FILE${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

