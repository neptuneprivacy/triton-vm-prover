#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# CUDA configuration
# ────────────────────────────────────────────────────────────────
export CUDA_PATH=/usr/local/cuda-13.0
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# Target GPU architecture
# put sm_100 for b200
# put sm_89 for 4090
ARCH="sm_90"

# Directory containing .cu files
SRC_DIR="twenty-first-local/twenty-first/cuda"
OUTPUT_DIR="built_kernels/"

# List of kernel base names (no .cu extension)
KERNELS=(
    "bfield_ntt"
    "xfield_ntt"
    "coset_scale"
    "aux_extend"
    "tip5_hash"
    "extract_rows"
    "quotient_evaluation"
    "big_package"
)

# ────────────────────────────────────────────────────────────────
# Compilation loop
# ────────────────────────────────────────────────────────────────
echo ">>> Using CUDA from: $CUDA_PATH"
$CUDA_PATH/bin/nvcc --version

quotient_constraints_file=$(find ./target/release -name "*quotient_constraints.cuh*" -print -quit)
if [[ -z "$quotient_constraints_file" ]]; then
    echo "❌ quotient_constraints.cuh not found under target/release"
    exit 1
fi

quotient_constraints_dir=$(dirname "$quotient_constraints_file")
echo ">>> Using quotient_constraints.cuh from: $quotient_constraints_dir"

for kernel in "${KERNELS[@]}"; do
    src="${SRC_DIR}/${kernel}.cu"
    out="${OUTPUT_DIR}/${kernel}.cubin"

    if [[ ! -f "$src" ]]; then
        echo "⚠️  Skipping ${kernel}: source not found at ${src}"
        continue
    fi

    echo "🔧 Compiling ${kernel}.cu → ${kernel}.cubin"
    "$CUDA_PATH/bin/nvcc" -O3 -arch="${ARCH}" -cubin "$src" -Xptxas="-O3 -v" -o "$out" -I "${SRC_DIR}/" -I "${quotient_constraints_dir}"

    echo "✅ Done: $out"
done

echo "🎉 All kernels compiled successfully."
