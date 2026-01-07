#!/bin/bash
#
# Run GPU prover against a dump directory with Rust test data
# and compare results step by step.
#
# Usage:
#   ./scripts/run_gpu_vs_rust_comparison.sh /tmp/gpu_prover_failed_<jobid>_<ts>
#
# Prerequisites:
#   - Dump directory with: program.json, nondet.json (or claim.json), rust_test_data/
#   - Built GPU prover: build/triton_vm_prove_gpu_full
#
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

DUMP_DIR="${1:?Usage: $0 <dump_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_PROVER="$PROJECT_DIR/build/triton_vm_prove_gpu_full"
TRITON_CLI="$PROJECT_DIR/../triton-cli-1.0.0/target/release/triton-cli"

if [ ! -d "$DUMP_DIR" ]; then
    echo -e "${RED}Error: $DUMP_DIR is not a directory${NC}"
    exit 1
fi

RUST_TEST_DATA="$DUMP_DIR/rust_test_data"
if [ ! -d "$RUST_TEST_DATA" ]; then
    echo -e "${RED}Error: No rust_test_data directory found in $DUMP_DIR${NC}"
    echo "Run ./run_triton_cli_rust_prove_from_dump.sh $DUMP_DIR first to generate test data"
    exit 1
fi

if [ ! -x "$GPU_PROVER" ]; then
    echo -e "${YELLOW}Building GPU prover...${NC}"
    cd "$PROJECT_DIR"
    ./run_gpu_prover.sh --build-only 2>/dev/null || {
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc) triton_vm_prove_gpu_full
    }
fi

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  GPU Prover vs Rust Test Data: Step-by-Step Comparison         ║${NC}"
echo -e "${CYAN}╠════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  Dump: $DUMP_DIR"
echo -e "${CYAN}║  Rust data: $RUST_TEST_DATA"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Load Rust parameters for reference
echo -e "${YELLOW}Loading Rust reference parameters...${NC}"
python3 - "$RUST_TEST_DATA" << 'PYEOF'
import json
import sys
import os

rust_dir = sys.argv[1]

# Load parameters
params_file = os.path.join(rust_dir, "02_parameters.json")
if os.path.exists(params_file):
    with open(params_file) as f:
        params = json.load(f)
        print(f"  Padded height:     {params.get('padded_height')}")
        print(f"  Log2 padded:       {params.get('log2_padded_height')}")
        print(f"  FRI domain:        {params.get('fri_domain_length')}")
        print(f"  Expansion factor:  {params.get('expansion_factor')}")

# Load Merkle roots for comparison
merkle_main = os.path.join(rust_dir, "06_main_tables_merkle.json")
if os.path.exists(merkle_main):
    with open(merkle_main) as f:
        d = json.load(f)
        print(f"\n  Rust Main Merkle:  {d['merkle_root'][:40]}...")

merkle_aux = os.path.join(rust_dir, "09_aux_tables_merkle.json")
if os.path.exists(merkle_aux):
    with open(merkle_aux) as f:
        d = json.load(f)
        print(f"  Rust Aux Merkle:   {d['aux_merkle_root'][:40]}...")

merkle_quot = os.path.join(rust_dir, "13_quotient_merkle.json")
if os.path.exists(merkle_quot):
    with open(merkle_quot) as f:
        d = json.load(f)
        print(f"  Rust Quot Merkle:  {d['quotient_merkle_root'][:40]}...")

# Load LDE first row samples
lde_main = os.path.join(rust_dir, "05_main_tables_lde.json")
if os.path.exists(lde_main):
    with open(lde_main) as f:
        d = json.load(f)
        if 'first_row' in d:
            first_row = d['first_row']
            print(f"\n  Rust LDE first row:")
            print(f"    [0]:   {first_row[0]}")
            print(f"    [100]: {first_row[100] if len(first_row) > 100 else 'N/A'}")
            print(f"    [378]: {first_row[378] if len(first_row) > 378 else 'N/A'} (LAST)")
PYEOF

echo ""
echo -e "${YELLOW}──────────────────────────────────────────────────────────────────${NC}"
echo -e "${YELLOW}  Running GPU prover with Rust test data comparison...${NC}"
echo -e "${YELLOW}──────────────────────────────────────────────────────────────────${NC}"
echo ""

# Prepare input files
PROGRAM_JSON="$DUMP_DIR/program.json"
NONDET_JSON="$DUMP_DIR/nondet.json"

# Get public input from claim.json
PUBLIC_INPUT=""
if [ -f "$DUMP_DIR/claim.json" ]; then
    PUBLIC_INPUT=$(python3 -c "
import json
with open('$DUMP_DIR/claim.json') as f:
    d = json.load(f)
    print(','.join(str(x) for x in d.get('input', [])))
" 2>/dev/null || echo "")
fi

# Fallback if no public input found
if [ -z "$PUBLIC_INPUT" ]; then
    PUBLIC_INPUT="316156163747956022,16160927538755301888,8992491622874874781,329695387193843274,9230046987694183548"
    echo -e "${YELLOW}Warning: Using default public input${NC}"
fi

CLAIM_OUT="$DUMP_DIR/gpu_compare.claim"
PROOF_OUT="$DUMP_DIR/gpu_compare.proof"

# Environment variables for comparison
export TVM_RUST_TEST_DATA_DIR="$RUST_TEST_DATA"
export TVM_USE_RUST_RANDOMIZERS=1  # Use same randomizers as Rust
export TVM_DEBUG_MERKLE_ROOT=1
export TVM_DEBUG_CHALLENGES=1
export TRITON_FIXED_SEED=1
export TVM_USE_TBB=1
export TVM_USE_TASKFLOW=1
export TVM_USE_RUST_TRACE=1
export TVM_VERIFY_TBB=1

echo "Running: $GPU_PROVER"
echo "  Program:   $PROGRAM_JSON"
echo "  Input:     $PUBLIC_INPUT"
echo "  NonDet:    $NONDET_JSON"
echo "  Test data: $RUST_TEST_DATA"
echo ""

# Run GPU prover
"$GPU_PROVER" "$PROGRAM_JSON" "$PUBLIC_INPUT" "$CLAIM_OUT" "$PROOF_OUT" "$NONDET_JSON" "$PROGRAM_JSON" 2>&1 | while read line; do
    # Highlight validation results
    if echo "$line" | grep -q "✅\|MATCH"; then
        echo -e "${GREEN}$line${NC}"
    elif echo "$line" | grep -q "❌\|MISMATCH\|Error"; then
        echo -e "${RED}$line${NC}"
    elif echo "$line" | grep -q "⚠️\|Warning"; then
        echo -e "${YELLOW}$line${NC}"
    elif echo "$line" | grep -q "DBG\|Rust\|GPU"; then
        echo -e "${CYAN}$line${NC}"
    else
        echo "$line"
    fi
done

echo ""
echo -e "${YELLOW}──────────────────────────────────────────────────────────────────${NC}"
echo -e "${YELLOW}  Verification${NC}"
echo -e "${YELLOW}──────────────────────────────────────────────────────────────────${NC}"

if [ -f "$CLAIM_OUT" ] && [ -f "$PROOF_OUT" ]; then
    echo "Verifying GPU-generated proof with triton-cli..."
    if "$TRITON_CLI" verify --claim "$CLAIM_OUT" --proof "$PROOF_OUT" 2>&1; then
        echo -e "${GREEN}✅ GPU proof VERIFIED by triton-cli${NC}"
    else
        echo -e "${RED}❌ GPU proof FAILED verification${NC}"
        
        # Also verify Rust proof for comparison
        RUST_CLAIM="$DUMP_DIR/claim.rust.json"
        RUST_PROOF="$DUMP_DIR/proof.rust.bin"
        if [ -f "$RUST_CLAIM" ] && [ -f "$RUST_PROOF" ]; then
            echo ""
            echo "Verifying Rust-generated proof for comparison..."
            if "$TRITON_CLI" verify --claim "$RUST_CLAIM" --proof "$RUST_PROOF" 2>&1; then
                echo -e "${GREEN}✅ Rust proof verifies (reference)${NC}"
            fi
        fi
    fi
else
    echo -e "${RED}❌ Proof files not generated${NC}"
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Comparison Complete${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo "  GPU claim: $CLAIM_OUT"
echo "  GPU proof: $PROOF_OUT"
echo ""

