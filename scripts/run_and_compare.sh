#!/bin/bash
# Run GPU prover on dump and compare with Rust test data step-by-step

set -e

DUMP_DIR="${1:?Usage: $0 <dump_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_PROVER="${SCRIPT_DIR}/../build/triton_vm_prove_gpu_full"
TRITON_CLI="${SCRIPT_DIR}/../../triton-cli-1.0.0/target/release/triton-cli"

if [ ! -d "$DUMP_DIR" ]; then
    echo "Error: $DUMP_DIR is not a directory"
    exit 1
fi

RUST_TEST_DATA="$DUMP_DIR/rust_test_data"
if [ ! -d "$RUST_TEST_DATA" ]; then
    echo "Error: No rust_test_data directory found in $DUMP_DIR"
    exit 1
fi

echo ""
echo "======================================================================"
echo "  GPU Prover vs Rust: Live Comparison"
echo "======================================================================"
echo "  Dump: $DUMP_DIR"
echo "  Rust data: $RUST_TEST_DATA"
echo ""

# Load test data parameters
echo "Loading Rust reference data..."
python3 - "$RUST_TEST_DATA" << 'PYEOF'
import json
import sys
import os

rust_dir = sys.argv[1]

# Load parameters
with open(os.path.join(rust_dir, "02_parameters.json")) as f:
    params = json.load(f)
    print(f"  Padded height: {params['padded_height']}")
    print(f"  FRI domain:    {params['fri_domain_length']}")

# Load randomizer info
with open(os.path.join(rust_dir, "trace_randomizer_all_columns.json")) as f:
    rand_data = json.load(f)
    info = rand_data['randomizer_info']
    print(f"  Randomizers:   {info['num_trace_randomizers']}")
    print(f"  Seed:          {info['seed_hex'][:32]}...")

# Load main LDE reference
with open(os.path.join(rust_dir, "05_main_tables_lde.json")) as f:
    lde = json.load(f)
    first_row = lde['first_row']
    print(f"\n  Rust LDE reference:")
    print(f"    [0]:   {first_row[0]}")
    print(f"    [378]: {first_row[378]}")

# Load Merkle root
with open(os.path.join(rust_dir, "06_main_tables_merkle.json")) as f:
    merkle = json.load(f)
    print(f"\n  Rust Merkle root: {merkle['merkle_root'][:32]}...")

# Load challenges
with open(os.path.join(rust_dir, "07_fiat_shamir_challenges.json")) as f:
    ch = json.load(f)
    vals = ch['challenge_values']
    print(f"\n  Rust challenges (first 3):")
    for i, c in enumerate(vals[:3]):
        print(f"    [{i}]: {c[:50]}...")
PYEOF

echo ""
echo "----------------------------------------------------------------------"
echo "  Running GPU prover with validation..."
echo "----------------------------------------------------------------------"
echo ""

# Run GPU prover with debug output
export TVM_USE_RUST_RANDOMIZERS=1
export TVM_RUST_TEST_DATA_DIR="$RUST_TEST_DATA"
export TVM_DEBUG_MAIN_LDE_FIRST_ROW=1
export TVM_DEBUG_MAIN_LDE_COL0_POINT=1
export TVM_DEBUG_MAIN_LDE_PARAMS=1
export TVM_DEBUG_MERKLE_ROOT=1
export TVM_DEBUG_CHALLENGES=1

PROGRAM_JSON="$DUMP_DIR/program.json"
NONDET_JSON="$DUMP_DIR/nondet.json"
PUBLIC_INPUT=$(cat "$DUMP_DIR/public_input.txt" 2>/dev/null || echo "")

# Parse public input
if [ -f "$DUMP_DIR/claim.json" ]; then
    PUBLIC_INPUT=$(python3 -c "
import json
with open('$DUMP_DIR/claim.json') as f:
    d = json.load(f)
    print(','.join(str(x) for x in d.get('input', [])))
" 2>/dev/null || echo "")
fi

if [ -z "$PUBLIC_INPUT" ]; then
    # Try to get from program.json or input files
    PUBLIC_INPUT="316156163747956022,16160927538755301888,8992491622874874781,329695387193843274,9230046987694183548"
fi

CLAIM_OUT="$DUMP_DIR/gpu_test.claim"
PROOF_OUT="$DUMP_DIR/gpu_test.proof"

echo "Running: $GPU_PROVER ..."
"$GPU_PROVER" "$PROGRAM_JSON" "$PUBLIC_INPUT" "$CLAIM_OUT" "$PROOF_OUT" "$NONDET_JSON" "$PROGRAM_JSON" 2>&1 | tee "$DUMP_DIR/gpu_output.log" | while read line; do
    # Highlight important comparisons
    if echo "$line" | grep -q "DBG.*Main LDE col378"; then
        echo -e "\033[1;33m$line\033[0m"
    elif echo "$line" | grep -q "DBG.*Merkle root"; then
        echo -e "\033[1;33m$line\033[0m"
    elif echo "$line" | grep -q "Challenge"; then
        echo -e "\033[1;33m$line\033[0m"
    elif echo "$line" | grep -q "OOD self-check"; then
        echo -e "\033[1;31m$line\033[0m"
    elif echo "$line" | grep -q "MISMATCH\|diff="; then
        echo -e "\033[1;31m$line\033[0m"
    elif echo "$line" | grep -q "MATCH\|✓"; then
        echo -e "\033[1;32m$line\033[0m"
    else
        echo "$line"
    fi
done

echo ""
echo "----------------------------------------------------------------------"
echo "  Verification"
echo "----------------------------------------------------------------------"

if [ -f "$CLAIM_OUT" ] && [ -f "$PROOF_OUT" ]; then
    echo "Running triton-cli verify..."
    "$TRITON_CLI" verify "$CLAIM_OUT" "$PROOF_OUT" 2>&1 | head -50 || true
fi

echo ""
echo "======================================================================"
echo "  Done. Check $DUMP_DIR/gpu_output.log for full output."
echo "======================================================================"

