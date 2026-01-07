#!/bin/bash
#
# Component Verification Script
#
# Verifies each component of the STARK proof generation pipeline
# Tests with input 19 (fast) and optionally input 20/21
#
# Usage:
#   ./scripts/verify_component.sh [component_name] [input]
#
# Examples:
#   ./scripts/verify_component.sh all 19
#   ./scripts/verify_component.sh aux_table 19
#   ./scripts/verify_component.sh quotient 20

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
COMPONENT="${1:-all}"
INPUT="${2:-19}"
PROGRAM="spin_input21.tasm"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Component Verification Test                                ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Component: ${YELLOW}$COMPONENT${NC}"
echo -e "${BLUE}║  Program:   ${YELLOW}$PROGRAM${NC}"
echo -e "${BLUE}║  Input:    ${YELLOW}$INPUT${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if program exists
if [ ! -f "$PROGRAM" ]; then
    echo -e "${RED}Error: Program file not found: $PROGRAM${NC}"
    exit 1
fi

# Set environment variables for optimal performance
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-96}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_SCHEDULE=dynamic

# Optimal OpenMP configuration (from benchmark)
export TRITON_OMP_UPLOAD=1
export TRITON_OMP_INIT=0
export TRITON_OMP_QUOTIENT=1
export TRITON_OMP_PROCESSOR=0

# GPU configuration
export TRITON_GPU_PHASE1=1
export TRITON_AUX_CPU=1
export TRITON_GPU_DEGREE_LOWERING=1
export TRITON_GPU_U32=1
export TRITON_GPU_IGNORE_MEMCHECK=1

# Component-specific environment variables
case "$COMPONENT" in
    "trace"|"vm")
        export TVM_DEBUG_TRACE=1
        echo -e "${CYAN}Testing: VM Trace Execution${NC}"
        ;;
    "main_table")
        export TVM_DEBUG_MAIN_TABLE=1
        echo -e "${CYAN}Testing: Main Table Creation${NC}"
        ;;
    "aux_table")
        export TVM_DEBUG_AUX_TABLE=1
        export TRITON_PROFILE_AUX=1
        echo -e "${CYAN}Testing: Aux Table Computation${NC}"
        ;;
    "quotient")
        export TVM_DEBUG_QUOTIENT=1
        export TRITON_PROFILE_QUOT=1
        echo -e "${CYAN}Testing: Quotient Computation${NC}"
        ;;
    "fri")
        export TVM_DEBUG_FRI=1
        echo -e "${CYAN}Testing: FRI Protocol${NC}"
        ;;
    "all")
        echo -e "${CYAN}Testing: All Components (Full Pipeline)${NC}"
        ;;
    *)
        echo -e "${YELLOW}Unknown component: $COMPONENT${NC}"
        echo "Available components: trace, main_table, aux_table, quotient, fri, all"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}Environment:${NC}"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  TRITON_OMP_UPLOAD=$TRITON_OMP_UPLOAD"
echo "  TRITON_OMP_INIT=$TRITON_OMP_INIT"
echo "  TRITON_OMP_QUOTIENT=$TRITON_OMP_QUOTIENT"
echo "  TRITON_OMP_PROCESSOR=$TRITON_OMP_PROCESSOR"
echo ""

# Run the prover
echo -e "${CYAN}━━━ Running Prover ━━━${NC}"
START_TIME=$(date +%s.%N)

if ! ./run_gpu_prover.sh "$PROGRAM" "$INPUT" --multi-gpu --gpu-count=2 > /tmp/prover_output.log 2>&1; then
    echo -e "${RED}✗ Proof generation failed!${NC}"
    echo ""
    echo "Last 50 lines of output:"
    tail -50 /tmp/prover_output.log
    exit 1
fi

END_TIME=$(date +%s.%N)
PROVE_TIME=$(awk "BEGIN {printf \"%.3f\", $END_TIME - $START_TIME}")

echo -e "${GREEN}✓ Proof generated in ${PROVE_TIME}s${NC}"
echo ""

# Extract claim and proof files from output
# Try multiple patterns to find the files
CLAIM_FILE=$(grep -E "Claim:|claim:" /tmp/prover_output.log | grep -oP "(Claim:|claim:)\s+\K[^\s]+" | tail -1)
PROOF_FILE=$(grep -E "Proof:|proof:" /tmp/prover_output.log | grep -oP "(Proof:|proof:)\s+\K[^\s]+" | tail -1)

# If still not found, try alternative patterns
if [ -z "$CLAIM_FILE" ]; then
    CLAIM_FILE=$(grep -oP "Claim:\s+\K[^\s]+" /tmp/prover_output.log | tail -1)
fi
if [ -z "$PROOF_FILE" ]; then
    PROOF_FILE=$(grep -oP "Proof:\s+\K[^\s]+" /tmp/prover_output.log | tail -1)
fi

# If still not found, try to find the most recent claim/proof files
if [ -z "$CLAIM_FILE" ] || [ ! -f "$CLAIM_FILE" ]; then
    CLAIM_FILE=$(ls -t /tmp/claim_*.bin 2>/dev/null | head -1)
fi
if [ -z "$PROOF_FILE" ] || [ ! -f "$PROOF_FILE" ]; then
    PROOF_FILE=$(ls -t /tmp/proof_*.bin 2>/dev/null | head -1)
fi

# Verify files exist
if [ -z "$CLAIM_FILE" ] || [ ! -f "$CLAIM_FILE" ]; then
    echo -e "${RED}Error: Could not find claim file${NC}"
    echo "  Searched in: /tmp/claim_*.bin"
    echo "  Last 10 lines of prover output:"
    tail -10 /tmp/prover_output.log
    exit 1
fi

if [ -z "$PROOF_FILE" ] || [ ! -f "$PROOF_FILE" ]; then
    echo -e "${RED}Error: Could not find proof file${NC}"
    echo "  Searched in: /tmp/proof_*.bin"
    echo "  Last 10 lines of prover output:"
    tail -10 /tmp/prover_output.log
    exit 1
fi

echo "  Found claim file: $CLAIM_FILE"
echo "  Found proof file: $PROOF_FILE"
echo ""

# Verify with triton-cli
echo -e "${CYAN}━━━ Verifying Proof ━━━${NC}"
TRITON_CLI_DIR="$PROJECT_DIR/../triton-cli-1.0.0"
TRITON_CLI="$TRITON_CLI_DIR/target/release/triton-cli"

if [ ! -f "$TRITON_CLI" ]; then
    echo -e "${YELLOW}Building triton-cli...${NC}"
    cd "$TRITON_CLI_DIR"
    cargo build --release 2>&1 | tail -3
    cd "$PROJECT_DIR"
fi

# Show verification command
echo "  Command: $TRITON_CLI verify --claim $CLAIM_FILE --proof $PROOF_FILE"
echo ""

# Run verification with timeout and capture output
# Use a temp file to capture output while also showing it
TEMP_VERIFY_OUTPUT=$(mktemp)
if timeout 30 "$TRITON_CLI" verify --claim "$CLAIM_FILE" --proof "$PROOF_FILE" 2>&1 | tee "$TEMP_VERIFY_OUTPUT"; then
    VERIFY_STATUS=0
else
    VERIFY_STATUS=$?
fi

VERIFY_OUTPUT=$(cat "$TEMP_VERIFY_OUTPUT")
rm -f "$TEMP_VERIFY_OUTPUT"

echo ""

# Check for verification success (triton-cli outputs "proof verified" or "✅ proof verified")
if [ $VERIFY_STATUS -eq 0 ] && ([[ "$VERIFY_OUTPUT" == *"verified"* ]] || [[ "$VERIFY_OUTPUT" == *"✅"* ]] || [[ "$VERIFY_OUTPUT" == *"proof verified"* ]]); then
    echo -e "${GREEN}✓ Verification PASSED${NC}"
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ${GREEN}SUCCESS${BLUE}                                                     ║${NC}"
    echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║  Component: ${YELLOW}$COMPONENT${NC}"
    echo -e "${BLUE}║  Input:     ${YELLOW}$INPUT${NC}"
    echo -e "${BLUE}║  Prove:    ${PROVE_TIME}s${NC}"
    echo -e "${BLUE}║  Status:   ${GREEN}VERIFIED${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}✗ Verification FAILED${NC}"
    echo ""
    echo "Verification output:"
    echo "$VERIFY_OUTPUT"
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  FAILURE                                                     ║${NC}"
    echo -e "${RED}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${RED}║  Component: ${YELLOW}$COMPONENT${NC}"
    echo -e "${RED}║  Input:     ${YELLOW}$INPUT${NC}"
    echo -e "${RED}║  Status:   ${RED}VERIFICATION FAILED${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

