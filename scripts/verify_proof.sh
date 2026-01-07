#!/bin/bash
#
# Quick Proof Verification Script
#
# Verifies a claim/proof pair with triton-cli
#
# Usage:
#   ./scripts/verify_proof.sh <claim_file> <proof_file>
#   ./scripts/verify_proof.sh <claim_file>  # (will auto-find matching proof)
#
# Examples:
#   ./scripts/verify_proof.sh /tmp/claim_20251219_213310.bin /tmp/proof_20251219_213310.bin
#   ./scripts/verify_proof.sh /tmp/claim_20251219_213310.bin

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CLAIM_FILE="$1"
PROOF_FILE="$2"

# If proof file not provided, try to find matching one
if [ -z "$PROOF_FILE" ] && [ -n "$CLAIM_FILE" ]; then
    # Extract timestamp from claim filename
    TIMESTAMP=$(basename "$CLAIM_FILE" | grep -oP 'claim_\K[0-9_]+' || echo "")
    if [ -n "$TIMESTAMP" ]; then
        PROOF_FILE=$(dirname "$CLAIM_FILE")/proof_${TIMESTAMP}.bin
    fi
fi

# Validate inputs
if [ -z "$CLAIM_FILE" ] || [ -z "$PROOF_FILE" ]; then
    echo -e "${RED}Usage: $0 <claim_file> [proof_file]${NC}"
    echo "  If proof_file is omitted, will try to find matching proof by timestamp"
    exit 1
fi

if [ ! -f "$CLAIM_FILE" ]; then
    echo -e "${RED}Error: Claim file not found: $CLAIM_FILE${NC}"
    exit 1
fi

if [ ! -f "$PROOF_FILE" ]; then
    echo -e "${RED}Error: Proof file not found: $PROOF_FILE${NC}"
    exit 1
fi

# Find triton-cli
TRITON_CLI_DIR="$PROJECT_DIR/../triton-cli-1.0.0"
TRITON_CLI="$TRITON_CLI_DIR/target/release/triton-cli"

if [ ! -f "$TRITON_CLI" ]; then
    echo -e "${YELLOW}Building triton-cli...${NC}"
    cd "$TRITON_CLI_DIR"
    cargo build --release 2>&1 | tail -3
    cd "$PROJECT_DIR"
fi

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Proof Verification                                          ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Claim: ${YELLOW}$CLAIM_FILE${NC}"
echo -e "${BLUE}║  Proof: ${YELLOW}$PROOF_FILE${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show file sizes
CLAIM_SIZE=$(ls -lh "$CLAIM_FILE" | awk '{print $5}')
PROOF_SIZE=$(ls -lh "$PROOF_FILE" | awk '{print $5}')
echo "  Claim size: $CLAIM_SIZE"
echo "  Proof size: $PROOF_SIZE"
echo ""

# Verify
echo -e "${CYAN}━━━ Verifying Proof ━━━${NC}"
START_TIME=$(date +%s.%N)

VERIFY_OUTPUT=$("$TRITON_CLI" verify --claim "$CLAIM_FILE" --proof "$PROOF_FILE" 2>&1)
VERIFY_STATUS=$?

END_TIME=$(date +%s.%N)
VERIFY_TIME=$(awk "BEGIN {printf \"%.3f\", $END_TIME - $START_TIME}")

# Display output
echo "$VERIFY_OUTPUT"
echo ""

# Check result
if [ $VERIFY_STATUS -eq 0 ] && ([[ "$VERIFY_OUTPUT" == *"verified"* ]] || [[ "$VERIFY_OUTPUT" == *"✅"* ]]); then
    echo -e "${GREEN}✓ Verification PASSED${NC} (${VERIFY_TIME}s)"
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ${GREEN}SUCCESS${BLUE}                                                     ║${NC}"
    echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║  Status:   ${GREEN}VERIFIED${NC}"
    echo -e "${BLUE}║  Verify:   ${VERIFY_TIME}s${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}✗ Verification FAILED${NC}"
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  FAILURE                                                     ║${NC}"
    echo -e "${RED}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${RED}║  Status:   ${RED}VERIFICATION FAILED${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

