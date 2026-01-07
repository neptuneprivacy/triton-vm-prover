#!/usr/bin/env bash
#
# Use a prover-server dump directory (program.json + nondet.json + public_input.txt)
# to generate a PURE RUST claim/proof via triton-cli, then verify it.
# Also dumps intermediate test data for GPU prover comparison.
#
# Usage:
#   ./run_triton_cli_rust_prove_from_dump.sh /tmp/gpu_prover_failed_<jobid>_<ts>
#
# Optional env:
#   TRITON_CLI=/path/to/triton-cli
#   SKIP_TEST_DATA=1  - skip test data generation
#
set -euo pipefail

DUMP_DIR="${1:-}"
if [[ -z "$DUMP_DIR" ]]; then
  echo "Usage: $0 /tmp/gpu_prover_failed_<jobid>_<ts>" >&2
  exit 1
fi
if [[ ! -d "$DUMP_DIR" ]]; then
  echo "Error: directory not found: $DUMP_DIR" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_CLI_DEFAULT="$SCRIPT_DIR/../triton-cli-1.0.0/target/release/triton-cli"
TRITON_CLI="${TRITON_CLI:-$TRITON_CLI_DEFAULT}"

if [[ ! -x "$TRITON_CLI" ]]; then
  echo "Error: triton-cli not found/executable at: $TRITON_CLI" >&2
  echo "Build it with: cd $SCRIPT_DIR/../triton-cli-1.0.0 && cargo build --release" >&2
  exit 1
fi

VMSTATE_JSON="$DUMP_DIR/initial_state.json"
CLAIM_JSON="$DUMP_DIR/claim.rust.json"
PROOF_BIN="$DUMP_DIR/proof.rust.bin"
TEST_DATA_DIR="$DUMP_DIR/rust_test_data"

echo "[rust-prove] dump dir: $DUMP_DIR"
echo "[rust-prove] triton-cli: $TRITON_CLI"
echo "[rust-prove] vmstate:   $VMSTATE_JSON"
echo "[rust-prove] claim:     $CLAIM_JSON"
echo "[rust-prove] proof:     $PROOF_BIN"
echo "[rust-prove] test data: $TEST_DATA_DIR"
echo

echo "[rust-prove] building vmstate json from dump..."
cd "$SCRIPT_DIR/rust/prover_server"
cargo run --release --bin dump-to-vmstate -- "$DUMP_DIR" "$VMSTATE_JSON"

echo
echo "[rust-prove] generating claim/proof via triton-cli prove..."
# Enable test data dumping to capture intermediate Rust prover values
mkdir -p "$TEST_DATA_DIR"
if [[ "${SKIP_TEST_DATA:-0}" != "1" ]]; then
  echo "[rust-prove] Enabling intermediate test data generation..."
  export TVM_DUMP_TEST_DATA="$TEST_DATA_DIR"
  export TVM_LIGHT_DUMP_MODE=0   # Full data, not sampled
  export TVM_DUMP_DETAILED=1     # Include detailed files
  export TVM_GENERATE_LIGHT_TEST_DATA=1
  export TVM_TEST_DATA_DIR="$TEST_DATA_DIR"
fi

"$TRITON_CLI" prove --initial-state "$VMSTATE_JSON" --claim "$CLAIM_JSON" --proof "$PROOF_BIN"

echo
echo "[rust-prove] verifying via triton-cli verify..."
"$TRITON_CLI" verify --claim "$CLAIM_JSON" --proof "$PROOF_BIN"

echo
echo "[rust-prove] DONE"
echo

if [[ -d "$TEST_DATA_DIR" ]]; then
  echo "[rust-prove] Test data files generated:"
  ls -la "$TEST_DATA_DIR"/ | head -30
  echo "..."
  echo "Total files: $(ls -1 "$TEST_DATA_DIR" | wc -l)"
fi
