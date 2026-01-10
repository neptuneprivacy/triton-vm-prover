#!/usr/bin/env bash
#
# Re-run GPU prover from a dumped repro directory produced by prover-server.
#
# Usage:
#   ./run_gpu_prover_from_dump.sh /tmp/gpu_prover_failed_<jobid>_<ts>
#
# Optional env:
#   TRITON_GPU_PROVER_PATH=/path/to/triton_vm_prove_gpu_full
#   TRITON_CLI=/path/to/triton-cli
#
set -euo pipefail

DIR="${1:-}"
if [[ -z "$DIR" ]]; then
  echo "Usage: $0 /path/to/gpu_prover_failed_<jobid>_<ts>" >&2
  exit 1
fi
if [[ ! -d "$DIR" ]]; then
  echo "Error: directory not found: $DIR" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_PROVER_DEFAULT="$SCRIPT_DIR/build/triton_vm_prove_gpu_full"
TRITON_CLI_DEFAULT="$SCRIPT_DIR/../triton-cli-1.0.0/target/release/triton-cli"

GPU_PROVER="${TRITON_GPU_PROVER_PATH:-$GPU_PROVER_DEFAULT}"
TRITON_CLI="${TRITON_CLI:-$TRITON_CLI_DEFAULT}"

PROGRAM_JSON="$DIR/program.json"
NONDET_JSON="$DIR/nondet.json"
PUBLIC_INPUT="$(cat "$DIR/public_input.txt" 2>/dev/null || true)"
CLAIM_BIN="$DIR/program.claim"
PROOF_BIN="$DIR/program.proof"

if [[ ! -f "$PROGRAM_JSON" ]]; then
  echo "Error: missing $PROGRAM_JSON" >&2
  exit 1
fi
if [[ ! -f "$NONDET_JSON" ]]; then
  echo "Error: missing $NONDET_JSON" >&2
  exit 1
fi

echo "[dump] dir: $DIR"
echo "[dump] gpu_prover: $GPU_PROVER"
echo "[dump] triton-cli: $TRITON_CLI"
echo "[dump] program.json: $PROGRAM_JSON"
echo "[dump] nondet.json:  $NONDET_JSON"
echo "[dump] public_input: ${PUBLIC_INPUT:-<empty>}"
echo

if [[ ! -x "$GPU_PROVER" ]]; then
  echo "Error: GPU prover not executable: $GPU_PROVER" >&2
  exit 1
fi

ARGS=("$PROGRAM_JSON" "$PUBLIC_INPUT" "$CLAIM_BIN" "$PROOF_BIN")

# Only pass nondet/program extra args if nondet.json is non-trivial (cheap check: file size > 5 bytes for "{}\n")
if [[ -s "$NONDET_JSON" ]] && [[ $(wc -c < "$NONDET_JSON") -gt 5 ]]; then
  ARGS+=("$NONDET_JSON" "$PROGRAM_JSON")
fi

echo "[dump] running: $GPU_PROVER ${ARGS[*]}"
"$GPU_PROVER" "${ARGS[@]}"

echo
echo "[dump] produced:"
ls -lh "$CLAIM_BIN" "$PROOF_BIN" || true

if [[ -x "$TRITON_CLI" ]]; then
  echo
  echo "[dump] verifying with triton-cli..."
  "$TRITON_CLI" verify --claim "$CLAIM_BIN" --proof "$PROOF_BIN"
else
  echo
  echo "[dump] NOTE: triton-cli not found/executable at: $TRITON_CLI"
  echo "[dump] Set TRITON_CLI=/path/to/triton-cli to auto-verify."
fi


