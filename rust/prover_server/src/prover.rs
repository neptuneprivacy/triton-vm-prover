//! Prover logic: parse Neptune's/XNT-Core's JSON, run GPU prover, return bincode proof
//!
//! This module supports two proving modes:
//! 1. Pure Rust proving (default) - uses triton_vm::stark::Stark::prove()
//! 2. GPU proving - calls the external GPU prover binary via subprocess
//!
//! Set TRITON_GPU_PROVER_PATH to enable GPU proving:
//!   export TRITON_GPU_PROVER_PATH=/path/to/triton_vm_prove_gpu_full

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use thiserror::Error;
use triton_vm::prelude::*;
use triton_vm::proof::{Claim, Proof};
use triton_vm::stark::Stark;

use crate::protocol::ProverRequest;

/// Environment variable for GPU prover binary path
const GPU_PROVER_PATH_ENV: &str = "TRITON_GPU_PROVER_PATH";

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("Failed to parse claim JSON: {0}")]
    ClaimParseError(#[source] serde_json::Error),
    
    #[error("Failed to parse program JSON: {0}")]
    ProgramParseError(#[source] serde_json::Error),
    
    #[error("Failed to parse non-determinism JSON: {0}")]
    NonDetParseError(#[source] serde_json::Error),
    
    #[error("Failed to parse max_log2 JSON: {0}")]
    MaxLog2ParseError(#[source] serde_json::Error),
    
    #[error("Failed to parse env_vars JSON: {0}")]
    EnvVarsParseError(#[source] serde_json::Error),
    
    #[error("Trace execution failed: {0}")]
    TraceExecutionFailed(String),
    
    #[error("Proving failed: {0}")]
    ProvingFailed(String),
    
    #[error("Proof serialization failed: {0}")]
    SerializationFailed(#[source] bincode::Error),
    
    #[error("Padded height {observed} exceeds limit {limit}")]
    PaddedHeightTooBig { observed: u32, limit: u32 },
    
    #[error("GPU prover error: {0}")]
    GpuProverError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result of a successful prove operation
pub struct ProveResult {
    pub proof_bincode: Vec<u8>,
    pub padded_height: u32,
}

/// Result that may indicate padded height too big
pub enum ProveOutcome {
    Success(ProveResult),
    PaddedHeightTooBig { observed_log2: u32 },
}

/// Environment variables configuration (matches Neptune's/XNT-Core's TritonVmEnvVars)
pub type EnvVarsConfig = HashMap<u8, Vec<(String, String)>>;

/// Check if GPU prover is available
/// 
/// Note: TRITON_FORCE_RUST_PROVER is ignored - GPU prover is always preferred when available
fn get_gpu_prover_path() -> Option<PathBuf> {
    // Always prefer GPU prover when TRITON_GPU_PROVER_PATH is set
    // TRITON_FORCE_RUST_PROVER is disabled - GPU prover will be used if available
    std::env::var(GPU_PROVER_PATH_ENV)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Parse the request and run the prover
/// 
/// This is the main entry point. It supports two modes:
/// 1. GPU proving (if TRITON_GPU_PROVER_PATH is set) - preferred
/// 2. Pure Rust proving (fallback when GPU prover is not available)
/// 
/// `gpu_device_id` specifies which GPU device to use (for multi-GPU setups)
/// `omp_num_threads` sets OMP_NUM_THREADS environment variable (None = use env/default)
/// `triton_omp_init` sets TRITON_OMP_INIT environment variable (None = use env/default)
pub fn prove_request(
    request: &ProverRequest,
    gpu_device_id: usize,
    omp_num_threads: Option<usize>,
    triton_omp_init: Option<bool>,
) -> Result<ProveOutcome, ProverError> {
    // Parse all JSON inputs
    let claim: Claim = serde_json::from_str(&request.claim_json)
        .map_err(ProverError::ClaimParseError)?;
    
    let program: Program = serde_json::from_str(&request.program_json)
        .map_err(ProverError::ProgramParseError)?;
    
    let non_determinism: NonDeterminism = serde_json::from_str(&request.nondet_json)
        .map_err(ProverError::NonDetParseError)?;
    
    let max_log2_padded_height: Option<u8> = serde_json::from_str(&request.max_log2_json)
        .map_err(ProverError::MaxLog2ParseError)?;
    
    let _env_vars: EnvVarsConfig = serde_json::from_str(&request.env_vars_json)
        .map_err(ProverError::EnvVarsParseError)?;
    
    tracing::info!(
        job_id = request.job_id,
        program_digest = %claim.program_digest,
        input_len = claim.input.len(),
        max_log2 = ?max_log2_padded_height,
        "Starting prove job"
    );
    
    // Verify program digest matches
    let computed_digest = program.hash();
    if computed_digest != claim.program_digest {
        return Err(ProverError::ProvingFailed(format!(
            "Program digest mismatch: claim has {:?}, program hashes to {:?}",
            claim.program_digest, computed_digest
        )));
    }
    
    // Run trace execution
    let public_input = PublicInput::new(claim.input.clone());
    let (aet, output) = VM::trace_execution(program.clone(), public_input, non_determinism.clone())
        .map_err(|e| ProverError::TraceExecutionFailed(format!("{:?}", e)))?;
    
    // Check output matches claim
    if output != claim.output {
        return Err(ProverError::ProvingFailed(format!(
            "Output mismatch: claim expects {:?}, execution produced {:?}",
            claim.output, output
        )));
    }
    
    // Check padded height limit
    let padded_height = aet.padded_height();
    let log2_padded_height = padded_height.ilog2() as u8;
    
    // Check NonDeterminism content
    let has_individual_tokens = !non_determinism.individual_tokens.is_empty();
    let has_digests = !non_determinism.digests.is_empty();
    let has_ram = !non_determinism.ram.is_empty();
    let has_nondet = has_individual_tokens || has_digests || has_ram;
    
    tracing::info!(
        padded_height = padded_height,
        log2_padded_height = log2_padded_height,
        "Trace execution complete"
    );
    
    // Debug logging for large proofs (log2=21)
    if log2_padded_height >= 21 {
        tracing::warn!(
            log2_padded_height = log2_padded_height,
            has_nondet = has_nondet,
            individual_tokens = non_determinism.individual_tokens.len(),
            digests = non_determinism.digests.len(),
            ram_entries = non_determinism.ram.len(),
            "[LARGE PROOF] Detected padded_height >= 2^21"
        );
    }
    
    if let Some(limit) = max_log2_padded_height {
        if log2_padded_height > limit {
            tracing::warn!(
                observed = log2_padded_height,
                limit = limit,
                "Padded height exceeds limit"
            );
            return Ok(ProveOutcome::PaddedHeightTooBig {
                observed_log2: log2_padded_height as u32,
            });
        }
    }
    
    // Check if GPU prover is available
    let proof_bincode = if let Some(gpu_prover_path) = get_gpu_prover_path() {
        tracing::info!("[PROVER] Step 1: Starting GPU STARK proving on device {}...", gpu_device_id);
        tracing::info!("[PROVER] GPU prover path: {}", gpu_prover_path.display());
        
        prove_with_gpu(
            &gpu_prover_path,
            request.job_id as u64,
            &claim,
            &program,
            &non_determinism,
            &request.program_json,
            &request.nondet_json,
            gpu_device_id,
            omp_num_threads,
            triton_omp_init,
        )?
    } else {
        // Use pure Rust proving
        tracing::info!("[PROVER] Step 1: Starting STARK proving (Rust)...");
        
        let start_time = std::time::Instant::now();
        let stark = Stark::default();
        let proof = stark.prove(&claim, &aet)
            .map_err(|e| ProverError::ProvingFailed(format!("{:?}", e)))?;
        let prove_duration = start_time.elapsed();
        
        tracing::info!(
            proof_len = proof.0.len(),
            duration_ms = prove_duration.as_millis(),
            "[PROVER] Step 2: STARK proof computation complete"
        );
        
        tracing::info!("[PROVER] Step 3: Serializing proof to bincode...");
        let serialize_start = std::time::Instant::now();
        
        // Serialize proof to bincode (exactly as Neptune/XNT-Core expects)
        let proof_bincode = bincode::serialize(&proof)
            .map_err(ProverError::SerializationFailed)?;
        
        let serialize_duration = serialize_start.elapsed();
        
        tracing::info!(
            bincode_len = proof_bincode.len(),
            duration_ms = serialize_duration.as_millis(),
            "[PROVER] Step 4: Proof serialization complete"
        );
        
        proof_bincode
    };
    
    tracing::info!(
        "[PROVER] All steps complete, ready to send response"
    );
    
    Ok(ProveOutcome::Success(ProveResult {
        proof_bincode,
        padded_height: log2_padded_height as u32,
    }))
}

/// Call the GPU prover binary as a subprocess
/// 
/// The GPU prover expects:
///   Input: Program (JSON or TASM), public input, and optionally NonDeterminism JSON
///   Output: Proof file (bincode format) and claim file
/// 
/// For Neptune/XNT-Core integration, we:
/// 1. Write the program JSON to a temp file
/// 2. Write the NonDeterminism JSON to a temp file (if non-trivial)
/// 3. Call GPU prover with the JSON files
/// 4. Read the generated proof file
/// 5. Return the proof bytes (already in bincode format from Rust FFI encoding)
fn prove_with_gpu(
    gpu_prover_path: &PathBuf,
    neptune_job_id: u64,
    claim: &Claim,
    _program: &Program,
    non_determinism: &NonDeterminism,
    program_json: &str,
    nondet_json: &str,
    gpu_device_id: usize,
    omp_num_threads: Option<usize>,
    triton_omp_init: Option<bool>,
) -> Result<Vec<u8>, ProverError> {
    use std::fs;
    
    let start_time = std::time::Instant::now();
    
    // Create temp directory for files
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("gpu_prover_{}_{}", neptune_job_id, ts));
    fs::create_dir_all(&temp_dir)?;
    
    let program_json_path = temp_dir.join("program.json");
    let nondet_json_path = temp_dir.join("nondet.json");
    let claim_json_path = temp_dir.join("claim.json");
    let public_input_path = temp_dir.join("public_input.txt");
    let claim_path = temp_dir.join("program.claim");
    let proof_path = temp_dir.join("program.proof");
    
    // Write program JSON to file
    fs::write(&program_json_path, program_json)?;
    
    tracing::info!(
        "[PROVER] [GPU] Wrote program JSON to {} ({} bytes)",
        program_json_path.display(),
        program_json.len()
    );
    
    // Check if NonDeterminism is non-trivial (has RAM or secret input)
    let has_nondet = !non_determinism.individual_tokens.is_empty()
        || !non_determinism.digests.is_empty()
        || !non_determinism.ram.is_empty();
    
    // Always write nondet.json for repro convenience (even if empty/trivial)
    fs::write(&nondet_json_path, nondet_json)?;
    tracing::info!(
        "[PROVER] [GPU] Wrote NonDeterminism JSON to {} ({} bytes)",
        nondet_json_path.display(),
        nondet_json.len()
    );

    // Also persist claim.json for repro/debug
    if let Ok(claim_json_pretty) = serde_json::to_string_pretty(claim) {
        let _ = fs::write(&claim_json_path, claim_json_pretty);
    }
    
    // Format public input as comma-separated
    let public_input_str: String = if claim.input.is_empty() {
        String::new()
    } else {
        claim.input
            .iter()
            .map(|bfe| bfe.value().to_string())
            .collect::<Vec<_>>()
            .join(",")
    };

    // Save public input string for repro/debug
    let _ = fs::write(&public_input_path, &public_input_str);

    // Helper: persist failure artifacts and write a repro script.
    let persist_failure = |reason: &str, gpu_stdout: &str, gpu_stderr: &str| -> std::path::PathBuf {
        let failed_dir = std::env::temp_dir().join(format!("gpu_prover_failed_{}_{}", neptune_job_id, ts));
        let _ = fs::create_dir_all(&failed_dir);

        // Best-effort rename temp_dir -> failed_dir (atomic if same filesystem)
        let _ = fs::rename(&temp_dir, &failed_dir);

        let _ = fs::write(failed_dir.join("gpu_stdout.log"), gpu_stdout);
        let _ = fs::write(failed_dir.join("gpu_stderr.log"), gpu_stderr);
        let _ = fs::write(failed_dir.join("reason.txt"), reason);

        // Repro script:
        // - runs GPU prover with the same arguments
        // - verifies claim/proof with triton-cli (if available)
        let repro_sh = format!(
            r#"#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

GPU_PROVER="{gpu}"
TRITON_CLI_DEFAULT="{triton_cli_default}"

PROGRAM_JSON="$DIR/program.json"
NONDET_JSON="$DIR/nondet.json"
PUBLIC_INPUT="$(cat "$DIR/public_input.txt" 2>/dev/null || true)"
CLAIM_BIN="$DIR/program.claim"
PROOF_BIN="$DIR/program.proof"

echo "[repro] dir: $DIR"
echo "[repro] gpu_prover: $GPU_PROVER"
echo "[repro] program_json: $PROGRAM_JSON"
echo "[repro] nondet_json: $NONDET_JSON"
echo "[repro] public_input: ${{PUBLIC_INPUT:-<empty>}}"
echo ""

if [[ ! -x "$GPU_PROVER" ]]; then
  echo "[repro] ERROR: GPU prover not executable at: $GPU_PROVER" >&2
  exit 1
fi

ARGS=("$PROGRAM_JSON" "$PUBLIC_INPUT" "$CLAIM_BIN" "$PROOF_BIN")

# Only pass nondet/program extra args if nondet.json is non-trivial (cheap check: file size > 5 bytes for "{{}}\\n")
if [[ -s "$NONDET_JSON" ]] && [[ $(wc -c < "$NONDET_JSON") -gt 5 ]]; then
  ARGS+=("$NONDET_JSON" "$PROGRAM_JSON")
fi

echo "[repro] running: $GPU_PROVER ${{ARGS[*]}}"
"$GPU_PROVER" "${{ARGS[@]}}"

echo ""
echo "[repro] produced:"
ls -lh "$CLAIM_BIN" "$PROOF_BIN" || true

TRITON_CLI="${{TRITON_CLI:-$TRITON_CLI_DEFAULT}}"
if [[ -x "$TRITON_CLI" ]]; then
  echo ""
  echo "[repro] verifying with triton-cli: $TRITON_CLI"
  "$TRITON_CLI" verify --claim "$CLAIM_BIN" --proof "$PROOF_BIN"
else
  echo ""
  echo "[repro] NOTE: triton-cli not found at: $TRITON_CLI"
  echo "[repro] Set TRITON_CLI=/path/to/triton-cli to auto-verify."
fi
"#,
            gpu = gpu_prover_path.display(),
            triton_cli_default = "/home/speedy/Documents/workspace-alt/workspace/triton-cli-1.0.0/target/release/triton-cli"
        );
        let repro_path = failed_dir.join("repro.sh");
        let _ = fs::write(&repro_path, repro_sh);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&repro_path, fs::Permissions::from_mode(0o755));
        }

        failed_dir
    };
    
    // Build GPU prover command
    // Usage: triton_vm_prove_gpu_full <program.json> <public_input> <output_claim> <output_proof> [nondet.json] [program.json]
    // Set CUDA_VISIBLE_DEVICES to restrict to the assigned GPU device
    let mut cmd = Command::new(gpu_prover_path);
    cmd.arg(&program_json_path)
        .arg(&public_input_str)
        .arg(&claim_path)
        .arg(&proof_path);
    
    // Add NonDeterminism and Program JSON arguments if NonDeterminism is present
    if has_nondet {
        cmd.arg(&nondet_json_path);
        cmd.arg(&program_json_path);
    }
    
    // Set CUDA_VISIBLE_DEVICES to use only the assigned GPU device
    cmd.env("CUDA_VISIBLE_DEVICES", gpu_device_id.to_string());
    
    // Set OpenMP thread count if specified
    if let Some(threads) = omp_num_threads {
        cmd.env("OMP_NUM_THREADS", threads.to_string());
    }
    
    // Set TRITON_OMP_INIT if specified
    if let Some(init_enabled) = triton_omp_init {
        cmd.env("TRITON_OMP_INIT", if init_enabled { "1" } else { "0" });
    }
    
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    tracing::info!(
        "[PROVER] [GPU] Calling GPU prover on device {}: {} {} {} {} {}{}",
        gpu_device_id,
        gpu_prover_path.display(),
        program_json_path.display(),
        if public_input_str.is_empty() { "<empty>" } else { &public_input_str },
        claim_path.display(),
        proof_path.display(),
        if has_nondet { format!(" {} {}", nondet_json_path.display(), program_json_path.display()) } else { String::new() }
    );
    
    let output = cmd.output()?;
    
    // Log GPU prover output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    if !stdout.is_empty() {
        for line in stdout.lines() {
            tracing::info!("[PROVER] [GPU stdout] {}", line);
        }
    }
    if !stderr.is_empty() {
        for line in stderr.lines() {
            tracing::warn!("[PROVER] [GPU stderr] {}", line);
        }
    }
    
    if !output.status.success() {
        // Check if this is a program execution error (expected for invalid programs)
        let stderr_lower = stderr.to_lowercase();
        let is_program_error = stderr_lower.contains("opstack underflow")
            || stderr_lower.contains("trace execution failed")
            || stderr_lower.contains("vm error")
            || stderr_lower.contains("assertion failed");

        let failed_dir = persist_failure(
            &format!("gpu_prover_exit_code={:?}", output.status.code()),
            &stdout,
            &stderr,
        );
        tracing::error!(
            "[PROVER] [GPU] GPU prover failed; saved repro to {}",
            failed_dir.display()
        );

        return Err(ProverError::GpuProverError(format!(
            "GPU prover failed with exit code: {:?}\nstderr: {}{}\n(repro saved to: {})",
            output.status.code(),
            stderr,
            if is_program_error { "\n(This appears to be a program execution error, not a prover bug)" } else { "" },
            failed_dir.display(),
        )));
    }
    
    // Read proof file
    if !proof_path.exists() {
        let failed_dir = persist_failure("gpu_prover_missing_proof_file", &stdout, &stderr);
        return Err(ProverError::GpuProverError(
            format!(
                "GPU prover did not create proof file (repro saved to: {})",
                failed_dir.display()
            )
        ));
    }
    
    // The proof file from GPU prover is encoded using Rust FFI (BFieldCodec + bincode)
    // We need to read it and then re-serialize using bincode::serialize for Neptune/XNT-Core
    let proof_bytes = fs::read(&proof_path)?;
    
    let gpu_duration = start_time.elapsed();
    tracing::info!(
        proof_len = proof_bytes.len(),
        duration_ms = gpu_duration.as_millis(),
        "[PROVER] [GPU] Proof file read successfully"
    );
    
    // The GPU prover already outputs bincode-serialized proof via Rust FFI
    // Validate that the proof is valid bincode (but skip verification - it will be verified by the client)
    let _proof: Proof = match bincode::deserialize::<Proof>(&proof_bytes) {
        Ok(p) => {
            tracing::info!(
                proof_items = p.0.len(),
                "[PROVER] [GPU] Proof deserialized successfully (verification skipped - will be verified by client)"
            );
            p
        }
        Err(e) => {
            let failed_dir = persist_failure(
                &format!("gpu_proof_deserialize_failed: {e}"),
                &stdout,
                &stderr,
            );
            return Err(ProverError::GpuProverError(format!(
                "Failed to deserialize GPU proof: {} (repro saved to: {})",
                e,
                failed_dir.display()
            )));
        }
    };
    
    // Cleanup temp files
    let _ = fs::remove_dir_all(&temp_dir);
    
    Ok(proof_bytes)
}

/// Async wrapper for prove_request (runs in blocking thread pool)
pub async fn prove_request_async(
    request: ProverRequest,
    gpu_device_id: usize,
    omp_num_threads: Option<usize>,
    triton_omp_init: Option<bool>,
) -> Result<ProveOutcome, ProverError> {
    // Run proving in a blocking thread since it's CPU-intensive
    tokio::task::spawn_blocking(move || prove_request(&request, gpu_device_id, omp_num_threads, triton_omp_init))
        .await
        .map_err(|e| ProverError::ProvingFailed(format!("Task join error: {:?}", e)))?
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_halt_program() {
        let claim_json = r#"{"program_digest":"49390b5279de3843c90d85289b12a2e65004866a98d03cfdbca7eb0c91bafd6962c094958c115b7e","version":0,"input":[],"output":[]}"#;
        let program_json = r#"{"instructions":["Halt"],"address_to_label":{},"debug_information":{"breakpoints":[false],"type_hints":{},"assertion_context":{}}}"#;
        let nondet_json = r#"{"individual_tokens":[],"digests":[],"ram":{}}"#;
        
        let claim: Claim = serde_json::from_str(claim_json).unwrap();
        let program: Program = serde_json::from_str(program_json).unwrap();
        let nondet: NonDeterminism = serde_json::from_str(nondet_json).unwrap();
        
        // Verify program matches claim digest
        assert_eq!(program.hash(), claim.program_digest);
        
        // Run trace execution
        let public_input = PublicInput::new(claim.input.clone());
        let (aet, output) = VM::trace_execution(program, public_input, nondet).unwrap();
        
        assert_eq!(output, claim.output);
        assert!(aet.padded_height() > 0);
    }
}

