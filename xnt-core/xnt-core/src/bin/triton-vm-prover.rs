//! Triton VM Prover with GPU support
//!
//! This binary handles proof generation for xnt-core. It supports two modes:
//!
//! 1. CPU mode (default): Uses Triton VM's native CPU prover
//! 2. GPU mode: When TRITON_GPU_PROVER_PATH is set, calls the external GPU prover
//!
//! Usage:
//!   # CPU mode (default)
//!   triton-vm-prover
//!
//!   # GPU mode
//!   TRITON_GPU_PROVER_PATH=/path/to/triton_vm_prove_gpu_full triton-vm-prover

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::io::BufRead;
use std::io::Write;
use std::process::{Command, Stdio};

use neptune_privacy::application::config::triton_vm_env_vars::TritonVmEnvVars;
use neptune_privacy::protocol::proof_abstractions::tasm::prover_job::PROOF_PADDED_HEIGHT_TOO_BIG_PROCESS_OFFSET_ERROR_CODE;
use tasm_lib::triton_vm::config::overwrite_lde_trace_caching_to;
use tasm_lib::triton_vm::config::CacheDecision;
use tasm_lib::triton_vm::prelude::Program;
use tasm_lib::triton_vm::proof::Claim;
use tasm_lib::triton_vm::proof::Proof;
use tasm_lib::triton_vm::stark::Stark;
use tasm_lib::triton_vm::vm::NonDeterminism;
use tasm_lib::triton_vm::vm::VM;
use thread_priority::set_current_thread_priority;
use thread_priority::ThreadPriority;

// Environment variable for GPU prover binary path
const GPU_PROVER_PATH_ENV: &str = "TRITON_GPU_PROVER_PATH";

// TODO: Replace by value exposed in Triton VM
const LDE_TRACE_ENV_VAR: &str = "TVM_LDE_TRACE";

fn set_environment_variables(env_vars: &[(String, String)]) {
    // Set environment variables for this spawned process only, does not apply
    // globally. Documentation of `set_var` shows it's for the currently
    // running process only.
    // This is only intended to set two environment variables: TVM_LDE_TRACE and
    // RAYON_NUM_THREADS, depending on the padded height of the algebraic
    // execution trace.
    for (key, value) in env_vars {
        eprintln!("Setting env variable for Triton VM: {key}={value}");

        // SAFETY:
        // - "The exact requirement is: you must ensure that there are no
        //   other threads concurrently writing or reading(!) the
        //   environment through functions or global variables other than
        //   the ones in this module." At this place, this program is
        //   single-threaded. Generation of algebraic execution trace is
        //   done, and proving hasn't started yet.
        unsafe {
            std::env::set_var(key, value);
        }

        // In case Triton VM has already set the cache decision prior to
        // the environment variable being set here, we override it through
        // a publicly exposed function.
        if key == LDE_TRACE_ENV_VAR {
            let maybe_overwrite = value.to_ascii_lowercase();
            let cache_lde_trace_overwrite = match maybe_overwrite.as_str() {
                "cache" => Some(CacheDecision::Cache),
                "no_cache" => Some(CacheDecision::NoCache),
                _ => None,
            };
            if let Some(cache_lde_trace_overwrite) = cache_lde_trace_overwrite {
                eprintln!("overwriting cache lde trace to: {cache_lde_trace_overwrite:?}");
                overwrite_lde_trace_caching_to(cache_lde_trace_overwrite);
            }
        }
    }
}

/// Execute proof generation using CPU (original Triton VM behavior)
fn execute_cpu(
    claim: Claim,
    program: Program,
    non_determinism: NonDeterminism,
    max_log2_padded_height: Option<u8>,
    env_vars: TritonVmEnvVars,
) -> Proof {
    let stark: Stark = Stark::default();

    let (aet, _) = VM::trace_execution(program, (&claim.input).into(), non_determinism).unwrap();
    let log2_padded_height = aet.padded_height().ilog2() as u8;

    // Use std-err for logging purposes since spawner (caller) doesn't get the
    // log outputs but can capture std-err.
    eprintln!("[CPU] actual log2 padded height for proof: {log2_padded_height}");

    if max_log2_padded_height.is_some_and(|max| log2_padded_height > max) {
        eprintln!(
            "[CPU] Canceling prover because padded height exceeds max value of {}",
            max_log2_padded_height.unwrap()
        );
        // Exit with error code indicating 1) AET padded height too big, and 2)
        // the log2 padded height. Guaranteed to be in the range [200-232].
        std::process::exit(
            PROOF_PADDED_HEIGHT_TOO_BIG_PROCESS_OFFSET_ERROR_CODE + i32::from(log2_padded_height),
        );
    }

    let env_vars = env_vars
        .get(&log2_padded_height)
        .map(|x| x.to_owned())
        .unwrap_or_default();

    set_environment_variables(&env_vars);

    stark.prove(&claim, &aet).unwrap()
}

/// Execute proof generation using GPU prover
fn execute_gpu(
    gpu_prover_path: &str,
    claim: &Claim,
    program: &Program,
    non_determinism: &NonDeterminism,
    max_log2_padded_height: Option<u8>,
) -> Result<Vec<u8>, String> {
    use std::fs;

    eprintln!("[GPU] =========================================");
    eprintln!("[GPU] Using GPU prover: {}", gpu_prover_path);

    // Run trace execution to determine padded height for environment variable configuration
    eprintln!("[GPU] Running trace execution to determine padded height...");
    let (aet, output) = VM::trace_execution(
        program.clone(),
        (&claim.input).into(),
        non_determinism.clone(),
    )
    .map_err(|e| format!("Failed to run trace execution: {:?}", e))?;

    // Verify output matches claim
    if output != claim.output {
        return Err(format!(
            "Output mismatch: claim expects {:?}, execution produced {:?}",
            claim.output, output
        ));
    }

    let padded_height = aet.padded_height();
    let log2_padded_height = padded_height.ilog2() as u8;
    eprintln!("[GPU] actual log2 padded height for proof: {log2_padded_height}");

    // Check padded height limit if specified
    if let Some(limit) = max_log2_padded_height {
        if log2_padded_height > limit {
            eprintln!(
                "[GPU] Canceling prover because padded height exceeds max value of {}",
                limit
            );
            // Exit with error code indicating 1) AET padded height too big, and 2)
            // the log2 padded height. Guaranteed to be in the range [200-232].
            std::process::exit(
                PROOF_PADDED_HEIGHT_TOO_BIG_PROCESS_OFFSET_ERROR_CODE + i32::from(log2_padded_height),
            );
        }
    }

    // Set TRITON_GPU_LDE_FRUGAL based on padded height threshold
    // Threshold: 2^22 (log2 = 22)
    // - If log2 >= 22: TRITON_GPU_LDE_FRUGAL=1
    // - If log2 < 22: TRITON_GPU_LDE_FRUGAL=0
    const LDE_FRUGAL_THRESHOLD_LOG2: u8 = 22;
    let lde_frugal_value = if log2_padded_height >= LDE_FRUGAL_THRESHOLD_LOG2 {
        "1"
    } else {
        "0"
    };
    eprintln!(
        "[GPU] Setting TRITON_GPU_LDE_FRUGAL={} (log2_padded_height={}, threshold={})",
        lde_frugal_value, log2_padded_height, LDE_FRUGAL_THRESHOLD_LOG2
    );

    // Create temp directory for files
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("triton_vm_prover_{}", ts));
    fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp dir: {}", e))?;

    let program_json_path = temp_dir.join("program.json");
    let nondet_json_path = temp_dir.join("nondet.json");
    let claim_path = temp_dir.join("program.claim");
    let proof_path = temp_dir.join("program.proof");

    // Serialize program and nondeterminism to JSON
    let program_json = serde_json::to_string(program)
        .map_err(|e| format!("Failed to serialize program: {}", e))?;
    let nondet_json = serde_json::to_string(non_determinism)
        .map_err(|e| format!("Failed to serialize nondeterminism: {}", e))?;

    // Write files
    fs::write(&program_json_path, &program_json)
        .map_err(|e| format!("Failed to write program.json: {}", e))?;
    fs::write(&nondet_json_path, &nondet_json)
        .map_err(|e| format!("Failed to write nondet.json: {}", e))?;

    eprintln!(
        "[GPU] Wrote program JSON: {} ({} bytes)",
        program_json_path.display(),
        program_json.len()
    );
    eprintln!(
        "[GPU] Wrote NonDeterminism JSON: {} ({} bytes)",
        nondet_json_path.display(),
        nondet_json.len()
    );

    // Format public input as comma-separated BFieldElements
    let public_input_str: String = if claim.input.is_empty() {
        String::new()
    } else {
        claim
            .input
            .iter()
            .map(|bfe| bfe.value().to_string())
            .collect::<Vec<_>>()
            .join(",")
    };

    // Check if NonDeterminism is non-trivial (has data)
    let has_nondet = !non_determinism.individual_tokens.is_empty()
        || !non_determinism.digests.is_empty()
        || !non_determinism.ram.is_empty();

    // Build GPU prover command
    // Usage: triton_vm_prove_gpu_full <program.json> <public_input> <output_claim> <output_proof> [nondet.json] [program.json]
    let mut cmd = Command::new(gpu_prover_path);
    cmd.arg(&program_json_path)
        .arg(&public_input_str)
        .arg(&claim_path)
        .arg(&proof_path);

    // Add NonDeterminism arguments if present
    if has_nondet {
        cmd.arg(&nondet_json_path);
        cmd.arg(&program_json_path);
    }

    // Set TRITON_GPU_LDE_FRUGAL environment variable for the GPU prover process
    cmd.env("TRITON_GPU_LDE_FRUGAL", lde_frugal_value);

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    eprintln!(
        "[GPU] Executing: {} {} \"{}\" {} {}{}",
        gpu_prover_path,
        program_json_path.display(),
        if public_input_str.is_empty() {
            "<empty>"
        } else {
            &public_input_str
        },
        claim_path.display(),
        proof_path.display(),
        if has_nondet {
            format!(
                " {} {}",
                nondet_json_path.display(),
                program_json_path.display()
            )
        } else {
            String::new()
        }
    );

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute GPU prover: {}", e))?;

    // Log GPU prover output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.is_empty() {
        for line in stdout.lines() {
            eprintln!("[GPU stdout] {}", line);
        }
    }
    if !stderr.is_empty() {
        for line in stderr.lines() {
            eprintln!("[GPU stderr] {}", line);
        }
    }

    if !output.status.success() {
        // Save failure artifacts for debugging
        let failed_dir = std::env::temp_dir().join(format!("triton_vm_prover_failed_{}", ts));
        
        // Create failed directory
        let _ = fs::create_dir_all(&failed_dir);
        
        // Copy all input files to failed directory for debugging
        // These should exist since we created them before running the prover
        if program_json_path.exists() {
            let _ = fs::copy(&program_json_path, failed_dir.join("program.json"));
        }
        if nondet_json_path.exists() {
            let _ = fs::copy(&nondet_json_path, failed_dir.join("nondet.json"));
        }
        if claim_path.exists() {
            let _ = fs::copy(&claim_path, failed_dir.join("claim.bin"));
        }
        
        // Save claim as JSON for easier inspection (always available from memory)
        let claim_json = serde_json::to_string_pretty(claim)
            .unwrap_or_else(|e| format!("Failed to serialize claim: {:?}", e));
        let _ = fs::write(failed_dir.join("claim.json"), claim_json);
        
        // Save stdout and stderr logs
        let _ = fs::write(failed_dir.join("gpu_stdout.log"), stdout.as_bytes());
        let _ = fs::write(failed_dir.join("gpu_stderr.log"), stderr.as_bytes());
        
        // Save debug information summary
        let debug_info = format!(
            "GPU Prover Failure Debug Information\n\
            ====================================\n\n\
            Timestamp: {}\n\
            GPU Prover Path: {}\n\
            Exit Code: {:?}\n\
            Log2 Padded Height: {}\n\
            Padded Height: {}\n\
            TRITON_GPU_LDE_FRUGAL: {}\n\
            Max Log2 Padded Height: {:?}\n\
            Public Input: {}\n\
            Has NonDeterminism: {}\n\
            Program JSON Size: {} bytes\n\
            NonDet JSON Size: {} bytes\n\n\
            Command executed:\n\
            {} {} \"{}\" {} {}{}\n\n\
            All input files and logs are saved in this directory.",
            ts,
            gpu_prover_path,
            output.status.code(),
            log2_padded_height,
            padded_height,
            lde_frugal_value,
            max_log2_padded_height,
            if public_input_str.is_empty() { "<empty>".to_string() } else { format!("{} values", claim.input.len()) },
            has_nondet,
            program_json.len(),
            nondet_json.len(),
            gpu_prover_path,
            program_json_path.display(),
            if public_input_str.is_empty() { "<empty>" } else { &public_input_str },
            claim_path.display(),
            proof_path.display(),
            if has_nondet {
                format!(" {} {}", nondet_json_path.display(), program_json_path.display())
            } else {
                String::new()
            }
        );
        let _ = fs::write(failed_dir.join("debug_info.txt"), debug_info);

        eprintln!(
            "[GPU] GPU prover failed; all artifacts saved to {}",
            failed_dir.display()
        );
        eprintln!(
            "[GPU] Debug info: log2_padded_height={}, TRITON_GPU_LDE_FRUGAL={}, exit_code={:?}",
            log2_padded_height,
            lde_frugal_value,
            output.status.code()
        );

        return Err(format!(
            "GPU prover failed with exit code: {:?}\nstderr: {}\n(artifacts saved to: {})",
            output.status.code(),
            stderr,
            failed_dir.display(),
        ));
    }

    // Read proof file
    if !proof_path.exists() {
        return Err(format!(
            "GPU prover did not create proof file: {}",
            proof_path.display()
        ));
    }

    let proof_bytes =
        fs::read(&proof_path).map_err(|e| format!("Failed to read proof file: {}", e))?;

    eprintln!(
        "[GPU] Proof file read successfully ({} bytes)",
        proof_bytes.len()
    );

    // Validate proof can be deserialized
    let proof: Proof = bincode::deserialize(&proof_bytes)
        .map_err(|e| format!("Failed to deserialize GPU proof: {}", e))?;

    // Check padded height limit if specified
    if let Some(limit) = max_log2_padded_height {
        let padded_height = proof
            .padded_height()
            .map_err(|e| format!("Failed to get padded height: {}", e))?;
        let log2_padded_height = padded_height.ilog2() as u8;

        if log2_padded_height > limit {
            eprintln!(
                "[GPU] Padded height {} exceeds limit {}",
                log2_padded_height, limit
            );
            // Cleanup
            let _ = fs::remove_dir_all(&temp_dir);
            // Exit with the same error code xnt-core expects
            std::process::exit(
                PROOF_PADDED_HEIGHT_TOO_BIG_PROCESS_OFFSET_ERROR_CODE + log2_padded_height as i32,
            );
        }
    }

    // Cleanup temp files
    let _ = fs::remove_dir_all(&temp_dir);

    eprintln!("[GPU] =========================================");

    Ok(proof_bytes)
}

fn main() {
    // Run with low priority so xnt-core remains responsive
    set_current_thread_priority(ThreadPriority::Min).unwrap();

    // Read inputs from stdin (5 JSON lines)
    let stdin = std::io::stdin();
    let mut iterator = stdin.lock().lines();

    let claim: Claim = serde_json::from_str(&iterator.next().unwrap().unwrap()).unwrap();
    let program: Program = serde_json::from_str(&iterator.next().unwrap().unwrap()).unwrap();
    let non_determinism: NonDeterminism =
        serde_json::from_str(&iterator.next().unwrap().unwrap()).unwrap();
    let max_log2_padded_height: Option<u8> =
        serde_json::from_str(&iterator.next().unwrap().unwrap()).unwrap();
    let env_variables: TritonVmEnvVars =
        serde_json::from_str(&iterator.next().unwrap().unwrap()).unwrap();

    // Check if GPU prover is available
    let proof_bytes = if let Ok(gpu_prover_path) = std::env::var(GPU_PROVER_PATH_ENV) {
        // GPU mode: use external GPU prover
        eprintln!("[GPU] GPU prover enabled via {}", GPU_PROVER_PATH_ENV);
        match execute_gpu(
            &gpu_prover_path,
            &claim,
            &program,
            &non_determinism,
            max_log2_padded_height,
        ) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("[GPU] ERROR: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // CPU mode: use native Triton VM prover
        eprintln!(
            "[CPU] Using CPU prover (set {} to enable GPU)",
            GPU_PROVER_PATH_ENV
        );
        let proof = execute_cpu(
            claim,
            program,
            non_determinism,
            max_log2_padded_height,
            env_variables,
        );
        eprintln!("[CPU] triton-vm: completed proof");
        bincode::serialize(&proof).unwrap()
    };

    // Write proof to stdout
    let mut stdout = std::io::stdout();
    stdout.write_all(&proof_bytes).unwrap();
    stdout.flush().unwrap();
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use tasm_lib::triton_vm;
    use tasm_lib::triton_vm::isa::triton_asm;

    use super::*;

    #[test]
    fn setting_tvm_env_vars_works() {
        let program = triton_asm!(halt);
        let program = Program::new(&program);
        let claim = Claim::about_program(&program);
        let non_determinism = NonDeterminism::default();
        let max_log2_padded_height = None;
        let mut env_vars = TritonVmEnvVars::default();
        env_vars.insert(
            8,
            vec![
                (LDE_TRACE_ENV_VAR.to_owned(), "no_cache".to_owned()),
                ("RAYON_NUM_THREADS".to_owned(), "3".to_owned()),
            ],
        );

        let proof = execute_cpu(
            claim.clone(),
            program,
            non_determinism,
            max_log2_padded_height,
            env_vars,
        );

        assert!(triton_vm::verify(Stark::default(), &claim, &proof));

        // Verify that env variables were actually set
        assert_eq!(
            "no_cache",
            std::env::var(LDE_TRACE_ENV_VAR).expect("Env variable for LDE trace must be set")
        );
        assert_eq!(
            "3",
            std::env::var("RAYON_NUM_THREADS").expect("Env variable for num threads must be set")
        );
    }

    #[test]
    fn make_halt_proof() {
        let program = triton_asm!(halt);
        let program = Program::new(&program);
        let claim = Claim::about_program(&program);
        let non_determinism = NonDeterminism::default();
        let max_log2_padded_height = None;
        let env_vars = TritonVmEnvVars::default();
        let proof = execute_cpu(
            claim.clone(),
            program,
            non_determinism,
            max_log2_padded_height,
            env_vars,
        );

        assert!(triton_vm::verify(Stark::default(), &claim, &proof));
    }

    #[test]
    fn halt_program() {
        let program = triton_asm!(halt);
        let program = Program::new(&program);
        let claim = Claim::about_program(&program);
        let non_determinism = NonDeterminism::default();

        println!("{}", serde_json::to_string(&claim).unwrap());
        println!("{}", serde_json::to_string(&program).unwrap());
        println!("{}", serde_json::to_string(&non_determinism).unwrap());
    }
}
