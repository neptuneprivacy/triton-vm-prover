//! Modified triton-vm-prover with GPU prover socket proxy support
//!
//! When TRITON_VM_PROVER_SOCKET is set (e.g., "127.0.0.1:5555" or "/tmp/prover.sock"),
//! this binary forwards proving requests to the GPU prover server instead of
//! running locally.
//!
//! Usage:
//!   # Run locally (default behavior, same as upstream)
//!   triton-vm-prover
//!
//!   # Forward to GPU prover server
//!   TRITON_VM_PROVER_SOCKET=127.0.0.1:5555 triton-vm-prover
//!
//!   # Forward to Unix socket
//!   TRITON_VM_PROVER_SOCKET=/tmp/gpu-prover.sock triton-vm-prover

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::io::BufRead;
use std::io::Read;
use std::io::Write;
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;

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

// Socket protocol constants (must match prover_server/src/protocol.rs)
const MAGIC_REQUEST: u32 = 0x54564D50; // "TVMP"
const MAGIC_RESPONSE: u32 = 0x54564D52; // "TVMR"
const PROTOCOL_VERSION: u32 = 1;

const RESPONSE_STATUS_OK: u32 = 0;
const RESPONSE_STATUS_PADDED_HEIGHT_TOO_BIG: u32 = 1;
const RESPONSE_STATUS_ERROR: u32 = 2;

// Environment variable name for socket address
const SOCKET_ENV_VAR: &str = "TRITON_VM_PROVER_SOCKET";

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

/// Execute locally (original behavior)
fn execute_local(
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
    eprintln!("actual log2 padded height for proof: {log2_padded_height}");

    if max_log2_padded_height.is_some_and(|max| log2_padded_height > max) {
        eprintln!(
            "Canceling prover because padded height exceeds max value of {}",
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

/// Write a length-prefixed string
fn write_length_prefixed<W: Write>(writer: &mut W, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

/// Forward proving request to GPU prover server via socket
fn execute_via_socket(
    socket_addr: &str,
    claim_json: &str,
    program_json: &str,
    nondet_json: &str,
    max_log2_json: &str,
    env_vars_json: &str,
) -> Result<Vec<u8>, String> {
    eprintln!("[proxy] Connecting to GPU prover at {}", socket_addr);
    
    // Connect to server
    let mut stream: Box<dyn ReadWrite> = if socket_addr.starts_with('/') {
        // Unix socket
        #[cfg(unix)]
        {
            Box::new(UnixStream::connect(socket_addr)
                .map_err(|e| format!("Failed to connect to Unix socket {}: {}", socket_addr, e))?)
        }
        #[cfg(not(unix))]
        {
            return Err("Unix sockets not supported on this platform".to_string());
        }
    } else {
        // TCP socket
        Box::new(TcpStream::connect(socket_addr)
            .map_err(|e| format!("Failed to connect to TCP socket {}: {}", socket_addr, e))?)
    };
    
    eprintln!("[proxy] Connected, sending request");
    
    // Generate a job ID
    let job_id: u32 = std::process::id();
    
    // Write request header
    stream.write_all(&MAGIC_REQUEST.to_le_bytes())
        .map_err(|e| format!("Failed to write magic: {}", e))?;
    stream.write_all(&PROTOCOL_VERSION.to_le_bytes())
        .map_err(|e| format!("Failed to write version: {}", e))?;
    stream.write_all(&job_id.to_le_bytes())
        .map_err(|e| format!("Failed to write job_id: {}", e))?;
    
    // Write 5 JSON strings
    write_length_prefixed(&mut stream, claim_json)
        .map_err(|e| format!("Failed to write claim: {}", e))?;
    write_length_prefixed(&mut stream, program_json)
        .map_err(|e| format!("Failed to write program: {}", e))?;
    write_length_prefixed(&mut stream, nondet_json)
        .map_err(|e| format!("Failed to write nondet: {}", e))?;
    write_length_prefixed(&mut stream, max_log2_json)
        .map_err(|e| format!("Failed to write max_log2: {}", e))?;
    write_length_prefixed(&mut stream, env_vars_json)
        .map_err(|e| format!("Failed to write env_vars: {}", e))?;
    
    stream.flush().map_err(|e| format!("Failed to flush: {}", e))?;
    
    eprintln!("[proxy] Request sent (job_id={}), waiting for response...", job_id);
    
    // Read response
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    
    eprintln!("[proxy] [WAIT] Reading response header...");
    
    // Read and verify magic
    stream.read_exact(&mut buf4)
        .map_err(|e| format!("Failed to read response magic: {}", e))?;
    let magic = u32::from_le_bytes(buf4);
    if magic != MAGIC_RESPONSE {
        return Err(format!("Invalid response magic: expected 0x{:08X}, got 0x{:08X}", MAGIC_RESPONSE, magic));
    }
    eprintln!("[proxy] [WAIT] Response magic OK");
    
    // Read status
    stream.read_exact(&mut buf4)
        .map_err(|e| format!("Failed to read status: {}", e))?;
    let status = u32::from_le_bytes(buf4);
    eprintln!("[proxy] [WAIT] Response status: {}", status);
    
    // Read job_id (echo back)
    stream.read_exact(&mut buf4)
        .map_err(|e| format!("Failed to read job_id: {}", e))?;
    let response_job_id = u32::from_le_bytes(buf4);
    eprintln!("[proxy] [WAIT] Response job_id: {} (expected: {})", response_job_id, job_id);
    
    match status {
        RESPONSE_STATUS_OK => {
            eprintln!("[proxy] [WAIT] Status OK, reading proof length...");
            // Read proof length
            stream.read_exact(&mut buf8)
                .map_err(|e| format!("Failed to read proof length: {}", e))?;
            let proof_len = u64::from_le_bytes(buf8);
            
            eprintln!("[proxy] [WAIT] Proof length: {} bytes, receiving proof data...", proof_len);
            
            // Read proof bytes
            let mut proof_bincode = vec![0u8; proof_len as usize];
            stream.read_exact(&mut proof_bincode)
                .map_err(|e| format!("Failed to read proof: {}", e))?;
            
            eprintln!("[proxy] [DONE] Proof received successfully ({} bytes)", proof_bincode.len());
            Ok(proof_bincode)
        }
        RESPONSE_STATUS_PADDED_HEIGHT_TOO_BIG => {
            // Read observed log2
            stream.read_exact(&mut buf4)
                .map_err(|e| format!("Failed to read observed_log2: {}", e))?;
            let observed_log2 = u32::from_le_bytes(buf4);
            
            eprintln!("[proxy] Padded height too big: {}", observed_log2);
            
            // Exit with the same error code xnt-core expects
            std::process::exit(
                PROOF_PADDED_HEIGHT_TOO_BIG_PROCESS_OFFSET_ERROR_CODE + observed_log2 as i32
            );
        }
        RESPONSE_STATUS_ERROR => {
            // Read error message length
            stream.read_exact(&mut buf4)
                .map_err(|e| format!("Failed to read error length: {}", e))?;
            let msg_len = u32::from_le_bytes(buf4) as usize;
            
            // Read error message
            let mut msg_bytes = vec![0u8; msg_len];
            stream.read_exact(&mut msg_bytes)
                .map_err(|e| format!("Failed to read error message: {}", e))?;
            let message = String::from_utf8_lossy(&msg_bytes);
            
            // Check if this is a program execution error (invalid program)
            // These are expected during block composition and should not cause xnt-core to shut down
            let is_program_error = message.contains("OpStack underflow") 
                || message.contains("would go below")
                || message.contains("execution failed")
                || message.contains("invalid program");
            
            if is_program_error {
                eprintln!("[proxy] INFO: GPU prover reported program execution error: {}", message);
                // Return empty proof - xnt-core will handle this gracefully
                Ok(vec![])
            } else {
                // Fatal error - exit with code 1
                Err(format!("GPU prover error: {}", message))
            }
        }
        _ => {
            Err(format!("Unknown response status: {}", status))
        }
    }
}

/// Trait for unified Read + Write
trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

fn main() {
    // run with a low priority so that xnt-core can remain responsive.
    set_current_thread_priority(ThreadPriority::Min).unwrap();

    // Read 5 JSON lines from stdin
    let stdin = std::io::stdin();
    let mut iterator = stdin.lock().lines();
    
    let claim_json = iterator.next().unwrap().unwrap();
    let program_json = iterator.next().unwrap().unwrap();
    let nondet_json = iterator.next().unwrap().unwrap();
    let max_log2_json = iterator.next().unwrap().unwrap();
    let env_vars_json = iterator.next().unwrap().unwrap();
    
    // Check if we should forward to GPU prover server
    let proof_bytes = if let Ok(socket_addr) = std::env::var(SOCKET_ENV_VAR) {
        eprintln!("[proxy] =========================================");
        eprintln!("[proxy] Forwarding to GPU prover server at {}", socket_addr);
        eprintln!("[proxy] =========================================");
        
        match execute_via_socket(
            &socket_addr,
            &claim_json,
            &program_json,
            &nondet_json,
            &max_log2_json,
            &env_vars_json,
        ) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("[proxy] ERROR: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Parse JSON and execute locally (original behavior)
        let claim: Claim = serde_json::from_str(&claim_json).unwrap();
        let program: Program = serde_json::from_str(&program_json).unwrap();
        let non_determinism: NonDeterminism = serde_json::from_str(&nondet_json).unwrap();
        let max_log2_padded_height: Option<u8> = serde_json::from_str(&max_log2_json).unwrap();
        let env_variables: TritonVmEnvVars = serde_json::from_str(&env_vars_json).unwrap();

        let proof = execute_local(
            claim,
            program,
            non_determinism,
            max_log2_padded_height,
            env_variables,
        );
        
        eprintln!("triton-vm: completed proof");
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

        let proof = execute_local(
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
        let proof = execute_local(
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
