#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::Command;

use std::time::Instant;
use thread_priority::ThreadPriority;
use thread_priority::set_current_thread_priority;
use tracing::{error, info};
use triton_vm::prelude::{BFieldElement, Program};
use triton_vm::proof::Claim;
use triton_vm::stark::Stark;
use triton_vm::twenty_first::math::ntt;
use triton_vm::vm::{NonDeterminism, VM};

use rand::Rng;

fn test_ntt() {
    let mut rng = rand::thread_rng();
    const POWER_START: i32 = 256;
    const POWER_END: i32 = 1024 * 1024 * 2;

    let mut current_power = POWER_START;
    while current_power <= POWER_END {
        let mut arr: Vec<BFieldElement> = Vec::with_capacity(current_power as usize);
        let mut arr2: Vec<BFieldElement> = Vec::with_capacity(current_power as usize);
        for _ in 0..current_power {
            let val: u64 = rng.r#gen();
            arr.push(BFieldElement::new(val));
            arr2.push(BFieldElement::new(val));
        }

        eprintln!("TEST -> {}", current_power);
        ntt::ntt(&mut arr);
        ntt::ntt_scalar(&mut arr2);
        eprintln!("");

        let mut same = true;
        for i in 0..arr.len() {
            if arr[i].raw_u64() != arr2[i].raw_u64() {
                same = false;
                break;
            }
        }
        assert!(same == true);

        current_power *= 2;
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    // Run with low priority so that neptune-core can remain responsive.
    set_current_thread_priority(ThreadPriority::Min).unwrap();

    // Get input file path from command line
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 {
        if args[1] == "test_ntt" {
            test_ntt();
            return;
        }
    }
    if args.len() < 3 {
        eprintln!("Usage: {} <input_file> <output_file> [verifier_path]", args[0]);
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let verifier_path = args.get(3).map(|s| s.as_str());

    // Open the file and read three JSON lines
    let file = File::open(input_path).expect("failed to open input file");
    let mut lines = BufReader::new(file).lines();

    let claim: Claim = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    let program: Program = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    let non_determinism: NonDeterminism =
        serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();

    let default_stark: Stark = Stark::default();

    // Start GPU initialization in background to hide the 420ms latency
    // The initialization will complete during VM trace execution
    #[cfg(feature = "gpu")]
    let gpu_init_handle = {
        use triton_vm::twenty_first::math::ntt::gpu_ntt::get_gpu_context;
        info!("triton-vm: starting GPU initialization in background");
        std::thread::spawn(|| {
            get_gpu_context();
        })
    };

    // Start profiling
    triton_vm::profiler::start("Triton VM Proof Generation");

    info!("triton-vm: starting proof");
    let start = Instant::now();

    // Use same execution logic as triton-vm-prover (doesn't validate claim digest)
    // This allows replaying dumps where claim and program may not match exactly
    let (aet, _) = match VM::trace_execution(
        program,
        (&claim.input).into(),
        non_determinism
    ) {
        Ok(result) => result,
        Err(e) => {
            error!("VM execution failed: {}", e);
            std::process::exit(1);
        }
    };

    // GPU LDE caching strategy:
    // - Check TVM_LDE_TRACE environment variable first
    // - If not set and GPU is available, enable caching (optimized for B200 and high-VRAM systems)
    // - For smaller GPUs, set TVM_LDE_TRACE=no_cache to use JIT mode
    // Wait for GPU initialization to complete before starting proof generation
    // This ensures the 420ms initialization happened in parallel with VM trace execution
    #[cfg(feature = "gpu")]
    {
        info!("triton-vm: waiting for background GPU initialization to complete");
        gpu_init_handle.join().expect("GPU initialization thread panicked");
        info!("triton-vm: GPU initialization completed");
    }

    #[cfg(feature = "gpu")]
    {
        use std::env;
        use triton_vm::twenty_first::math::ntt::gpu_ntt::get_gpu_context;

        // Check environment variable
        let env_override = env::var("TVM_LDE_TRACE").ok();

        let enable_cache = match env_override.as_deref() {
            Some("cache") => {
                info!("triton-vm: LDE caching FORCED ENABLED via TVM_LDE_TRACE=cache");
                true
            }
            Some("no_cache") => {
                info!("triton-vm: LDE caching FORCED DISABLED via TVM_LDE_TRACE=no_cache");
                false
            }
            _ => {
                // Auto-detect: enable if GPU is available (optimized for B200)
                if get_gpu_context().is_some() {
                    info!("triton-vm: GPU detected - enabling LDE caching for optimal performance");
                    info!("triton-vm: (Set TVM_LDE_TRACE=no_cache if running out of GPU memory)");
                    true
                } else {
                    false
                }
            }
        };

        if enable_cache {
            triton_vm::config::overwrite_lde_trace_caching_to(triton_vm::config::CacheDecision::Cache);
            info!("triton-vm: LDE caching ENABLED - using 2M quotient domain with GPU-resident optimization");
        } else {
            triton_vm::config::overwrite_lde_trace_caching_to(triton_vm::config::CacheDecision::NoCache);
            info!("triton-vm: LDE caching DISABLED (JIT mode) - using 262K quotient domain");
        }
    }

    let proof = match default_stark.prove(&claim, &aet) {
        Ok(proof) => proof,
        Err(e) => {
            error!("Proof generation failed: {}", e);
            std::process::exit(1);
        }
    };
    let duration = start.elapsed();
    info!("triton-vm: completed proof in {:.2?}", duration);

    // Get and print the profiling results
    let profile = triton_vm::profiler::finish();
    eprintln!("\n{}", profile);

    // Print GPU timing summary (commented out to reduce log noise)
    // triton_vm::twenty_first::math::ntt::print_gpu_timing_summary();

    // Serialize to binary
    let as_bytes = bincode::serialize(&proof).unwrap();

    // Write proof to output.bin
    let mut out = File::create(output_path).expect("failed to create output.bin");
    out.write_all(&as_bytes).unwrap();
    out.flush().unwrap();

    info!("proof written to {}", output_path);

    // Verify the proof
    if let Some(verifier_bin) = verifier_path {
        // Use external verifier (unmodified triton-vm)
        info!("verifying with external verifier: {}", verifier_bin);

        // Save claim to a temporary JSON file
        let claim_path = "/tmp/claim.json";
        let claim_json = serde_json::to_string(&claim).unwrap();
        let mut claim_file = File::create(claim_path).expect("failed to create claim file");
        claim_file.write_all(claim_json.as_bytes()).unwrap();
        claim_file.flush().unwrap();

        // Run the external verifier
        let status = Command::new(verifier_bin)
            .arg(claim_path)
            .arg(output_path)
            .status()
            .expect("failed to execute verifier");

        if status.success() {
            info!("External verification SUCCESSFUL");
        } else {
            error!("External verification FAILED with exit code: {:?}", status.code());
            std::process::exit(1);
        }
    } else {
        // Use internal verification (may be optimized triton-vm)
        info!("verifying with internal verifier...");
        match default_stark.verify(&claim, &proof) {
            Ok(()) => {
                info!("Proof is correct");
            }
            Err(err) => {
                error!("Proof verification failed -> {:?}", err);
                std::process::exit(1);
            }
        }
    }
}
