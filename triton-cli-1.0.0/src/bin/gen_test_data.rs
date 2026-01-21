//! Test data generator for functional verification when converting Rust to C++.
//!
//! This tool captures intermediate states at each step of the proving process
//! for input 16 (or any specified input) to use as reference data when implementing
//! the C++ version.

use anyhow::{Context, Result, anyhow};
use serde_json;
use std::fs;
use std::path::PathBuf;
use triton_vm::prelude::*;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: gen_test_data <program.tasm> <input> [output_dir]");
        eprintln!("Example: gen_test_data spin.tasm 16 test_data");
        std::process::exit(1);
    }

    let program_path = &args[1];
    let input_str = &args[2];
    let default_output = "test_data".to_string();
    let output_dir = args.get(3).map(|s| s.as_str()).unwrap_or(&default_output);

    let input: Vec<u64> = input_str
        .split(',')
        .map(|s| s.trim().parse().context("Failed to parse input as u64"))
        .collect::<Result<Vec<_>>>()?;
    let input_bfe: Vec<BFieldElement> = input.iter().map(|&x| x.into()).collect();

    // Load program
    let program_source = fs::read_to_string(program_path)
        .with_context(|| format!("Failed to read program from {}", program_path))?;
    // Use map_err to work around lifetime issues (same pattern as args.rs)
    let program = Program::from_code(&program_source)
        .map_err(|err| anyhow!("Failed to parse program: {err}"))?;

    // Create output directory
    let output_path = PathBuf::from(output_dir);
    fs::create_dir_all(&output_path)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    // Set environment variable to enable test data dumping in prove()
    let output_path_str = output_path
        .to_str()
        .ok_or_else(|| anyhow!("Output path contains invalid UTF-8"))?;
    unsafe {
        std::env::set_var("TVM_DUMP_TEST_DATA", output_path_str);
    }

    println!("Generating test data for input: {:?}", input);
    println!("Output directory: {}", output_path.display());
    println!(
        "Test data dumping enabled via TVM_DUMP_TEST_DATA={}",
        output_path_str
    );

    // Step 1: Trace execution
    println!("\n[1/15] Trace execution...");
    let (aet, public_output) = VM::trace_execution(
        program.clone(),
        PublicInput::new(input_bfe.clone()),
        NonDeterminism::default(),
    )?;

    // Save AET summary
    let aet_data = serde_json::json!({
        "processor_trace_height": aet.processor_trace.nrows(),
        "processor_trace_width": aet.processor_trace.ncols(),
        "public_output": public_output.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "padded_height": aet.padded_height(),
    });
    fs::write(
        output_path.join("01_trace_execution.json"),
        serde_json::to_string_pretty(&aet_data)?,
    )?;

    // Save sample rows from processor trace
    let first_row: Vec<u64> = aet
        .processor_trace
        .row(0)
        .iter()
        .map(|x| x.value())
        .collect();
    let last_row: Vec<u64> = if aet.processor_trace.nrows() > 1 {
        aet.processor_trace
            .row(aet.processor_trace.nrows() - 1)
            .iter()
            .map(|x| x.value())
            .collect()
    } else {
        vec![]
    };

    let processor_trace_sample = serde_json::json!({
        "first_row": first_row,
        "last_row": last_row,
    });
    fs::write(
        output_path.join("01_trace_execution_sample.json"),
        serde_json::to_string_pretty(&processor_trace_sample)?,
    )?;

    // Step 2: Create claim and derive parameters
    println!("[2/15] Creating claim and deriving parameters...");
    let claim = Claim::about_program(&program).with_input(input_bfe.clone());
    let claim = claim.with_output(public_output.clone());

    let padded_height = aet.padded_height();
    let stark = Stark::default();
    let fri = stark.fri(padded_height)?;

    let params_data = serde_json::json!({
        "padded_height": padded_height,
        "log2_padded_height": padded_height.ilog2(),
        "fri_domain_length": fri.domain.length,
        "note": "Full domain information requires access to internal ProverDomains (not public API)",
    });
    fs::write(
        output_path.join("02_parameters.json"),
        serde_json::to_string_pretty(&params_data)?,
    )?;

    // Run full prove - this will automatically dump all intermediate states
    // because TVM_DUMP_TEST_DATA is set
    println!("\n[3-19/19] Running full prove (will dump all intermediate states)...");
    let proof = stark.prove(&claim, &aet)?;

    // Save proof metadata
    let proof_data = serde_json::json!({
        "padded_height": proof.padded_height()?,
        "note": "Full intermediate states require modifying prove() to dump at each step",
        "reference_log": "See reference.log for timing breakdown of each step",
    });
    fs::write(
        output_path.join("proof_metadata.json"),
        serde_json::to_string_pretty(&proof_data)?,
    )?;

    println!("\nTest data generation complete!");
    println!("Files saved to: {}", output_path.display());
    println!("\nNext steps:");
    println!("1. Review reference.log to understand all proving steps");
    println!("2. Consider modifying triton-vm/src/stark.rs prove() function");
    println!("   to dump intermediate states at each profiler!(start/stop) point");
    println!("3. Use this test data as reference for C++ implementation");

    Ok(())
}
