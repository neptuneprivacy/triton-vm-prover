//! Generate actual trace randomizer coefficients from Rust for C++ verification.

use std::fs;
use std::path::PathBuf;
use anyhow::{anyhow, Result};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use triton_vm::prelude::*;
use triton_vm::table::master_table::MasterMainTable;
use triton_vm::arithmetic_domain::ArithmeticDomain;
use triton_vm::stark::{Stark, ProverDomains};
use serde_json;

fn main() -> Result<()> {
    let output_dir = PathBuf::from("test_data_lde_cases");
    fs::create_dir_all(&output_dir)?;
    
    println!("Generating trace randomizer coefficients...\n");
    
    // Load program and execute
    let program_path = "spin.tasm";
    let program_source = fs::read_to_string(program_path)
        .map_err(|e| anyhow!("Failed to read program: {}", e))?;
    
    let program = Program::from_code(&program_source)
        .map_err(|e| anyhow!("Failed to parse program: {}", e))?;
    
    let public_input = PublicInput::new(vec![8_u64.into()]);
    let non_determinism = NonDeterminism::default();
    
    println!("[1/3] Executing program...");
    let (aet, public_output) = VM::trace_execution(program.clone(), public_input.clone(), non_determinism.clone())
        .map_err(|e| anyhow!("Execution failed: {}", e))?;
    
    let padded_height = aet.padded_height();
    
    // Create table
    let stark = Stark::default();
    let claim = Claim::about_program(&program)
        .with_input(public_input.individual_tokens.clone())
        .with_output(public_output.clone());
    
    let fri = stark.fri(padded_height)?;
    let domains = ProverDomains::derive(
        padded_height,
        stark.num_trace_randomizers,
        fri.domain,
        stark.max_degree(padded_height),
    );
    
    println!("[2/3] Creating table...");
    let mut rng = StdRng::from_entropy();
    let seed: <StdRng as SeedableRng>::Seed = {
        use rand::RngCore;
        let mut seed = <StdRng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        seed
    };
    let master_main_table = MasterMainTable::new(
        &aet,
        domains,
        stark.num_trace_randomizers,
        seed,
    );
    
    println!("[3/3] Extracting randomizer coefficients...");
    
    // Get trace domain
    let trace_domain = master_main_table.domains().trace;
    let trace_randomizer_seed = master_main_table.trace_randomizer_seed();
    let num_trace_randomizers = master_main_table.num_trace_randomizers();
    let quotient_domain = master_main_table.domains().quotient;
    
    // Extract first column's randomizer
    let trace_randomizer = master_main_table.trace_randomizer_for_column(0);
    let randomizer_coeffs: Vec<u64> = trace_randomizer.coefficients()
        .iter()
        .map(|x| x.value())
        .collect();
    
    // Also get the randomized interpolant
    let randomized_interpolant = master_main_table.randomized_column_interpolant(0);
    
    // Get trace table first column
    let trace_table = master_main_table.trace_table();
    let first_column: Vec<BFieldElement> = trace_table.column(0).to_vec();
    let column_interpolant = trace_domain.interpolate(&first_column);
    
    // Save comprehensive data
    let seed_hex: String = trace_randomizer_seed.iter()
        .map(|&b| format!("{:02x}", b))
        .collect();
    
    let data = serde_json::json!({
        "test_case": "trace_randomizer_coefficients",
        "description": "Actual trace randomizer coefficients from Rust for exact C++ matching",
        "trace_domain": {
            "length": trace_domain.length,
            "generator": trace_domain.generator.value(),
            "offset": trace_domain.offset.value(),
        },
        "quotient_domain": {
            "length": quotient_domain.length,
            "generator": quotient_domain.generator.value(),
            "offset": quotient_domain.offset.value(),
        },
        "randomizer_info": {
            "seed_bytes": trace_randomizer_seed.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            "seed_hex": seed_hex,
            "column_index": 0,
            "num_trace_randomizers": num_trace_randomizers,
        },
        "input": {
            "trace_column_index": 0,
            "trace_values": first_column.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        },
        "randomizer": {
            "coefficients": randomizer_coeffs,
            "num_coefficients": randomizer_coeffs.len(),
        },
        "interpolants": {
            "column_interpolant_coefficients": column_interpolant.coefficients()
                .iter()
                .map(|x| x.value())
                .collect::<Vec<u64>>(),
            "randomized_interpolant_coefficients_first_100": randomized_interpolant.coefficients()
                .iter()
                .take(100)
                .map(|x| x.value())
                .collect::<Vec<u64>>(),
        },
    });
    
    fs::write(output_dir.join("04_randomizer_coefficients.json"), serde_json::to_string_pretty(&data)?)?;
    println!("  ✓ Wrote 04_randomizer_coefficients.json");
    println!("     Seed: {}", seed_hex);
    println!("     Randomizer coefficients: {}", randomizer_coeffs.len());
    println!("     Column interpolant coefficients: {}", column_interpolant.coefficients().len());
    
    Ok(())
}

