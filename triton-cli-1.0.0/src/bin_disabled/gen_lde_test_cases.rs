//! Generate LDE test cases with zero randomizers for exact matching.

use std::fs;
use std::path::PathBuf;
use anyhow::{anyhow, Result};
use triton_vm::prelude::*;
use triton_vm::arithmetic_domain::ArithmeticDomain;
use serde_json;

fn main() -> Result<()> {
    let output_dir = PathBuf::from("test_data_lde_cases");
    fs::create_dir_all(&output_dir)?;
    
    println!("Generating LDE test cases for exact matching...\n");
    
    // Test Case 1: Zero Randomizers (pure interpolation + evaluation)
    println!("[1/2] Test Case 1: Zero Randomizers");
    generate_zero_randomizers_test(&output_dir)?;
    
    // Test Case 2: Document structure for fixed seed (C++ will need to implement randomizer)
    println!("\n[2/2] Test Case 2: Fixed Seed Structure");
    generate_fixed_seed_structure(&output_dir)?;
    
    println!("\n✅ All LDE test cases generated in: {}", output_dir.display());
    
    Ok(())
}

/// Test Case 1: LDE without randomizers (pure interpolation + evaluation)
/// This allows C++ to verify exact matching since no random values are involved.
fn generate_zero_randomizers_test(dir: &PathBuf) -> Result<()> {
    // Load existing padded table data
    let padded_table_path = "test_data_lde/04_main_tables_pad.json";
    let params_path = "test_data_lde/02_parameters.json";
    
    let padded_data: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(padded_table_path)
            .map_err(|e| anyhow!("Failed to read padded table: {}. Run gen_test_data first.", e))?
    )?;
    
    let params: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(params_path)?
    )?;
    
    println!("  Loading data...");
    
    // Extract parameters
    let trace_len = params["trace_domain"]["length"].as_u64().unwrap() as usize;
    let trace_gen = params["trace_domain"]["generator"].as_u64().unwrap();
    let quot_len = params["quotient_domain"]["length"].as_u64().unwrap() as usize;
    let quot_gen = params["quotient_domain"]["generator"].as_u64().unwrap();
    let quot_offset = params["quotient_domain"]["offset"].as_u64().unwrap();
    
    // Extract first column from padded table
    let padded_rows = padded_data["padded_table_data"].as_array().unwrap();
    let first_column: Vec<BFieldElement> = padded_rows.iter()
        .map(|row| BFieldElement::new(row[0].as_u64().unwrap()))
        .collect();
    
    println!("  Computing LDE without randomizers...");
    
    // Create domains
    let trace_domain = ArithmeticDomain::of_length(trace_len)?;
    let quot_domain = ArithmeticDomain::of_length(quot_len)?
        .with_offset(BFieldElement::new(quot_offset));
    
    // Step 1: Interpolate on trace domain (no randomizer)
    let interpolant = trace_domain.interpolate(&first_column);
    
    // Step 2: Evaluate on quotient domain
    let lde_column = quot_domain.evaluate(&interpolant);
    
    // Save test data
    let data = serde_json::json!({
        "test_case": "zero_randomizers",
        "description": "LDE computed without trace randomizers (pure interpolation + evaluation). This allows 100% exact matching with C++.",
        "trace_domain": {
            "length": trace_len,
            "generator": trace_gen,
            "offset": 1,
        },
        "quotient_domain": {
            "length": quot_len,
            "generator": quot_gen,
            "offset": quot_offset,
        },
        "input": {
            "trace_column_index": 0,
            "trace_values": first_column.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "num_values": first_column.len(),
        },
        "intermediate": {
            "coefficients_after_interpolation": interpolant.coefficients().iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "coefficients_first_16": interpolant.coefficients().iter().take(16).map(|x| x.value()).collect::<Vec<u64>>(),
        },
        "output": {
            "lde_values": lde_column.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "num_values": lde_column.len(),
            "lde_values_first_16": lde_column.iter().take(16).map(|x| x.value()).collect::<Vec<u64>>(),
        },
    });
    
    fs::write(dir.join("01_zero_randomizers.json"), serde_json::to_string_pretty(&data)?)?;
    println!("  ✓ Wrote 01_zero_randomizers.json");
    println!("     Input: {} trace values", first_column.len());
    println!("     Output: {} LDE values", lde_column.len());
    
    Ok(())
}

/// Test Case 2: Document the structure for fixed seed testing
/// This explains what's needed for C++ to match Rust's LDE with randomizers.
fn generate_fixed_seed_structure(dir: &PathBuf) -> Result<()> {
    // Load Rust's actual LDE output
    let rust_lde_path = "test_data_lde/05_main_tables_lde.json";
    let rust_lde_data: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(rust_lde_path)
            .map_err(|e| anyhow!("Failed to read Rust LDE: {}. Run gen_test_data first.", e))?
    )?;
    
    let params_path = "test_data_lde/02_parameters.json";
    let params: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(params_path)?
    )?;
    
    let padded_table_path = "test_data_lde/04_main_tables_pad.json";
    let padded_data: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(padded_table_path)?
    )?;
    
    // Extract first column
    let padded_rows = padded_data["padded_table_data"].as_array().unwrap();
    let first_column: Vec<u64> = padded_rows.iter()
        .map(|row| row[0].as_u64().unwrap())
        .collect();
    
    // Extract Rust's LDE first column
    let rust_lde_rows = rust_lde_data["lde_table_data"].as_array().unwrap();
    let rust_lde_column: Vec<u64> = rust_lde_rows.iter()
        .map(|row| row[0].as_u64().unwrap())
        .collect();
    
    let data = serde_json::json!({
        "test_case": "fixed_seed_structure",
        "description": "Structure documentation for testing LDE with fixed randomizer seed. C++ needs to implement the same randomizer generation algorithm as Rust.",
        "note": "Rust's LDE includes trace randomizers: interpolant + zerofier * randomizer. To match exactly, C++ needs the same randomizer polynomial coefficients.",
        "trace_domain": {
            "length": params["trace_domain"]["length"],
            "generator": params["trace_domain"]["generator"],
            "offset": params["trace_domain"]["offset"],
        },
        "quotient_domain": {
            "length": params["quotient_domain"]["length"],
            "generator": params["quotient_domain"]["generator"],
            "offset": params["quotient_domain"]["offset"],
        },
        "input": {
            "trace_column_index": 0,
            "trace_values": first_column,
            "num_values": first_column.len(),
        },
        "rust_output": {
            "lde_values": rust_lde_column,
            "num_values": rust_lde_column.len(),
            "lde_values_first_16": rust_lde_column.iter().take(16).copied().collect::<Vec<u64>>(),
        },
        "algorithm_steps": [
            "1. Interpolate trace column to get polynomial coefficients",
            "2. Generate trace randomizer polynomial (degree = num_trace_randomizers - 1)",
            "3. Compute zerofier polynomial for trace domain",
            "4. Compute: randomized_interpolant = interpolant + zerofier * randomizer",
            "5. Evaluate randomized_interpolant on quotient domain",
        ],
        "implementation_notes": {
            "randomizer_generation": "Use StdRng with seed derived from trace_randomizer_seed and column index",
            "zerofier": "x^n - offset^n where n = trace_domain.length",
            "zerofier_multiply": "Fast multiplication using shift_coefficients (multiply by x^n) then subtract",
        },
    });
    
    fs::write(dir.join("02_fixed_seed_structure.json"), serde_json::to_string_pretty(&data)?)?;
    println!("  ✓ Wrote 02_fixed_seed_structure.json");
    println!("     This documents the structure for fixed-seed testing");
    println!("     Rust LDE output has {} values", rust_lde_column.len());
    
    Ok(())
}
