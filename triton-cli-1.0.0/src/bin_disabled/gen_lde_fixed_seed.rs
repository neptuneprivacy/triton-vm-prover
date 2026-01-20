//! Generate LDE test data with fixed seed - extracts randomizer data from existing prove output.

use std::fs;
use std::path::PathBuf;
use anyhow::{anyhow, Result};
use triton_vm::prelude::*;
use triton_vm::arithmetic_domain::ArithmeticDomain;
use serde_json;

fn main() -> Result<()> {
    let output_dir = PathBuf::from("test_data_lde_cases");
    fs::create_dir_all(&output_dir)?;
    
    println!("Generating LDE fixed seed test data...\n");
    
    // Load existing test data
    let padded_table_path = "test_data_lde/04_main_tables_pad.json";
    let params_path = "test_data_lde/02_parameters.json";
    let rust_lde_path = "test_data_lde/05_main_tables_lde.json";
    
    let padded_data: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(padded_table_path)
            .map_err(|e| anyhow!("Failed to read padded table: {}. Run gen_test_data first.", e))?
    )?;
    
    let params: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(params_path)?
    )?;
    
    let rust_lde_data: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(rust_lde_path)?
    )?;
    
    // Extract first column from padded table
    let padded_rows = padded_data["padded_table_data"].as_array().unwrap();
    let first_column: Vec<BFieldElement> = padded_rows.iter()
        .map(|row| BFieldElement::new(row[0].as_u64().unwrap()))
        .collect();
    
    // Extract Rust's LDE first column
    let rust_lde_rows = rust_lde_data["lde_table_data"].as_array().unwrap();
    let rust_lde_column: Vec<u64> = rust_lde_rows.iter()
        .map(|row| row[0].as_u64().unwrap())
        .collect();
    
    // Get domain parameters
    let trace_len = params["trace_domain"]["length"].as_u64().unwrap() as usize;
    let trace_gen = params["trace_domain"]["generator"].as_u64().unwrap();
    let quot_len = params["quotient_domain"]["length"].as_u64().unwrap() as usize;
    let quot_gen = params["quotient_domain"]["generator"].as_u64().unwrap();
    let quot_offset = params["quotient_domain"]["offset"].as_u64().unwrap();
    
    println!("  Computing difference between zero-randomizer and Rust LDE...");
    
    // Compute zero-randomizer LDE
    let trace_domain = ArithmeticDomain::of_length(trace_len)?;
    let quot_domain = ArithmeticDomain::of_length(quot_len)?
        .with_offset(BFieldElement::new(quot_offset));
    
    let interpolant = trace_domain.interpolate(&first_column);
    let zero_lde = quot_domain.evaluate(&interpolant);
    
    // Compute the difference (this is the randomizer contribution)
    // Use BFieldElement arithmetic for proper modular subtraction
    let mut randomizer_contribution: Vec<u64> = Vec::new();
    for i in 0..rust_lde_column.len() {
        let rust_val = BFieldElement::new(rust_lde_column[i]);
        let zero_val = zero_lde[i];
        let diff = (rust_val - zero_val).value();
        randomizer_contribution.push(diff);
    }
    
    // Create comprehensive test data structure
    let data = serde_json::json!({
        "test_case": "fixed_seed_implementation",
        "description": "Complete data for implementing LDE with trace randomizers in C++",
        "algorithm": {
            "steps": [
                "1. Interpolate trace column: trace_values -> polynomial coefficients",
                "2. Generate trace randomizer polynomial (see randomizer_info)",
                "3. Compute zerofier for trace domain: x^n - offset^n",
                "4. Multiply: zerofier * randomizer",
                "5. Add: randomized_interpolant = interpolant + (zerofier * randomizer)",
                "6. Evaluate randomized_interpolant on quotient domain",
            ],
            "zerofier_formula": {
                "note": "x^n - offset^n where n = trace_domain.length",
                "evaluates_to_zero_on": "All points in trace domain",
            },
        },
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
        },
        "intermediate": {
            "zero_randomizer_lde": zero_lde.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        },
        "randomizer_info": {
            "note": "Randomizer contribution computed as: rust_lde - zero_randomizer_lde",
            "contribution_on_quotient_domain": randomizer_contribution,
            "implementation_note": "To generate randomizer: use StdRng with seed + column_index, generate num_trace_randomizers coefficients",
        },
        "output": {
            "rust_lde_with_randomizers": rust_lde_column,
            "expected_output": "rust_lde_with_randomizers should match C++ computation",
        },
    });
    
    fs::write(output_dir.join("03_fixed_seed_implementation.json"), serde_json::to_string_pretty(&data)?)?;
    println!("  ✓ Wrote 03_fixed_seed_implementation.json");
    
    println!("\n  Summary:");
    println!("    Zero-randomizer LDE: {} values", zero_lde.len());
    println!("    Rust LDE (with randomizers): {} values", rust_lde_column.len());
    println!("    Randomizer contribution computed");
    println!("\n  Note: This file contains all data needed to implement randomized LDE in C++");
    
    Ok(())
}
