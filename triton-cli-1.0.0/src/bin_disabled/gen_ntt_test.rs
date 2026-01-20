//! Generate NTT/LDE test data for C++ verification.

use std::fs;
use std::path::PathBuf;
use anyhow::{anyhow, Result};
use triton_vm::prelude::*;
use triton_vm::arithmetic_domain::ArithmeticDomain;
use triton_vm::stark::Stark;
use twenty_first::prelude::*;
use twenty_first::math::ntt::{intt, ntt};
use twenty_first::math::traits::PrimitiveRootOfUnity;

fn main() -> Result<()> {
    let output_dir = PathBuf::from("test_data_ntt");
    fs::create_dir_all(&output_dir)?;
    
    println!("Generating NTT test data for C++ verification...\n");
    
    // Test 1: Simple NTT on small input
    println!("[1/4] Simple NTT test (size 8)...");
    generate_simple_ntt_test(&output_dir)?;
    
    // Test 2: NTT roundtrip
    println!("[2/4] NTT roundtrip test...");
    generate_ntt_roundtrip_test(&output_dir)?;
    
    // Test 3: Coset evaluation
    println!("[3/4] Coset evaluation test...");
    generate_coset_test(&output_dir)?;
    
    // Test 4: Full LDE with real program data
    println!("[4/4] Full LDE with program data...");
    generate_full_lde_test(&output_dir)?;
    
    println!("\n✅ All NTT test data generated in: {}", output_dir.display());
    
    // List generated files
    for entry in fs::read_dir(&output_dir)? {
        let entry = entry?;
        let size = entry.metadata()?.len();
        println!("   {} ({} bytes)", entry.file_name().to_string_lossy(), size);
    }
    
    Ok(())
}

fn generate_simple_ntt_test(dir: &PathBuf) -> Result<()> {
    let n = 8usize;
    
    // Create simple input
    let mut values: Vec<BFieldElement> = (0..n).map(|i| BFieldElement::new(i as u64)).collect();
    
    // Get domain parameters
    let domain = ArithmeticDomain::of_length(n)?;
    
    // Save original values
    let original = values.clone();
    
    // Forward NTT (interpolation inverse)
    intt(&mut values);
    let coefficients = values.clone();
    
    // Inverse NTT to get back evaluations
    ntt(&mut values);
    
    // Verify roundtrip
    assert_eq!(values, original, "NTT roundtrip failed");
    
    let data = serde_json::json!({
        "test": "simple_ntt_8",
        "size": n,
        "domain_generator": domain.generator.value(),
        "original_values": original.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "coefficients_after_intt": coefficients.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "values_after_ntt": values.iter().map(|x| x.value()).collect::<Vec<u64>>(),
    });
    
    fs::write(dir.join("ntt_simple_8.json"), serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote ntt_simple_8.json");
    
    Ok(())
}

fn generate_ntt_roundtrip_test(dir: &PathBuf) -> Result<()> {
    let n = 512usize;
    
    // Create input with pattern similar to trace column
    let mut values: Vec<BFieldElement> = (0..n).map(|i| BFieldElement::new(i as u64)).collect();
    let original = values.clone();
    
    let domain = ArithmeticDomain::of_length(n)?;
    
    // INTT: evaluations -> coefficients
    intt(&mut values);
    let coefficients = values.clone();
    
    // NTT: coefficients -> evaluations
    ntt(&mut values);
    
    assert_eq!(values, original, "Roundtrip failed for n=512");
    
    let data = serde_json::json!({
        "test": "ntt_roundtrip_512",
        "size": n,
        "domain_generator": domain.generator.value(),
        "original_values": original.iter().take(16).map(|x| x.value()).collect::<Vec<u64>>(),
        "coefficients_first_16": coefficients.iter().take(16).map(|x| x.value()).collect::<Vec<u64>>(),
        "coefficients_all": coefficients.iter().map(|x| x.value()).collect::<Vec<u64>>(),
    });
    
    fs::write(dir.join("ntt_roundtrip_512.json"), serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote ntt_roundtrip_512.json");
    
    Ok(())
}

fn generate_coset_test(dir: &PathBuf) -> Result<()> {
    // Polynomial: f(x) = 1 + 2x + 3x^2 + 4x^3 (in coefficient form, padded to 8)
    let n = 8usize;
    let mut coeffs: Vec<BFieldElement> = vec![
        BFieldElement::new(1),
        BFieldElement::new(2),
        BFieldElement::new(3),
        BFieldElement::new(4),
        BFieldElement::new(0),
        BFieldElement::new(0),
        BFieldElement::new(0),
        BFieldElement::new(0),
    ];
    
    let domain = ArithmeticDomain::of_length(n)?;
    let offset = BFieldElement::new(7);
    
    // Evaluate on standard domain (offset = 1)
    let mut standard_evals = coeffs.clone();
    ntt(&mut standard_evals);
    
    // Evaluate on coset (offset = 7)
    // Method: scale coefficients by offset^i, then NTT
    let mut coset_coeffs = coeffs.clone();
    let mut scale = BFieldElement::new(1);
    for i in 0..n {
        coset_coeffs[i] = coset_coeffs[i] * scale;
        scale = scale * offset;
    }
    
    let mut coset_evals = coset_coeffs.clone();
    ntt(&mut coset_evals);
    
    // Verify by manual evaluation at first point
    let omega = domain.generator;
    let x0 = offset;  // First coset point: offset * omega^0 = offset
    let manual_eval = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 * x0 + coeffs[3] * x0 * x0 * x0;
    
    assert_eq!(coset_evals[0], manual_eval, "Coset evaluation mismatch");
    
    let data = serde_json::json!({
        "test": "coset_evaluation",
        "size": n,
        "domain_generator": domain.generator.value(),
        "offset": offset.value(),
        "original_coefficients": coeffs.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "scaled_coefficients": coset_coeffs.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "standard_domain_evaluations": standard_evals.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "coset_evaluations": coset_evals.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "manual_eval_at_offset": manual_eval.value(),
    });
    
    fs::write(dir.join("coset_evaluation.json"), serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote coset_evaluation.json");
    
    Ok(())
}

fn generate_full_lde_test(dir: &PathBuf) -> Result<()> {
    // Load the padded table from the existing test data
    let padded_table_path = "test_data_lde/04_main_tables_pad.json";
    let padded_data: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(padded_table_path)
            .map_err(|e| anyhow!("Failed to read padded table: {}. Run gen_test_data first.", e))?
    )?;
    
    let params_path = "test_data_lde/02_parameters.json";
    let params: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(params_path)?
    )?;
    
    let lde_path = "test_data_lde/05_main_tables_lde.json";
    let lde_data: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(lde_path)?
    )?;
    
    // Extract first column from padded table
    let padded_rows = padded_data["padded_table_data"].as_array().unwrap();
    let padded_height = padded_rows.len();
    let first_column: Vec<BFieldElement> = padded_rows.iter()
        .map(|row| BFieldElement::new(row[0].as_u64().unwrap()))
        .collect();
    
    println!("   Loaded padded table: {} rows", padded_height);
    
    // Get domain parameters
    let trace_gen = params["trace_domain"]["generator"].as_u64().unwrap();
    let quot_len = params["quotient_domain"]["length"].as_u64().unwrap() as usize;
    let quot_gen = params["quotient_domain"]["generator"].as_u64().unwrap();
    let quot_offset = params["quotient_domain"]["offset"].as_u64().unwrap();
    
    println!("   Trace domain: {} elements, gen={}", padded_height, trace_gen);
    println!("   Quotient domain: {} elements, gen={}, offset={}", quot_len, quot_gen, quot_offset);
    
    // Perform INTT on first column to get coefficients
    let mut coeffs = first_column.clone();
    intt(&mut coeffs);
    
    println!("   Interpolated {} coefficients", coeffs.len());
    println!("   First coefficients: {}, {}, {}...", 
             coeffs[0].value(), coeffs[1].value(), coeffs[2].value());
    
    // Zero-extend to quotient domain size
    let offset = BFieldElement::new(quot_offset);
    let mut extended_coeffs = vec![BFieldElement::new(0); quot_len];
    for (i, &c) in coeffs.iter().enumerate() {
        extended_coeffs[i] = c;
    }
    
    // Scale by offset powers for coset evaluation
    let mut scale = BFieldElement::new(1);
    for i in 0..quot_len {
        extended_coeffs[i] = extended_coeffs[i] * scale;
        scale = scale * offset;
    }
    
    // NTT to get coset evaluations
    let mut computed_lde = extended_coeffs.clone();
    ntt(&mut computed_lde);
    
    // Load Rust's actual LDE output
    let rust_lde_rows = lde_data["lde_table_data"].as_array().unwrap();
    let rust_lde_column: Vec<u64> = rust_lde_rows.iter()
        .map(|row| row[0].as_u64().unwrap())
        .collect();
    
    // Compare
    let matches = computed_lde.iter().zip(rust_lde_column.iter())
        .filter(|(a, b)| a.value() == **b)
        .count();
    
    println!("   Computed LDE match: {}/{}", matches, computed_lde.len());
    println!("   Computed first values: {}, {}...", computed_lde[0].value(), computed_lde[1].value());
    println!("   Rust first values: {}, {}...", rust_lde_column[0], rust_lde_column[1]);
    
    let data = serde_json::json!({
        "test": "full_lde_first_column",
        "trace_domain": {
            "length": padded_height,
            "generator": trace_gen,
            "offset": 1,
        },
        "quotient_domain": {
            "length": quot_len,
            "generator": quot_gen,
            "offset": quot_offset,
        },
        "trace_values_first_16": first_column.iter().take(16).map(|x| x.value()).collect::<Vec<u64>>(),
        "trace_values_all": first_column.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "coefficients_first_16": coeffs.iter().take(16).map(|x| x.value()).collect::<Vec<u64>>(),
        "coefficients_all": coeffs.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "computed_lde_first_16": computed_lde.iter().take(16).map(|x| x.value()).collect::<Vec<u64>>(),
        "rust_lde_first_16": rust_lde_column.iter().take(16).copied().collect::<Vec<u64>>(),
        "rust_lde_all": rust_lde_column.clone(),
        "match_count": matches,
        "total_count": computed_lde.len(),
    });
    
    fs::write(dir.join("full_lde_first_column.json"), serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote full_lde_first_column.json");
    
    Ok(())
}

