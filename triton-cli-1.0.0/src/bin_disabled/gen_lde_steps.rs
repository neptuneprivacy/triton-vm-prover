//! Generate LDE step-by-step I/O data for C++ verification.
//! 
//! This dumps:
//! 1. Padded main table (after create + pad)
//! 2. Each LDE step's input and output

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
    let output_dir = PathBuf::from("test_data_lde");
    fs::create_dir_all(&output_dir)?;
    
    println!("Generating LDE step-by-step I/O data for C++ verification...\n");
    
    // Load and execute the spin.tasm program with input 8
    let program_path = "spin.tasm";
    let program_source = fs::read_to_string(program_path)
        .map_err(|e| anyhow!("Failed to read program: {}", e))?;
    
    let program = Program::from_code(&program_source)
        .map_err(|e| anyhow!("Failed to parse program: {}", e))?;
    
    let public_input = PublicInput::new(vec![8_u64.into()]);
    let non_determinism = NonDeterminism::default();
    
    println!("[1/7] Executing program...");
    let (aet, public_output) = VM::trace_execution(program.clone(), public_input.clone(), non_determinism.clone())
        .map_err(|e| anyhow!("Execution failed: {}", e))?;
    
    let padded_height = aet.padded_height();
    println!("   Padded height: {}", padded_height);
    println!("   Processor trace: {} rows x {} cols", 
             aet.processor_trace.nrows(), aet.processor_trace.ncols());
    
    // Create Stark and derive domains
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
    
    println!("[2/7] Creating main table...");
    let mut rng = StdRng::from_entropy();
    let seed: <StdRng as SeedableRng>::Seed = {
        use rand::RngCore;
        let mut seed = <StdRng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        seed
    };
    let mut master_main_table = MasterMainTable::new(
        &aet,
        domains,
        stark.num_trace_randomizers,
        seed,
    );
    
    // Dump table after creation (before padding)
    dump_table_state(&output_dir, "01_after_create", &master_main_table)?;
    
    println!("[3/7] Padding main table...");
    master_main_table.pad();
    
    // Dump table after padding
    dump_table_state(&output_dir, "02_after_pad", &master_main_table)?;
    
    println!("[4/7] Running LDE (with step-by-step capture)...");
    
    // The LDE process in triton-vm does:
    // 1. Interpolation (trace -> polynomials)
    // 2. Resize (low-degree extension coefficients)
    // 3. Evaluation (polynomials -> quotient domain values)
    
    // Capture trace domain values before LDE
    dump_trace_table_data(&output_dir, "03_lde_input_trace", &master_main_table)?;
    
    // Dump domain parameters
    let trace_domain = master_main_table.domains().trace;
    let quotient_domain = master_main_table.domains().quotient;
    dump_domains(&output_dir, &trace_domain, &quotient_domain, &fri.domain)?;
    
    // Run LDE and capture output
    master_main_table.maybe_low_degree_extend_all_columns();
    
    // Dump LDE output
    dump_lde_output_from_table(&output_dir, "05_lde_output", &master_main_table)?;
    
    println!("[5/7] Computing Merkle root...");
    let merkle_tree = master_main_table.merkle_tree();
    let merkle_root = merkle_tree.root();
    dump_merkle_info(&output_dir, &merkle_tree, merkle_root)?;
    
    println!("[6/7] Generating polynomial sample data...");
    // Sample a few columns' worth of polynomial coefficients for verification
    dump_polynomial_samples(&output_dir, &master_main_table)?;
    
    println!("[7/7] Writing summary...");
    dump_summary(&output_dir, padded_height, &trace_domain, &quotient_domain, &fri, merkle_root)?;
    
    println!("\n✅ All LDE step data generated in: {}", output_dir.display());
    println!("\nGenerated files:");
    for entry in fs::read_dir(&output_dir)? {
        let entry = entry?;
        let size = entry.metadata()?.len();
        println!("   {} ({} bytes)", entry.file_name().to_string_lossy(), size);
    }
    
    Ok(())
}

fn dump_table_state(dir: &PathBuf, name: &str, table: &MasterMainTable) -> Result<()> {
    let trace = table.trace_table();
    let rows = trace.nrows();
    let cols = trace.ncols();
    
    // Dump shape and first/last rows
    let first_row: Vec<u64> = trace.row(0).iter().map(|x| x.value()).collect();
    let last_row: Vec<u64> = if rows > 1 {
        trace.row(rows - 1).iter().map(|x| x.value()).collect()
    } else {
        vec![]
    };
    
    // Dump all rows for full verification
    let all_rows: Vec<Vec<u64>> = (0..rows)
        .map(|r| trace.row(r).iter().map(|x| x.value()).collect())
        .collect();
    
    let data = serde_json::json!({
        "stage": name,
        "shape": [rows, cols],
        "num_columns": MasterMainTable::NUM_COLUMNS,
        "first_row": first_row,
        "last_row": last_row,
        "all_rows": all_rows,
    });
    
    let path = dir.join(format!("{}.json", name));
    fs::write(&path, serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote {} ({} rows x {} cols)", path.display(), rows, cols);
    
    Ok(())
}

fn dump_trace_table_data(dir: &PathBuf, name: &str, table: &MasterMainTable) -> Result<()> {
    let trace = table.trace_table();
    let rows = trace.nrows();
    let cols = trace.ncols();
    
    let all_rows: Vec<Vec<u64>> = (0..rows)
        .map(|r| trace.row(r).iter().map(|x| x.value()).collect())
        .collect();
    
    let data = serde_json::json!({
        "stage": name,
        "shape": [rows, cols],
        "data": all_rows,
    });
    
    let path = dir.join(format!("{}.json", name));
    fs::write(&path, serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote {} ({} rows x {} cols)", path.display(), rows, cols);
    
    Ok(())
}

fn dump_domains(
    dir: &PathBuf,
    trace_domain: &ArithmeticDomain,
    quotient_domain: &ArithmeticDomain,
    fri_domain: &ArithmeticDomain,
) -> Result<()> {
    let data = serde_json::json!({
        "trace_domain": {
            "length": trace_domain.length,
            "offset": trace_domain.offset.value(),
            "generator": trace_domain.generator.value(),
        },
        "quotient_domain": {
            "length": quotient_domain.length,
            "offset": quotient_domain.offset.value(),
            "generator": quotient_domain.generator.value(),
        },
        "fri_domain": {
            "length": fri_domain.length,
            "offset": fri_domain.offset.value(),
            "generator": fri_domain.generator.value(),
        },
        "expansion_factor": fri_domain.length / trace_domain.length,
    });
    
    let path = dir.join("04_domains.json");
    fs::write(&path, serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote {}", path.display());
    
    Ok(())
}

fn dump_lde_output_from_table(dir: &PathBuf, name: &str, table: &MasterMainTable) -> Result<()> {
    if let Some(lde) = table.quotient_domain_table() {
        let rows = lde.nrows();
        let cols = lde.ncols();
        
        // Dump full LDE table
        let all_rows: Vec<Vec<u64>> = (0..rows)
            .map(|r| lde.row(r).iter().map(|x| x.value()).collect())
            .collect();
        
        let data = serde_json::json!({
            "stage": name,
            "shape": [rows, cols],
            "data": all_rows,
        });
        
        let path = dir.join(format!("{}.json", name));
        fs::write(&path, serde_json::to_string_pretty(&data)?)?;
        println!("   Wrote {} ({} rows x {} cols)", path.display(), rows, cols);
    } else {
        println!("   Warning: LDE table not available");
    }
    
    Ok(())
}

fn dump_merkle_info(
    dir: &PathBuf,
    tree: &twenty_first::util_types::merkle_tree::MerkleTree,
    root: Digest,
) -> Result<()> {
    let data = serde_json::json!({
        "num_leafs": tree.num_leafs(),
        "root": [
            root.0[0].value(),
            root.0[1].value(),
            root.0[2].value(),
            root.0[3].value(),
            root.0[4].value(),
        ],
        "root_hex": root.to_hex(),
    });
    
    let path = dir.join("06_merkle.json");
    fs::write(&path, serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote {}", path.display());
    
    Ok(())
}

fn dump_polynomial_samples(dir: &PathBuf, table: &MasterMainTable) -> Result<()> {
    // Get interpolation polynomials for first few columns
    let trace = table.trace_table();
    let rows = trace.nrows();
    
    // Sample first 5 columns
    let mut samples = vec![];
    for col in 0..std::cmp::min(5, trace.ncols()) {
        let column: Vec<u64> = (0..rows)
            .map(|row| trace[[row, col]].value())
            .collect();
        samples.push(serde_json::json!({
            "column_index": col,
            "values": column,
        }));
    }
    
    let data = serde_json::json!({
        "num_samples": samples.len(),
        "trace_length": rows,
        "samples": samples,
    });
    
    let path = dir.join("07_polynomial_samples.json");
    fs::write(&path, serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote {}", path.display());
    
    Ok(())
}

fn dump_summary(
    dir: &PathBuf,
    padded_height: usize,
    trace_domain: &ArithmeticDomain,
    quotient_domain: &ArithmeticDomain,
    fri: &triton_vm::fri::Fri,
    merkle_root: Digest,
) -> Result<()> {
    let data = serde_json::json!({
        "padded_height": padded_height,
        "trace_domain_length": trace_domain.length,
        "quotient_domain_length": quotient_domain.length,
        "fri_domain_length": fri.domain.length,
        "expansion_factor": fri.domain.length / trace_domain.length,
        "num_main_columns": MasterMainTable::NUM_COLUMNS,
        "merkle_root_hex": merkle_root.to_hex(),
        "verification_steps": [
            "1. Load 02_after_pad.json as input to LDE",
            "2. Use domains from 04_domains.json",
            "3. Run interpolation: trace values -> polynomial coefficients",
            "4. Run resize: extend polynomial to quotient domain size",
            "5. Run evaluation: polynomial -> quotient domain values",
            "6. Compare output with 05_lde_output.json",
            "7. Hash rows and build Merkle tree",
            "8. Compare Merkle root with 06_merkle.json",
        ],
    });
    
    let path = dir.join("00_summary.json");
    fs::write(&path, serde_json::to_string_pretty(&data)?)?;
    println!("   Wrote {}", path.display());
    
    Ok(())
}

