//! Test data dumper for functional verification when converting Rust to C++.
//! 
//! This module provides functionality to dump intermediate states during proof
//! generation to JSON files for use as reference data in C++ implementation.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use serde_json;
use ndarray::{Axis, Array2};

use crate::table::master_table::{MasterAuxTable, MasterMainTable, MasterTable};
use crate::table::{MainRow, AuxiliaryRow, QuotientSegments};
use crate::aet::AlgebraicExecutionTrace;
use crate::stark::ProverDomains;
use twenty_first::prelude::*;

const ENV_VAR_DUMP_TEST_DATA: &str = "TVM_DUMP_TEST_DATA";
const ENV_VAR_LIGHT_MODE: &str = "TVM_LIGHT_DUMP_MODE";
const ENV_VAR_DUMP_DETAILED: &str = "TVM_DUMP_DETAILED";

/// Sample size for light mode - first N and last N elements
const LIGHT_SAMPLE_SIZE: usize = 100;

/// Check if detailed (very large) dumps should be generated.
/// These include: 15_linear_combination_detailed, 16_deep_detailed,
/// 17_combined_deep_polynomial_detailed, 18_fri_detailed, 18_fri_debug_data.
/// Default is FALSE (skip these huge files).
pub(crate) fn should_dump_detailed() -> bool {
    env::var(ENV_VAR_DUMP_DETAILED).ok()
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Get the output directory for test data dumping from environment variable.
/// Returns None if not set or empty.
pub(crate) fn get_test_data_dir() -> Option<PathBuf> {
    env::var(ENV_VAR_DUMP_TEST_DATA).ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Check if light mode sampling is enabled
fn is_light_mode() -> bool {
    env::var(ENV_VAR_LIGHT_MODE).ok()
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(true) // Default to light mode
}

/// Sample a vector: first N + last N elements (or all if small enough)
fn sample_vec<T: Clone>(vec: &[T], sample_size: usize) -> Vec<T> {
    if vec.len() <= sample_size * 2 {
        vec.to_vec()
    } else {
        let mut result = vec[..sample_size].to_vec();
        result.extend_from_slice(&vec[vec.len() - sample_size..]);
        result
    }
}

/// Sample rows from a 2D array: first N + last N rows
fn sample_rows<T: Clone>(rows: &[Vec<T>], sample_size: usize) -> Vec<Vec<T>> {
    if rows.len() <= sample_size * 2 {
        rows.to_vec()
    } else {
        let mut result = rows[..sample_size].to_vec();
        result.extend(rows[rows.len() - sample_size..].iter().cloned());
        result
    }
}

/// Ensure the test data directory exists
fn ensure_dir(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)
}

/// Dump data at a specific step number
fn dump_json(dir: &Path, step_num: u32, step_name: &str, data: &serde_json::Value) -> std::io::Result<()> {
    ensure_dir(dir)?;
    let filename = format!("{:02}_{}.json", step_num, step_name.replace(' ', "_").replace("&", "and"));
    let path = dir.join(filename);
    let content = serde_json::to_string_pretty(data)?;
    fs::write(path, content)
}

/// Step 1: Dump trace execution (AET) data
pub(crate) fn dump_trace_execution(dir: &Path, aet: &AlgebraicExecutionTrace, public_output: &[BFieldElement]) -> std::io::Result<()> {
    let data = serde_json::json!({
        "processor_trace_height": aet.processor_trace.nrows(),
        "processor_trace_width": aet.processor_trace.ncols(),
        "public_output": public_output.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "padded_height": aet.padded_height(),
    });
    dump_json(dir, 1, "trace_execution", &data)?;
    
    // Dump sample rows
    let first_row: Vec<u64> = aet.processor_trace.row(0).iter().map(|x| x.value()).collect();
    let last_row: Vec<u64> = if aet.processor_trace.nrows() > 1 {
        aet.processor_trace.row(aet.processor_trace.nrows() - 1)
            .iter().map(|x| x.value()).collect()
    } else {
        vec![]
    };
    
    let sample_data = serde_json::json!({
        "first_row": first_row,
        "last_row": last_row,
    });
    dump_json(dir, 1, "trace_execution_sample", &sample_data)
}

/// Step 2: Dump parameters with full domain details for LDE verification
pub(crate) fn dump_parameters(dir: &Path, padded_height: usize, fri_domain_length: usize, domains: &ProverDomains) -> std::io::Result<()> {
    let data = serde_json::json!({
        "padded_height": padded_height,
        "log2_padded_height": padded_height.ilog2(),
        "fri_domain_length": fri_domain_length,
        "expansion_factor": fri_domain_length / padded_height,
        "trace_domain": {
            "length": domains.trace.length,
            "offset": domains.trace.offset.value(),
            "generator": domains.trace.generator.value(),
        },
        "randomized_trace_domain": {
            "length": domains.randomized_trace.length,
            "offset": domains.randomized_trace.offset.value(),
            "generator": domains.randomized_trace.generator.value(),
        },
        "quotient_domain": {
            "length": domains.quotient.length,
            "offset": domains.quotient.offset.value(),
            "generator": domains.quotient.generator.value(),
        },
        "fri_domain": {
            "length": domains.fri.length,
            "offset": domains.fri.offset.value(),
            "generator": domains.fri.generator.value(),
        },
    });
    dump_json(dir, 2, "parameters", &data)
}

/// Step 3: Dump main table after creation
pub(crate) fn dump_main_table_create(dir: &Path, main_table: &MasterMainTable) -> std::io::Result<()> {
    let trace_table = main_table.trace_table();
    let first_row: Vec<u64> = trace_table.row(0).iter().map(|x| x.value()).collect();
    
    let data = serde_json::json!({
        "trace_table_shape": [trace_table.nrows(), trace_table.ncols()],
        "num_columns": MasterMainTable::NUM_COLUMNS,
        "first_row": first_row,
    });
    dump_json(dir, 3, "main_tables_create", &data)
}

/// Step 4: Dump main table after padding (with sampled data for LDE verification)
pub(crate) fn dump_main_table_pad(dir: &Path, main_table: &MasterMainTable) -> std::io::Result<()> {
    let trace_table = main_table.trace_table();
    let rows = trace_table.nrows();
    let cols = trace_table.ncols();
    
    // Collect all rows as vectors
    let all_rows: Vec<Vec<u64>> = trace_table.axis_iter(Axis(0))
        .map(|row| row.iter().map(|x| x.value()).collect())
        .collect();
    
    // Light mode: sample first/last N rows
    let sampled_rows = if is_light_mode() {
        sample_rows(&all_rows, LIGHT_SAMPLE_SIZE)
    } else {
        all_rows.clone()
    };
    
    let data = serde_json::json!({
        "trace_table_shape_after_pad": [rows, cols],
        "num_rows": rows,
        "num_columns": cols,
        "light_mode": is_light_mode(),
        "sample_size": if is_light_mode() { LIGHT_SAMPLE_SIZE } else { rows },
        "padded_table_data": sampled_rows,
        // Always include first and last row for validation
        "first_row": all_rows.first(),
        "last_row": all_rows.last(),
    });
    dump_json(dir, 4, "main_tables_pad", &data)
}

/// Step 5: Dump main table LDE (sampled for efficiency)
pub(crate) fn dump_main_table_lde(dir: &Path, main_table: &MasterMainTable) -> std::io::Result<()> {
    if let Some(lde_table) = main_table.quotient_domain_table() {
        let rows = lde_table.nrows();
        let cols = lde_table.ncols();
        
        // Collect all rows as vectors
        let all_rows: Vec<Vec<u64>> = lde_table.axis_iter(Axis(0))
            .map(|row| row.iter().map(|x| x.value()).collect())
            .collect();
        
        // Light mode: sample first/last N rows
        let sampled_rows = if is_light_mode() {
            sample_rows(&all_rows, LIGHT_SAMPLE_SIZE)
        } else {
            all_rows.clone()
        };
        
        let data = serde_json::json!({
            "lde_table_shape": [rows, cols],
            "light_mode": is_light_mode(),
            "sample_size": if is_light_mode() { LIGHT_SAMPLE_SIZE } else { rows },
            "lde_table_data": sampled_rows,
            // Always include first and last row for validation
            "first_row": all_rows.first(),
            "last_row": all_rows.last(),
            // Include some middle rows for verification
            "middle_row_index": rows / 2,
            "middle_row": all_rows.get(rows / 2),
        });
        dump_json(dir, 5, "main_tables_lde", &data)
    } else {
        let data = serde_json::json!({
            "note": "LDE table not cached (computed just-in-time)",
        });
        dump_json(dir, 5, "main_tables_lde", &data)
    }
}

/// Dump trace randomizer information for first column (for C++ verification)
pub(crate) fn dump_trace_randomizer_first_column(dir: &Path, main_table: &MasterMainTable) -> std::io::Result<()> {
    let trace_table = main_table.trace_table();
    let trace_domain = main_table.domains().trace;
    let trace_randomizer_seed = main_table.trace_randomizer_seed();
    let num_trace_randomizers = main_table.num_trace_randomizers();
    
    // Get first column
    let first_column: Vec<BFieldElement> = trace_table.column(0).to_vec();
    
    // Get randomizer for first column
    let trace_randomizer = main_table.trace_randomizer_for_column(0);
    let randomizer_coeffs: Vec<u64> = trace_randomizer.coefficients()
        .iter()
        .map(|x| x.value())
        .collect();
    
    // Get interpolants
    let column_interpolant = trace_domain.interpolate(first_column.as_slice());
    let randomized_interpolant = main_table.randomized_column_interpolant(0);
    
    let seed_hex: String = trace_randomizer_seed.iter()
        .map(|&b| format!("{:02x}", b))
        .collect();
    
    let data = serde_json::json!({
        "column_index": 0,
        "trace_domain": {
            "length": trace_domain.length,
            "generator": trace_domain.generator.value(),
            "offset": trace_domain.offset.value(),
        },
        "randomizer_info": {
            "seed_bytes": trace_randomizer_seed.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            "seed_hex": seed_hex,
            "num_trace_randomizers": num_trace_randomizers,
        },
        "randomizer_coefficients": randomizer_coeffs,
        "column_interpolant_coefficients": column_interpolant.coefficients()
            .iter()
            .map(|x| x.value())
            .collect::<Vec<u64>>(),
        "randomized_interpolant_coefficients_first_100": randomized_interpolant.coefficients()
            .iter()
            .take(100)
            .map(|x| x.value())
            .collect::<Vec<u64>>(),
    });
    
    // Write to a special file in test_data directory
    fs::create_dir_all(dir)?;
    let path = dir.join("trace_randomizer_column_0.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump trace randomizer coefficients for ALL columns (for C++ verification) - sampled
pub(crate) fn dump_trace_randomizer_all_columns(dir: &Path, main_table: &MasterMainTable) -> std::io::Result<()> {
    let trace_domain = main_table.domains().trace;
    let trace_randomizer_seed = main_table.trace_randomizer_seed();
    let num_trace_randomizers = main_table.num_trace_randomizers();
    let num_columns = MasterMainTable::NUM_COLUMNS;
    
    let seed_hex: String = trace_randomizer_seed.iter()
        .map(|&b| format!("{:02x}", b))
        .collect();
    
    // Even in light mode, dump ALL trace-randomizer coefficients.
    //
    // Rationale: randomizer coefficients are required to reproduce the exact
    // low-degree extensions and Merkle roots cross-implementation. The size is
    // small (num_columns * num_trace_randomizers u64s) and does not dominate
    // disk usage compared to other dumps.
    let columns_to_dump: Vec<usize> = (0..num_columns).collect();
    
    // Collect randomizer coefficients for selected columns
    let mut all_randomizers: Vec<serde_json::Value> = Vec::with_capacity(columns_to_dump.len());
    
    for col_idx in columns_to_dump.iter() {
        let trace_randomizer = main_table.trace_randomizer_for_column(*col_idx);
        let randomizer_coeffs: Vec<u64> = trace_randomizer.coefficients()
            .iter()
            .map(|x| x.value())
            .collect();
        
        // Do not sample coefficients; we need the full polynomial.
        let sampled_coeffs = randomizer_coeffs;

        all_randomizers.push(serde_json::json!({
            "column_index": col_idx,
            "randomizer_coefficients": sampled_coeffs,
        }));
    }
    
    let data = serde_json::json!({
        "trace_domain": {
            "length": trace_domain.length,
            "generator": trace_domain.generator.value(),
            "offset": trace_domain.offset.value(),
        },
        "light_mode": is_light_mode(),
        "randomizer_info": {
            "seed_bytes": trace_randomizer_seed.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            "seed_hex": seed_hex,
            "num_trace_randomizers": num_trace_randomizers,
            "num_columns": num_columns,
            "dumped_columns": columns_to_dump.len(),
        },
        "sampled_columns": all_randomizers,
    });
    
    // Write to a special file in test_data directory
    fs::create_dir_all(dir)?;
    let path = dir.join("trace_randomizer_all_columns.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Step 6: Dump main table Merkle tree
pub(crate) fn dump_main_table_merkle(dir: &Path, merkle_root: Digest, num_leafs: usize) -> std::io::Result<()> {
    let data = serde_json::json!({
        "merkle_root": merkle_root.to_hex(),
        "num_leafs": num_leafs,
    });
    dump_json(dir, 6, "main_tables_merkle", &data)
}

/// Dump claim data (before Fiat-Shamir) - encodes claim for C++ to absorb
pub(crate) fn dump_claim(dir: &Path, claim: &crate::proof::Claim) -> std::io::Result<()> {
    
    // Encode claim as it would be for Fiat-Shamir
    let encoded = claim.encode();
    
    let data = serde_json::json!({
        "program_digest": claim.program_digest.to_hex(),
        "version": claim.version,
        "input": claim.input.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "output": claim.output.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "encoded_for_fiat_shamir": encoded.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "encoded_length": encoded.len(),
    });
    dump_json(dir, 6, "claim", &data)
}

/// Dump Merkle root encoding for debugging
pub(crate) fn dump_merkle_root_encoding(dir: &Path, encoded: &[BFieldElement]) -> std::io::Result<()> {
    let data = serde_json::json!({
        "encoded": encoded.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "encoded_length": encoded.len(),
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join("merkle_root_encoding.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump sponge state for debugging Fiat-Shamir
pub(crate) fn dump_sponge_state(dir: &Path, step_name: &str, sponge: &twenty_first::prelude::Tip5) -> std::io::Result<()> {
    let state: Vec<u64> = sponge.state.iter().map(|x| x.value()).collect();
    let data = serde_json::json!({
        "state": state,
        "state_size": state.len(),
    });
    
    // Write to debug file (not numbered step)
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("sponge_state_{}.json", step_name));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Step 7: Dump Fiat-Shamir challenges
pub(crate) fn dump_fiat_shamir_challenges(dir: &Path, challenges: &[XFieldElement]) -> std::io::Result<()> {
    let data = serde_json::json!({
        "challenges_sample_count": challenges.len(),
        "challenge_values": challenges.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
    });
    dump_json(dir, 7, "fiat_shamir_challenges", &data)
}

/// Dump quotient combination weights (sampled after aux Merkle root)
pub(crate) fn dump_quotient_combination_weights(dir: &Path, weights: &[XFieldElement]) -> std::io::Result<()> {
    let data = serde_json::json!({
        "weights_count": weights.len(),
        "weight_values": weights.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join("quotient_combination_weights.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Step 7: Dump aux table after creation (extend step) - sampled
pub(crate) fn dump_aux_table_create(dir: &Path, aux_table: &MasterAuxTable) -> std::io::Result<()> {
    let trace_table = aux_table.trace_table();
    let num_rows = trace_table.nrows();
    let num_cols = trace_table.ncols();
    
    // Collect all rows as string vectors
    let all_rows: Vec<Vec<String>> = trace_table.axis_iter(Axis(0))
        .map(|row| row.iter()
            .map(|xfe| format!("{}", xfe))
            .collect())
        .collect();
    
    // Light mode: sample first/last N rows
    let sampled_rows = if is_light_mode() {
        sample_rows(&all_rows, LIGHT_SAMPLE_SIZE)
    } else {
        all_rows.clone()
    };
    
    // Dump aux table randomizer seed
    let trace_randomizer_seed = aux_table.trace_randomizer_seed();
    let num_trace_randomizers = aux_table.num_trace_randomizers();
    let seed_hex: String = trace_randomizer_seed.iter()
        .map(|&b| format!("{:02x}", b))
        .collect();
    
    let data = serde_json::json!({
        "aux_table_shape": [num_rows, num_cols],
        "num_columns": MasterAuxTable::NUM_COLUMNS,
        "light_mode": is_light_mode(),
        "sample_size": if is_light_mode() { LIGHT_SAMPLE_SIZE } else { num_rows },
        "sampled_rows": sampled_rows,
        "row_count": num_rows,
        "column_count": num_cols,
        // Always include first and last row
        "first_row": all_rows.first(),
        "last_row": all_rows.last(),
        "trace_randomizer_info": {
            "seed_bytes": trace_randomizer_seed.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            "seed_hex": seed_hex,
            "num_trace_randomizers": num_trace_randomizers,
        },
    });
    dump_json(dir, 7, "aux_tables_create", &data)
}

/// Dump aux table trace randomizer coefficients for ALL columns (for C++ verification)
pub(crate) fn dump_aux_trace_randomizer_all_columns(dir: &Path, aux_table: &MasterAuxTable) -> std::io::Result<()> {
    let trace_domain = aux_table.domains().trace;
    let trace_randomizer_seed = aux_table.trace_randomizer_seed();
    let num_trace_randomizers = aux_table.num_trace_randomizers();
    let num_columns = MasterAuxTable::NUM_COLUMNS;
    
    let seed_hex: String = trace_randomizer_seed.iter()
        .map(|&b| format!("{:02x}", b))
        .collect();
    
    // Collect randomizer coefficients for all columns
    let mut all_randomizers: Vec<serde_json::Value> = Vec::with_capacity(num_columns);
    
    for col_idx in 0..num_columns {
        let trace_randomizer = aux_table.trace_randomizer_for_column(col_idx);
        // Randomizer is Polynomial<XFieldElement>, export all three coefficients of each XFieldElement
        let randomizer_coeffs: Vec<Vec<u64>> = trace_randomizer.coefficients()
            .iter()
            .map(|xfe| {
                vec![
                    xfe.coefficients[0].value(),
                    xfe.coefficients[1].value(),
                    xfe.coefficients[2].value(),
                ]
            })
            .collect();
        
        all_randomizers.push(serde_json::json!({
            "column_index": col_idx,
            "randomizer_coefficients": randomizer_coeffs,
        }));
    }
    
    let data = serde_json::json!({
        "trace_domain": {
            "length": trace_domain.length,
            "generator": trace_domain.generator.value(),
            "offset": trace_domain.offset.value(),
        },
        "randomizer_info": {
            "seed_bytes": trace_randomizer_seed.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            "seed_hex": seed_hex,
            "num_trace_randomizers": num_trace_randomizers,
            "num_columns": num_columns,
        },
        "all_columns": all_randomizers,
    });
    
    // Write to a special file in test_data directory
    fs::create_dir_all(dir)?;
    let path = dir.join("aux_trace_randomizer_all_columns.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Step 8: Dump aux table LDE - sampled for efficiency
pub(crate) fn dump_aux_table_lde(dir: &Path, aux_table: &MasterAuxTable) -> std::io::Result<()> {
    if let Some(aux_lde_table) = aux_table.quotient_domain_table() {
        let num_rows = aux_lde_table.nrows();
        let num_cols = aux_lde_table.ncols();
        
        // Convert XFieldElement to string representation
        let all_rows: Vec<Vec<String>> = aux_lde_table.axis_iter(Axis(0))
            .map(|row| row.iter().map(|x| format!("{}", x)).collect())
            .collect();
        
        // Light mode: sample first/last N rows
        let sampled_rows = if is_light_mode() {
            sample_rows(&all_rows, LIGHT_SAMPLE_SIZE)
        } else {
            all_rows.clone()
        };
        
        // Dump aux table randomizer seed (similar to main table LDE)
        let trace_randomizer_seed = aux_table.trace_randomizer_seed();
        let num_trace_randomizers = aux_table.num_trace_randomizers();
        let seed_hex: String = trace_randomizer_seed.iter()
            .map(|&b| format!("{:02x}", b))
            .collect();
        
        // Dump intermediate values for first column (for debugging)
        let trace_table = aux_table.trace_table();
        let trace_domain = aux_table.domains().trace;
        let evaluation_domain = aux_table.evaluation_domain();
        
        // Get first column for intermediate value debugging
        let first_column: Vec<XFieldElement> = trace_table.column(0).to_vec();
        
        // Compute interpolant
        let column_interpolant = trace_domain.interpolate(first_column.as_slice());
        let interpolant_coeffs: Vec<Vec<u64>> = column_interpolant.coefficients()
            .iter()
            .map(|xfe| {
                let coeffs = xfe.coefficients;
                vec![coeffs[0].value(), coeffs[1].value(), coeffs[2].value()]
            })
            .collect();
        
        // Get randomizer
        let trace_randomizer = aux_table.trace_randomizer_for_column(0);
        let randomizer_coeffs: Vec<u64> = trace_randomizer.coefficients()
            .iter()
            .map(|xfe| {
                // Randomizer is XFieldElement, extract constant term (first coefficient)
                xfe.coefficients[0].value()
            })
            .collect();
        
        // Compute zerofier * randomizer
        let zerofier_times_randomizer = trace_domain.mul_zerofier_with(trace_randomizer);
        let zerofier_randomizer_coeffs: Vec<Vec<u64>> = zerofier_times_randomizer.coefficients()
            .iter()
            .map(|xfe| {
                let coeffs = xfe.coefficients;
                vec![coeffs[0].value(), coeffs[1].value(), coeffs[2].value()]
            })
            .collect();
        
        // Compute randomized interpolant
        let randomized_interpolant = aux_table.randomized_column_interpolant(0);
        let randomized_coeffs: Vec<Vec<u64>> = randomized_interpolant.coefficients()
            .iter()
            .map(|xfe| {
                let coeffs = xfe.coefficients;
                vec![coeffs[0].value(), coeffs[1].value(), coeffs[2].value()]
            })
            .collect();
        
        // Evaluate first few points for comparison
        let first_few_evals: Vec<Vec<u64>> = (0..std::cmp::min(10, evaluation_domain.length))
            .map(|i| {
                let eval: XFieldElement = randomized_interpolant.evaluate(evaluation_domain.domain_value(i as u32));
                let coeffs = eval.coefficients;
                vec![coeffs[0].value(), coeffs[1].value(), coeffs[2].value()]
            })
            .collect();
        
        let data = serde_json::json!({
            "aux_lde_table_shape": [num_rows, num_cols],
            "light_mode": is_light_mode(),
            "sample_size": if is_light_mode() { LIGHT_SAMPLE_SIZE } else { num_rows },
            "aux_lde_table_data": sampled_rows,
            // Always include first and last row
            "first_row": all_rows.first(),
            "last_row": all_rows.last(),
            "middle_row_index": num_rows / 2,
            "middle_row": all_rows.get(num_rows / 2),
            "trace_randomizer_info": {
                "seed_bytes": trace_randomizer_seed.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
                "seed_hex": seed_hex,
                "num_trace_randomizers": num_trace_randomizers,
            },
            "intermediate_values_column_0": {
                "trace_domain": {
                    "length": trace_domain.length,
                    "offset": trace_domain.offset.value(),
                },
                "evaluation_domain": {
                    "length": evaluation_domain.length,
                    "offset": evaluation_domain.offset.value(),
                },
                "first_column_trace_values": first_column.iter().take(10).map(|xfe| format!("{}", xfe)).collect::<Vec<String>>(),
                "interpolant_coefficients": sample_rows(&interpolant_coeffs, LIGHT_SAMPLE_SIZE),
                "randomizer_coefficients": sample_vec(&randomizer_coeffs, LIGHT_SAMPLE_SIZE),
                "zerofier_times_randomizer_coefficients": sample_rows(&zerofier_randomizer_coeffs, LIGHT_SAMPLE_SIZE),
                "randomized_interpolant_coefficients": sample_rows(&randomized_coeffs, LIGHT_SAMPLE_SIZE),
                "first_few_evaluations": first_few_evals,
            },
        });
        dump_json(dir, 8, "aux_tables_lde", &data)
    } else {
        let data = serde_json::json!({
            "note": "Aux LDE table not cached (computed just-in-time)",
        });
        dump_json(dir, 8, "aux_tables_lde", &data)
    }
}

/// Step 9: Dump aux table Merkle tree
pub(crate) fn dump_aux_table_merkle(dir: &Path, merkle_root: Digest, num_leafs: usize) -> std::io::Result<()> {
    let data = serde_json::json!({
        "aux_merkle_root": merkle_root.to_hex(),
        "num_leafs": num_leafs,
    });
    dump_json(dir, 9, "aux_tables_merkle", &data)
}

/// Step 9b: Dump aux row digests (for debugging Merkle tree mismatches)
/// Only dumps when TVM_DEBUG_ROW_HASHES=1
pub(crate) fn dump_aux_row_digests(dir: &Path, digests: &[Digest], num_leafs: usize) -> std::io::Result<()> {
    // Only dump if debug flag is set
    if std::env::var("TVM_DEBUG_ROW_HASHES").unwrap_or_default() != "1" {
        return Ok(());
    }
    
    // Sample indices: first few, middle, last few
    let mut indices: Vec<usize> = vec![0, 1, 2, 10, 100, 1000];
    if num_leafs > 10000 { indices.push(10000); }
    if num_leafs > 100000 { indices.push(100000); }
    if num_leafs > 1000000 { indices.push(1000000); }
    indices.push(num_leafs / 2);
    indices.push(num_leafs - 2);
    indices.push(num_leafs - 1);
    indices.sort();
    indices.dedup();
    indices.retain(|&i| i < num_leafs);
    
    let sampled_digests: Vec<Vec<u64>> = indices.iter()
        .map(|&i| digests[i].0.iter().map(|x| x.value()).collect())
        .collect();
    
    let data = serde_json::json!({
        "indices": indices,
        "digests": sampled_digests,
        "num_leafs": num_leafs,
    });
    
    let path = dir.join("09_aux_row_digests.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Step 10: Dump quotient calculation
pub(crate) fn dump_quotient_calculation(dir: &Path, cached: bool) -> std::io::Result<()> {
    let data = serde_json::json!({
        "cached": cached,
        "note": if cached { "Quotient calculated from cached tables" } else { "Quotient calculated just-in-time" },
    });
    dump_json(dir, 10, "quotient_calculation", &data)
}

/// Dump quotient calculation with intermediate values (zerofier inverses and quotient codeword)
pub(crate) fn dump_quotient_calculation_with_intermediates(
    dir: &Path,
    cached: bool,
    initial_zerofier_inverse: &ndarray::Array1<twenty_first::prelude::BFieldElement>,
    consistency_zerofier_inverse: &ndarray::Array1<twenty_first::prelude::BFieldElement>,
    transition_zerofier_inverse: &ndarray::Array1<twenty_first::prelude::BFieldElement>,
    terminal_zerofier_inverse: &ndarray::Array1<twenty_first::prelude::BFieldElement>,
    quotient_codeword: &ndarray::Array1<twenty_first::prelude::XFieldElement>,
    first_row_constraints: Option<(Vec<twenty_first::prelude::XFieldElement>, Vec<twenty_first::prelude::XFieldElement>, Vec<twenty_first::prelude::XFieldElement>, Vec<twenty_first::prelude::XFieldElement>, twenty_first::prelude::XFieldElement, twenty_first::prelude::XFieldElement, twenty_first::prelude::XFieldElement, twenty_first::prelude::XFieldElement, twenty_first::prelude::XFieldElement)>,
    quotient_combination_weights: Option<&[twenty_first::prelude::XFieldElement]>,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "cached": cached,
        "note": if cached { "Quotient calculated from cached tables" } else { "Quotient calculated just-in-time" },
        "zerofier_inverses": {
            "initial": initial_zerofier_inverse.iter().take(10).map(|x| x.value()).collect::<Vec<u64>>(),
            "consistency": consistency_zerofier_inverse.iter().take(10).map(|x| x.value()).collect::<Vec<u64>>(),
            "transition": transition_zerofier_inverse.iter().take(10).map(|x| x.value()).collect::<Vec<u64>>(),
            "terminal": terminal_zerofier_inverse.iter().take(10).map(|x| x.value()).collect::<Vec<u64>>(),
            "length": initial_zerofier_inverse.len(),
        },
        "quotient_codeword": {
            "first_10_values": quotient_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "all_values": quotient_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": quotient_codeword.len(),
        },
    });
    
    // Add first row constraint evaluation if provided
    let mut data_with_constraints = data;
    if let Some((init_constraints, cons_constraints, tran_constraints, term_constraints, init_inner, cons_inner, tran_inner, term_inner, quot_val)) = first_row_constraints {
        let init_section_end = init_constraints.len();
        let cons_section_end = init_section_end + cons_constraints.len();
        let tran_section_end = cons_section_end + tran_constraints.len();
        
        let mut constraint_eval = serde_json::json!({
            "initial_constraints": init_constraints.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "initial_constraints_count": init_constraints.len(),
            "initial_inner_product": format!("{}", init_inner),
            "consistency_constraints": cons_constraints.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "consistency_constraints_count": cons_constraints.len(),
            "consistency_inner_product": format!("{}", cons_inner),
            "transition_constraints": tran_constraints.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "transition_constraints_count": tran_constraints.len(),
            "transition_inner_product": format!("{}", tran_inner),
            "terminal_constraints": term_constraints.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "terminal_constraints_count": term_constraints.len(),
            "terminal_inner_product": format!("{}", term_inner),
            "quotient_value": format!("{}", quot_val),
        });
        
        // Add weights if provided
        if let Some(weights) = quotient_combination_weights {
            if weights.len() >= tran_section_end {
                constraint_eval["initial_weights"] = serde_json::json!(weights[..init_section_end].iter().map(|x| format!("{}", x)).collect::<Vec<String>>());
                constraint_eval["consistency_weights"] = serde_json::json!(weights[init_section_end..cons_section_end].iter().map(|x| format!("{}", x)).collect::<Vec<String>>());
                constraint_eval["transition_weights"] = serde_json::json!(weights[cons_section_end..tran_section_end].iter().map(|x| format!("{}", x)).collect::<Vec<String>>());
                constraint_eval["terminal_weights"] = serde_json::json!(weights[tran_section_end..].iter().map(|x| format!("{}", x)).collect::<Vec<String>>());
            }
        }
        
        data_with_constraints["first_row_constraint_evaluation"] = constraint_eval;
    }
    
    dump_json(dir, 10, "quotient_calculation", &data_with_constraints)
}

/// Step 11: Dump quotient LDE - sampled for efficiency
pub(crate) fn dump_quotient_lde(dir: &Path, quotient_segments: Option<&Array2<XFieldElement>>) -> std::io::Result<()> {
    let data = if let Some(quotient_table) = quotient_segments {
        let num_rows = quotient_table.nrows();
        let num_cols = quotient_table.ncols();
        
        // Convert XFieldElement to string representation
        let all_rows: Vec<Vec<String>> = quotient_table.axis_iter(Axis(0))
            .map(|row| row.iter().map(|x| format!("{}", x)).collect())
            .collect();
        
        // Light mode: sample first/last N rows
        let sampled_rows = if is_light_mode() {
            sample_rows(&all_rows, LIGHT_SAMPLE_SIZE)
        } else {
            all_rows.clone()
        };
        
        serde_json::json!({
            "quotient_segments_shape": [num_rows, num_cols],
            "light_mode": is_light_mode(),
            "sample_size": if is_light_mode() { LIGHT_SAMPLE_SIZE } else { num_rows },
            "quotient_segments_data": sampled_rows,
            // Always include first and last row
            "first_row": all_rows.first(),
            "last_row": all_rows.last(),
            "middle_row_index": num_rows / 2,
            "middle_row": all_rows.get(num_rows / 2),
        })
    } else {
        serde_json::json!({
            "note": "Quotient segments data not available",
        })
    };
    dump_json(dir, 11, "quotient_lde", &data)
}

/// Step 12: Dump hash rows of quotient segments - sampled
pub(crate) fn dump_quotient_hash_rows(dir: &Path, digests: &[Digest]) -> std::io::Result<()> {
    let all_digests: Vec<String> = digests.iter().map(|d| d.to_hex()).collect();
    
    // Light mode: sample first/last N digests
    let sampled_digests = if is_light_mode() {
        sample_vec(&all_digests, LIGHT_SAMPLE_SIZE)
    } else {
        all_digests.clone()
    };
    
    let data = serde_json::json!({
        "num_quotient_segment_digests": digests.len(),
        "light_mode": is_light_mode(),
        "sample_size": if is_light_mode() { LIGHT_SAMPLE_SIZE } else { digests.len() },
        "row_digests": sampled_digests,
        // Always include first and last digest
        "first_digest": all_digests.first(),
        "last_digest": all_digests.last(),
        "middle_digest_index": digests.len() / 2,
        "middle_digest": all_digests.get(digests.len() / 2),
    });
    dump_json(dir, 12, "quotient_hash_rows", &data)
}

/// Step 13: Dump quotient Merkle tree
pub(crate) fn dump_quotient_merkle(dir: &Path, merkle_root: Digest, num_leafs: usize) -> std::io::Result<()> {
    let data = serde_json::json!({
        "quotient_merkle_root": merkle_root.to_hex(),
        "num_leafs": num_leafs,
    });
    dump_json(dir, 13, "quotient_merkle", &data)
}

/// Step 14: Dump out-of-domain rows
pub(crate) fn dump_out_of_domain_rows(
    dir: &Path,
    ood_point_curr: XFieldElement,
    ood_point_next: XFieldElement,
    ood_main_row_curr: &[BFieldElement],
    ood_aux_row_curr: &[XFieldElement],
    ood_main_row_next: &[BFieldElement],
    ood_aux_row_next: &[XFieldElement],
    ood_quotient_segments: &[XFieldElement; 4],
    challenges: &crate::challenges::Challenges,
) -> std::io::Result<()> {
    // Also write to a debug file
    let debug_path = dir.join("debug_called.txt");
    std::fs::write(debug_path, "dump_out_of_domain_rows was called").unwrap();
    let data = serde_json::json!({
        "out_of_domain_point_curr_row": format!("{}", ood_point_curr),
        "out_of_domain_point_next_row": format!("{}", ood_point_next),
        "out_of_domain_main_row_curr": ood_main_row_curr.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "out_of_domain_aux_row_curr": ood_aux_row_curr.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
        "out_of_domain_main_row_next": ood_main_row_next.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "out_of_domain_aux_row_next": ood_aux_row_next.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
        "out_of_domain_quotient_segments": ood_quotient_segments.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
    });
    dump_json(dir, 14, "out_of_domain_rows", &data)?;

    // Also dump constraint evaluation data for C++ FFI verification
    // Use the same test point as C++ test
    let test_point = XFieldElement::new([
        12345u64.into(),
        67890u64.into(),
        11111u64.into()
    ]);

    crate::test_data_io::dump_constraint_evaluation_test(
        dir,
        ood_main_row_curr,
        ood_aux_row_curr,
        ood_main_row_next,
        ood_aux_row_next,
        challenges,
        &test_point
    )?;

    Ok(())
}

/// Step 15: Dump linear combination info
pub(crate) fn dump_linear_combination(
    dir: &Path,
    codeword_length: usize,
    main_weights: &[XFieldElement],
    aux_weights: &[XFieldElement],
    quotient_segments_weights: &[XFieldElement],
    linear_comb_curr: XFieldElement,
    linear_comb_next: XFieldElement,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "combination_codeword_length": codeword_length,
        "main_weights": main_weights,
        "aux_weights": aux_weights,
        "quotient_segments_weights": quotient_segments_weights,
        "linear_comb_curr": linear_comb_curr,
        "linear_comb_next": linear_comb_next,
    });
    dump_json(dir, 15, "linear_combination", &data)
}

/// Step 15b: Dump detailed linear combination intermediate values
pub(crate) fn dump_linear_combination_detailed(
    dir: &Path,
    main_combination_poly: &twenty_first::prelude::Polynomial<'_, XFieldElement>,
    aux_combination_poly: &twenty_first::prelude::Polynomial<'_, XFieldElement>,
    main_and_aux_combination_polynomial: &twenty_first::prelude::Polynomial<'_, XFieldElement>,
    main_and_aux_codeword: &[XFieldElement],
    quotient_segments_combination_polynomial: &twenty_first::prelude::Polynomial<'_, XFieldElement>,
    quotient_segments_combination_codeword: &[XFieldElement],
    short_domain_length: usize,
    short_domain_offset: BFieldElement,
    out_of_domain_point_curr_row: XFieldElement,
    out_of_domain_point_next_row: XFieldElement,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "main_combination_polynomial": {
            "coefficients": main_combination_poly.coefficients(),
            "degree": main_combination_poly.degree(),
            "first_10_coefficients": main_combination_poly.coefficients().iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
        },
        "aux_combination_polynomial": {
            "coefficients": aux_combination_poly.coefficients(),
            "degree": aux_combination_poly.degree(),
            "first_10_coefficients": aux_combination_poly.coefficients().iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
        },
        "main_and_aux_combination_polynomial": {
            "coefficients": main_and_aux_combination_polynomial.coefficients(),
            "degree": main_and_aux_combination_polynomial.degree(),
            "first_10_coefficients": main_and_aux_combination_polynomial.coefficients().iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
        },
        "main_and_aux_codeword": {
            "length": main_and_aux_codeword.len(),
            "first_20_values": main_and_aux_codeword.iter().take(20).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "all_values": main_and_aux_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
        },
        "quotient_segments_combination_polynomial": {
            "coefficients": quotient_segments_combination_polynomial.coefficients(),
            "degree": quotient_segments_combination_polynomial.degree(),
            "first_10_coefficients": quotient_segments_combination_polynomial.coefficients().iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
        },
        "quotient_segments_combination_codeword": {
            "length": quotient_segments_combination_codeword.len(),
            "first_20_values": quotient_segments_combination_codeword.iter().take(20).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "all_values": quotient_segments_combination_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
        },
        "short_domain": {
            "length": short_domain_length,
            "offset": short_domain_offset.value(),
        },
        "out_of_domain_points": {
            "curr_row": format!("{}", out_of_domain_point_curr_row),
            "next_row": format!("{}", out_of_domain_point_next_row),
        },
    });
    dump_json(dir, 15, "linear_combination_detailed", &data)
}

/// Step 16: Dump DEEP info
pub(crate) fn dump_deep(dir: &Path, deep_codeword_length: usize) -> std::io::Result<()> {
    let data = serde_json::json!({
        "deep_codeword_length": deep_codeword_length,
    });
    dump_json(dir, 16, "deep", &data)
}

/// Step 16: Dump detailed DEEP codewords for verification
pub(crate) fn dump_deep_detailed(
    dir: &Path,
    main_and_aux_curr_row_deep_codeword: &[XFieldElement],
    main_and_aux_next_row_deep_codeword: &[XFieldElement],
    quotient_segments_curr_row_deep_codeword: &[XFieldElement],
    combined_deep_codeword: &[XFieldElement],
    out_of_domain_point_curr_row: XFieldElement,
    out_of_domain_point_next_row: XFieldElement,
    out_of_domain_curr_row_main_and_aux_value: XFieldElement,
    out_of_domain_next_row_main_and_aux_value: XFieldElement,
    out_of_domain_curr_row_quot_segments_value: XFieldElement,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "main_and_aux_curr_row_deep_codeword": {
            "values": main_and_aux_curr_row_deep_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "first_10_values": main_and_aux_curr_row_deep_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": main_and_aux_curr_row_deep_codeword.len(),
        },
        "main_and_aux_next_row_deep_codeword": {
            "values": main_and_aux_next_row_deep_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "first_10_values": main_and_aux_next_row_deep_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": main_and_aux_next_row_deep_codeword.len(),
        },
        "quotient_segments_curr_row_deep_codeword": {
            "values": quotient_segments_curr_row_deep_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "first_10_values": quotient_segments_curr_row_deep_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": quotient_segments_curr_row_deep_codeword.len(),
        },
        "combined_deep_codeword": {
            "values": combined_deep_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "first_10_values": combined_deep_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": combined_deep_codeword.len(),
        },
        "out_of_domain_point_curr_row": out_of_domain_point_curr_row,
        "out_of_domain_point_next_row": out_of_domain_point_next_row,
        "out_of_domain_curr_row_main_and_aux_value": out_of_domain_curr_row_main_and_aux_value,
        "out_of_domain_next_row_main_and_aux_value": out_of_domain_next_row_main_and_aux_value,
        "out_of_domain_curr_row_quot_segments_value": out_of_domain_curr_row_quot_segments_value,
    });
    dump_json(dir, 16, "deep_detailed", &data)
}

/// Step 17: Dump combined DEEP polynomial
pub(crate) fn dump_combined_deep_polynomial(dir: &Path, fri_domain_length: usize) -> std::io::Result<()> {
    let data = serde_json::json!({
        "fri_combination_codeword_length": fri_domain_length,
    });
    dump_json(dir, 17, "combined_deep_polynomial", &data)
}

/// Step 17: Dump detailed combined DEEP polynomial for verification
pub(crate) fn dump_combined_deep_polynomial_detailed(
    dir: &Path,
    deep_codeword: &[XFieldElement],
    fri_combination_codeword: &[XFieldElement],
    deep_weights: &[XFieldElement],
    applied_lde: bool,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "deep_codeword": {
            "values": deep_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "first_10_values": deep_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": deep_codeword.len(),
        },
        "fri_combination_codeword": {
            "values": fri_combination_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "first_10_values": fri_combination_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": fri_combination_codeword.len(),
        },
        "deep_weights": deep_weights.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
        "applied_lde": applied_lde,
    });
    dump_json(dir, 17, "combined_deep_polynomial_detailed", &data)
}

/// Step 18: Dump FRI info
pub(crate) fn dump_fri(dir: &Path, num_revealed_indices: usize) -> std::io::Result<()> {
    let data = serde_json::json!({
        "num_revealed_indices": num_revealed_indices,
    });
    dump_json(dir, 18, "fri", &data)
}

/// Step 18: Dump detailed FRI data for verification
pub(crate) fn dump_fri_detailed(
    dir: &Path,
    fri_combination_codeword: &[XFieldElement],
    revealed_indices: &[usize],
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "fri_combination_codeword": {
            "values": fri_combination_codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "first_10_values": fri_combination_codeword.iter().take(10).map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": fri_combination_codeword.len(),
        },
        "revealed_indices": revealed_indices,
        "num_revealed_indices": revealed_indices.len(),
    });
    dump_json(dir, 18, "fri_detailed", &data)
}

/// Step 18: Dump FRI debug data (Merkle roots and folding challenges)
pub(crate) fn dump_fri_debug_data(
    dir: &Path,
    merkle_roots: &[Digest],
    folding_challenges: &[XFieldElement],
    folded_codewords: &[Vec<XFieldElement>],
    last_codeword: Option<&[XFieldElement]>,
    last_polynomial: Option<&[XFieldElement]>,
    sponge_state_before_query: Option<&[BFieldElement; 16]>,
) -> std::io::Result<()> {
    let mut data = serde_json::json!({
        "merkle_roots": merkle_roots.iter().map(|d| d.to_string()).collect::<Vec<String>>(),
        "folding_challenges": folding_challenges.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
        "num_merkle_roots": merkle_roots.len(),
        "num_folding_challenges": folding_challenges.len(),
    });
    
    // Dump folded codewords (after each fold, before creating next round)
    data["folded_codewords"] = serde_json::json!(folded_codewords.iter().enumerate().map(|(i, codeword)| {
        serde_json::json!({
            "round": i + 1,
            "values": codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": codeword.len(),
            "first_5_values": codeword.iter().take(5).map(|x| format!("{}", x)).collect::<Vec<String>>(),
        })
    }).collect::<Vec<_>>());
    
    if let Some(codeword) = last_codeword {
        data["last_codeword"] = serde_json::json!({
            "values": codeword.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": codeword.len(),
        });
    }
    
    if let Some(polynomial) = last_polynomial {
        data["last_polynomial"] = serde_json::json!({
            "values": polynomial.iter().map(|x| format!("{}", x)).collect::<Vec<String>>(),
            "length": polynomial.len(),
        });
    }
    
    if let Some(sponge_state) = sponge_state_before_query {
        data["sponge_state_before_query"] = serde_json::json!({
            "state": sponge_state.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "state_size": sponge_state.len(),
        });
    }
    
    dump_json(dir, 18, "fri_debug_data", &data)
}

/// Step 19: Dump open trace leafs
pub(crate) fn dump_open_trace_leafs(dir: &Path, num_revealed_rows: usize) -> std::io::Result<()> {
    let data = serde_json::json!({
        "num_revealed_main_rows": num_revealed_rows,
        "num_revealed_aux_rows": num_revealed_rows,
        "num_revealed_quotient_rows": num_revealed_rows,
    });
    dump_json(dir, 19, "open_trace_leafs", &data)
}

/// Step 19: Dump detailed open trace leafs for verification
pub(crate) fn dump_open_trace_leafs_detailed(
    dir: &Path,
    revealed_indices: &[usize],
    revealed_main_rows: &[MainRow<BFieldElement>],
    revealed_aux_rows: &[AuxiliaryRow],
    revealed_quotient_segments: &[QuotientSegments],
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "revealed_indices": revealed_indices,
        "num_revealed_rows": revealed_indices.len(),
        "revealed_main_rows": revealed_main_rows.iter().map(|row| {
            row.iter().map(|&bfe| bfe.value()).collect::<Vec<u64>>()
        }).collect::<Vec<Vec<u64>>>(),
        "revealed_aux_rows": revealed_aux_rows.iter().map(|row| {
            row.iter().map(|xfe| format!("{}", xfe)).collect::<Vec<String>>()
        }).collect::<Vec<Vec<String>>>(),
        "revealed_quotient_segments": revealed_quotient_segments.iter().map(|seg| {
            seg.iter().map(|xfe| format!("{}", xfe)).collect::<Vec<String>>()
        }).collect::<Vec<Vec<String>>>(),
    });
    dump_json(dir, 19, "open_trace_leafs_detailed", &data)
}

