//! Step Input/Output test data for 100% accurate C++ verification.
//! 
//! This module captures complete inputs and outputs for each step,
//! enabling C++ to compute the same step and verify exact match.

use std::fs;
use std::path::Path;
use serde_json;
use ndarray::Axis;

use crate::table::master_table::{MasterAuxTable, MasterMainTable, MasterTable};
use crate::table::auxiliary_table::Evaluable;
use crate::stark::ProverDomains;
use twenty_first::prelude::*;

/// Dump complete step I/O for Tip5 hash computation
pub fn dump_tip5_hash_test(
    dir: &Path,
    test_name: &str,
    input_elements: &[BFieldElement],
    output_digest: Digest,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "test": test_name,
        "input": {
            "elements": input_elements.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "count": input_elements.len(),
        },
        "output": {
            "digest": [
                output_digest.0[0].value(),
                output_digest.0[1].value(),
                output_digest.0[2].value(),
                output_digest.0[3].value(),
                output_digest.0[4].value(),
            ],
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("tip5_{}.json", test_name));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump complete step I/O for Merkle tree construction
pub fn dump_merkle_tree_test(
    dir: &Path,
    test_name: &str,
    leaf_digests: &[Digest],
    root: Digest,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "test": test_name,
        "input": {
            "leaf_count": leaf_digests.len(),
            "leaves": leaf_digests.iter().map(|d| [
                d.0[0].value(),
                d.0[1].value(),
                d.0[2].value(),
                d.0[3].value(),
                d.0[4].value(),
            ]).collect::<Vec<_>>(),
        },
        "output": {
            "root": [
                root.0[0].value(),
                root.0[1].value(),
                root.0[2].value(),
                root.0[3].value(),
                root.0[4].value(),
            ],
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("merkle_{}.json", test_name));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump complete step I/O for domain computation
pub fn dump_domain_test(
    dir: &Path,
    test_name: &str,
    length: usize,
    generator: BFieldElement,
    offset: BFieldElement,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "test": test_name,
        "input": {
            "length": length,
        },
        "output": {
            "generator": generator.value(),
            "offset": offset.value(),
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("domain_{}.json", test_name));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump ProverDomains computation I/O
pub fn dump_prover_domains_test(
    dir: &Path,
    padded_height: usize,
    num_trace_randomizers: usize,
    fri_domain_length: usize,
    domains: &ProverDomains,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "test": "prover_domains",
        "input": {
            "padded_height": padded_height,
            "num_trace_randomizers": num_trace_randomizers,
            "fri_domain_length": fri_domain_length,
        },
        "output": {
            "trace": {
                "length": domains.trace.length,
                "offset": domains.trace.offset.value(),
                "generator": domains.trace.generator.value(),
            },
            "randomized_trace": {
                "length": domains.randomized_trace.length,
                "offset": domains.randomized_trace.offset.value(),
                "generator": domains.randomized_trace.generator.value(),
            },
            "quotient": {
                "length": domains.quotient.length,
                "offset": domains.quotient.offset.value(),
                "generator": domains.quotient.generator.value(),
            },
            "fri": {
                "length": domains.fri.length,
                "offset": domains.fri.offset.value(),
                "generator": domains.fri.generator.value(),
            },
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join("prover_domains.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump main table row hash I/O for verifying table commitment
pub fn dump_row_hash_test(
    dir: &Path,
    row_index: usize,
    row_data: &[BFieldElement],
    row_hash: Digest,
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "test": format!("row_hash_{}", row_index),
        "input": {
            "row_index": row_index,
            "row_data": row_data.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "num_columns": row_data.len(),
        },
        "output": {
            "hash": [
                row_hash.0[0].value(),
                row_hash.0[1].value(),
                row_hash.0[2].value(),
                row_hash.0[3].value(),
                row_hash.0[4].value(),
            ],
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("row_hash_{}.json", row_index));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump FRI folding step I/O
pub fn dump_fri_fold_test(
    dir: &Path,
    round: usize,
    codeword_before: &[XFieldElement],
    folding_challenge: XFieldElement,
    codeword_after: &[XFieldElement],
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "test": format!("fri_fold_round_{}", round),
        "input": {
            "round": round,
            "codeword_length": codeword_before.len(),
            "codeword": codeword_before.iter().map(|x| [
                x.coefficients[0].value(),
                x.coefficients[1].value(),
                x.coefficients[2].value(),
            ]).collect::<Vec<_>>(),
            "folding_challenge": [
                folding_challenge.coefficients[0].value(),
                folding_challenge.coefficients[1].value(),
                folding_challenge.coefficients[2].value(),
            ],
        },
        "output": {
            "folded_codeword_length": codeword_after.len(),
            "folded_codeword": codeword_after.iter().map(|x| [
                x.coefficients[0].value(),
                x.coefficients[1].value(),
                x.coefficients[2].value(),
            ]).collect::<Vec<_>>(),
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("fri_fold_round_{}.json", round));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump ProofStream Fiat-Shamir I/O
pub fn dump_fiat_shamir_sample_test(
    dir: &Path,
    test_name: &str,
    sponge_state_before: &[BFieldElement; 16],
    num_scalars: usize,
    sampled_scalars: &[XFieldElement],
    sponge_state_after: &[BFieldElement; 16],
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "test": test_name,
        "input": {
            "sponge_state": sponge_state_before.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "num_scalars": num_scalars,
        },
        "output": {
            "sampled_scalars": sampled_scalars.iter().map(|x| [
                x.coefficients[0].value(),
                x.coefficients[1].value(),
                x.coefficients[2].value(),
            ]).collect::<Vec<_>>(),
            "sponge_state_after": sponge_state_after.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("fiat_shamir_{}.json", test_name));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump complete main table trace data
pub fn dump_main_table_full(
    dir: &Path,
    main_table: &MasterMainTable,
) -> std::io::Result<()> {
    let trace_table = main_table.trace_table();
    
    // Dump all rows
    let all_rows: Vec<Vec<u64>> = trace_table.axis_iter(Axis(0))
        .map(|row| row.iter().map(|x| x.value()).collect())
        .collect();
    
    let data = serde_json::json!({
        "shape": [trace_table.nrows(), trace_table.ncols()],
        "num_columns": MasterMainTable::NUM_COLUMNS,
        "data": all_rows,
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join("main_table_full.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump LDE table row hashes for Merkle verification
pub fn dump_lde_row_hashes(
    dir: &Path,
    table_name: &str,
    row_hashes: &[Digest],
) -> std::io::Result<()> {
    let data = serde_json::json!({
        "table": table_name,
        "num_rows": row_hashes.len(),
        "row_hashes": row_hashes.iter().map(|d| [
            d.0[0].value(),
            d.0[1].value(),
            d.0[2].value(),
            d.0[3].value(),
            d.0[4].value(),
        ]).collect::<Vec<_>>(),
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}_row_hashes.json", table_name));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump trace randomizer information for fixed seed testing
pub fn dump_trace_randomizer_test(
    dir: &Path,
    column_index: usize,
    trace_randomizer_seed: &[u8; 32],
    num_trace_randomizers: usize,
    randomizer_coefficients: &[BFieldElement],
) -> std::io::Result<()> {
    // Convert seed bytes to hex string manually
    let seed_hex: String = trace_randomizer_seed.iter()
        .map(|&b| format!("{:02x}", b))
        .collect();
    
    let data = serde_json::json!({
        "test": format!("trace_randomizer_column_{}", column_index),
        "input": {
            "column_index": column_index,
            "seed_bytes": trace_randomizer_seed.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            "seed_hex": seed_hex,
            "num_trace_randomizers": num_trace_randomizers,
        },
        "output": {
            "randomizer_coefficients": randomizer_coefficients.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "num_coefficients": randomizer_coefficients.len(),
        },
    });
    
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("trace_randomizer_column_{}.json", column_index));
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// Dump constraint evaluation data for C++ FFI verification
pub fn dump_constraint_evaluation_test(
    dir: &Path,
    main_row_curr: &[BFieldElement],
    aux_row_curr: &[XFieldElement],
    main_row_next: &[BFieldElement],
    aux_row_next: &[XFieldElement],
    challenges: &crate::challenges::Challenges,
    test_point: &XFieldElement,
) -> std::io::Result<()> {
    println!("Debug: main_row_curr len: {}, aux_row_curr len: {}", main_row_curr.len(), aux_row_curr.len());

    // Evaluate initial constraints
    let initial_constraints = MasterAuxTable::evaluate_initial_constraints(
        main_row_curr.into(),
        aux_row_curr.into(),
        challenges,
    );

    // Evaluate full OOD quotient with default weights
    let weights = vec![XFieldElement::new([bfe!(1), bfe!(0), bfe!(0)]); 596]; // 81 + 94 + 398 + 23 total constraints
    let trace_domain_length = 512; // From test data
    let trace_domain_generator = BFieldElement::generator();
    let trace_domain_generator_inverse = trace_domain_generator.inverse();

    // Compute zerofier inverses
    let initial_zerofier_inv = (*test_point - bfe!(1)).inverse();
    let consistency_zerofier_inv = (test_point.mod_pow_u32(trace_domain_length as u32) - bfe!(1)).inverse();
    let except_last_row = *test_point - trace_domain_generator_inverse;
    let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
    let terminal_zerofier_inv = except_last_row.inverse();

    // Evaluate all constraints
    let consistency_constraints = MasterAuxTable::evaluate_consistency_constraints(
        (&main_row_curr[..]).into(),
        (&aux_row_curr[..]).into(),
        challenges,
    );
    let transition_constraints = MasterAuxTable::evaluate_transition_constraints(
        (&main_row_curr[..]).into(),
        (&aux_row_curr[..]).into(),
        (&main_row_next[..]).into(),
        (&aux_row_next[..]).into(),
        challenges,
    );
    let terminal_constraints = MasterAuxTable::evaluate_terminal_constraints(
        (&main_row_curr[..]).into(),
        (&aux_row_curr[..]).into(),
        challenges,
    );

    // Apply zerofiers
    let divide = |constraints: Vec<_>, z_inv| constraints.into_iter().map(move |c| c * z_inv);
    let initial_quotients = divide(initial_constraints, initial_zerofier_inv);
    let consistency_quotients = divide(consistency_constraints, consistency_zerofier_inv);
    let transition_quotients = divide(transition_constraints, transition_zerofier_inv);
    let terminal_quotients = divide(terminal_constraints, terminal_zerofier_inv);

    // Collect raw constraints
    let raw_initial_constraints = initial_quotients.collect::<Vec<_>>();
    let raw_initial_constraints_clone = raw_initial_constraints.clone();

    // Collect raw constraints (without zerofier division)
    let initial_quotients_vec = raw_initial_constraints.clone();
    let consistency_quotients_vec = consistency_quotients.collect::<Vec<_>>();
    let transition_quotients_vec = transition_quotients.collect::<Vec<_>>();
    let terminal_quotients_vec = terminal_quotients.collect::<Vec<_>>();

    let quotient_summands = initial_quotients_vec.iter()
        .chain(consistency_quotients_vec.iter())
        .chain(transition_quotients_vec.iter())
        .chain(terminal_quotients_vec.iter())
        .cloned()
        .collect::<Vec<_>>();

    // Inner product with weights
    let ood_quotient_value = quotient_summands.iter()
        .zip(weights.iter())
        .map(|(constraint, weight)| *constraint * *weight)
        .sum::<XFieldElement>();

    // Format data for JSON
    let data = serde_json::json!({
        "initial_constraints": initial_quotients_vec.into_iter().map(|x| format!("({}·x² + {}·x + {})",
            x.coefficients[2].value(),
            x.coefficients[1].value(),
            x.coefficients[0].value()
        )).collect::<Vec<_>>(),
        "raw_initial_constraints": raw_initial_constraints.into_iter().map(|x| format!("({}·x² + {}·x + {})",
            x.coefficients[2].value(),
            x.coefficients[1].value(),
            x.coefficients[0].value()
        )).collect::<Vec<_>>(),
        "raw_initial_coeffs": raw_initial_constraints_clone.into_iter().map(|x| vec![
            x.coefficients[0].value(),
            x.coefficients[1].value(),
            x.coefficients[2].value()
        ]).collect::<Vec<Vec<u64>>>(),
        "ood_quotient": format!("({}·x² + {}·x + {})",
            ood_quotient_value.coefficients[2].value(),
            ood_quotient_value.coefficients[1].value(),
            ood_quotient_value.coefficients[0].value()
        ),
        "main_curr_row": main_row_curr.iter().map(|x| x.value()).collect::<Vec<u64>>(),
        "aux_curr_row": aux_row_curr.iter().map(|x| format!("({}·x² + {}·x + {})",
            x.coefficients[2].value(),
            x.coefficients[1].value(),
            x.coefficients[0].value()
        )).collect::<Vec<_>>(),
        "challenges": challenges.challenges.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
        "test_point": format!("({}·x² + {}·x + {})",
            test_point.coefficients[2].value(),
            test_point.coefficients[1].value(),
            test_point.coefficients[0].value()
        )
    });

    fs::create_dir_all(dir)?;
    let path = dir.join("rust_constraint_data.json");
    fs::write(path, serde_json::to_string_pretty(&data)?)
}
