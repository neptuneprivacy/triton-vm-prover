//! Generate quotient computation debug output for comparison with C++.
//! 
//! This dumps intermediate values from quotient computation:
//! 1. Zerofier inverses
//! 2. Quotient codeword
//! 3. First row constraint values
//! 4. Polynomial coefficients
//! 5. Segment polynomials
//! 6. Out-of-domain evaluation

use std::fs;
use std::path::PathBuf;
use anyhow::{anyhow, Result};
use triton_vm::prelude::*;
use triton_vm::table::master_table::MasterMainTable;
use triton_vm::table::master_table::MasterAuxTable;
use triton_vm::table::master_table::all_quotients_combined;
use triton_vm::table::master_table::{
    initial_quotient_zerofier_inverse,
    consistency_quotient_zerofier_inverse,
    transition_quotient_zerofier_inverse,
    terminal_quotient_zerofier_inverse,
};
use triton_vm::table::auxiliary_table::Evaluable;
use triton_vm::arithmetic_domain::ArithmeticDomain;
use triton_vm::stark::{Stark, ProverDomains, Prover};
use triton_vm::challenges::Challenges;
use triton_vm::proof_stream::ProofStream;
use triton_vm::twenty_first::prelude::Polynomial;
use triton_vm::twenty_first::math::traits::ModPowU32;
use serde_json;
use num_traits::identities::One;

fn split_polynomial_into_segments<const N: usize, FF: triton_vm::prelude::FiniteField>(
    polynomial: Polynomial<FF>,
) -> [Polynomial<'static, FF>; N] {
    let mut segments = Vec::with_capacity(N);
    let coefficients = polynomial.into_coefficients();
    for segment_index in 0..N {
        let segment_coefficients = coefficients.iter().skip(segment_index).step_by(N);
        let segment = Polynomial::new(segment_coefficients.copied().collect());
        segments.push(segment);
    }
    segments.try_into().unwrap()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <program.tasm> <public_input> [output_dir]", args[0]);
        eprintln!("Example: {} spin_input8.tasm 8 /tmp/rust_debug_quotient", args[0]);
        std::process::exit(1);
    }
    
    let program_path = &args[1];
    let public_input_str = &args[2];
    let output_dir = if args.len() >= 4 {
        PathBuf::from(&args[3])
    } else {
        PathBuf::from("rust_debug_quotient")
    };
    
    fs::create_dir_all(&output_dir)?;
    
    println!("Generating Rust quotient debug output...");
    println!("  Program: {}", program_path);
    println!("  Public input: {}", public_input_str);
    println!("  Output dir: {}", output_dir.display());
    println!();
    
    // Parse program
    let program_source = fs::read_to_string(program_path)
        .map_err(|e| anyhow!("Failed to read program: {}", e))?;
    let program = Program::from_code(&program_source)
        .map_err(|e| anyhow!("Failed to parse program: {}", e))?;
    
    // Parse public input
    let public_input_value: u64 = public_input_str.parse()
        .map_err(|e| anyhow!("Failed to parse public input: {}", e))?;
    let public_input = PublicInput::new(vec![public_input_value.into()]);
    let non_determinism = NonDeterminism::default();
    
    // Execute program
    println!("[1/6] Executing program...");
    let (aet, public_output) = VM::trace_execution(
        program.clone(),
        public_input.clone(),
        non_determinism.clone()
    ).map_err(|e| anyhow!("Execution failed: {}", e))?;
    
    let padded_height = aet.padded_height();
    println!("   Padded height: {}", padded_height);
    
    // Create Stark and derive domains
    let stark = Stark::default();
    let claim = Claim::about_program(&program)
        .with_input(public_input.individual_tokens.clone())
        .with_output(public_output.clone());
    
    println!("[2/6] Deriving domains...");
    let fri = stark.fri(padded_height)?;
    let domains = ProverDomains::derive(
        padded_height,
        stark.num_trace_randomizers,
        fri.domain,
        stark.max_degree(padded_height),
    );
    
    println!("   Trace domain length: {}", domains.trace.length);
    println!("   Quotient domain length: {}", domains.quotient.length);
    println!("   FRI domain length: {}", domains.fri.length);
    
    // Create main table
    println!("[3/6] Creating main table...");
    let mut main_table = MasterMainTable::new(&aet, domains.clone(), stark.num_trace_randomizers, [0u8; 32]);

    // Pad main table
    main_table.pad();

    // Sample challenges (matching C++ Fiat-Shamir)
    println!("[4/6] Sampling challenges...");
    let mut proof_stream = ProofStream::default();
    proof_stream.alter_fiat_shamir_state_with(&claim.encode());
    let challenges_scalars = proof_stream.sample_scalars(Challenges::SAMPLE_COUNT);
    let challenges = Challenges::new(challenges_scalars, &claim);

    // Create aux table by extending main table
    let aux_table = main_table.extend(&challenges);

    // Get quotient domain table for main table
    let main_quotient = main_table.quotient_domain_table()
        .ok_or_else(|| anyhow!("Failed to get quotient domain main table"))?;

    // For aux table, we'll use an empty array for now (debug purposes)
    // TODO: Figure out proper aux table quotient domain access
    let aux_quotient = ndarray::ArrayView2::from_shape((0, 0), &[]).unwrap();
    
    // Get quotient weights
    let quotient_weights = proof_stream.sample_scalars(MasterAuxTable::NUM_CONSTRAINTS);
    
    // Dump zerofier inverses
    println!("[5/6] Computing zerofier inverses...");
    let init_inv = initial_quotient_zerofier_inverse(domains.quotient);
    let cons_inv = consistency_quotient_zerofier_inverse(domains.trace, domains.quotient);
    let tran_inv = transition_quotient_zerofier_inverse(domains.trace, domains.quotient);
    let term_inv = terminal_quotient_zerofier_inverse(domains.trace, domains.quotient);
    
    let mut zf_json = serde_json::Map::new();
    zf_json.insert("quotient_len".to_string(), serde_json::Value::Number(domains.quotient.length.into()));
    zf_json.insert("trace_len".to_string(), serde_json::Value::Number(domains.trace.length.into()));
    
    let init_inv_vec: Vec<u64> = init_inv.iter().take(10).map(|x| x.value()).collect();
    let cons_inv_vec: Vec<u64> = cons_inv.iter().take(10).map(|x| x.value()).collect();
    let tran_inv_vec: Vec<u64> = tran_inv.iter().take(10).map(|x| x.value()).collect();
    let term_inv_vec: Vec<u64> = term_inv.iter().take(10).map(|x| x.value()).collect();
    
    zf_json.insert("init_inv".to_string(), serde_json::to_value(init_inv_vec)?);
    zf_json.insert("cons_inv".to_string(), serde_json::to_value(cons_inv_vec)?);
    zf_json.insert("tran_inv".to_string(), serde_json::to_value(tran_inv_vec)?);
    zf_json.insert("term_inv".to_string(), serde_json::to_value(term_inv_vec)?);
    
    fs::write(
        output_dir.join("quotient_zerofier_inverses.json"),
        serde_json::to_string_pretty(&zf_json)?
    )?;
    println!("   ✓ Dumped zerofier inverses");
    
    // Compute quotient codeword
    println!("[6/6] Computing quotient codeword...");
    let quotient_codeword = all_quotients_combined(
        main_quotient.view(),
        aux_quotient.view(),
        domains.trace,
        domains.quotient,
        &challenges,
        &quotient_weights,
    );
    
    // Dump quotient codeword
    let mut qc_json = serde_json::Map::new();
    qc_json.insert("quotient_len".to_string(), serde_json::Value::Number(quotient_codeword.len().into()));
    let qv_array: Vec<serde_json::Value> = quotient_codeword.iter().take(20).map(|xfe| {
        let mut obj = serde_json::Map::new();
        obj.insert("c0".to_string(), serde_json::Value::Number(xfe.coefficients[0].value().into()));
        obj.insert("c1".to_string(), serde_json::Value::Number(xfe.coefficients[1].value().into()));
        obj.insert("c2".to_string(), serde_json::Value::Number(xfe.coefficients[2].value().into()));
        serde_json::Value::Object(obj)
    }).collect();
    qc_json.insert("quotient_values".to_string(), serde_json::Value::Array(qv_array));
    
    fs::write(
        output_dir.join("quotient_codeword.json"),
        serde_json::to_string_pretty(&qc_json)?
    )?;
    println!("   ✓ Dumped quotient codeword");
    
    // Dump first row constraint values
    let first_row_main = main_quotient.row(0);
    let first_row_aux = aux_quotient.row(0);
    let second_row_main = main_quotient.row(1 % main_quotient.nrows());
    let second_row_aux = aux_quotient.row(1 % aux_quotient.nrows());
    
    // Evaluate constraints for first row
    let initial_constraints = MasterAuxTable::evaluate_initial_constraints(
        first_row_main,
        first_row_aux,
        &challenges,
    );
    let consistency_constraints = MasterAuxTable::evaluate_consistency_constraints(
        first_row_main,
        first_row_aux,
        &challenges,
    );
    let transition_constraints = MasterAuxTable::evaluate_transition_constraints(
        first_row_main,
        first_row_aux,
        second_row_main,
        second_row_aux,
        &challenges,
    );
    let terminal_constraints = MasterAuxTable::evaluate_terminal_constraints(
        first_row_main,
        first_row_aux,
        &challenges,
    );
    
    let mut cr_json = serde_json::Map::new();
    
    let xfe_to_json = |xfe: &XFieldElement| -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("c0".to_string(), serde_json::Value::Number(xfe.coefficients[0].value().into()));
        obj.insert("c1".to_string(), serde_json::Value::Number(xfe.coefficients[1].value().into()));
        obj.insert("c2".to_string(), serde_json::Value::Number(xfe.coefficients[2].value().into()));
        serde_json::Value::Object(obj)
    };
    
    let initial_array: Vec<serde_json::Value> = initial_constraints.iter().map(xfe_to_json).collect();
    let consistency_array: Vec<serde_json::Value> = consistency_constraints.iter().map(xfe_to_json).collect();
    let transition_array: Vec<serde_json::Value> = transition_constraints.iter().map(xfe_to_json).collect();
    let terminal_array: Vec<serde_json::Value> = terminal_constraints.iter().map(xfe_to_json).collect();
    
    cr_json.insert("initial_constraints".to_string(), serde_json::Value::Array(initial_array));
    cr_json.insert("consistency_constraints".to_string(), serde_json::Value::Array(consistency_array));
    cr_json.insert("transition_constraints".to_string(), serde_json::Value::Array(transition_array));
    cr_json.insert("terminal_constraints".to_string(), serde_json::Value::Array(terminal_array));
    cr_json.insert("quotient_value".to_string(), xfe_to_json(&quotient_codeword[0]));
    
    fs::write(
        output_dir.join("quotient_first_row_constraints.json"),
        serde_json::to_string_pretty(&cr_json)?
    )?;
    println!("   ✓ Dumped first row constraints");
    
    // Dump weights and challenges
    let mut wc_json = serde_json::Map::new();
    wc_json.insert("num_weights".to_string(), serde_json::Value::Number(quotient_weights.len().into()));
    let weights_array: Vec<serde_json::Value> = quotient_weights.iter().take(20).map(xfe_to_json).collect();
    wc_json.insert("weights".to_string(), serde_json::Value::Array(weights_array));
    
    wc_json.insert("num_challenges".to_string(), serde_json::Value::Number(Challenges::SAMPLE_COUNT.into()));
    let challenges_array: Vec<serde_json::Value> = challenges.challenges.iter().take(10).map(xfe_to_json).collect();
    wc_json.insert("challenges".to_string(), serde_json::Value::Array(challenges_array));
    
    fs::write(
        output_dir.join("quotient_weights_challenges.json"),
        serde_json::to_string_pretty(&wc_json)?
    )?;
    println!("   ✓ Dumped weights and challenges");
    
    // Interpolate quotient codeword
    let quotient_poly = domains.quotient.interpolate(&quotient_codeword);
    let poly_coeffs = quotient_poly.clone().into_coefficients();
    
    let mut pc_json = serde_json::Map::new();
    pc_json.insert("num_coeffs".to_string(), serde_json::Value::Number(poly_coeffs.len().into()));
    let coeffs_array: Vec<serde_json::Value> = poly_coeffs.iter().take(50).map(xfe_to_json).collect();
    pc_json.insert("coeffs".to_string(), serde_json::Value::Array(coeffs_array));
    
    fs::write(
        output_dir.join("quotient_poly_coeffs.json"),
        serde_json::to_string_pretty(&pc_json)?
    )?;
    println!("   ✓ Dumped polynomial coefficients");
    
    // Split into segments
    let segments = split_polynomial_into_segments::<4, XFieldElement>(quotient_poly.clone());
    
    let mut sp_json = serde_json::Map::new();
    sp_json.insert("num_segments".to_string(), serde_json::Value::Number(segments.len().into()));
    let segments_array: Vec<serde_json::Value> = segments.iter().enumerate().map(|(idx, seg)| {
        let mut seg_obj = serde_json::Map::new();
        seg_obj.insert("segment_index".to_string(), serde_json::Value::Number(idx.into()));
        let coeffs = seg.coefficients();
        seg_obj.insert("num_coeffs".to_string(), serde_json::Value::Number(coeffs.len().into()));
        let coeffs_array: Vec<serde_json::Value> = coeffs.iter().take(20).map(xfe_to_json).collect();
        seg_obj.insert("coeffs".to_string(), serde_json::Value::Array(coeffs_array));
        serde_json::Value::Object(seg_obj)
    }).collect();
    sp_json.insert("segments".to_string(), serde_json::Value::Array(segments_array));
    
    fs::write(
        output_dir.join("quotient_segment_polynomials.json"),
        serde_json::to_string_pretty(&sp_json)?
    )?;
    println!("   ✓ Dumped segment polynomials");
    
    // Compute out-of-domain evaluation
    // The out-of-domain point is sampled from proof stream after quotient codeword merkle root
    // For comparison, we need to match the exact point used in C++
    // In practice, this comes from the proof stream after enqueuing the quotient codeword merkle root
    // We'll use a deterministic value for now - in real comparison, extract from C++ debug output
    let ood_point = XFieldElement::new([bfe!(123456789), bfe!(0), bfe!(0)]); // Placeholder
    
    let z4 = ood_point.mod_pow_u32(4);
    let mut ood_evaluations = Vec::new();
    let mut point_power = XFieldElement::new([bfe!(1), bfe!(0), bfe!(0)]);
    for seg in &segments {
        let eval = seg.evaluate(z4);
        ood_evaluations.push((eval, point_power));
        point_power = point_power * ood_point;
    }
    
    let sum_of_segments: XFieldElement = ood_evaluations.iter()
        .map(|(eval, power)| *eval * *power)
        .sum();
    
    // Note: We can't easily compute the AIR quotient value here without the FFI
    // But we can dump what we have
    let mut ood_json = serde_json::Map::new();
    ood_json.insert("out_of_domain_point".to_string(), xfe_to_json(&ood_point));
    ood_json.insert("point_to_4".to_string(), xfe_to_json(&z4));
    
    let seg_evals_array: Vec<serde_json::Value> = ood_evaluations.iter().enumerate().map(|(idx, (eval, power))| {
        let mut seg_obj = serde_json::Map::new();
        seg_obj.insert("segment_index".to_string(), serde_json::Value::Number(idx.into()));
        seg_obj.insert("segment_value_at_point4".to_string(), xfe_to_json(eval));
        seg_obj.insert("point_power".to_string(), xfe_to_json(power));
        let contrib = *eval * *power;
        seg_obj.insert("contribution".to_string(), xfe_to_json(&contrib));
        serde_json::Value::Object(seg_obj)
    }).collect();
    ood_json.insert("segment_evaluations".to_string(), serde_json::Value::Array(seg_evals_array));
    ood_json.insert("sum_of_segments".to_string(), xfe_to_json(&sum_of_segments));
    
    fs::write(
        output_dir.join("quotient_ood_evaluation.json"),
        serde_json::to_string_pretty(&ood_json)?
    )?;
    println!("   ✓ Dumped out-of-domain evaluation");
    
    println!();
    println!("✅ All debug files written to: {}", output_dir.display());
    
    Ok(())
}

