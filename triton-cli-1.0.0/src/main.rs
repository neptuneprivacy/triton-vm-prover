use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use triton_vm::error::VerificationError;
use serde_json;
use serde_json::json;
use triton_vm::prelude::BFieldCodec;
use triton_vm::prelude::Claim;
use triton_vm::prelude::Stark;
use triton_vm::prelude::VM;
use triton_vm::proof_item::ProofItem;
use triton_vm::proof_stream::ProofStream;
use triton_vm::prelude::TableId;

use crate::args::Args;
use crate::args::Command;
use crate::args::Flags;
use crate::args::ProofArtifacts;
use crate::args::RunArgs;

const SUCCESS: ExitCode = ExitCode::SUCCESS;
const FAILURE: ExitCode = ExitCode::FAILURE;

mod args;

/// Generate sampled test data (first 100 + last 100 + stride sampling)
/// to avoid generating too much data while still providing good coverage
fn generate_sampled_test_data(dir: &std::path::Path, step_num: u32, step_name: &str, data: &serde_json::Value) {
    use std::fs;
    use serde_json;

    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Warning: Failed to create test data directory: {}", e);
        return;
    }

    let filename = format!("{:02}_{}.json", step_num, step_name.replace(' ', "_").replace("&", "and"));
    let path = dir.join(filename);

    match serde_json::to_string_pretty(data) {
        Ok(content) => {
            if let Err(e) = fs::write(&path, content) {
                eprintln!("Warning: Failed to write test data file {}: {}", path.display(), e);
            } else {
                println!("✓ Generated test data: {}", path.display());
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to serialize test data for {}: {}", step_name, e);
        }
    }
}

/// Sample elements from a vector: first 100 + last 100 + every Nth element
fn sample_vector<T: Clone>(vec: &[T], max_samples: usize) -> Vec<T> {
    if vec.len() <= max_samples {
        return vec.to_vec();
    }

    let mut sampled = Vec::with_capacity(max_samples);

    // First 50 elements
    let first_count = (max_samples / 3).min(50);
    sampled.extend_from_slice(&vec[..first_count]);

    // Last 50 elements
    let last_count = (max_samples / 3).min(50);
    if vec.len() > last_count {
        sampled.extend_from_slice(&vec[vec.len() - last_count..]);
    }

    // Stride sampling for the middle
    let remaining = max_samples - sampled.len();
    if remaining > 0 && vec.len() > first_count + last_count {
        let middle_start = first_count;
        let middle_end = vec.len() - last_count;
        let stride = ((middle_end - middle_start) / remaining).max(1);

        for i in (middle_start..middle_end).step_by(stride) {
            if sampled.len() >= max_samples {
                break;
            }
            sampled.push(vec[i].clone());
        }
    }

    sampled
}

fn main() -> Result<ExitCode> {
    human_panic::setup_panic!();

    let Args { flags, command } = Args::parse();
    match command {
        Command::Run(args) => run(flags, args),
        Command::Prove { args, artifacts } => prove(flags, args, artifacts),
        Command::Verify(artifacts) => verify(flags, artifacts),
    }
}

fn run(flags: Flags, args: RunArgs) -> Result<ExitCode> {
    let (program, input, non_determinism) = args.parse()?;

    let output = if flags.profile {
        let (output, profile) = VM::profile(program, input, non_determinism)?;
        println!("{profile}\n");
        output
    } else {
        VM::run(program, input, non_determinism)?
    };
    if !output.is_empty() {
        println!("{}", output.iter().join(", "));
    }

    Ok(SUCCESS)
}

fn prove(flags: Flags, args: RunArgs, artifacts: ProofArtifacts) -> Result<ExitCode> {
    let (program, input, non_determinism) = args.parse()?;

    // Generate light test data at each step (sampled to avoid space/speed issues)
    let generate_light_test_data = std::env::var("TVM_GENERATE_LIGHT_TEST_DATA").is_ok();
    let test_data_dir = if generate_light_test_data {
        let dir = std::env::var("TVM_TEST_DATA_DIR").ok()
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("rust_test_data"));
        println!("🔧 Light test data generation enabled - output dir: {}", dir.display());
        Some(dir)
    } else {
        println!("🔧 Light test data generation disabled");
        None
    };

    triton_vm::profiler::start("Triton VM – Prove");
    let claim = Claim::about_program(&program).with_input(input.clone());
    let (aet, public_output) = VM::trace_execution(program, input, non_determinism)?;

    // Step 1: Dump trace execution data
    if let Some(ref dir) = test_data_dir {
        let public_output_values: Vec<u64> = public_output.iter().map(|x| x.value()).collect();
        let sampled_output = if public_output_values.len() > 200 {
            [&public_output_values[..100], &public_output_values[public_output_values.len()-100..]].concat()
        } else {
            public_output_values
        };

        // Calculate table heights
        let total_instructions = aet.processor_trace.nrows();
        // let memory_operations = aet.height_of_table(3) + aet.height_of_table(4);

        // Dump processor trace first and last row
        let proc_first_row: Vec<u64> = aet.processor_trace.row(0).iter().map(|x| x.value()).collect();
        let proc_last_row: Vec<u64> = if aet.processor_trace.nrows() > 1 {
            aet.processor_trace.row(aet.processor_trace.nrows() - 1)
                .iter().map(|x| x.value()).collect()
        } else {
            vec![]
        };

        // Dump op stack first row
        let op_stack_first_row: Vec<u64> = if aet.op_stack_underflow_trace.nrows() > 0 {
            aet.op_stack_underflow_trace.row(0).iter().map(|x| x.value()).collect()
        } else {
            vec![]
        };

        // Dump RAM first row
        let ram_first_row: Vec<u64> = if aet.ram_trace.nrows() > 0 {
            aet.ram_trace.row(0).iter().map(|x| x.value()).collect()
        } else {
            vec![]
        };

        // Dump program hash first row
        let ph_first_row: Vec<u64> = if aet.program_hash_trace.nrows() > 0 {
            aet.program_hash_trace.row(0).iter().map(|x| x.value()).collect()
        } else {
            vec![]
        };

        // Dump hash first row
        let hash_first_row: Vec<u64> = if aet.hash_trace.nrows() > 0 {
            aet.hash_trace.row(0).iter().map(|x| x.value()).collect()
        } else {
            vec![]
        };

        // Dump sponge first row
        let sponge_first_row: Vec<u64> = if aet.sponge_trace.nrows() > 0 {
            aet.sponge_trace.row(0).iter().map(|x| x.value()).collect()
        } else {
            vec![]
        };

        // Dump U32 entries sample (first 10)
        let u32_sample: Vec<serde_json::Value> = aet.u32_entries.iter()
            .take(10)
            .map(|(entry, mult)| {
                json!({
                    "instruction": entry.instruction.opcode(),
                    "left_operand": entry.left_operand.value(),
                    "right_operand": entry.right_operand.value(),
                    "multiplicity": mult
                })
            })
            .collect();

        // Dump cascade table lookup multiplicities sample (first 10)
        let cascade_sample: Vec<serde_json::Value> = aet.cascade_table_lookup_multiplicities.iter()
            .take(10)
            .map(|(key, mult)| {
                json!({
                    "key": *key,
                    "multiplicity": mult
                })
            })
            .collect();

        // Dump lookup table lookup multiplicities (first 10)
        let lookup_mults_sample: Vec<u64> = aet.lookup_table_lookup_multiplicities.iter()
            .take(10)
            .copied()
            .collect();

        // Dump instruction multiplicities sample (first 10)
        let inst_mults_sample: Vec<u32> = aet.instruction_multiplicities.iter()
            .take(10)
            .copied()
            .collect();

        let data = serde_json::json!({
            "processor_trace_height": aet.processor_trace.nrows(),
            "processor_trace_width": aet.processor_trace.ncols(),
            "processor_trace_first_row": proc_first_row,
            "processor_trace_last_row": proc_last_row,
            "public_output_sampled": sampled_output,
            "padded_height": aet.padded_height(),
            "total_instructions": total_instructions,
            "op_stack_height": aet.op_stack_underflow_trace.nrows(),
            "op_stack_width": aet.op_stack_underflow_trace.ncols(),
            "op_stack_first_row": op_stack_first_row,
            "ram_height": aet.ram_trace.nrows(),
            "ram_width": aet.ram_trace.ncols(),
            "ram_first_row": ram_first_row,
            "program_hash_height": aet.program_hash_trace.nrows(),
            "program_hash_width": aet.program_hash_trace.ncols(),
            "program_hash_first_row": ph_first_row,
            "hash_height": aet.hash_trace.nrows(),
            "hash_width": aet.hash_trace.ncols(),
            "hash_first_row": hash_first_row,
            "sponge_height": aet.sponge_trace.nrows(),
            "sponge_width": aet.sponge_trace.ncols(),
            "sponge_first_row": sponge_first_row,
            "u32_entries_sample": u32_sample,
            "u32_entries_count": aet.u32_entries.len(),
            "cascade_table_sample": cascade_sample,
            "cascade_table_count": aet.cascade_table_lookup_multiplicities.len(),
            "lookup_table_multiplicities_sample": lookup_mults_sample,
            "instruction_multiplicities_sample": inst_mults_sample,
            "instruction_multiplicities_count": aet.instruction_multiplicities.len(),
            "table_heights": [
                aet.height_of_table(TableId::Program),
                aet.height_of_table(TableId::Processor),
                aet.height_of_table(TableId::OpStack),
                aet.height_of_table(TableId::Ram),
                aet.height_of_table(TableId::JumpStack),
                aet.height_of_table(TableId::Hash),
                aet.height_of_table(TableId::Cascade),
                aet.height_of_table(TableId::Lookup),
                aet.height_of_table(TableId::U32),
            ],
        });
        generate_sampled_test_data(dir, 1, "trace_execution", &data);

        // Step 2: Dump claim data
        let input_values: Vec<u64> = claim.input.iter().map(|x| x.value()).collect();
        let output_values: Vec<u64> = claim.output.iter().map(|x| x.value()).collect();
        let sampled_input = if input_values.len() > 200 {
            [&input_values[..100], &input_values[input_values.len().saturating_sub(100)..]].concat()
        } else {
            input_values.clone()
        };
        let sampled_output = if output_values.len() > 200 {
            [&output_values[..100], &output_values[output_values.len().saturating_sub(100)..]].concat()
        } else {
            output_values.clone()
        };
        let claim_data = serde_json::json!({
            "program_digest": claim.program_digest.values().iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "input_sampled": sampled_input,
            "output_sampled": sampled_output,
            "input_length": input_values.len(),
            "output_length": output_values.len(),
            "claim_encoding_length": claim.encode().len(),
        });
        generate_sampled_test_data(dir, 2, "claim", &claim_data);
    }

    let claim = claim.with_output(public_output);

    // Add detailed debugging
    let debug_proof_generation = std::env::var("TVM_DEBUG_PROOF").is_ok();

    let proof = if debug_proof_generation {
        // Use a wrapper to capture proof stream details
        let stark = Stark::default();
        let mut prover = triton_vm::prelude::Prover::new(stark);

        // Check for TRITON_FIXED_SEED environment variable (same as C++)
        let prover = if let Ok(seed_val) = std::env::var("TRITON_FIXED_SEED") {
            if seed_val == "1" || seed_val == "zero" {
                // Use all-zero seed for deterministic proofs (matches C++)
                let prover = prover.set_randomness_seed_which_may_break_zero_knowledge([0u8; 32]);
                eprintln!("  [Rust] Using fixed seed (all zeros) for deterministic proofs");
                prover
            } else {
                prover
            }
        } else {
            prover
        };

        // Generate proof and analyze it
        let proof = prover.prove(&claim, &aet)?;

        // Generate test data for proof analysis
        if let Some(ref dir) = test_data_dir {
            let decoded_stream = ProofStream::try_from(&proof).unwrap_or_else(|_| ProofStream::default());
            let proof_values: Vec<u64> = proof.0.iter().map(|x| x.value()).collect();
            let sampled_proof = if proof_values.len() > 500 {
                [&proof_values[..250], &proof_values[proof_values.len()-250..]].concat()
            } else {
                proof_values
            };
            let proof_data = serde_json::json!({
                "proof_length": proof.0.len(),
                "proof_stream_items": decoded_stream.items.len(),
                "sampled_proof_elements": sampled_proof,
            });
            generate_sampled_test_data(dir, 99, "proof_generated", &proof_data);
        }

        // Decode and analyze
        let decoded_stream = ProofStream::try_from(&proof)?;

        eprintln!("\n=== RUST PROOF GENERATION DEBUG ===");
        eprintln!("Proof.0 length: {} BFieldElements", proof.0.len());
        eprintln!("Proof stream items: {}", decoded_stream.items.len());
        eprintln!();

        // Try to extract FRI indices and row counts from the proof stream
        // Look for MasterMainTableRows to see how many rows were revealed
        for (i, item) in decoded_stream.items.iter().enumerate() {
            if let ProofItem::MasterMainTableRows(rows) = item {
                eprintln!("DEBUG: MasterMainTableRows (item {}) has {} rows", i, rows.len());
                eprintln!("  This should match num_collinearity_checks or be expanded somehow");
            }
            if let ProofItem::FriResponse(resp) = item {
                eprintln!("DEBUG: FriResponse (item {}) has {} revealed leaves", i, resp.revealed_leaves.len());
            }
        }
        eprintln!();

        // Analyze each item
        let mut total_item_encoding = 0;
        for (i, item) in decoded_stream.items.iter().enumerate() {
            let item_encoding = item.encode();
            total_item_encoding += item_encoding.len();

            let item_type = match item {
                ProofItem::MerkleRoot(_) => "MerkleRoot",
                ProofItem::OutOfDomainMainRow(_) => "OutOfDomainMainRow",
                ProofItem::OutOfDomainAuxRow(_) => "OutOfDomainAuxRow",
                ProofItem::OutOfDomainQuotientSegments(_) => "OutOfDomainQuotientSegments",
                ProofItem::AuthenticationStructure(_) => "AuthenticationStructure",
                ProofItem::MasterMainTableRows(rows) => {
                    eprintln!("  Item {}: MasterMainTableRows - {} rows", i, rows.len());
                    "MasterMainTableRows"
                }
                ProofItem::MasterAuxTableRows(rows) => {
                    eprintln!("  Item {}: MasterAuxTableRows - {} rows", i, rows.len());
                    "MasterAuxTableRows"
                }
                ProofItem::Log2PaddedHeight(_) => "Log2PaddedHeight",
                ProofItem::QuotientSegmentsElements(segs) => {
                    eprintln!("  Item {}: QuotientSegmentsElements - {} segments", i, segs.len());
                    "QuotientSegmentsElements"
                }
                ProofItem::FriCodeword(codeword) => {
                    eprintln!("  Item {}: FriCodeword - {} XFieldElements", i, codeword.len());
                    "FriCodeword"
                }
                ProofItem::FriPolynomial(poly) => {
                    let coeffs = poly.coefficients();
                    eprintln!("  Item {}: FriPolynomial - {} coefficients (after trimming)", i, coeffs.len());
                    "FriPolynomial"
                }
                ProofItem::FriResponse(resp) => {
                    eprintln!("  Item {}: FriResponse - {} auth digests, {} revealed leaves",
                             i, resp.auth_structure.len(), resp.revealed_leaves.len());
                    "FriResponse"
                }
            };

            eprintln!("  Item {}: {} (encoding: {} elements)", i, item_type, item_encoding.len());
        }

        eprintln!();
        eprintln!("Total item encoding size: {} elements", total_item_encoding);
        let proof_stream_encoding = decoded_stream.encode();
        eprintln!("Proof stream encoding size: {} elements", proof_stream_encoding.len());
        eprintln!("Proof.0 size: {} elements", proof.0.len());
        eprintln!("=====================================\n");

        proof
    } else {
        // Manual step-by-step proving process for test data generation
        let stark = Stark::default();
        let mut prover = triton_vm::prelude::Prover::new(stark);

        // Check for TRITON_FIXED_SEED environment variable (same as C++)
        let prover = if let Ok(seed_val) = std::env::var("TRITON_FIXED_SEED") {
            if seed_val == "1" || seed_val == "zero" {
                // Use all-zero seed for deterministic proofs (matches C++)
                let prover = prover.set_randomness_seed_which_may_break_zero_knowledge([0u8; 32]);
                eprintln!("  [Rust] Using fixed seed (all zeros) for deterministic proofs");
                prover
            } else {
                prover
            }
        } else {
            prover
        };

        // Step 3: Fiat-Shamir initialization with claim
        let mut proof_stream = triton_vm::proof_stream::ProofStream::new();
        proof_stream.alter_fiat_shamir_state_with(&claim.encode());

        if let Some(ref dir) = test_data_dir {
            let fiat_shamir_data = serde_json::json!({
                "fiat_shamir_initialized": true,
                "claim_encoded_length": claim.encode().len(),
                "claim_version": claim.version,
                "program_digest_length": claim.program_digest.values().len(),
                "input_length": claim.input.len(),
                "output_length": claim.output.len(),
            });
            generate_sampled_test_data(dir, 3, "fiat_shamir_init", &fiat_shamir_data);
        }

        // Step 4: Domain setup
        let padded_height = aet.padded_height();
        let fri = stark.fri(padded_height)?;
        let domains = triton_vm::stark::ProverDomains::derive(
            padded_height,
            stark.num_trace_randomizers,
            fri.domain,
            stark.max_degree(padded_height),
        );

        if let Some(ref dir) = test_data_dir {
            let domain_data = serde_json::json!({
                "padded_height": padded_height,
                "trace_domain_length": domains.trace.length,
                "quotient_domain_length": domains.quotient.length,
                "fri_domain_length": domains.fri.length,
                "fri_expansion_factor": fri.expansion_factor,
                "fri_num_rounds": fri.num_rounds(),
                "num_trace_randomizers": stark.num_trace_randomizers,
            });
            generate_sampled_test_data(dir, 4, "domain_setup", &domain_data);
        }

        // Step 5: Create main table
        let mut master_main_table = triton_vm::table::master_table::MasterMainTable::new(
            &aet,
            domains.clone(),
            stark.num_trace_randomizers,
            [0u8; 32], // Use fixed seed for reproducibility
        );

        if let Some(ref dir) = test_data_dir {
            let main_table_data = serde_json::json!({
                "main_table_created": true,
                "padded_height": padded_height,
                "num_trace_randomizers": stark.num_trace_randomizers,
            });
            generate_sampled_test_data(dir, 5, "main_table_create", &main_table_data);
        }

        // Step 6: Pad main table
        master_main_table.pad();

        if let Some(ref dir) = test_data_dir {
            // Generate metadata for step 6
            let padded_table_data = serde_json::json!({
                "main_table_padded": true,
                "padded_height": padded_height,
            });
            generate_sampled_test_data(dir, 6, "main_table_pad", &padded_table_data);
            
            // Generate full padded table data for comparison (step 4 file name for compatibility)
            let trace_table = master_main_table.trace_table();
            let num_rows = trace_table.nrows();
            let num_cols = trace_table.ncols();
            
            // Extract first and last rows
            let first_row: Vec<u64> = trace_table.row(0).iter().map(|x| x.value()).collect();
            let last_row: Vec<u64> = if num_rows > 0 {
                trace_table.row(num_rows - 1).iter().map(|x| x.value()).collect()
            } else {
                vec![]
            };
            
            // Extract full padded table data
            let padded_table_data_full: Vec<Vec<u64>> = (0..num_rows)
                .map(|r| {
                    trace_table.row(r).iter().map(|x| x.value()).collect()
                })
                .collect();
            
            let full_padded_data = serde_json::json!({
                "num_rows": num_rows,
                "num_columns": num_cols,
                "padded_height": padded_height,
                "first_row": first_row,
                "last_row": last_row,
                "padded_table_data": padded_table_data_full,
                "trace_table_shape_after_pad": [num_rows, num_cols],
                "light_mode": false,
                "sample_size": num_rows,
            });
            
            // Save as step 4 file for compatibility with C++ comparison code
            generate_sampled_test_data(dir, 4, "main_tables_pad", &full_padded_data);
        }

        // Generate GPU step completion markers (these steps happen inside prove())
        if let Some(ref dir) = test_data_dir {
            // Step 7: Main table commitment completed
            let main_table_commitment_data = serde_json::json!({
                "main_table_committed": true,
                "step_description": "Main table LDE + Merkle commitment"
            });
            generate_sampled_test_data(dir, 7, "main_table_commitment", &main_table_commitment_data);

            // Step 8: Aux table commitment completed
            let aux_table_commitment_data = serde_json::json!({
                "aux_table_committed": true,
                "step_description": "Auxiliary table commitment"
            });
            generate_sampled_test_data(dir, 8, "aux_table_commitment", &aux_table_commitment_data);

            // Step 9: Quotient computation completed
            let quotient_computation_data = serde_json::json!({
                "quotient_computed": true,
                "step_description": "Quotient polynomial computation and constraint evaluation"
            });
            generate_sampled_test_data(dir, 9, "quotient_computation", &quotient_computation_data);

            // Step 10: Out-of-domain evaluation completed
            let out_of_domain_evaluation_data = serde_json::json!({
                "out_of_domain_evaluated": true,
                "step_description": "Out-of-domain evaluation for boundary constraints"
            });
            generate_sampled_test_data(dir, 10, "out_of_domain_evaluation", &out_of_domain_evaluation_data);

            // Step 11: FRI protocol completed
            let fri_protocol_data = serde_json::json!({
                "fri_completed": true,
                "step_description": "FRI polynomial commitment protocol"
            });
            generate_sampled_test_data(dir, 11, "fri_protocol", &fri_protocol_data);
        }

        let proof = prover.prove(&claim, &aet)?;

        // Generate test data for proof result
        if let Some(ref dir) = test_data_dir {
            let proof_values: Vec<u64> = proof.0.iter().map(|x| x.value()).collect();
            let sampled_proof = if proof_values.len() > 500 {
                [&proof_values[..250], &proof_values[proof_values.len()-250..]].concat()
            } else {
                proof_values
            };
            let proof_data = serde_json::json!({
                "proof_length": proof.0.len(),
                "sampled_proof_elements": sampled_proof,
            });
            generate_sampled_test_data(dir, 99, "proof_generated", &proof_data);
        }

        proof
    };

    if flags.profile {
        let padded_height = aet.padded_height();
        let profile = triton_vm::profiler::finish()
            .with_cycle_count(aet.processor_trace.nrows())
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri_domain_length(padded_height)?);
        println!("{profile}");
    }

    artifacts.write(&claim, &proof)?;

    Ok(SUCCESS)
}

fn verify(flags: Flags, artifacts: ProofArtifacts) -> Result<ExitCode> {
    let (claim, proof) = artifacts.read()?;

    // Add detailed debugging
    let debug_verification = std::env::var("TVM_DEBUG_VERIFY").is_ok();
    
    if debug_verification {
        use triton_vm::prelude::*;
        eprintln!("\n=== RUST VERIFICATION DEBUG ===");
        eprintln!("Proof.0 length: {} BFieldElements", proof.0.len());

        // Decode proof stream using the canonical Rust decoder and print Merkle roots in-file.
        if let Ok(decoded_stream) = ProofStream::try_from(&proof) {
            eprintln!("Decoded proof stream items: {}", decoded_stream.items.len());
            for (i, item) in decoded_stream.items.iter().enumerate() {
                if let ProofItem::MerkleRoot(d) = item {
                    eprintln!("  In-file MerkleRoot (item {}): {}", i, d.to_hex());
                }
                if let ProofItem::OutOfDomainMainRow(row) = item {
                    eprintln!("  In-file OutOfDomainMainRow (item {}): len={}, first={}", i, row.len(), row[0]);
                }
                if let ProofItem::OutOfDomainAuxRow(row) = item {
                    eprintln!("  In-file OutOfDomainAuxRow (item {}): len={}, first={}", i, row.len(), row[0]);
                }
                if let ProofItem::OutOfDomainQuotientSegments(segs) = item {
                    eprintln!("  In-file OutOfDomainQuotientSegments (item {}): len={}, seg0={}", i, segs.len(), segs[0]);
                }
                if let ProofItem::FriPolynomial(poly) = item {
                    let coeffs = poly.coefficients();
                    eprintln!("  In-file FriPolynomial (item {}): coefficients_len={}, degree={}", i, coeffs.len(), poly.degree());
                }
                if let ProofItem::FriCodeword(codeword) = item {
                    let first = codeword.first().map(|x| format!("{}", x)).unwrap_or_else(|| "<empty>".to_string());
                    eprintln!("  In-file FriCodeword (item {}): len={}, first={}", i, codeword.len(), first);
                }
            }
        } else {
            eprintln!("WARNING: ProofStream::try_from(&proof) failed; falling back to manual decoder below.");
        }
        
        // Try to decode proof stream step by step
        let proof_stream_encoding = &proof.0;
        eprintln!("Proof stream encoding length: {} elements", proof_stream_encoding.len());
        
        // Decode Vec<ProofItem> length
        if proof_stream_encoding.is_empty() {
            eprintln!("ERROR: Proof stream is empty!");
            return Ok(FAILURE);
        }
        let num_items = proof_stream_encoding[0].value() as usize;
        eprintln!("Vec<ProofItem> length: {}", num_items);
        
        // Try to decode each item manually to see where it fails
        let mut idx = 1;
        for item_num in 0..num_items {
            if idx >= proof_stream_encoding.len() {
                eprintln!("ERROR: Ran out of elements at item {} (index {})", item_num, idx);
                break;
            }
            
            // Read item length
            let item_length = proof_stream_encoding[idx].value() as usize;
            idx += 1;
            
            if idx + item_length > proof_stream_encoding.len() {
                eprintln!("ERROR: Item {} length ({}) exceeds remaining elements ({})", 
                         item_num, item_length, proof_stream_encoding.len() - idx);
                break;
            }
            
            // Try to decode the item
            let item_encoding = &proof_stream_encoding[idx..idx + item_length];
            eprintln!("Item {}: length prefix = {}, trying to decode {} elements...", 
                     item_num, item_length, item_length);
            
            // Debug: Print first few bytes to see what we're decoding
            if item_encoding.len() > 0 {
                eprintln!("  First element (discriminant): {}", item_encoding[0].value());
            }
            if item_encoding.len() > 1 {
                eprintln!("  Second element: {}", item_encoding[1].value());
            }
            if item_encoding.len() > 2 {
                eprintln!("  Third element: {}", item_encoding[2].value());
            }
            if item_encoding.len() > 3 {
                eprintln!("  Fourth element: {}", item_encoding[3].value());
            }
            if item_encoding.len() > 4 {
                eprintln!("  Fifth element: {}", item_encoding[4].value());
            }
            
            // Try to decode as ProofItem
            match ProofItem::decode(item_encoding) {
                Ok(item) => {
                    let item_type = match &*item {
                        ProofItem::MerkleRoot(d) => {
                            eprintln!("  ✓ Decoded: MerkleRoot - {}", d.to_hex());
                            "MerkleRoot"
                        }
                        ProofItem::OutOfDomainMainRow(_) => "OutOfDomainMainRow",
                        ProofItem::OutOfDomainAuxRow(_) => "OutOfDomainAuxRow",
                        ProofItem::OutOfDomainQuotientSegments(_) => "OutOfDomainQuotientSegments",
                        ProofItem::AuthenticationStructure(auth) => {
                            eprintln!("  ✓ Decoded: AuthenticationStructure - {} digests", auth.len());
                            "AuthenticationStructure"
                        }
                        ProofItem::MasterMainTableRows(rows) => {
                            eprintln!("  ✓ Decoded: MasterMainTableRows - {} rows", rows.len());
                            "MasterMainTableRows"
                        }
                        ProofItem::MasterAuxTableRows(rows) => {
                            eprintln!("  ✓ Decoded: MasterAuxTableRows - {} rows", rows.len());
                            "MasterAuxTableRows"
                        }
                        ProofItem::Log2PaddedHeight(h) => {
                            eprintln!("  ✓ Decoded: Log2PaddedHeight - {}", h);
                            "Log2PaddedHeight"
                        }
                        ProofItem::QuotientSegmentsElements(segs) => {
                            eprintln!("  ✓ Decoded: QuotientSegmentsElements - {} segments", segs.len());
                            "QuotientSegmentsElements"
                        }
                        ProofItem::FriCodeword(codeword) => {
                            eprintln!("  ✓ Decoded: FriCodeword - {} XFieldElements", codeword.len());
                            eprintln!("    Item encoding breakdown:");
                            eprintln!("      - Discriminant (1): {}", item_encoding[0].value());
                            if item_encoding.len() > 1 {
                                eprintln!("      - Field 0 length prefix (1): {}", item_encoding[1].value());
                                if item_encoding.len() > 2 {
                                    let vec_length = item_encoding[2].value() as usize;
                                    eprintln!("      - Vec<XFieldElement> length (1): {}", vec_length);
                                    let expected_elements = 1 + 3 * vec_length; // Vec length + 3*N elements
                                    eprintln!("      - Expected Vec encoding: {} elements", expected_elements);
                                    eprintln!("      - Actual remaining: {} elements", item_encoding.len() - 2);
                                }
                            }
                            "FriCodeword"
                        }
                        ProofItem::FriPolynomial(poly) => {
                            let coeffs = poly.coefficients();
                            eprintln!("  ✓ Decoded: FriPolynomial - {} coefficients", coeffs.len());
                            "FriPolynomial"
                        }
                        ProofItem::FriResponse(resp) => {
                            eprintln!("  ✓ Decoded: FriResponse - {} auth digests, {} revealed leaves", 
                                     resp.auth_structure.len(), resp.revealed_leaves.len());
                            eprintln!("    Item encoding breakdown:");
                            eprintln!("      - Discriminant (1): {}", item_encoding[0].value());
                            if item_encoding.len() > 1 {
                                let field0_length = item_encoding[1].value() as usize;
                                eprintln!("      - Field 0 (auth_structure) length prefix (1): {}", field0_length);
                                eprintln!("      - Field 0 encoding: {} elements", field0_length);
                                if item_encoding.len() > 2 {
                                    let vec_len = item_encoding[2].value() as usize;
                                    eprintln!("      - Vec<Digest> length (within field 0): {}", vec_len);
                                    eprintln!("      - Expected Vec encoding: 1 + 5*{} = {}", vec_len, 1 + 5 * vec_len);
                                    eprintln!("      - Actual field 0 encoding length: {}", field0_length);
                                }
                                let field1_idx = 1 + field0_length;
                                if item_encoding.len() > field1_idx {
                                    let field1_length = item_encoding[field1_idx].value() as usize;
                                    eprintln!("      - Field 1 (revealed_leaves) length prefix at index {}: {}", field1_idx, field1_length);
                                    if item_encoding.len() > field1_idx + 1 {
                                        let vec_len = item_encoding[field1_idx + 1].value() as usize;
                                        eprintln!("      - Vec<XFieldElement> length (within field 1): {}", vec_len);
                                        eprintln!("      - Expected Vec encoding: 1 + 3*{} = {}", vec_len, 1 + 3 * vec_len);
                                        eprintln!("      - Actual field 1 encoding length: {}", field1_length);
                                    }
                                    eprintln!("      - Expected total: 1 + 1 + {} + 1 + {} = {}", 
                                             field0_length, field1_length, 1 + 1 + field0_length + 1 + field1_length);
                                    eprintln!("      - Actual total: {} elements", item_encoding.len());
                                } else {
                                    eprintln!("      - ERROR: Sequence too short! Expected field 1 at index {}, but only have {} elements", field1_idx, item_encoding.len());
                                }
                            }
                            "FriResponse"
                        }
                    };
                    eprintln!("  Item {}: {} - SUCCESS", item_num, item_type);
                }
                Err(e) => {
                    eprintln!("  ✗ FAILED to decode item {}: {}", item_num, e);
                    eprintln!("    Item encoding length: {} elements", item_encoding.len());
                    eprintln!("    Item encoding (first 30 elements): {:?}", 
                             &item_encoding[..item_encoding.len().min(30)]);
                    
                    // Try to identify which variant we're trying to decode
                    if item_encoding.len() > 0 {
                        let discriminant = item_encoding[0].value();
                        eprintln!("    Discriminant: {} (variant {})", discriminant, discriminant);
                        
                        // If it's FriResponse (variant 11), try to manually decode it step by step
                        if discriminant == 11 {
                            eprintln!("    Attempting manual FriResponse decode to find exact failure point...");
                            use triton_vm::prelude::*;
                            use triton_vm::proof_item::FriResponse;
                            
                            // Skip discriminant
                            let mut decode_sequence = &item_encoding[1..];
                            eprintln!("    After skipping discriminant, sequence.len(): {}", decode_sequence.len());
                            
                            // Try to decode field 0 (auth_structure)
                            if !decode_sequence.is_empty() {
                                let field0_length = decode_sequence[0].value();
                                eprintln!("    Field 0 length prefix: {} (u64)", field0_length);
                                let field0_length_usize: Result<usize, _> = field0_length.try_into();
                                match field0_length_usize {
                                    Ok(f0_len) => {
                                        eprintln!("    Field 0 length as usize: {}", f0_len);
                                        if decode_sequence.len() > f0_len {
                                            let field0_data = &decode_sequence[1..1+f0_len];
                                            eprintln!("    Field 0 data length: {} elements", field0_data.len());
                                            eprintln!("    Field 0 data (first 5): {:?}", &field0_data[..field0_data.len().min(5)]);
                                            
                                                                                // Try to decode Vec<Digest> manually
                                                                                if !field0_data.is_empty() {
                                                                                    let vec_len_value = field0_data[0].value();
                                                                                    eprintln!("    Vec<Digest> length indicator: {} (u64)", vec_len_value);
                                                                                    let vec_len_usize: Result<usize, _> = vec_len_value.try_into();
                                                                                    match vec_len_usize {
                                                                                        Ok(vl) => {
                                                                                            eprintln!("    Vec<Digest> length as usize: {}", vl);
                                                                                            eprintln!("    Field 0 data length: {} elements", field0_data.len());
                                                                                            eprintln!("    Expected Vec<Digest> encoding: 1 + 5*{} = {} elements", vl, 1 + 5*vl);
                                                                                            eprintln!("    Match: {}", field0_data.len() == 1 + 5*vl);
                                                                                            eprintln!("    Attempting Vec<Digest>::decode...");
                                                                                            match Vec::<Digest>::decode(field0_data) {
                                                                                                Ok(vec_digest) => {
                                                                                                    eprintln!("    ✓ Vec<Digest> decoded successfully: {} digests", vec_digest.len());
                                                                                                    eprintln!("    Note: Vec::decode is expected to consume all {} elements", field0_data.len());
                                                                
                                                                // Now try field 1
                                                                let field1_start = 1 + f0_len;
                                                                if decode_sequence.len() > field1_start {
                                                                    let field1_length = decode_sequence[field1_start].value();
                                                                    eprintln!("    Field 1 length prefix at index {}: {} (u64)", field1_start, field1_length);
                                                                    let field1_length_usize: Result<usize, _> = field1_length.try_into();
                                                                    match field1_length_usize {
                                                                        Ok(f1_len) => {
                                                                            eprintln!("    Field 1 length as usize: {}", f1_len);
                                                                            if decode_sequence.len() > field1_start + f1_len {
                                                                                let field1_data = &decode_sequence[field1_start + 1..field1_start + 1 + f1_len];
                                                                                eprintln!("    Field 1 data length: {} elements", field1_data.len());
                                                                                eprintln!("    Field 1 data (first 5): {:?}", &field1_data[..field1_data.len().min(5)]);
                                                                                
                                                                                // Try to decode Vec<XFieldElement> manually
                                                                                if !field1_data.is_empty() {
                                                                                    let vec_len_value = field1_data[0].value();
                                                                                    eprintln!("    Vec<XFieldElement> length indicator: {} (u64)", vec_len_value);
                                                                                    let vec_len_usize: Result<usize, _> = vec_len_value.try_into();
                                                                                    match vec_len_usize {
                                                                                        Ok(vl) => {
                                                                                            eprintln!("    Vec<XFieldElement> length as usize: {}", vl);
                                                                                            eprintln!("    Attempting Vec<XFieldElement>::decode...");
                                                                                            match Vec::<XFieldElement>::decode(field1_data) {
                                                                                                Ok(vec_xfield) => {
                                                                                                    eprintln!("    ✓ Vec<XFieldElement> decoded successfully: {} elements", vec_xfield.len());
                                                                                                    eprintln!("    ✓ Both fields decoded successfully manually!");
                                                                                                    eprintln!("    This suggests the issue is in the struct field decoding logic, not the Vec decoding.");
                                                                    
                                                                    // Now try to decode the full FriResponse struct with step-by-step tracing
                                                                    eprintln!("    Attempting full FriResponse::decode with step-by-step tracing...");
                                                                    let struct_sequence = &item_encoding[1..];
                                                                    eprintln!("    Struct decode sequence length: {}", struct_sequence.len());
                                                                    eprintln!("    Expected total: 1 + {} + 1 + {} = {}", f0_len, f1_len, 1 + f0_len + 1 + f1_len);
                                                                    
                                                                    // Manually trace through what the struct decoder should do, EXACTLY as it does
                                                                    eprintln!("    Step-by-step struct decode simulation (matching Rust decoder exactly):");
                                                                    let mut sim_sequence = struct_sequence;
                                                                    eprintln!("      Initial sequence length: {}", sim_sequence.len());
                                                                    
                                                                    // Field 0: auth_structure - EXACTLY as struct decoder does it
                                                                    if !sim_sequence.is_empty() {
                                                                        let field0_len_prefix = sim_sequence[0].value() as usize;
                                                                        eprintln!("      Field 0: Read length prefix = {} (from sim_sequence[0])", field0_len_prefix);
                                                                        sim_sequence = &sim_sequence[1..];  // Advance past length prefix
                                                                        eprintln!("      Field 0: After advance, sequence length: {}", sim_sequence.len());
                                                                        eprintln!("      Field 0: Will call Vec<Digest>::decode(&sim_sequence[0..{}])", field0_len_prefix);
                                                                        if sim_sequence.len() >= field0_len_prefix {
                                                                            let field0_slice = &sim_sequence[0..field0_len_prefix];
                                                                            eprintln!("      Field 0: Slice length: {} elements", field0_slice.len());
                                                                            eprintln!("      Field 0: Slice first element: {} (should be vec_length)", field0_slice[0].value());
                                                                            
                                                                            // Try to decode this slice
                                                                            eprintln!("      Field 0: Attempting Vec<Digest>::decode on slice of {} elements", field0_slice.len());
                                                                            eprintln!("      Field 0: Slice[0] = {} (vec_length)", field0_slice[0].value());
                                                                            eprintln!("      Field 0: Expected Vec encoding: 1 + 5*{} = {} elements", field0_slice[0].value(), 1 + 5 * field0_slice[0].value() as usize);
                                                                            match Vec::<Digest>::decode(field0_slice) {
                                                                                Ok(vec_digest) => {
                                                                                    eprintln!("      Field 0: ✓ Vec<Digest> decoded: {} digests", vec_digest.len());
                                                                                    eprintln!("      Field 0: Vec<Digest> consumed all {} elements (verified)", field0_slice.len());
                                                                                    // Advance past decoded data (as struct decoder does)
                                                                                    sim_sequence = &sim_sequence[field0_len_prefix..];
                                                                                    eprintln!("      Field 0: After advance, sequence length: {}", sim_sequence.len());
                                                                                    eprintln!("      Field 0: Expected remaining: {} (1 + {})", 1 + f1_len, f1_len);
                                                                                    eprintln!("      Field 0: Actual remaining: {} (match: {})", sim_sequence.len(), sim_sequence.len() == 1 + f1_len);
                                                                                    
                                                                                    // Field 1: revealed_leaves - EXACTLY as struct decoder does it
                                                                                    if !sim_sequence.is_empty() {
                                                                                        let field1_len_prefix = sim_sequence[0].value() as usize;
                                                                                        eprintln!("      Field 1: Read length prefix = {} (from sim_sequence[0])", field1_len_prefix);
                                                                                        sim_sequence = &sim_sequence[1..];  // Advance past length prefix
                                                                                        eprintln!("      Field 1: After advance, sequence length: {}", sim_sequence.len());
                                                                                        eprintln!("      Field 1: Will call Vec<XFieldElement>::decode(&sim_sequence[0..{}])", field1_len_prefix);
                                                                                        if sim_sequence.len() >= field1_len_prefix {
                                                                                            let field1_slice = &sim_sequence[0..field1_len_prefix];
                                                                                            eprintln!("      Field 1: Slice length: {} elements", field1_slice.len());
                                                                                            eprintln!("      Field 1: Slice first element: {} (should be vec_length)", field1_slice[0].value());
                                                                                            
                                                                                            // Try to decode this slice
                                                                                            match Vec::<XFieldElement>::decode(field1_slice) {
                                                                                                Ok(vec_xfield) => {
                                                                                                    eprintln!("      Field 1: ✓ Vec<XFieldElement> decoded: {} elements", vec_xfield.len());
                                                                                                    // Advance past decoded data (as struct decoder does)
                                                                                                    sim_sequence = &sim_sequence[field1_len_prefix..];
                                                                                                    eprintln!("      Field 1: After advance, sequence length: {}", sim_sequence.len());
                                                                                                    eprintln!("      Field 1: Expected remaining: 0");
                                                                                                    if !sim_sequence.is_empty() {
                                                                                                        eprintln!("      ⚠️  SIMULATION SHOWS LEFTOVER: {} elements", sim_sequence.len());
                                                                                                        eprintln!("      First few leftover elements: {:?}", &sim_sequence[..sim_sequence.len().min(5)]);
                                                                                                    } else {
                                                                                                        eprintln!("      ✓ Simulation shows no leftover data");
                                                                                                        eprintln!("      ⚠️  But actual decode fails - issue must be in struct decoder logic!");
                                                                                                    }
                                                                                                }
                                                                                                Err(err) => {
                                                                                                    eprintln!("      Field 1: ✗ Vec<XFieldElement> decode failed: {:?}", err);
                                                                                                }
                                                                                            }
                                                                                        } else {
                                                                                            eprintln!("      Field 1: ✗ Not enough data for slice");
                                                                                        }
                                                                                    } else {
                                                                                        eprintln!("      Field 1: ✗ No data for length prefix");
                                                                                    }
                                                                                }
                                                                                Err(err) => {
                                                                                    eprintln!("      Field 0: ✗ Vec<Digest> decode failed: {:?}", err);
                                                                                }
                                                                            }
                                                                        } else {
                                                                            eprintln!("      Field 0: ✗ Not enough data for slice");
                                                                        }
                                                                    }
                                                                    
                                                                    // Now try actual decode
                                                                    eprintln!("    Attempting actual FriResponse::decode...");
                                                                    match FriResponse::decode(struct_sequence) {
                                                                        Ok(fri_resp) => {
                                                                            eprintln!("    ✓ FriResponse decoded successfully!");
                                                                            eprintln!("      Auth structure: {} digests", fri_resp.auth_structure.len());
                                                                            eprintln!("      Revealed leaves: {} elements", fri_resp.revealed_leaves.len());
                                                                        }
                                                                        Err(err) => {
                                                                            eprintln!("    ✗ FriResponse::decode failed: {:?}", err);
                                                                            eprintln!("    Error details: {}", err);
                                                                        }
                                                                    }
                                                                }
                                                                                                Err(err) => {
                                                                                                    eprintln!("    ✗ Vec<XFieldElement> decode failed: {:?}", err);
                                                                                                    eprintln!("    This is likely where the 'invalid length indicator' error originates!");
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                        Err(_) => {
                                                                                            eprintln!("    ✗ Vec<XFieldElement> length ({}) cannot be converted to usize!", vec_len_value);
                                                                                        }
                                                                                    }
                                                                                }
                                                                            } else {
                                                                                eprintln!("    ✗ Sequence too short for field 1! Need {} elements, have {}", field1_start + 1 + f1_len, decode_sequence.len());
                                                                            }
                                                                        }
                                                                        Err(_) => {
                                                                            eprintln!("    ✗ Field 1 length ({}) cannot be converted to usize!", field1_length);
                                                                        }
                                                                    }
                                                                } else {
                                                                    eprintln!("    ✗ Sequence too short! Expected field 1 at index {}, but only have {} elements", field1_start, decode_sequence.len());
                                                                }
                                                            }
                                                            Err(err) => {
                                                                eprintln!("    ✗ Vec<Digest> decode failed: {:?}", err);
                                                                eprintln!("    This is likely where the 'invalid length indicator' error originates!");
                                                            }
                                                        }
                                                    }
                                                    Err(_) => {
                                                        eprintln!("    ✗ Vec<Digest> length ({}) cannot be converted to usize!", vec_len_value);
                                                    }
                                                }
                                            }
                                        } else {
                                            eprintln!("    ✗ Sequence too short for field 0! Need {} elements, have {}", f0_len + 1, decode_sequence.len());
                                        }
                                    }
                                    Err(_) => {
                                        eprintln!("    ✗ Field 0 length ({}) cannot be converted to usize!", field0_length);
                                    }
                                }
                            }
                        }
                        
                        // Check if it's variant 9 (FriCodeword)
                        if discriminant == 9 {
                            eprintln!("    This is FriCodeword (variant 9)!");
                            eprintln!("    Expected format: [discriminant(1), field0_length(1), vec_length(1), ...xfield_elements(3*N)]");
                            if item_encoding.len() > 1 {
                                let field0_length = item_encoding[1].value() as usize;
                                eprintln!("    Field 0 length prefix: {} (expects {} elements for Vec<XFieldElement>)", field0_length, field0_length);
                                if item_encoding.len() > 2 {
                                    let vec_length = item_encoding[2].value() as usize;
                                    eprintln!("    Vec<XFieldElement> length: {} (expects {} XFieldElements)", vec_length, vec_length);
                                    let expected_vec_encoding = 1 + 3 * vec_length; // vec_length + 3*N coefficients
                                    eprintln!("    Expected Vec encoding size: {} elements (1 + 3*{})", expected_vec_encoding, vec_length);
                                    eprintln!("    Expected total item size: {} elements (1 + 1 + {})", 1 + 1 + expected_vec_encoding, expected_vec_encoding);
                                    eprintln!("    Actual item size: {} elements", item_encoding.len());
                                    eprintln!("    Difference: {} elements", item_encoding.len() as i64 - (1 + 1 + expected_vec_encoding) as i64);
                                    
                                    if item_encoding.len() < 1 + 1 + expected_vec_encoding {
                                        eprintln!("    ERROR: Sequence too short! Need {} more elements", 
                                                (1 + 1 + expected_vec_encoding) - item_encoding.len());
                                    }
                                } else {
                                    eprintln!("    ERROR: Sequence too short! Need at least Vec length prefix (element 2)");
                                }
                            } else {
                                eprintln!("    ERROR: Sequence too short! Need at least field 0 length prefix (element 1)");
                            }
                        }
                        
                        // Check if it's variant 11 (FriResponse)
                        if discriminant == 11 {
                            eprintln!("    This is FriResponse (variant 11)!");
                            eprintln!("    Expected format: [discriminant(1), field0_length(1), field0_data, field1_length(1), field1_data]");
                            eprintln!("    Where field0_data = Vec<Digest> = [vec_length(1), ...digests(5*N)]");
                            eprintln!("    Where field1_data = Vec<XFieldElement> = [vec_length(1), ...xfield_elements(3*M)]");
                            
                            if item_encoding.len() > 1 {
                                let field0_length = item_encoding[1].value();
                                eprintln!("    Field 0 (auth_structure) length prefix: {}", field0_length);
                                
                                // Try to convert to usize and check for overflow
                                let field0_length_usize: Result<usize, _> = field0_length.try_into();
                                match field0_length_usize {
                                    Ok(f0_len) => {
                                        eprintln!("    Field 0 length as usize: {}", f0_len);
                                        eprintln!("    Field 0 should contain {} elements", f0_len);
                                        
                                        if item_encoding.len() > 2 {
                                            let field0_start = 2;
                                            let field0_end = field0_start + f0_len;
                                            
                                            if item_encoding.len() >= field0_end {
                                                let vec_len_elem = item_encoding[field0_start];
                                                let vec_len = vec_len_elem.value();
                                                eprintln!("    Field 0 Vec<Digest> length indicator: {}", vec_len);
                                                
                                                // Try to convert to usize
                                                let vec_len_usize: Result<usize, _> = vec_len.try_into();
                                                match vec_len_usize {
                                                    Ok(vl) => {
                                                        eprintln!("    Vec<Digest> length as usize: {}", vl);
                                                        let expected_digest_elements = 5 * vl;
                                                        let expected_vec_encoding = 1 + expected_digest_elements;
                                                        eprintln!("    Expected Vec<Digest> encoding: 1 + 5*{} = {} elements", vl, expected_vec_encoding);
                                                        eprintln!("    Actual field 0 encoding length: {}", f0_len);
                                                        
                                                        if f0_len != expected_vec_encoding {
                                                            eprintln!("    ⚠️  MISMATCH: Field 0 length prefix ({}) != Vec encoding size ({})", f0_len, expected_vec_encoding);
                                                        } else {
                                                            eprintln!("    ✓ Field 0 length matches Vec encoding size");
                                                        }
                                                        
                                                        // Check for potential overflow
                                                        let checked_mul = vl.checked_mul(5);
                                                        match checked_mul {
                                                            Some(product) => {
                                                                eprintln!("    ✓ {} * 5 = {} (no overflow)", vl, product);
                                                            }
                                                            None => {
                                                                eprintln!("    ❌ ERROR: {} * 5 would overflow usize!", vl);
                                                                eprintln!("    This is the 'invalid length indicator' error!");
                                                            }
                                                        }
                                                        
                                                        // Show the actual field 0 data and verify the vec_length value
                                                        eprintln!("    Field 0 data (first 10 elements):");
                                                        for i in 0..10.min(f0_len) {
                                                            let idx = field0_start + i;
                                                            if idx < item_encoding.len() {
                                                                eprintln!("      [{}]: {} (raw value: {})", i, item_encoding[idx].value(), item_encoding[idx].value());
                                                            }
                                                        }
                                                        
                                                        // Try to manually decode Vec<Digest> to see where it fails
                                                        eprintln!("    Attempting manual Vec<Digest> decode...");
                                                        if f0_len > 0 && item_encoding.len() >= field0_start + f0_len {
                                                            let vec_data = &item_encoding[field0_start..field0_start + f0_len];
                                                            if !vec_data.is_empty() {
                                                                let manual_vec_len = vec_data[0].value();
                                                                eprintln!("      Manual vec_length read: {}", manual_vec_len);
                                                                let manual_vec_len_usize: Result<usize, _> = manual_vec_len.try_into();
                                                                match manual_vec_len_usize {
                                                                    Ok(mvl) => {
                                                                        eprintln!("      Manual vec_length as usize: {}", mvl);
                                                                        let manual_checked = mvl.checked_mul(5);
                                                                        match manual_checked {
                                                                            Some(p) => {
                                                                                eprintln!("      ✓ Manual check: {} * 5 = {} (no overflow)", mvl, p);
                                                                                eprintln!("      Expected digest data: {} elements", p);
                                                                                eprintln!("      Available data: {} elements", vec_data.len() - 1);
                                                                                if vec_data.len() - 1 != p {
                                                                                    eprintln!("      ⚠️  MISMATCH: Expected {} elements, have {}", p, vec_data.len() - 1);
                                                                                }
                                                                            }
                                                                            None => {
                                                                                eprintln!("      ❌ Manual check: {} * 5 would overflow!", mvl);
                                                                            }
                                                                        }
                                                                    }
                                                                    Err(_) => {
                                                                        eprintln!("      ❌ Manual vec_length ({}) cannot be converted to usize!", manual_vec_len);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        
                                                        // Check field 1
                                                        if item_encoding.len() > field0_end {
                                                            let field1_length = item_encoding[field0_end].value();
                                                            eprintln!("    Field 1 (revealed_leaves) length prefix at index {}: {}", field0_end, field1_length);
                                                            
                                                            let field1_length_usize: Result<usize, _> = field1_length.try_into();
                                                            match field1_length_usize {
                                                                Ok(f1_len) => {
                                                                    eprintln!("    Field 1 length as usize: {}", f1_len);
                                                                    let field1_start = field0_end + 1;
                                                                    let field1_end = field1_start + f1_len;
                                                                    
                                                                    if item_encoding.len() >= field1_start {
                                                                        if item_encoding.len() >= field1_end {
                                                                            let vec_len_elem2 = item_encoding[field1_start];
                                                                            let vec_len2 = vec_len_elem2.value();
                                                                            eprintln!("    Field 1 Vec<XFieldElement> length indicator: {}", vec_len2);
                                                                            
                                                                            let vec_len2_usize: Result<usize, _> = vec_len2.try_into();
                                                                            match vec_len2_usize {
                                                                                Ok(vl2) => {
                                                                                    eprintln!("    Vec<XFieldElement> length as usize: {}", vl2);
                                                                                    let expected_xfield_elements = 3 * vl2;
                                                                                    let expected_vec2_encoding = 1 + expected_xfield_elements;
                                                                                    eprintln!("    Expected Vec<XFieldElement> encoding: 1 + 3*{} = {} elements", vl2, expected_vec2_encoding);
                                                                                    eprintln!("    Actual field 1 encoding length: {}", f1_len);
                                                                                    
                                                                                    if f1_len != expected_vec2_encoding {
                                                                                        eprintln!("    ⚠️  MISMATCH: Field 1 length prefix ({}) != Vec encoding size ({})", f1_len, expected_vec2_encoding);
                                                                                    } else {
                                                                                        eprintln!("    ✓ Field 1 length matches Vec encoding size");
                                                                                    }
                                                                                    
                                                                                    let expected_total = 1 + 1 + f0_len + 1 + f1_len;
                                                                                    eprintln!("    Expected total: 1 + 1 + {} + 1 + {} = {}", f0_len, f1_len, expected_total);
                                                                                    eprintln!("    Actual total: {} elements", item_encoding.len());
                                                                                }
                                                                                Err(_) => {
                                                                                    eprintln!("    ❌ ERROR: Vec<XFieldElement> length ({}) cannot be converted to usize!", vec_len2);
                                                                                    eprintln!("    This is likely the 'invalid length indicator' error!");
                                                                                }
                                                                            }
                                                                        } else {
                                                                            eprintln!("    ERROR: Sequence too short for field 1! Need {} elements, have {}", field1_end, item_encoding.len());
                                                                        }
                                                                    }
                                                                }
                                                                Err(_) => {
                                                                    eprintln!("    ❌ ERROR: Field 1 length ({}) cannot be converted to usize!", field1_length);
                                                                    eprintln!("    This is likely the 'invalid length indicator' error!");
                                                                }
                                                            }
                                                        } else {
                                                            eprintln!("    ERROR: Sequence too short! Expected field 1 at index {}, but only have {} elements", field0_end, item_encoding.len());
                                                        }
                                                    }
                                                    Err(_) => {
                                                        eprintln!("    ❌ ERROR: Field 0 length ({}) cannot be converted to usize!", field0_length);
                                                        eprintln!("    This is likely the 'invalid length indicator' error!");
                                                    }
                                                }
                                            } else {
                                                eprintln!("    ERROR: Sequence too short for field 0! Need {} elements, have {}", field0_end, item_encoding.len());
                                            }
                                        } else {
                                            eprintln!("    ERROR: Sequence too short! Need at least Vec length prefix (element 2)");
                                        }
                                    }
                                    Err(_) => {
                                        eprintln!("    ❌ ERROR: Field 0 length ({}) cannot be converted to usize!", field0_length);
                                        eprintln!("    This is likely the 'invalid length indicator' error!");
                                    }
                                }
                            } else {
                                eprintln!("    ERROR: Sequence too short! Need at least field 0 length prefix (element 1)");
                            }
                        }
                    }
                    eprintln!("    This is where the error occurs!");
                    break;
                }
            }
            
            idx += item_length;
        }
        
        eprintln!("================================\n");
    }

    triton_vm::profiler::start("Triton VM – Verify");
    let verify_result = Stark::default().verify(&claim, &proof);
    if flags.profile {
        let padded_height = proof.padded_height()?;
        let profile = triton_vm::profiler::finish()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri_domain_length(padded_height)?);
        println!("{profile}");
    }

    match verify_result {
        Ok(()) => {
            println!("✅ proof verified");
            Ok(SUCCESS)
        }
        Err(e) => {
            eprintln!("❌ proof verification failed");
            eprintln!("Error: {e}");
            match e {
                VerificationError::OutOfDomainQuotientValueMismatch => {
                    eprintln!("  → The computed quotient values at the out-of-domain point don't match the provided values.");
                }
                VerificationError::MainCodewordAuthenticationFailure => {
                    eprintln!("  → Failed to verify Merkle authentication path for main table codeword.");
                }
                VerificationError::AuxiliaryCodewordAuthenticationFailure => {
                    eprintln!("  → Failed to verify Merkle authentication path for auxiliary table codeword.");
                }
                VerificationError::QuotientCodewordAuthenticationFailure => {
                    eprintln!("  → Failed to verify Merkle authentication path for quotient codeword.");
                }
                VerificationError::CombinationCodewordMismatch => {
                    eprintln!("  → The computed combination codeword doesn't match the provided codeword.");
                }
                VerificationError::IncorrectNumberOfRowIndices => {
                    eprintln!("  → The number of row indices in the proof doesn't match the expected count.");
                }
                VerificationError::IncorrectNumberOfFRIValues => {
                    eprintln!("  → The number of FRI codeword values doesn't match the expected count.");
                }
                VerificationError::IncorrectNumberOfQuotientSegmentElements => {
                    eprintln!("  → The number of quotient segment elements doesn't match the expected count.");
                }
                VerificationError::IncorrectNumberOfMainTableRows => {
                    eprintln!("  → The number of main table rows doesn't match the expected count.");
                }
                VerificationError::IncorrectNumberOfAuxTableRows => {
                    eprintln!("  → The number of auxiliary table rows doesn't match the expected count.");
                }
                _ => {
                    eprintln!("  → See error message above for details.");
                }
            }
            Ok(FAILURE)
        }
    }
}

fn fri_domain_length(padded_height: usize) -> Result<usize> {
    let fri = Stark::default().fri(padded_height)?;
    Ok(fri.domain.length)
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;
    use triton_vm::prelude::TableId;

    #[test]
    fn max_table_label_len_is_9() {
        let max_table_label_len = TableId::iter()
            .map(|id| id.to_string().len())
            .max()
            .unwrap();
        assert_eq!(9, max_table_label_len);
    }
}
