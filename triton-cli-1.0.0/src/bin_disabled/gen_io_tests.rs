//! Generate complete input/output test data for C++ verification.
//! 
//! This generates focused test cases where C++ can:
//! 1. Load the inputs
//! 2. Compute the output
//! 3. Compare 100% with Rust output

use std::fs;
use std::path::PathBuf;
use anyhow::Result;
use triton_vm::prelude::*;
use triton_vm::arithmetic_domain::ArithmeticDomain;
use triton_vm::stark::Stark;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::util_types::merkle_tree::MerkleTree;
use serde_json;

fn main() -> Result<()> {
    let output_dir = PathBuf::from("test_data_io");
    fs::create_dir_all(&output_dir)?;
    
    println!("Generating I/O test data for C++ verification...\n");
    
    // 1. Tip5 hash tests
    generate_tip5_tests(&output_dir)?;
    
    // 2. Domain computation tests
    generate_domain_tests(&output_dir)?;
    
    // 3. Merkle tree tests
    generate_merkle_tests(&output_dir)?;
    
    // 4. FRI folding tests  
    generate_fri_tests(&output_dir)?;
    
    // 5. BFieldElement arithmetic tests
    generate_bfield_tests(&output_dir)?;
    
    // 6. XFieldElement arithmetic tests
    generate_xfield_tests(&output_dir)?;
    
    println!("\n✅ All I/O test data generated in: {}", output_dir.display());
    Ok(())
}

fn generate_tip5_tests(dir: &PathBuf) -> Result<()> {
    println!("[1/6] Generating Tip5 hash tests...");
    
    let tip5_dir = dir.join("tip5");
    fs::create_dir_all(&tip5_dir)?;
    
    // Test 1: Hash empty
    let empty: Vec<BFieldElement> = vec![];
    let hash_empty = Tip5::hash_varlen(&empty);
    dump_tip5_test(&tip5_dir, "hash_empty", &empty, hash_empty)?;
    
    // Test 2: Hash single element
    let single = vec![BFieldElement::new(12345)];
    let hash_single = Tip5::hash_varlen(&single);
    dump_tip5_test(&tip5_dir, "hash_single", &single, hash_single)?;
    
    // Test 3: Hash 10 elements
    let ten: Vec<BFieldElement> = (0..10).map(|i| BFieldElement::new(i * 7 + 3)).collect();
    let hash_ten = Tip5::hash_varlen(&ten);
    dump_tip5_test(&tip5_dir, "hash_ten", &ten, hash_ten)?;
    
    // Test 4: Hash pair of digests
    let d1 = Digest::new([
        BFieldElement::new(1),
        BFieldElement::new(2),
        BFieldElement::new(3),
        BFieldElement::new(4),
        BFieldElement::new(5),
    ]);
    let d2 = Digest::new([
        BFieldElement::new(6),
        BFieldElement::new(7),
        BFieldElement::new(8),
        BFieldElement::new(9),
        BFieldElement::new(10),
    ]);
    let hash_pair = Tip5::hash_pair(d1, d2);
    dump_tip5_pair_test(&tip5_dir, "hash_pair", d1, d2, hash_pair)?;
    
    // Test 5: Hash 379 elements (main table row size)
    let row_379: Vec<BFieldElement> = (0..379).map(|i| BFieldElement::new(i * 13 + 7)).collect();
    let hash_row = Tip5::hash_varlen(&row_379);
    dump_tip5_test(&tip5_dir, "hash_row_379", &row_379, hash_row)?;
    
    println!("   ✓ Generated 5 Tip5 tests");
    Ok(())
}

fn generate_domain_tests(dir: &PathBuf) -> Result<()> {
    println!("[2/6] Generating domain computation tests...");
    
    let domain_dir = dir.join("domain");
    fs::create_dir_all(&domain_dir)?;
    
    // Test various domain sizes
    for log2_len in [8, 9, 10, 12] {
        let len = 1usize << log2_len;
        let domain = ArithmeticDomain::of_length(len)?;
        dump_domain_test(&domain_dir, len, domain)?;
    }
    
    // Test domain with offset (use generator() as FRI does)
    let domain_4096 = ArithmeticDomain::of_length(4096)?;
    let offset = BFieldElement::generator();
    let domain_with_offset = domain_4096.with_offset(offset);
    dump_domain_with_offset_test(&domain_dir, 4096, offset, domain_with_offset)?;
    
    // Test ProverDomains
    let padded_height = 512;
    let num_trace_randomizers = 2;
    let fri_domain = ArithmeticDomain::of_length(4096)?;
    let fri_offset = BFieldElement::generator();
    let fri_domain = fri_domain.with_offset(fri_offset);
    let _stark = Stark::default();
    
    // Note: ProverDomains::derive is private, so we'll compute the expected values
    let trace_domain = ArithmeticDomain::of_length(padded_height)?;
    let randomized_trace_len = padded_height + num_trace_randomizers;
    let randomized_trace_domain = ArithmeticDomain::of_length(randomized_trace_len.next_power_of_two())?;
    
    dump_prover_domains_test(&domain_dir, padded_height, num_trace_randomizers, 
                             trace_domain, randomized_trace_domain, fri_domain)?;
    
    println!("   ✓ Generated domain computation tests");
    Ok(())
}

fn generate_merkle_tests(dir: &PathBuf) -> Result<()> {
    println!("[3/6] Generating Merkle tree tests...");
    
    let merkle_dir = dir.join("merkle");
    fs::create_dir_all(&merkle_dir)?;
    
    // Test 1: 4 leaves
    let leaves_4: Vec<Digest> = (0..4).map(|i| {
        Digest::new([
            BFieldElement::new(i * 5 + 1),
            BFieldElement::new(i * 5 + 2),
            BFieldElement::new(i * 5 + 3),
            BFieldElement::new(i * 5 + 4),
            BFieldElement::new(i * 5 + 5),
        ])
    }).collect();
    let tree_4 = MerkleTree::par_new(&leaves_4)?;
    dump_merkle_test(&merkle_dir, "tree_4_leaves", &leaves_4, tree_4.root())?;
    
    // Test 2: 8 leaves
    let leaves_8: Vec<Digest> = (0..8).map(|i| {
        Digest::new([
            BFieldElement::new(i * 11 + 1),
            BFieldElement::new(i * 11 + 2),
            BFieldElement::new(i * 11 + 3),
            BFieldElement::new(i * 11 + 4),
            BFieldElement::new(i * 11 + 5),
        ])
    }).collect();
    let tree_8 = MerkleTree::par_new(&leaves_8)?;
    dump_merkle_test(&merkle_dir, "tree_8_leaves", &leaves_8, tree_8.root())?;
    
    // Test 3: Authentication path
    dump_merkle_auth_path(&merkle_dir, &tree_8, 3)?;
    
    println!("   ✓ Generated Merkle tree tests");
    Ok(())
}

fn generate_fri_tests(dir: &PathBuf) -> Result<()> {
    println!("[4/6] Generating FRI folding tests...");
    
    let fri_dir = dir.join("fri");
    fs::create_dir_all(&fri_dir)?;
    
    // Create a simple codeword and fold it
    let codeword: Vec<XFieldElement> = (0..16).map(|i| {
        XFieldElement::new([
            BFieldElement::new(i * 7 + 1),
            BFieldElement::new(i * 3 + 2),
            BFieldElement::new(i * 11 + 3),
        ])
    }).collect();
    
    let challenge = XFieldElement::new([
        BFieldElement::new(12345),
        BFieldElement::new(67890),
        BFieldElement::new(11111),
    ]);
    
    // Manual FRI fold (split and fold)
    let domain = ArithmeticDomain::of_length(16)?;
    let half_n = 8;
    
    // Compute folded codeword
    let mut folded = Vec::with_capacity(half_n);
    for i in 0..half_n {
        let omega_i = domain.domain_value(i as u32);
        let left = codeword[i];
        let right = codeword[i + half_n];
        
        // FRI fold formula: (left + right + challenge * omega_i^(-1) * (left - right)) / 2
        let omega_inv = omega_i.inverse();
        let sum = left + right;
        let diff = left - right;
        let scaled_diff = XFieldElement::from(omega_inv) * diff * challenge;
        let folded_val = (sum + scaled_diff) * XFieldElement::from(BFieldElement::new(2)).inverse();
        folded.push(folded_val);
    }
    
    dump_fri_fold_test(&fri_dir, &codeword, challenge, &folded)?;
    
    println!("   ✓ Generated FRI folding tests");
    Ok(())
}

fn generate_bfield_tests(dir: &PathBuf) -> Result<()> {
    println!("[5/6] Generating BFieldElement arithmetic tests...");
    
    let bfield_dir = dir.join("bfield");
    fs::create_dir_all(&bfield_dir)?;
    
    // Test values
    let a = BFieldElement::new(12345678901234567890);
    let b = BFieldElement::new(9876543210987654321);
    let c = BFieldElement::new(42);
    
    let data = serde_json::json!({
        "modulus": BFieldElement::P,
        "tests": {
            "addition": {
                "a": a.value(),
                "b": b.value(),
                "result": (a + b).value(),
            },
            "subtraction": {
                "a": a.value(),
                "b": b.value(),
                "result": (a - b).value(),
            },
            "multiplication": {
                "a": a.value(),
                "b": b.value(),
                "result": (a * b).value(),
            },
            "negation": {
                "a": c.value(),
                "result": (-c).value(),
            },
            "inverse": {
                "a": c.value(),
                "result": c.inverse().value(),
            },
            "power": {
                "base": c.value(),
                "exponent": 10,
                "result": c.mod_pow(10).value(),
            },
            "primitive_root_512": {
                "n": 512,
                "result": BFieldElement::primitive_root_of_unity(512).unwrap().value(),
            },
            "primitive_root_4096": {
                "n": 4096,
                "result": BFieldElement::primitive_root_of_unity(4096).unwrap().value(),
            },
        }
    });
    
    fs::write(bfield_dir.join("arithmetic.json"), serde_json::to_string_pretty(&data)?)?;
    
    println!("   ✓ Generated BFieldElement arithmetic tests");
    Ok(())
}

fn generate_xfield_tests(dir: &PathBuf) -> Result<()> {
    println!("[6/6] Generating XFieldElement arithmetic tests...");
    
    let xfield_dir = dir.join("xfield");
    fs::create_dir_all(&xfield_dir)?;
    
    let a = XFieldElement::new([
        BFieldElement::new(123456),
        BFieldElement::new(789012),
        BFieldElement::new(345678),
    ]);
    let b = XFieldElement::new([
        BFieldElement::new(111111),
        BFieldElement::new(222222),
        BFieldElement::new(333333),
    ]);
    
    let data = serde_json::json!({
        "shah_polynomial": "x^3 - x + 1 = 0, so x^3 = x - 1",
        "tests": {
            "addition": {
                "a": [a.coefficients[0].value(), a.coefficients[1].value(), a.coefficients[2].value()],
                "b": [b.coefficients[0].value(), b.coefficients[1].value(), b.coefficients[2].value()],
                "result": [
                    (a + b).coefficients[0].value(),
                    (a + b).coefficients[1].value(),
                    (a + b).coefficients[2].value(),
                ],
            },
            "multiplication": {
                "a": [a.coefficients[0].value(), a.coefficients[1].value(), a.coefficients[2].value()],
                "b": [b.coefficients[0].value(), b.coefficients[1].value(), b.coefficients[2].value()],
                "result": [
                    (a * b).coefficients[0].value(),
                    (a * b).coefficients[1].value(),
                    (a * b).coefficients[2].value(),
                ],
            },
            "inverse": {
                "a": [a.coefficients[0].value(), a.coefficients[1].value(), a.coefficients[2].value()],
                "result": [
                    a.inverse().coefficients[0].value(),
                    a.inverse().coefficients[1].value(),
                    a.inverse().coefficients[2].value(),
                ],
            },
            "x_cubed_equals_x_minus_one": {
                "x": [0, 1, 0],  // x
                "x_cubed": {
                    // x^3 = x - 1, so x^3 has coefficients [-1, 1, 0] = [P-1, 1, 0]
                    "result": [
                        (BFieldElement::new(0) - BFieldElement::new(1)).value(),
                        1,
                        0,
                    ],
                }
            },
        }
    });
    
    fs::write(xfield_dir.join("arithmetic.json"), serde_json::to_string_pretty(&data)?)?;
    
    println!("   ✓ Generated XFieldElement arithmetic tests");
    Ok(())
}

// Helper functions for dumping test data

fn dump_tip5_test(dir: &PathBuf, name: &str, input: &[BFieldElement], output: Digest) -> Result<()> {
    let data = serde_json::json!({
        "test": name,
        "input": {
            "elements": input.iter().map(|x| x.value()).collect::<Vec<u64>>(),
            "count": input.len(),
        },
        "output": {
            "digest": [
                output.0[0].value(),
                output.0[1].value(),
                output.0[2].value(),
                output.0[3].value(),
                output.0[4].value(),
            ],
        },
    });
    fs::write(dir.join(format!("{}.json", name)), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn dump_tip5_pair_test(dir: &PathBuf, name: &str, d1: Digest, d2: Digest, output: Digest) -> Result<()> {
    let data = serde_json::json!({
        "test": name,
        "input": {
            "left": [d1.0[0].value(), d1.0[1].value(), d1.0[2].value(), d1.0[3].value(), d1.0[4].value()],
            "right": [d2.0[0].value(), d2.0[1].value(), d2.0[2].value(), d2.0[3].value(), d2.0[4].value()],
        },
        "output": {
            "digest": [
                output.0[0].value(),
                output.0[1].value(),
                output.0[2].value(),
                output.0[3].value(),
                output.0[4].value(),
            ],
        },
    });
    fs::write(dir.join(format!("{}.json", name)), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn dump_domain_test(dir: &PathBuf, length: usize, domain: ArithmeticDomain) -> Result<()> {
    let data = serde_json::json!({
        "test": format!("domain_{}", length),
        "input": {
            "length": length,
        },
        "output": {
            "length": domain.length,
            "offset": domain.offset.value(),
            "generator": domain.generator.value(),
        },
    });
    fs::write(dir.join(format!("domain_{}.json", length)), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn dump_domain_with_offset_test(dir: &PathBuf, length: usize, offset: BFieldElement, domain: ArithmeticDomain) -> Result<()> {
    let data = serde_json::json!({
        "test": format!("domain_{}_with_offset", length),
        "input": {
            "length": length,
            "offset": offset.value(),
        },
        "output": {
            "length": domain.length,
            "offset": domain.offset.value(),
            "generator": domain.generator.value(),
        },
    });
    fs::write(dir.join(format!("domain_{}_with_offset.json", length)), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn dump_prover_domains_test(
    dir: &PathBuf,
    padded_height: usize,
    num_trace_randomizers: usize,
    trace: ArithmeticDomain,
    randomized_trace: ArithmeticDomain,
    fri: ArithmeticDomain,
) -> Result<()> {
    let data = serde_json::json!({
        "test": "prover_domains",
        "input": {
            "padded_height": padded_height,
            "num_trace_randomizers": num_trace_randomizers,
        },
        "output": {
            "trace": {
                "length": trace.length,
                "offset": trace.offset.value(),
                "generator": trace.generator.value(),
            },
            "randomized_trace": {
                "length": randomized_trace.length,
                "offset": randomized_trace.offset.value(),
                "generator": randomized_trace.generator.value(),
            },
            "fri": {
                "length": fri.length,
                "offset": fri.offset.value(),
                "generator": fri.generator.value(),
            },
        },
    });
    fs::write(dir.join("prover_domains.json"), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn dump_merkle_test(dir: &PathBuf, name: &str, leaves: &[Digest], root: Digest) -> Result<()> {
    let data = serde_json::json!({
        "test": name,
        "input": {
            "leaf_count": leaves.len(),
            "leaves": leaves.iter().map(|d| [
                d.0[0].value(), d.0[1].value(), d.0[2].value(), d.0[3].value(), d.0[4].value()
            ]).collect::<Vec<_>>(),
        },
        "output": {
            "root": [
                root.0[0].value(), root.0[1].value(), root.0[2].value(), root.0[3].value(), root.0[4].value()
            ],
        },
    });
    fs::write(dir.join(format!("{}.json", name)), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn dump_merkle_auth_path(dir: &PathBuf, tree: &MerkleTree, leaf_index: usize) -> Result<()> {
    let auth_path = tree.authentication_structure(&[leaf_index])?;
    
    let data = serde_json::json!({
        "test": format!("auth_path_leaf_{}", leaf_index),
        "input": {
            "num_leaves": tree.num_leafs(),
            "leaf_index": leaf_index,
            "root": [
                tree.root().0[0].value(),
                tree.root().0[1].value(),
                tree.root().0[2].value(),
                tree.root().0[3].value(),
                tree.root().0[4].value(),
            ],
        },
        "output": {
            "auth_path": auth_path.iter().map(|d| [
                d.0[0].value(), d.0[1].value(), d.0[2].value(), d.0[3].value(), d.0[4].value()
            ]).collect::<Vec<_>>(),
        },
    });
    fs::write(dir.join(format!("auth_path_leaf_{}.json", leaf_index)), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn dump_fri_fold_test(
    dir: &PathBuf,
    codeword: &[XFieldElement],
    challenge: XFieldElement,
    folded: &[XFieldElement],
) -> Result<()> {
    let data = serde_json::json!({
        "test": "fri_fold",
        "input": {
            "codeword_length": codeword.len(),
            "codeword": codeword.iter().map(|x| [
                x.coefficients[0].value(),
                x.coefficients[1].value(),
                x.coefficients[2].value(),
            ]).collect::<Vec<_>>(),
            "challenge": [
                challenge.coefficients[0].value(),
                challenge.coefficients[1].value(),
                challenge.coefficients[2].value(),
            ],
        },
        "output": {
            "folded_length": folded.len(),
            "folded": folded.iter().map(|x| [
                x.coefficients[0].value(),
                x.coefficients[1].value(),
                x.coefficients[2].value(),
            ]).collect::<Vec<_>>(),
        },
    });
    fs::write(dir.join("fri_fold.json"), serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

