// Temporary debug script to understand proof encoding
use triton_vm::prelude::*;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load program
    let program_source = fs::read_to_string("spin.tasm")?;
    let program = Program::from_code(&program_source)?;
    
    // Create claim
    let input = vec![BFieldElement::new(8)];
    let claim = Claim::about_program(&program).with_input(input.clone());
    
    // Execute and prove
    let (aet, public_output) = VM::trace_execution(program, input, NonDeterminism::default())?;
    let claim = claim.with_output(public_output);
    
    // Generate proof
    let proof = Stark::default().prove(&claim, &aet)?;
    
    // Decode proof stream to analyze structure
    let proof_stream = ProofStream::try_from(&proof)?;
    
    println!("=== PROOF STREAM ANALYSIS ===");
    println!("Total items: {}", proof_stream.items.len());
    println!();
    
    // Analyze each item
    let mut total_encoding_size = 0;
    for (i, item) in proof_stream.items.iter().enumerate() {
        let item_encoding = item.encode();
        total_encoding_size += item_encoding.len();
        
        println!("Item {}: {:?}", i, item);
        println!("  Encoding size: {} elements", item_encoding.len());
        
        // Show first few elements for large items
        if item_encoding.len() > 10 {
            println!("  First 5 elements: {:?}", &item_encoding[..5]);
            println!("  Last 5 elements: {:?}", &item_encoding[item_encoding.len()-5..]);
        } else {
            println!("  All elements: {:?}", item_encoding);
        }
        println!();
    }
    
    println!("=== SUMMARY ===");
    println!("Total items: {}", proof_stream.items.len());
    println!("Total item encoding size: {} elements", total_encoding_size);
    println!("Proof stream encoding size: {} elements", proof_stream.encode().len());
    println!("Proof.0 size: {} elements", proof.0.len());
    println!();
    
    // Show proof stream encoding structure
    let proof_stream_encoding = proof_stream.encode();
    println!("Proof stream encoding structure:");
    println!("  Element 0 (Vec<ProofItem> length): {}", proof_stream_encoding[0].value());
    println!("  Total encoding size: {} elements", proof_stream_encoding.len());
    println!();
    
    // Parse items with length prefixes
    let mut idx = 1;
    for i in 0..proof_stream.items.len() {
        if idx >= proof_stream_encoding.len() {
            break;
        }
        let item_length = proof_stream_encoding[idx].value() as usize;
        idx += 1;
        println!("  Item {} length prefix: {} elements", i, item_length);
        if idx + item_length <= proof_stream_encoding.len() {
            println!("    First 3 elements: {:?}", 
                &proof_stream_encoding[idx..idx+std::cmp::min(3, item_length)]
                    .iter().map(|e| e.value()).collect::<Vec<_>>());
        }
        idx += item_length;
    }
    
    Ok(())
}
