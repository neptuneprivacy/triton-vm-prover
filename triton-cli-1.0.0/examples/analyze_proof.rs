use triton_vm::prelude::*;
use triton_vm::proof_stream::ProofStream;
use triton_vm::proof_item::ProofItem;
use std::fs::File;
use std::io::BufReader;
use bincode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load proof
    let proof_file = File::open("debug_rust.proof")?;
    let proof: Proof = bincode::deserialize_from(BufReader::new(proof_file))?;
    
    println!("=== PROOF FILE ANALYSIS ===");
    println!("Proof.0 length: {} elements", proof.0.len());
    println!();
    
    // Decode proof stream
    let proof_stream = ProofStream::try_from(&proof)?;
    
    println!("=== PROOF STREAM ANALYSIS ===");
    println!("Total items: {}", proof_stream.items.len());
    println!();
    
    // Analyze each item
    let mut total_item_encoding = 0;
    for (i, item) in proof_stream.items.iter().enumerate() {
        let item_encoding = item.encode();
        total_item_encoding += item_encoding.len();
        
        let item_type = match item {
            ProofItem::MerkleRoot(_) => "MerkleRoot",
            ProofItem::OutOfDomainMainRow(_) => "OutOfDomainMainRow",
            ProofItem::OutOfDomainAuxRow(_) => "OutOfDomainAuxRow",
            ProofItem::OutOfDomainQuotientSegments(_) => "OutOfDomainQuotientSegments",
            ProofItem::AuthenticationStructure(_) => "AuthenticationStructure",
            ProofItem::MasterMainTableRows(_) => "MasterMainTableRows",
            ProofItem::MasterAuxTableRows(_) => "MasterAuxTableRows",
            ProofItem::Log2PaddedHeight(_) => "Log2PaddedHeight",
            ProofItem::QuotientSegmentsElements(_) => "QuotientSegmentsElements",
            ProofItem::FriCodeword(_) => "FriCodeword",
            ProofItem::FriPolynomial(_) => "FriPolynomial",
            ProofItem::FriResponse(_) => "FriResponse",
        };
        
        println!("Item {}: {} (encoding: {} elements)", i, item_type, item_encoding.len());
    }
    
    println!();
    println!("=== SUMMARY ===");
    println!("Total items: {}", proof_stream.items.len());
    println!("Total item encoding size: {} elements", total_item_encoding);
    
    // Encode proof stream
    let proof_stream_encoding = proof_stream.encode();
    println!("Proof stream encoding size: {} elements", proof_stream_encoding.len());
    println!("Proof.0 size: {} elements", proof.0.len());
    println!();
    
    // Show structure
    println!("Proof stream encoding structure:");
    println!("  Element 0 (Vec<ProofItem> length): {}", proof_stream_encoding[0].value());
    
    // Parse with length prefixes
    let mut idx = 1;
    let mut item_sizes = Vec::new();
    for i in 0..proof_stream.items.len() {
        if idx >= proof_stream_encoding.len() {
            break;
        }
        let item_length = proof_stream_encoding[idx].value() as usize;
        item_sizes.push((i, item_length));
        idx += 1 + item_length;
    }
    
    println!("  Per-item length prefixes and sizes:");
    for (i, size) in item_sizes {
        println!("    Item {}: length prefix = {} elements", i, size);
    }
    
    println!();
    println!("Total with prefixes: 1 (Vec length) + {} (item prefixes) + {} (item encodings) = {} elements",
        proof_stream.items.len(),
        total_item_encoding,
        1 + proof_stream.items.len() + total_item_encoding);
    
    Ok(())
}
