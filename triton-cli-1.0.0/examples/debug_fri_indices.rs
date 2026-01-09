use triton_vm::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a proof and analyze FRI indices
    let program_source = std::fs::read_to_string("spin.tasm")?;
    let program = Program::from_code(&program_source)?;
    let input = PublicInput::new(vec![BFieldElement::new(8)]);
    let claim = Claim::about_program(&program).with_input(input.individual_tokens.clone());
    
    let (aet, public_output) = VM::trace_execution(program, input, NonDeterminism::default())?;
    let claim = claim.with_output(public_output);
    
    let proof = Stark::default().prove(&claim, &aet)?;
    let proof_stream = ProofStream::try_from(&proof)?;
    
    println!("=== FRI INDICES ANALYSIS ===");
    
    // Calculate expected num_collinearity_checks
    let padded_height = aet.padded_height();
    let stark = Stark::default();
    let fri = stark.fri(padded_height)?;
    let security_level = 160;
    let log2_fri_expansion_factor = 3; // log2(8) = 3
    let num_collinearity_checks = std::cmp::max(1, security_level / log2_fri_expansion_factor);
    
    println!("Expected num_collinearity_checks: {}", num_collinearity_checks);
    println!();
    
    // Check what was actually revealed
    for (i, item) in proof_stream.items.iter().enumerate() {
        match item {
            ProofItem::FriResponse(resp) => {
                println!("FriResponse item {}: {} revealed leaves", i, resp.revealed_leaves.len());
                println!("  Auth structure: {} digests", resp.auth_structure.len());
            }
            ProofItem::MasterMainTableRows(rows) => {
                println!("MasterMainTableRows item {}: {} rows", i, rows.len());
                println!("  Expected: {} rows (from num_collinearity_checks)", num_collinearity_checks);
                if rows.len() != num_collinearity_checks {
                    println!("  ⚠️  MISMATCH: Expected {}, got {}", num_collinearity_checks, rows.len());
                }
            }
            ProofItem::MasterAuxTableRows(rows) => {
                println!("MasterAuxTableRows item {}: {} rows", i, rows.len());
            }
            ProofItem::QuotientSegmentsElements(segs) => {
                println!("QuotientSegmentsElements item {}: {} segments", i, segs.len());
            }
            _ => {}
        }
    }
    
    println!();
    println!("Key question: How do {} FRI indices become {} revealed rows?", 
             num_collinearity_checks, 
             proof_stream.items.iter()
                 .find_map(|item| {
                     if let ProofItem::MasterMainTableRows(rows) = item {
                         Some(rows.len())
                     } else {
                         None
                     }
                 })
                 .unwrap_or(0));
    
    Ok(())
}
