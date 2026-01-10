use triton_vm::twenty_first::prelude::*;
use num_traits::Zero;

fn main() {
    // Test the complex expression at index 34
    // ((((((((((challenges[0] + main_row[33]) * challenges[0]) + main_row[34]) * challenges[0]) + main_row[35]) * challenges[0]) + main_row[36]) * challenges[0]) + main_row[37]) + (constant * challenges[62]))
    
    // Load test data
    use std::fs;
    let challenges_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("test_data/07_fiat_shamir_challenges.json").unwrap()
    ).unwrap();
    
    let main_lde_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("test_data/05_main_tables_lde.json").unwrap()
    ).unwrap();
    
    // Parse challenges
    let mut challenges = Vec::new();
    if let Some(challenge_vals) = challenges_json["challenge_values"].as_array() {
        for val_str in challenge_vals {
            let s = val_str.as_str().unwrap();
            // Parse XFieldElement from string
            use regex::Regex;
            let re = Regex::new(r"\((\d+)·x²\s*\+\s*(\d+)·x\s*\+\s*(\d+)\)").unwrap();
            if let Some(caps) = re.captures(s) {
                let c2: u64 = caps[1].parse().unwrap();
                let c1: u64 = caps[2].parse().unwrap();
                let c0: u64 = caps[3].parse().unwrap();
                challenges.push(XFieldElement::new([
                    BFieldElement::new(c0),
                    BFieldElement::new(c1),
                    BFieldElement::new(c2),
                ]));
            }
        }
    }
    
    // Get main_row[33-37] from row 0
    let main_row_0 = &main_lde_json["lde_table_data"][0];
    let main_33: u64 = main_row_0[33].as_u64().unwrap();
    let main_34: u64 = main_row_0[34].as_u64().unwrap();
    let main_35: u64 = main_row_0[35].as_u64().unwrap();
    let main_36: u64 = main_row_0[36].as_u64().unwrap();
    let main_37: u64 = main_row_0[37].as_u64().unwrap();
    
    let main_row_33 = XFieldElement::new([BFieldElement::new(main_33), BFieldElement::zero(), BFieldElement::zero()]);
    let main_row_34 = XFieldElement::new([BFieldElement::new(main_34), BFieldElement::zero(), BFieldElement::zero()]);
    let main_row_35 = XFieldElement::new([BFieldElement::new(main_35), BFieldElement::zero(), BFieldElement::zero()]);
    let main_row_36 = XFieldElement::new([BFieldElement::new(main_36), BFieldElement::zero(), BFieldElement::zero()]);
    let main_row_37 = XFieldElement::new([BFieldElement::new(main_37), BFieldElement::zero(), BFieldElement::zero()]);
    
    let constant = BFieldElement::from_raw_u64(18446744065119617026);
    
    // Evaluate the complex expression
    let ch0 = challenges[0];
    let ch62 = challenges[62];
    
    let expr = ((((((((((ch0 + main_row_33) * ch0) + main_row_34) * ch0) + main_row_35) * ch0) + main_row_36) * ch0) + main_row_37) + (constant * ch62));
    
    println!("Complex expression result: {}", expr);
    println!("Canonical: ({}, {}, {})", 
             expr.coefficients[0].value(),
             expr.coefficients[1].value(),
             expr.coefficients[2].value());
    
    // Expected from test data
    println!("Expected: (15943041338466224938, 3463472446969474349, 4400907353741795892)");
}
