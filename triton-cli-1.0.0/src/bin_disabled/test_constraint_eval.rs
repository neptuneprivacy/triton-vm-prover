use triton_vm::twenty_first::prelude::*;
use triton_vm::table::master_table::MasterAuxTable;
use ndarray::ArrayView1;

fn main() {
    // Load the actual test data to get the exact values
    use std::fs;
    let quotient_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("test_data/10_quotient_calculation.json").unwrap()
    ).unwrap();
    
    let aux_lde_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("test_data/08_aux_tables_lde.json").unwrap()
    ).unwrap();
    
    let main_lde_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("test_data/05_main_tables_lde.json").unwrap()
    ).unwrap();
    
    // Get row 0 values
    let aux_row_0 = &aux_lde_json["aux_lde_table_data"][0];
    let main_row_0 = &main_lde_json["lde_table_data"][0];
    
    // Parse aux_row[2]
    let aux_row_2_str = aux_row_0[2].as_str().unwrap();
    println!("aux_row[2] from test data: {}", aux_row_2_str);
    
    // For now, let's just test the simple addition
    // aux_row[2] + BFieldElement::from_raw_u64(18446744065119617026)
    
    // Create aux_row[2] from the string
    // Format: (c2·x² + c1·x + c0)
    use regex::Regex;
    let re = Regex::new(r"\((\d+)·x²\s*\+\s*(\d+)·x\s*\+\s*(\d+)\)").unwrap();
    if let Some(caps) = re.captures(aux_row_2_str) {
        let c2: u64 = caps[1].parse().unwrap();
        let c1: u64 = caps[2].parse().unwrap();
        let c0: u64 = caps[3].parse().unwrap();
        
        // These are canonical values from the dump
        // We need to convert them to Montgomery for internal representation
        let aux_row_2 = XFieldElement::new([
            BFieldElement::new(c0),
            BFieldElement::new(c1),
            BFieldElement::new(c2),
        ]);
        
        println!("aux_row[2] as XFieldElement: {}", aux_row_2);
        
        let constant = BFieldElement::from_raw_u64(18446744065119617026);
        println!("Constant (Montgomery): {}", constant.raw_u64());
        println!("Constant (Canonical): {}", constant.value());
        
        let result = aux_row_2 + constant;
        println!("Result: {}", result);
        println!("Result (canonical): ({}, {}, {})", 
                 result.coefficients[0].value(),
                 result.coefficients[1].value(),
                 result.coefficients[2].value());
        
        // Expected from test data
        let expected_str = quotient_json["first_row_constraint_evaluation"]["initial_constraints"][34].as_str().unwrap();
        println!("\nExpected from test data: {}", expected_str);
    }
}
