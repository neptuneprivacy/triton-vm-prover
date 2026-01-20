use triton_vm::twenty_first::prelude::*;

fn main() {
    // Test: aux_row[2] + BFieldElement::from_raw_u64(18446744065119617026)
    
    // aux_row[2] from test data (canonical values)
    // (15657451969740507096·x² + 15112644096662456908·x + 05189108744467977098)
    let aux_c0_canonical = 5189108744467977098u64;
    let aux_c1_canonical = 15112644096662456908u64;
    let aux_c2_canonical = 15657451969740507096u64;
    
    // Convert to Montgomery (BFieldElement::new converts canonical to Montgomery)
    let aux_row_2 = XFieldElement::new([
        BFieldElement::new(aux_c0_canonical),
        BFieldElement::new(aux_c1_canonical),
        BFieldElement::new(aux_c2_canonical),
    ]);
    
    println!("aux_row[2] (from canonical):");
    println!("  Internal (Montgomery): ({}, {}, {})",
             aux_row_2.coefficients[0].raw_u64(),
             aux_row_2.coefficients[1].raw_u64(),
             aux_row_2.coefficients[2].raw_u64());
    println!("  Display (canonical): {}", aux_row_2);
    
    let constant = BFieldElement::from_raw_u64(18446744065119617026);
    println!("\nConstant from_raw_u64(18446744065119617026):");
    println!("  Internal (Montgomery): {}", constant.raw_u64());
    println!("  Display (canonical): {}", constant.value());
    
    // Add them (addition happens in Montgomery)
    let result = aux_row_2 + constant;
    println!("\nResult (aux_row[2] + constant):");
    println!("  Internal (Montgomery): ({}, {}, {})",
             result.coefficients[0].raw_u64(),
             result.coefficients[1].raw_u64(),
             result.coefficients[2].raw_u64());
    println!("  Display (canonical): {}", result);
    println!("  Canonical values: ({}, {}, {})",
             result.coefficients[0].value(),
             result.coefficients[1].value(),
             result.coefficients[2].value());
    
    // Expected from test data
    println!("\nExpected from test data:");
    println!("  (15943041338466224938, 3463472446969474349, 4400907353741795892)");
    
    let expected_c0 = 15943041338466224938u64;
    let expected_c1 = 3463472446969474349u64;
    let expected_c2 = 4400907353741795892u64;
    
    let actual_c0 = result.coefficients[0].value();
    let actual_c1 = result.coefficients[1].value();
    let actual_c2 = result.coefficients[2].value();
    
    println!("\nComparison:");
    println!("  c0: actual={}, expected={}, match={}", actual_c0, expected_c0, actual_c0 == expected_c0);
    println!("  c1: actual={}, expected={}, match={}", actual_c1, expected_c1, actual_c1 == expected_c1);
    println!("  c2: actual={}, expected={}, match={}", actual_c2, expected_c2, actual_c2 == expected_c2);
}
