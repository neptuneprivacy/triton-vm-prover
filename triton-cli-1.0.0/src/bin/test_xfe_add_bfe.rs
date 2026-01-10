use triton_vm::twenty_first::prelude::*;

fn main() {
    // Test: aux_row[2] + BFieldElement::from_raw_u64(18446744065119617026)
    
    // Create aux_row[2] from the test data
    // From test data: aux_row[2] = (5189108744467977098, 15112644096662456908, 15657451969740507096)
    let aux_row_2 = XFieldElement::new([
        BFieldElement::new(5189108744467977098),
        BFieldElement::new(15112644096662456908),
        BFieldElement::new(15657451969740507096),
    ]);
    
    println!("aux_row[2]: {}", aux_row_2);
    println!("aux_row[2] (hex): 0x{:016x} 0x{:016x} 0x{:016x}", 
             aux_row_2.coefficients[0].value(),
             aux_row_2.coefficients[1].value(),
             aux_row_2.coefficients[2].value());
    
    // Create the constant
    let constant = BFieldElement::from_raw_u64(18446744065119617026);
    println!("\nConstant from_raw_u64(18446744065119617026):");
    println!("  Montgomery: {}", constant.raw_u64());
    println!("  Canonical: {}", constant.value());
    println!("  Canonical (hex): 0x{:016x}", constant.value());
    
    // Add them
    let result = aux_row_2 + constant;
    println!("\nResult (aux_row[2] + constant):");
    println!("  {}", result);
    println!("  (hex): 0x{:016x} 0x{:016x} 0x{:016x}", 
             result.coefficients[0].value(),
             result.coefficients[1].value(),
             result.coefficients[2].value());
    
    // Expected from test data
    println!("\nExpected from test data:");
    println!("  (15943041338466224938, 3463472446969474349, 4400907353741795892)");
    println!("  (hex): 0x{:016x} 0x{:016x} 0x{:016x}", 
             15943041338466224938u64,
             3463472446969474349u64,
             4400907353741795892u64);
}
